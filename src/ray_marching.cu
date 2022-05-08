#include <iostream>

#include "utils.h"
#include "common.h"
#include "ray_marching.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <helper_math.h>
#include <kdtree/kdtree_flann.h>
#include <stb_image_write.h>
#include <thrust/device_vector.h>

__device__ glm::mat4     VIEW;
__device__ float         FOV_RADIANS;
__device__ const float4 *VERTICES;
__device__ const float4 *COLORS;
__device__ const float  *RADII;
__device__ size_t        POINTCLOUD_SIZE;

__device__ float *FRUSTRUM_EDGE_PTS_WORLD_TMP;
__device__ size_t FRUSTRUM_VERTICES_CNT;
__device__ size_t *FRUSTRUM_VERTICES_IDX;

GLuint TEXTURE_HANDLE;

PointcloudRayMarcher *PointcloudRayMarcher::instance = nullptr;

surface<void, cudaSurfaceType2D> surfaceWrite; // NOLINT(cert-err58-cpp)
cudaGraphicsResource_t           cuda_image_resource_handle;
cudaArray_t                      cuda_image;


struct RayHit {
    float distance = 1.0f / 0.0f;  // MAX_FLOAT
    size_t index = 0;
};

__device__ RayHit distance_function(float3 pos);
__device__ long int ray_march(const float3 &ray_origin, const float3 &ray_dir);

template<typename T>
__device__ float3 make_float3(const T &v) {
  return make_float3(v.x, v.y, v.z);
}


__global__ void render()
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= SCREEN_WIDTH || y >= SCREEN_HEIGHT) {
    return;
  }
  auto resolution = make_float2(SCREEN_WIDTH, SCREEN_HEIGHT);
  auto coordinates = make_float2((float)x, (float)y);
  auto uv = (2.0f * coordinates - resolution);
  // In world coords
  auto camera_rot = glm::transpose(glm::mat3(VIEW));
  auto camera_pos = - camera_rot * glm::vec3(VIEW[3]);
  auto pane_dist = SCREEN_WIDTH / (2.0f * tanf(0.5f * FOV_RADIANS));
  auto ro = make_float3(camera_pos);
  auto rd = make_float3(glm::normalize(camera_rot * glm::vec3(uv.x, uv.y, -pane_dist)));
  auto color_index = ray_march(ro, rd);
  // Great life-saving trick for debugging purposes when not writing to the whole picture. Leaving as a memento.
  // auto finalColor = make_float4(x / (SCREEN_WIDTH-1), y / (SCREEN_HEIGHT-1), 1, 1);
  auto finalColor = BACKGROUND_COLOR;
  if (color_index >= 0) {
    finalColor = COLORS[color_index];
  }
#pragma diag_suppress 1215  // Deprecated symbol
  surf2Dwrite(finalColor, surfaceWrite, x * sizeof(float4), y);
#pragma diag_default 1215  // Deprecated symbol get back default behavior
  __syncthreads();
}

void PointcloudRayMarcher::render_to_texture(
  const glm::mat4 &view,
  float fov_radians) {
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(VIEW, &view, sizeof(view)) );
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(FOV_RADIANS, &fov_radians, sizeof(fov_radians)) );

  // Generate homogeneous point for a frustrum-edge-lying point
  auto v = [fov_radians] (float x, float y){
    auto x_factor = glm::tan(fov_radians / 2);
    return glm::vec3(x, y, -2 * x_factor / SCREEN_WIDTH);  // Camera looking to -z
  };
  // Due to computational cost, we're performing the test when moved to world origin.
  auto cam_to_world_rot = glm::transpose(glm::mat3(view));
  frustrum_edge_pts_world_tmp[0] = glm::vec4(cam_to_world_rot[2] + v(-1, -1), 1.0f);
  frustrum_edge_pts_world_tmp[1] = glm::vec4(cam_to_world_rot[2] + v(1, -1), 1.0f);
  frustrum_edge_pts_world_tmp[2] = glm::vec4(cam_to_world_rot[2] + v(1, 1), 1.0f);
  frustrum_edge_pts_world_tmp[3] = glm::vec4(cam_to_world_rot[2] + v(-1, 1), 1.0f);

  frustrum_pcd_size = thrust::copy_if(
    thrust::counting_iterator<size_t>(0), thrust::counting_iterator<size_t>(vertices.size()),
    vertices.begin(),
    frustrum_vertices_idx.begin(),
    [cam_pos = -view[3], cam_dir = glm::vec4(cam_to_world_rot[2], 0)] __device__ (const glm::vec4 &vertex){
      auto pt = vertex - cam_pos;
      auto in = true;
      // Frustrum
      for (int i = 0; i < 4; ++i) {
        // Vector pairing {{0, 1}, {1, 2}, {2, 3}, {3, 0}}
        auto v2 = FRUSTRUM_EDGE_PTS_WORLD_TMP + 4 * (i);
        auto v1 = FRUSTRUM_EDGE_PTS_WORLD_TMP + 4 * ((i + 1) % 4);
        // Plane through origin, (v2 x v1) . pt
        in &= (((*(v2+1) * *(v1+2) - *(v2+2) * *(v1+1)) * pt.x +
                (*(v2+2) * *v1     - *v2     * *(v1+2)) * pt.y +
                (*v2     * *(v1+1) - *(v2+1) * *v1)     * pt.z) > 0);
      }
      pt -= cam_dir * ZNEAR;
      in &= (pt.x * cam_dir.x + pt.y * cam_dir.y + pt.z * cam_dir.z > 0);
      pt -= cam_dir * (ZFAR - ZNEAR);
      in &= (pt.x * - cam_dir.x + pt.y * - cam_dir.y + pt.z * - cam_dir.z > 0);
      return in;
    }
  ) - frustrum_vertices_idx.begin();
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(FRUSTRUM_VERTICES_CNT, &frustrum_pcd_size, sizeof(frustrum_pcd_size)) );

  CHECK_ERROR_CUDA(
    cudaGraphicsGLRegisterImage(
      &cuda_image_resource_handle,
      TEXTURE_HANDLE,
      GL_TEXTURE_2D,
      cudaGraphicsRegisterFlagsSurfaceLoadStore
    )
  );
  CHECK_ERROR_CUDA( cudaGraphicsMapResources(1, &cuda_image_resource_handle) );
  CHECK_ERROR_CUDA( cudaGraphicsSubResourceGetMappedArray(&cuda_image, cuda_image_resource_handle, 0, 0) );
  CHECK_ERROR_CUDA( cudaBindSurfaceToArray(surfaceWrite, cuda_image) );
  dim3 block_dim(32, 32, 1);
  dim3 grid_dim(ceil(SCREEN_WIDTH / block_dim.x), ceil(SCREEN_HEIGHT / block_dim.y), 1);
  render<<< grid_dim, block_dim >>>();
  CHECK_ERROR_CUDA();
  CHECK_ERROR_CUDA( cudaGraphicsUnmapResources(1, &cuda_image_resource_handle) );
  CHECK_ERROR_CUDA( cudaGraphicsUnregisterResource(cuda_image_resource_handle) );
}

void PointcloudRayMarcher::save_png(const std::string &filename) {
  auto raw_data = texture.get_texture_data<float4>();
  auto png = std::vector<unsigned char>(4 * SCREEN_WIDTH * SCREEN_HEIGHT);  // 4=RGBA
  auto begin = (const float*)raw_data.data();
  auto end = (const float*)(raw_data.data() + raw_data.size());
  std::transform(begin, end, png.begin(), [](const float &val){ return (unsigned char)(val * 255.0f); });
  // OpenGL expects the 0.0 coordinate on the y-axis to be on the bottom side of the image, but images usually
  // have 0.0 at the top of the y-axis. For now, this unifies output with the visualisation on the screen.
  stbi_flip_vertically_on_write(true);
  stbi_write_png(filename.c_str(), SCREEN_WIDTH, SCREEN_HEIGHT, 4, png.data(), 4 * SCREEN_WIDTH);  // 4=RGBA
}

/*
 * Not thread safe.
 */
PointcloudRayMarcher *PointcloudRayMarcher::get_instance(
  const thrust::device_vector<glm::vec4> &vertices,
  const thrust::device_vector<glm::vec4> &colors,
  const thrust::device_vector<float> &radii,
  const Texture2D &texture) {
  TEXTURE_HANDLE = texture.get_id();
  if(instance == nullptr) {
    instance = new PointcloudRayMarcher(vertices, colors, radii, texture);
  }
  return instance;
}

PointcloudRayMarcher::PointcloudRayMarcher(
  const thrust::device_vector<glm::vec4> &vertices,
  const thrust::device_vector<glm::vec4> &colors,
  const thrust::device_vector<float> &radii,
  const Texture2D &texture) : vertices(vertices), colors(colors), radii(radii), texture(texture) {
  auto ptr = reinterpret_cast<const float4 *>(vertices.data().get());
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(VERTICES, &ptr, sizeof(ptr)) );
  ptr = reinterpret_cast<const float4 *>(colors.data().get());
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(COLORS, &ptr, sizeof(ptr)) );
  auto ptr_f = radii.data().get();
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(RADII, &ptr_f, sizeof(ptr_f)) );
  auto pointcloud_size = vertices.size();
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(POINTCLOUD_SIZE, &pointcloud_size, sizeof(pointcloud_size)) );
  auto ptr_v = reinterpret_cast<float *>(frustrum_edge_pts_world_tmp.data().get());
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(FRUSTRUM_EDGE_PTS_WORLD_TMP, &ptr_v, sizeof(ptr_v)) );
  frustrum_vertices_idx.resize(pointcloud_size);
  auto ptr_s = frustrum_vertices_idx.data().get();
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(FRUSTRUM_VERTICES_IDX, &ptr_s, sizeof(ptr_s)) );
}

__device__ long int ray_march(const float3 &ray_origin, const float3 &ray_dir) {
  float total_distance_travelled = 0.0f;
  float3 current_position;
  RayHit res;

  for (int i = 0; i < MAX_STEPS; ++i) {
    current_position = ray_origin + ray_dir * total_distance_travelled;
    res = distance_function(current_position);
    total_distance_travelled += res.distance;
    if (res.distance < MIN_DIST) {
      return (long int)res.index;
    }
    if (total_distance_travelled > ZFAR)
      break;
  }

  return -1;
}

__device__ RayHit distance_function(float3 pos) {
  float dist;
  RayHit hit;
  size_t index;
  for (size_t i = 0; i < FRUSTRUM_VERTICES_CNT; ++i) {
    index = FRUSTRUM_VERTICES_IDX[i];
    dist = length(make_float3(VERTICES[index]) - pos) - RADII[index];
    if (dist < hit.distance) {
      hit.distance = dist;
      hit.index = index;
    }
  }
  return hit;
}
