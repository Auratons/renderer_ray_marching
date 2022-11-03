#include <iostream>

#include "utils.h"
#include "common.h"
#include "ray_marching.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <kdtree/kdtree_flann.h>
#include <stb_image_write.h>
#include <thrust/device_vector.h>

__device__ glm::mat4     CAMERA_MATRIX;
__device__ float         FOV_RADIANS;
__device__ const float4 *VERTICES;
__device__ const float4 *COLORS;
__device__ const float  *RADII;
__device__ size_t        POINTCLOUD_SIZE;
__device__ GLsizei       TEXTURE_WIDTH;
__device__ GLsizei       TEXTURE_HEIGHT;

__device__ flann::cuda::kd_tree_builder_detail::SplitInfo *GPU_SPLITS;
__device__ int *GPU_CHILD1;
__device__ int *GPU_PARENT;
__device__ float4 *GPU_AABB_MIN;
__device__ float4 *GPU_AABB_MAX;
__device__ float4 *GPU_POINTS;
__device__ int *GPU_VIND;
__device__ int *INDICES;
__device__ float *DISTANCES;
__device__ float4 *QUERY;
__device__ int KNN;

__device__ float4 *FRUSTUM_EDGE_PTS_WORLD_CS;

GLuint TEXTURE_HANDLE;

PointcloudRayMarcher *PointcloudRayMarcher::instance = nullptr;

cudaGraphicsResource_t cuda_image_resource_handle;
cudaArray_t            cuda_image;
cudaSurfaceObject_t    g_surfaceObj;
cudaResourceDesc       g_resourceDesc;


struct RayHit {
    float distance = 1.0f / 0.0f;  // MAX_FLOAT
    int index = 0;
};

__device__ RayHit distance_function(float3 pos, int linearized_index);
__device__ long int ray_march(
    const float3 &ray_origin,
    const float3 &ray_dir,
    float dist_to_far_plane,
    float dist_to_near_plane,
    int linearized_index);

template<typename T>
__device__ __host__ float3 make_float3(const T &v) {
  return make_float3(v.x, v.y, v.z);
}


__global__ void render(cudaSurfaceObject_t surface)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= TEXTURE_WIDTH || y >= TEXTURE_HEIGHT) {
    return;
  }
  auto resolution = make_float2((float)TEXTURE_WIDTH, (float)TEXTURE_HEIGHT);
  auto coordinates = make_float2((float)x, (float)y);
  auto uv = (2.0f * coordinates - resolution) / 2.0f;
  // In world coords
  auto pane_dist = (float)TEXTURE_WIDTH / (2.0f * tanf(0.5f * FOV_RADIANS));
  auto cam_dir = make_float3(- CAMERA_MATRIX[2]);
  auto ray_origin = make_float3(CAMERA_MATRIX[3]); // camera position
  auto ray_direction = make_float3(glm::normalize(CAMERA_MATRIX * glm::vec4(uv.x, uv.y, -pane_dist, 0.0f)));
  auto dist_to_far_plane = ZFAR / dot(cam_dir, ray_direction);  // Both vectors are normalized
  auto dist_to_near_plane = ZNEAR / dot(cam_dir, ray_direction);  // Both vectors are normalized
  auto color_index = ray_march(ray_origin, ray_direction, dist_to_far_plane, dist_to_near_plane, y * TEXTURE_WIDTH + x);
  // Great life-saving trick for debugging purposes when not writing to the whole picture. Leaving as a memento.
  // auto finalColor = make_float4(x / ((float)(TEXTURE_WIDTH-1)), y / ((float)(TEXTURE_HEIGHT-1)), 1, 1);
  auto finalColor = BACKGROUND_COLOR;
  if (color_index >= 0) {
    finalColor = COLORS[color_index];
  }
  surf2Dwrite(finalColor, surface, x * (int)sizeof(float4), y);
}

void PointcloudRayMarcher::render_to_texture(
  const glm::mat4 &camera_matrix,
  float fov_radians) {
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(CAMERA_MATRIX, &camera_matrix, sizeof(camera_matrix)) );
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(FOV_RADIANS, &fov_radians, sizeof(fov_radians)) );

  // Generate homogeneous point for a frustum-edge-lying point
  auto v = [&] (float x, float y){
    return glm::vec4(
      x * (0.5f * (float)texture.width),
      y * (0.5f * (float)texture.height),
      -(float)texture.width / (2.0f * tanf(0.5f * fov_radians)),
      0.0f
    );
  };
  frustum_edge_pts_world_cs[0] = camera_matrix * v(-1, -1);
  frustum_edge_pts_world_cs[1] = camera_matrix * v(1, -1);
  frustum_edge_pts_world_cs[2] = camera_matrix * v(1, 1);
  frustum_edge_pts_world_cs[3] = camera_matrix * v(-1, 1);

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
  memset(&g_resourceDesc, 0, sizeof(cudaResourceDesc));
  g_resourceDesc.resType = cudaResourceTypeArray;
  g_resourceDesc.res.array.array = cuda_image;
  CHECK_ERROR_CUDA(cudaCreateSurfaceObject(&g_surfaceObj, &g_resourceDesc));
  dim3 block_dim(16, 16, 1);
  dim3 grid_dim((texture.width + block_dim.x - 1) / block_dim.x, (texture.height + block_dim.y - 1) / block_dim.y, 1);
  render<<< grid_dim, block_dim >>>(g_surfaceObj);
  CHECK_ERROR_CUDA();
  CHECK_ERROR_CUDA( cudaGraphicsUnmapResources(1, &cuda_image_resource_handle) );
  CHECK_ERROR_CUDA( cudaGraphicsUnregisterResource(cuda_image_resource_handle) );
}

void PointcloudRayMarcher::save_png(const std::string &filename) {
  auto raw_data = texture.get_texture_data<float4>();
  auto png = std::vector<unsigned char>(4 * texture.width * texture.height);  // 4=RGBA
  auto begin = (const float*)raw_data.data();
  auto end = (const float*)(raw_data.data() + raw_data.size());
  std::transform(begin, end, png.begin(), [](const float &val){ return (unsigned char)(val * 255.0f); });
  // OpenGL expects the 0.0 coordinate on the y-axis to be on the bottom side of the image, but images usually
  // have 0.0 at the top of the y-axis. For now, this unifies output with the visualisation on the screen.
  stbi_flip_vertically_on_write(true);
  stbi_write_png(filename.c_str(), texture.width, texture.height, 4, png.data(), 4 * texture.width);  // 4=RGBA
}

/*
 * Not thread safe.
 */
PointcloudRayMarcher *PointcloudRayMarcher::get_instance(
  const thrust::device_vector<glm::vec4> &vertices,
  const thrust::device_vector<glm::vec4> &colors,
  const thrust::device_vector<float> &radii,
  const Texture2D &texture,
  const kdtree::KDTreeFlann &tree) {
  TEXTURE_HANDLE = texture.get_id();
  if(instance == nullptr) {
    instance = new PointcloudRayMarcher(vertices, colors, radii, texture, tree);
  }
  return instance;
}

PointcloudRayMarcher::PointcloudRayMarcher(
  const thrust::device_vector<glm::vec4> &vertices,
  const thrust::device_vector<glm::vec4> &colors,
  const thrust::device_vector<float> &radii,
  const Texture2D &texture,
  const kdtree::KDTreeFlann &tree) : vertices(vertices), colors(colors), radii(radii), texture(texture), tree(tree) {
  auto ptr = reinterpret_cast<const float4 *>(vertices.data().get());
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(VERTICES, &ptr, sizeof(ptr)) );
  ptr = reinterpret_cast<const float4 *>(colors.data().get());
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(COLORS, &ptr, sizeof(ptr)) );
  auto ptr_f = radii.data().get();
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(RADII, &ptr_f, sizeof(ptr_f)) );
  auto pointcloud_size = vertices.size();
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(POINTCLOUD_SIZE, &pointcloud_size, sizeof(pointcloud_size)) );
  auto ptr_f4 = reinterpret_cast<float4 *>(frustum_edge_pts_world_cs.data().get());
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(FRUSTUM_EDGE_PTS_WORLD_CS, &ptr_f4, sizeof(ptr_f4)) );
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(TEXTURE_WIDTH, &texture.width, sizeof(texture.width)) );
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(TEXTURE_HEIGHT, &texture.height, sizeof(texture.height)) );
  auto gpu_splits = thrust::raw_pointer_cast(&((*tree.flann_index_->gpu_helper_->gpu_splits_)[0]));
  auto gpu_child1 = thrust::raw_pointer_cast(&((*tree.flann_index_->gpu_helper_->gpu_child1_)[0]));
  auto gpu_parent = thrust::raw_pointer_cast(&((*tree.flann_index_->gpu_helper_->gpu_parent_)[0]));
  auto gpu_aabb_min = thrust::raw_pointer_cast(&((*tree.flann_index_->gpu_helper_->gpu_aabb_min_)[0]));
  auto gpu_aabb_max = thrust::raw_pointer_cast(&((*tree.flann_index_->gpu_helper_->gpu_aabb_max_)[0]));
  auto gpu_points = thrust::raw_pointer_cast(&((*tree.flann_index_->gpu_helper_->gpu_points_)[0]));
  auto gpu_vind = thrust::raw_pointer_cast(&((*tree.flann_index_->gpu_helper_->gpu_vind_)[0]));
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(GPU_SPLITS, &gpu_splits, sizeof(gpu_splits)) );
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(GPU_CHILD1, &gpu_child1, sizeof(gpu_child1)) );
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(GPU_PARENT, &gpu_parent, sizeof(gpu_parent)) );
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(GPU_AABB_MIN, &gpu_aabb_min, sizeof(gpu_aabb_min)) );
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(GPU_AABB_MAX, &gpu_aabb_max, sizeof(gpu_aabb_max)) );
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(GPU_POINTS, &gpu_points, sizeof(gpu_points)) );
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(GPU_VIND, &gpu_vind, sizeof(gpu_vind)) );
  auto pixel_count = texture.width * texture.height;
  indices.resize(pixel_count * knn);
  distances.resize(pixel_count * knn);
  query.resize(pixel_count);
  auto ptr_i = indices.data().get();
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(INDICES, &ptr_i, sizeof(ptr_i)) );
  auto ptr_v = distances.data().get();
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(DISTANCES, &ptr_v, sizeof(ptr_v)) );
  ptr_f4 = query.data().get();
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(QUERY, &ptr_f4, sizeof(ptr_f4)) );
  CHECK_ERROR_CUDA( cudaMemcpyToSymbol(KNN, &knn, sizeof(knn)) );
}

__device__ long int ray_march(
    const float3 &ray_origin,
    const float3 &ray_dir,
    float dist_to_far_plane,
    float dist_to_near_plane,
    int linearized_index) {
  float total_distance_travelled = 0.0f;
  float3 current_position;
  RayHit res;

  for (int i = 0; i < MAX_STEPS; ++i) {
    current_position = ray_origin + ray_dir * total_distance_travelled;
    res = distance_function(current_position, linearized_index);
    total_distance_travelled += res.distance;
    if (res.distance < MIN_DIST) {
      if (total_distance_travelled < dist_to_near_plane)
        continue;
      return (long int)res.index;
    }
    if (total_distance_travelled > dist_to_far_plane)
      break;
  }

  return -1;
}


__device__ bool is_inside(const float4 &vertex) {
    auto cam_pos = make_float3(CAMERA_MATRIX[3]);
    auto cam_dir = make_float3(- CAMERA_MATRIX[2]);
    auto pt = make_float3(vertex) - cam_pos;
    auto in = true;
    // frustum
// From a mysterious reason, this doesn't work even though exactly the same code worked
// as a lambda in thrust::copy_if in a previous version of the code. Fortunately, near
// and far plane clipping works so back-cone-outliers are discarded. If we take sometimes
// a frustum outlier near border planes of view frustum, we must take it. The speedup here
// with KD tree in distance function is substantial over a simple for cycle through all
// points inside.
//    for (int i = 0; i < 4; ++i) {
//      // Vector pairing {{0, 1}, {1, 2}, {2, 3}, {3, 0}}
//      auto v2 = make_float3(FRUSTUM_EDGE_PTS_WORLD_CS[i]);
//      auto v1 = make_float3(FRUSTUM_EDGE_PTS_WORLD_CS[(i + 1) % 4]);
//      // Plane through origin, (v2 x v1) . pt
//      in &= (((v2.y * v1.z - v2.z * v1.y) * pt.x +
//              (v2.z * v1.x - v2.x * v1.z) * pt.y +
//              (v2.x * v1.y - v2.y * v1.x) * pt.z) < 0);
//      in &= (dot(cross(v2, v1), pt) < 0);
//    }
    pt -= (cam_dir * ZNEAR);
    in &= (pt.x * cam_dir.x + pt.y * cam_dir.y + pt.z * cam_dir.z > 0);
    pt -= (cam_dir * (ZFAR - ZNEAR));
    in &= (pt.x * -cam_dir.x + pt.y * -cam_dir.y + pt.z * -cam_dir.z > 0);
    return in;
};

__device__ RayHit distance_function(float3 pos, int linearized_index) {
  float dist;
  RayHit hit;
  int index;
  auto start_of_threads_memory_offset = linearized_index * KNN;

  QUERY[linearized_index] = make_float4(pos, 0);

  flann::KDTreeCuda3dIndex<flann::L2<float> >::knnSearchOneQuery(
    GPU_SPLITS,
    GPU_CHILD1,
    GPU_PARENT,
    GPU_AABB_MIN,
    GPU_AABB_MAX,
    GPU_POINTS,
    QUERY[linearized_index],
    INDICES + start_of_threads_memory_offset,
    DISTANCES + start_of_threads_memory_offset,
    KNN);

  for (int i = 0; i < KNN; ++i) {
    auto kth_nn_idx = start_of_threads_memory_offset + i;
    // transforms indices in the internal data set back to the original indices
    index = (INDICES[kth_nn_idx] >= 0) ? GPU_VIND[INDICES[kth_nn_idx]] : INDICES[kth_nn_idx];
    if (index < 0) continue;
    dist = sqrtf(DISTANCES[kth_nn_idx]) - RADII[index];
    if (dist < hit.distance && is_inside(VERTICES[index])) {
      hit.distance = dist;
      hit.index = index;
    }
  }
  return hit;
}
