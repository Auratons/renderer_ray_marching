#include <iostream>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <helper_math.h>

#include "ray_marching.h"

#define ZFAR 100
#define MAX_STEPS 128
#define MIN_DIST 0.001
#define BACKGROUND_COLOR make_float4(1)

#define SCREEN_WIDTH 1024.0
#define SCREEN_HEIGHT 768.0

__device__ glm::mat4     MODEL;
__device__ glm::mat4     VIEW;
__device__ float         FOV_RADIANS;
__device__ const float3 *VERTICES;
__device__ const float3 *COLORS;
__device__ const float  *RADII;
__device__ size_t        POINTCLOUD_SIZE;

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

__device__ float3 make_float3(const glm::vec4 &v) {
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
  auto uv = (2.0f * coordinates - resolution) / resolution;
  auto ro = make_float3(VIEW[3]);
  auto rd_glm = glm::normalize(VIEW * MODEL * glm::vec4(uv.x, uv.y, FOV_RADIANS, 0));
  auto rd = make_float3(rd_glm.x, rd_glm.y, rd_glm.z);
  auto color_index = ray_march(ro, rd);
  auto finalColor = BACKGROUND_COLOR;
  if (color_index >= 0) {
    finalColor = make_float4(COLORS[color_index], 1.0);
  }
#pragma diag_suppress 1215  // Deprecated symbol
  surf2Dwrite(finalColor, surfaceWrite, x * sizeof(float4), y);
#pragma diag_default 1215  // Deprecated symbol get back default behavior
  __syncthreads();
}

void PointcloudRayMarcher::render_to_texture(
  const glm::mat4 &model,
  const glm::mat4 &view,
  float fov_radians) {
  cudaMemcpyToSymbol(MODEL, &model, sizeof(model));
  cudaCheckError();
  cudaMemcpyToSymbol(VIEW, &view, sizeof(view));
  cudaCheckError();
  cudaMemcpyToSymbol(FOV_RADIANS, &fov_radians, sizeof(fov_radians));
  cudaCheckError();

  dim3 block_dim(32, 32, 1);
  dim3 grid_dim(SCREEN_WIDTH / block_dim.x, SCREEN_HEIGHT / block_dim.y, 1);

  cudaGraphicsGLRegisterImage(
    &cuda_image_resource_handle,
    TEXTURE_HANDLE,
    GL_TEXTURE_2D,
    cudaGraphicsRegisterFlagsSurfaceLoadStore
  );
  cudaCheckError();
  cudaGraphicsMapResources(1, &cuda_image_resource_handle);
  cudaCheckError();
  cudaGraphicsSubResourceGetMappedArray(&cuda_image, cuda_image_resource_handle, 0, 0);
  cudaCheckError();
  cudaBindSurfaceToArray(surfaceWrite, cuda_image);
  cudaCheckError();
  render<<< grid_dim, block_dim >>>();
  cudaCheckError();

  cudaGraphicsUnmapResources(1, &cuda_image_resource_handle);
  cudaCheckError();
  cudaGraphicsUnregisterResource(cuda_image_resource_handle);
  cudaCheckError();
}

/*
 * Not thread safe.
 */
PointcloudRayMarcher *PointcloudRayMarcher::get_instance(
  const float3 *vertices,
  const float3 *colors,
  const float *radii,
  size_t pointcloud_size,
  GLuint texture_handle) {
  TEXTURE_HANDLE = texture_handle;
  if(instance == nullptr) {
    instance = new PointcloudRayMarcher(vertices, colors, radii, pointcloud_size);
  }
  return instance;
}

/*
 * Not thread safe.
 */
PointcloudRayMarcher *PointcloudRayMarcher::get_instance() {
  if(instance == nullptr) {
    throw std::runtime_error("Renderer not initialized.");
  }
  return instance;
}

PointcloudRayMarcher::PointcloudRayMarcher(
  const float3 *vertices,
  const float3 *colors,
  const float *radii,
  size_t pointcloud_size) {
  cudaMemcpyToSymbol(VERTICES, &vertices, sizeof(vertices));
  cudaCheckError();
  cudaMemcpyToSymbol(COLORS, &colors, sizeof(colors));
  cudaCheckError();
  cudaMemcpyToSymbol(RADII, &radii, sizeof(radii));
  cudaCheckError();
  cudaMemcpyToSymbol(POINTCLOUD_SIZE, &pointcloud_size, sizeof(pointcloud_size));
  cudaCheckError();
}

__device__ long int ray_march(const float3 &ray_origin, const float3 &ray_dir) {
  float total_distance_travelled = 0.0f;

  for (int i = 0; i < MAX_STEPS; i++) {
    auto current_position = ray_origin + ray_dir * total_distance_travelled;
    auto res = distance_function(current_position);
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
  float dist, radius;
  float3 position;
  RayHit hit;
  for (size_t i = 0; i < POINTCLOUD_SIZE; ++i) {
    position = VERTICES[i];
    radius = RADII[i];
    dist = length(position - pos) - radius;
    if (dist < hit.distance) {
      hit.distance = dist;
      hit.index = i;
    }
  }
  return hit;
}

#pragma clang diagnostic pop