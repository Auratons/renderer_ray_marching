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

surface<void, cudaSurfaceType2D> surfaceWrite; // NOLINT(cert-err58-cpp)


struct RayHit {
    float distance = 1.0f / 0.0f;  // MAX_FLOAT
    size_t index = 0;
};

__device__ RayHit distance_function(const float3 &pos, size_t num_v, const float3 *vertices, const float *radii) {
  float dist, radius;
  float3 position;
  RayHit hit;
  for (size_t i = 0; i < num_v; ++i) {
    position = vertices[i];
    radius = radii[i];
    dist = length(position - pos) - radius;
    if (dist < hit.distance) {
      hit.distance = dist;
      hit.index = i;
    }
  }
  return hit;
}

__device__ long int ray_march(const float3 &rayOrigin, const float3 &rayDir, size_t num_v, const float3 *vertices, const float *radii)
{
  float total_distance_travelled = 0.0f;

  for (int i = 0; i < MAX_STEPS; i++) {
    auto current_position = rayOrigin + rayDir * total_distance_travelled;
    RayHit res = distance_function(current_position, num_v, vertices, radii);
    total_distance_travelled += res.distance;
    if (res.distance < MIN_DIST) {
      return (long int)res.index;
    }
    if (total_distance_travelled > ZFAR)
      break;
  }

  return -1;
}


__global__ void rendering(size_t num_v, const float3 *vertices, const float3 *colors, const float *radii)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= SCREEN_WIDTH || y >= SCREEN_HEIGHT) {
    return;
  }
  auto resolution = make_float2(SCREEN_WIDTH, SCREEN_HEIGHT);
  auto coordinates = make_float2((float)x, (float)y);
  auto uv = (2.0f * coordinates - resolution) / resolution.y;
  auto ro = make_float3(0.0f, 0.0f, -8.0f);
  auto rd = normalize(make_float3(uv, 2.0f));
  auto color_index = ray_march(ro, rd, num_v, vertices, radii);
  auto finalColor = BACKGROUND_COLOR;
  if (color_index >= 0) {
    finalColor = make_float4(colors[color_index], 1.0);
  }
  surf2Dwrite(finalColor, surfaceWrite, x * sizeof(float4), y);
  __syncthreads();
}

void launch_kernel(cudaArray_t cuda_image_array, size_t num_v, const float3 *vertices, const float3 *colors, const float *radii) {
  dim3 block_dim(32, 32, 1);
  dim3 grid_dim(SCREEN_WIDTH / block_dim.x, SCREEN_HEIGHT / block_dim.y, 1);

  //Bind voxel array to a writable CUDA surface
  cudaBindSurfaceToArray(surfaceWrite, cuda_image_array);
  cudaCheckError();
  rendering<<< grid_dim, block_dim >>>(num_v, vertices, colors, radii);
  cudaCheckError();
}
