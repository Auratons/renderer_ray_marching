#ifndef POINTCLOUD_RENDERER_RAY_MARCHING_H
#define POINTCLOUD_RENDERER_RAY_MARCHING_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <thrust/device_vector.h>

#define cudaCheckError() { \
  cudaError_t err = cudaGetLastError(); \
  if(err != cudaSuccess) { \
    std::cerr << "CUDA ERROR: " __FILE__ << " " << __LINE__ << " " << cudaGetErrorString(err) << std::endl; \
    exit(1); \
  } \
}

void launch_kernel(cudaArray_t cuda_image_array, size_t num_v, const float3 *vertices, const float3 *colors, const float *radii);

#endif //POINTCLOUD_RENDERER_RAY_MARCHING_H
