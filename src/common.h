#ifndef POINTCLOUD_RENDERER_COMMON_H
#define POINTCLOUD_RENDERER_COMMON_H

#define ZFAR 100.0f
#define ZNEAR 0.1f
#define MAX_STEPS 128
#define MIN_DIST 0.001f
#define BACKGROUND_COLOR make_float4(1)

#define SCREEN_WIDTH 1024.0
#define SCREEN_HEIGHT 768.0

#define cudaCheckError(EXPRESSION) { \
  EXPRESSION; \
  cudaError_t err = cudaGetLastError(); \
  if(err != cudaSuccess) { \
    std::cerr << "CUDA ERROR: " __FILE__ << " " << __LINE__ << " " << cudaGetErrorString(err) << std::endl; \
    exit(1); \
  } \
}

#endif //POINTCLOUD_RENDERER_COMMON_H
