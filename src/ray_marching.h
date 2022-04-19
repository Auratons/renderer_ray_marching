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

class PointcloudRayMarcher {
private:
    static PointcloudRayMarcher *instance;

protected:
  PointcloudRayMarcher(
    const float3 *vertices,
    const float3 *colors,
    const float *radii,
    size_t pointcloud_size
  );

public:
  PointcloudRayMarcher(PointcloudRayMarcher &other) = delete;
  void operator=(const PointcloudRayMarcher &) = delete;

  static PointcloudRayMarcher *get_instance(
    const float3 *vertices,
    const float3 *colors,
    const float *radii,
    size_t pointcloud_size,
    GLuint texture_handle);

    static PointcloudRayMarcher *get_instance();

  void render_to_texture(const glm::mat4 &model, const glm::mat4 &view, float fov_radians);
};

#endif //POINTCLOUD_RENDERER_RAY_MARCHING_H
