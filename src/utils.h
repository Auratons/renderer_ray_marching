#ifndef POINTCLOUD_RENDERER_UTILS_H
#define POINTCLOUD_RENDERER_UTILS_H

#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <thrust/device_vector.h>


#define CHECK_ERROR_CUDA(EXPRESSION) { \
  EXPRESSION; \
  cudaError_t err = cudaGetLastError(); \
  if(err != cudaSuccess) { \
    std::cerr << "CUDA ERROR: " __FILE__ << " " << __LINE__ << " " << cudaGetErrorString(err) << std::endl; \
    exit(1); \
  } \
}

#define CHECK_ERROR_GL(EXPRESSION) { \
  EXPRESSION; \
  GLenum err = glGetError(); \
  if(err != GL_NO_ERROR) { \
    std::cerr << "GL ERROR: " __FILE__ << " " << __LINE__ << " " << gluErrorString(err) << std::endl; \
    exit(1); \
  } \
}

#undef glBufferData
template <class T>
inline void glBufferData(GLenum target, const std::vector<T>& v, GLenum usage) {
  glad_glBufferData(target, v.size() * sizeof(T), &v[0], usage);
}


std::string gluErrorString(GLenum);

// A pack of fuctions for compute shader related information.
void print_workgroup_count();
void print_workgroup_size();
int print_invocations();

thrust::device_vector<float> compute_radii(const thrust::device_vector<glm::vec4> &vertices);

std::vector<bool> filter_view_frustrum(const glm::mat4 &view, const std::vector<glm::vec3> &pts, float ratio, float fov_rad);

#endif //POINTCLOUD_RENDERER_UTILS_H
