#ifndef POINTCLOUD_RENDERER_UTILS_H
#define POINTCLOUD_RENDERER_UTILS_H

#include <array>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#undef glBufferData

template <class T>
inline void glBufferData(GLenum target, const std::vector<T>& v, GLenum usage) {
  glad_glBufferData(target, v.size() * sizeof(T), &v[0], usage);
}

std::vector<float> compute_radii(std::vector<std::array<double, 3>> &vertices);

#endif //POINTCLOUD_RENDERER_UTILS_H
