#ifndef POINTCLOUD_RENDERER_UTILS_H
#define POINTCLOUD_RENDERER_UTILS_H

#include <array>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#undef glBufferData

template <class T>
inline void glBufferData(GLenum target, const std::vector<T>& v, GLenum usage) {
  glad_glBufferData(target, v.size() * sizeof(T), &v[0], usage);
}

std::vector<float> compute_radii(const std::vector<glm::vec4> &vertices);

std::vector<bool> filter_view_frustrum(const glm::mat4 &view, const std::vector<glm::vec3> &pts, float ratio, float fov_rad);

#endif //POINTCLOUD_RENDERER_UTILS_H
