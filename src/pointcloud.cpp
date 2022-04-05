#include <algorithm>
#include <exception>
#include <vector>

#include <glad/glad.h>

#include "pointcloud.h"
#include "utils.h"

#define POINT_SIZE 3
#define COLOR_SIZE 3
#define RADIUS_SIZE 1

Pointcloud::Pointcloud(
    const std::vector<std::array<double, POINT_SIZE>> &points,
    const std::vector<std::array<unsigned char, COLOR_SIZE>> &colors,
    const std::vector<float> &radii) {
  if (points.size() - colors.size() || points.size() - radii.size()) {
    throw std::runtime_error("Pointcloud doesn't have the same number of points, colors and radii.");
  }

  // (POINT_SIZE + 1) here is used so that we can utilize std430 in shader
  // code with struct Point { vec4 pos; vec4 color_radius; } with correct alignment.
  constexpr size_t stride = (POINT_SIZE + 1) + COLOR_SIZE + RADIUS_SIZE;
  constexpr auto op = [](auto c){ return c / 255.0f; };
  auto vbo_data = std::vector<GLfloat>(stride * points.size(), 1.0f);  // Homogeneous one.
  // Separate for are hopefully more cache-friendly.
  for (size_t idx = 0; idx < points.size(); ++idx)
    std::copy(points[idx].begin(), points[idx].end(), &vbo_data[idx * stride]);
  for (size_t idx = 0; idx < points.size(); ++idx)
    std::transform(colors[idx].begin(), colors[idx].end(), &vbo_data[idx * stride + (POINT_SIZE + 1)], op);
  for (size_t idx = 0; idx < points.size(); ++idx)
    vbo_data[idx * stride + (POINT_SIZE + 1) + COLOR_SIZE] = radii[idx];

  glGenBuffers(1, &ssbo);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
  glBufferData(GL_SHADER_STORAGE_BUFFER, vbo_data, GL_STATIC_DRAW);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}
