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

  constexpr size_t stride = POINT_SIZE + COLOR_SIZE + RADIUS_SIZE;
  constexpr auto op = [](auto c){ return c / 255.0f; };
  auto vbo_data = std::vector<GLfloat>(stride * points.size());
  // Separate for are hopefully more cache-friendly.
  for (size_t idx = 0; idx < points.size(); ++idx)
    std::copy(points[idx].begin(), points[idx].end(), &vbo_data[idx * stride]);
  for (size_t idx = 0; idx < points.size(); ++idx)
    std::transform(colors[idx].begin(), colors[idx].end(), &vbo_data[idx * stride + POINT_SIZE], op);
  for (size_t idx = 0; idx < points.size(); ++idx)
    vbo_data[idx * stride + POINT_SIZE + COLOR_SIZE] = radii[idx];

  // VAO
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  // VBO
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, vbo_data, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0); // NOLINT(modernize-use-nullptr)
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(6 * sizeof(float)));
  glEnableVertexAttribArray(2);

  glBindVertexArray(0);
}

Pointcloud::~Pointcloud() {
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo);
}
