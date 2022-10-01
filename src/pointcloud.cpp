#include <algorithm>
#include <exception>
#include <vector>

#include <glad/glad.h>
#include <happly.h>

#include "pointcloud.h"
#include "utils.h"

#define POINT_SIZE 3
#define COLOR_SIZE 3
#define RADIUS_SIZE 1

Pointcloud::Pointcloud(
    const std::vector<glm::vec3> &points,
    const std::vector<glm::vec3> &colors,
    const std::vector<float> &radii) {
  if (points.size() - colors.size() || points.size() - radii.size()) {
    throw std::runtime_error("Pointcloud doesn't have the same number of points, colors and radii.");
  }

  // (POINT_SIZE + 1) here is used so that we can utilize std430 in shader
  // code with struct Point { vec4 pos; vec4 color_radius; } with correct alignment.
  constexpr size_t stride = (POINT_SIZE + 1) + COLOR_SIZE + RADIUS_SIZE;
  auto vbo_data = std::vector<GLfloat>(stride * points.size());
  // Separate for are hopefully more cache-friendly.
  for (size_t idx = 0; idx < points.size(); ++idx)
    std::copy((float*)&points[idx], (float*)&points[idx] + 3, &vbo_data[idx * stride]);
  for (size_t idx = 0; idx < points.size(); ++idx)
    std::copy((float*)&colors[idx], (float*)&colors[idx] + 3, &vbo_data[idx * stride + (POINT_SIZE + 1)]);
  for (size_t idx = 0; idx < points.size(); ++idx) {
    vbo_data[idx * stride + POINT_SIZE] = 1.0f;  // Homogeneous one.
    vbo_data[idx * stride + (POINT_SIZE + 1) + COLOR_SIZE] = radii[idx];
  }

  glGenBuffers(1, &ssbo);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
  glBufferData(GL_SHADER_STORAGE_BUFFER, vbo_data, GL_STATIC_DRAW);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

std::tuple<std::vector<glm::vec4>, std::vector<glm::vec4>> Pointcloud::load_ply(const std::string &pcd_path) {
  happly::PLYData ply(pcd_path);
  auto input_vertices = ply.getVertexPositions();
  auto input_colors = ply.getVertexColors();
  auto all_vertices = std::vector<glm::vec4>(input_vertices.size());
  auto all_colors = std::vector<glm::vec4>(input_colors.size());
  transform(
          input_vertices.begin(), input_vertices.end(), all_vertices.begin(),
          [] (const std::array<double, 3> &pt){ return glm::vec4(pt[0], pt[1], pt[2], 1.0f); }
  );
  transform(
          input_colors.begin(), input_colors.end(), all_colors.begin(),
          [] (const std::array<unsigned char, 3> &pt){ return glm::vec4(pt[0] / 255.0f, pt[1] / 255.0f, pt[2] / 255.0f, 1.0f); }
  );
  return std::make_tuple(all_vertices, all_colors);
}
