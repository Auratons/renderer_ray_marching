#ifndef POINTCLOUD_RENDERER_RAY_MARCHING_H
#define POINTCLOUD_RENDERER_RAY_MARCHING_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <thrust/device_vector.h>

#include "texture.h"

class PointcloudRayMarcher {
private:
    static PointcloudRayMarcher *instance;
    const thrust::device_vector<glm::vec4> vertices;
    const thrust::device_vector<glm::vec4> colors;
    const thrust::device_vector<float> radii;
    thrust::device_vector<glm::vec4> frustum_edge_pts_world_cs = thrust::device_vector<glm::vec4>(4);
    thrust::device_vector<size_t> frustum_vertices_idx;
    size_t frustum_pcd_size = 0;
    Texture2D texture;

protected:
  PointcloudRayMarcher(
    const thrust::device_vector<glm::vec4> &vertices,
    const thrust::device_vector<glm::vec4> &colors,
    const thrust::device_vector<float> &radii,
    const Texture2D &texture
  );

public:
  PointcloudRayMarcher(PointcloudRayMarcher &) = delete;
  void operator=(const PointcloudRayMarcher &) = delete;

  static PointcloudRayMarcher *get_instance(
    const thrust::device_vector<glm::vec4> &vertices,
    const thrust::device_vector<glm::vec4> &colors,
    const thrust::device_vector<float> &radii,
    const Texture2D &texture);

  void render_to_texture(const glm::mat4 &view, float fov_radians);

  [[nodiscard]] const Texture2D& get_texture() const noexcept {
    return texture;
  }

  void save_png(const std::string &path);
};

#endif //POINTCLOUD_RENDERER_RAY_MARCHING_H
