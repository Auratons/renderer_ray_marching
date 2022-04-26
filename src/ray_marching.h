#ifndef POINTCLOUD_RENDERER_RAY_MARCHING_H
#define POINTCLOUD_RENDERER_RAY_MARCHING_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <thrust/device_vector.h>

class PointcloudRayMarcher {
private:
    static PointcloudRayMarcher *instance;
    const thrust::device_vector<glm::vec4> vertices;
    const thrust::device_vector<glm::vec4> colors;
    const thrust::device_vector<float> radii;
    thrust::device_vector<glm::vec4> frustrum_edge_pts_world_tmp = thrust::device_vector<glm::vec4>(4);
    thrust::device_vector<size_t> frustrum_vertices_idx;
    size_t frustrum_pcd_size = 0;

protected:
  PointcloudRayMarcher(
    const thrust::device_vector<glm::vec4> &vertices,
    const thrust::device_vector<glm::vec4> &colors,
    const thrust::device_vector<float> &radii
  );

public:
  PointcloudRayMarcher(PointcloudRayMarcher &) = delete;
  void operator=(const PointcloudRayMarcher &) = delete;

  static PointcloudRayMarcher *get_instance(
    const thrust::device_vector<glm::vec4> &vertices,
    const thrust::device_vector<glm::vec4> &colors,
    const thrust::device_vector<float> &radii,
    GLuint texture_handle);

  void render_to_texture(const glm::mat4 &view, float fov_radians);
};

#endif //POINTCLOUD_RENDERER_RAY_MARCHING_H
