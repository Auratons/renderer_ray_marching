#ifndef POINTCLOUD_RENDERER_RAY_MARCHING_H
#define POINTCLOUD_RENDERER_RAY_MARCHING_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <kdtree/kdtree_flann.h>
#include <thrust/device_vector.h>

#include "texture.h"

class PointcloudRayMarcher {
private:
    static PointcloudRayMarcher *instance;
    const thrust::device_vector<glm::vec4> vertices;
    const thrust::device_vector<glm::vec4> colors;
    const thrust::device_vector<float> radii;
    const int knn = 10;
    thrust::device_vector<glm::vec4> frustum_edge_pts_world_cs = thrust::device_vector<glm::vec4>(4);
    thrust::device_vector<size_t> frustum_vertices_idx;
    const kdtree::KDTreeFlann &tree;
    thrust::device_vector<int> indices;
    thrust::device_vector<float> distances;
    thrust::device_vector<float4> query;

protected:
  PointcloudRayMarcher(
    const thrust::device_vector<glm::vec4> &vertices,
    const thrust::device_vector<glm::vec4> &colors,
    const thrust::device_vector<float> &radii,
    const kdtree::KDTreeFlann &tree
  );

public:
  PointcloudRayMarcher(PointcloudRayMarcher &) = delete;
  void operator=(const PointcloudRayMarcher &) = delete;

  static PointcloudRayMarcher *get_instance(
    const thrust::device_vector<glm::vec4> &vertices,
    const thrust::device_vector<glm::vec4> &colors,
    const thrust::device_vector<float> &radii,
    const kdtree::KDTreeFlann &tree);

  void render_to_texture(glm::mat4 view, float fov_radians, const Texture2D &texture);
};

#endif //POINTCLOUD_RENDERER_RAY_MARCHING_H
