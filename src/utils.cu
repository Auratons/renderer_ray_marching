#include <algorithm>
#include <ostream>

#include <glm/glm.hpp>
#include <kdtree/kdtree_flann.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils.h"

std::vector<bool> filter_view_frustrum(const glm::mat4 &view, const std::vector<glm::vec3> &pts, float ratio, float fov_rad) {
  // Generate homogeneous point for a frustrum-edge-lying point
  auto v = [fov_rad, ratio](float x, float y){
    auto x_factor = glm::tan(fov_rad);
    return glm::vec3(x * x_factor, y * x_factor / ratio, -1);  // Camera looking to -z
  };
  // Frustrum edge pts (frustrum tip is in 0,0,0) in camera space
  auto edge_cam = std::vector<glm::vec3>({v(-1, -1), v(1, -1), v(1, 1), v(-1, 1)});
  // Frustrum edge pts (frustrum tip is in camera position) in world space
  auto centered_edge_world = thrust::device_vector<glm::vec3>(edge_cam.begin(), edge_cam.end());
  // Due to computational cost, we're performing the test when moved to world origin.
  auto cam_to_world_rot = glm::transpose(glm::mat3(view));
  thrust::transform(
    centered_edge_world.begin(), centered_edge_world.end(), centered_edge_world.begin(),
    [cam_to_world_rot] __device__ (const glm::vec3 &pt) { return cam_to_world_rot * pt; }
  );

  auto is_in_frustrum = thrust::device_vector<bool>(pts.size());
  auto pts_device = thrust::device_vector<glm::vec3>(pts.begin(), pts.end());
  thrust::transform(
    pts_device.begin(), pts_device.end(), pts_device.begin(),
    [cam_pos = -glm::vec3(view[3])] __device__ (const glm::vec3 &pt){ return pt - cam_pos; }
  );
  auto centered_edge_world_ptr = (float*)centered_edge_world.data().get();
  auto frustrum_face_side_test = [centered_edge_world_ptr] __device__ (const glm::vec3 &pt){
    auto in = true;
    for (int i = 0; i < 4; ++i) {
      // Vector pairing {{0, 1}, {1, 2}, {2, 3}, {3, 0}}
      auto v2 = centered_edge_world_ptr + 3 * (i);
      auto v1 = centered_edge_world_ptr + 3 * ((i + 1) % 4);
      // Plane through origin, (v2 x v1) . pt
      in &= (((*(v2+1) * *(v1+2) - *(v2+2) * *(v1+1)) * pt.x +
              (*(v2+2) * *v1     - *v2     * *(v1+2)) * pt.y +
              (*v2     * *(v1+1) - *(v2+1) * *v1)     * pt.z) > 0);
    }
    return in;
  };
  thrust::transform(pts_device.begin(), pts_device.end(), is_in_frustrum.begin(), frustrum_face_side_test);
  return std::vector<bool>{is_in_frustrum.begin(), is_in_frustrum.end()};
}

std::vector<float> compute_radii(const std::vector<glm::vec4> &vertices) {
  const auto stride = 4;
  auto points = thrust::device_vector<float>(vertices.size() * stride);  // Homogeneous one.
  for (size_t idx = 0; idx < vertices.size(); ++idx)
    thrust::copy((float*)&vertices[idx], (float*)&vertices[idx] + 4, &points[idx * stride]);
  auto query = thrust::device_vector<float>(points.begin(), points.end());

  thrust::device_vector<int> indices;
  thrust::device_vector<float> distances;
  kdtree::KDTreeSearchParams params(2);
  auto tree = kdtree::KDTreeFlann();
  tree.Build(points);
  tree.Search(query, params, indices, distances);

//  thrust::copy(distances.begin(), distances.end(), ostream_iterator<float>(cout, "\n"));
  auto radii = std::vector<float>(vertices.size());
  for (size_t i = 1; i < indices.size(); i+=2) {
    radii[indices[i]] = std::clamp(distances[i] / 2, 0.0f, 5.0f);
  }
  return radii;
}
