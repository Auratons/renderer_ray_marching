#include <kdtree/kdtree_flann.h>
#include <thrust/device_vector.h>

#include "utils.h"

std::vector<float> compute_radii(std::vector<std::array<double, 3>> &vertices) {
  const auto stride = 4;
  auto points = thrust::device_vector<float>(vertices.size() * stride, 1.0f);  // Homogeneous one.
  for (size_t idx = 0; idx < vertices.size(); ++idx)
    thrust::copy(vertices[idx].begin(), vertices[idx].end(), &points[idx * stride]);
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
    radii[indices[i]] = distances[i] / 2;
  }
  return radii;
}
