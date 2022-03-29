#include <kdtree/kdtree_flann.h>
#include <thrust/device_vector.h>

#include "utils.h"

std::vector<Point> generate_vertex_buffer(
    std::vector<std::array<double, 3>> &vertices,
    std::vector<std::array<unsigned char, 3>> &colors
) {
  auto vbo_data = std::vector<Point>(colors.size());
  for (size_t idx = 0; idx < vertices.size(); ++idx) {
    vbo_data[idx] = Point(vertices[idx], colors[idx], 10.0f);
  }

  auto tree = kdtree::KDTreeFlann();
  auto points = thrust::device_vector<float>(colors.size() * 4);
  for (size_t i = 0; i < vertices.size(); ++i) {
    points[4*i] = vertices[i][0];
    points[4*i+1] = vertices[i][1];
    points[4*i+2] = vertices[i][2];
    points[4*i+3] = 1.0f;
  }
  tree.Build(points);
  auto query = thrust::device_vector<float>(points.begin(), points.end());
  thrust::device_vector<int> indices;
  thrust::device_vector<float> distances;
  kdtree::KDTreeSearchParams params(2);
  tree.Search(query, params, indices, distances);

//  thrust::copy(distances.begin(), distances.end(), ostream_iterator<float>(cout, "\n"));
  thrust::pair<thrust::device_vector<float>::iterator,thrust::device_vector<float>::iterator> tuple;
  tuple = thrust::minmax_element(distances.begin(),distances.end());
  auto low_data = *(tuple.first);
  auto high_data = *tuple.second;
  auto low_pixels = 1.0f;
  auto high_pixels = 200.0f;
//  std::cout << low_data << std::endl;
//  std::cout << high_data << std::endl;

  for (size_t i = 1; i < indices.size(); i+=2) {
    vbo_data[indices[i]].radius = std::clamp(
      low_pixels + (distances[i] / 2 - low_data) * (high_pixels - low_pixels) / (high_data - low_data),
      low_pixels,
      high_pixels
    );
//    if (i % 1001 == 0)
//      std::cout << vbo_data[indices[i]].radius  << std::endl;
  }
  return vbo_data;
}
