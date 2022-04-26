#include <algorithm>
#include <ostream>
#include <string>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <kdtree/kdtree_flann.h>
#include <thrust/device_vector.h>

#include "utils.h"

namespace glm {
  std::ostream &operator<<(std::ostream &out, const glm::vec3 &v) {
    out << v.x << " " << v.y << " " << v.z;
    return out;
  }
}

std::string gluErrorString(GLenum x) {
  char last_error_buffer[20];
  switch(x) {
    case GL_NO_ERROR: return "GL_NO_ERROR: No error has been recorded";
    case GL_INVALID_ENUM: return "GL_INVALID_ENUM: An unacceptable value is specified for an enumerated argument. The offending command is ignored and has no other side effect than to set the error flag";
    case GL_INVALID_VALUE: return "GL_INVALID_VALUE: A numeric argument is out of range. The offending command is ignored and has no other side effect than to set the error flag";
    case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION: The specified operation is not allowed in the current state. The offending command is ignored and has no other side effect than to set the error flag";
#ifdef LEGACY_OPENGL
      case GL_STACK_OVERFLOW: return "GL_STACK_OVERFLOW: An attempt has been made to perform an operation that would cause an internal stack to overflow";
    case GL_STACK_UNDERFLOW: return "GL_STACK_UNDERFLOW: An attempt has been made to perform an operation that would cause an internal stack to underflow";
#endif
    case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY: There is not enough memory left to execute the command. The state of the GL is undefined, except for the state of the error flags, after this error is recorded";
    case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION: The framebuffer object is not complete. The offending command is ignored and has no other side effect than to set the error flag";
#ifdef ARB_KHR_robustness
      // https://www.opengl.org/wiki/OpenGL_Error
    case GL_CONTEXT_LOST: return "GL_CONTEXT_LOST: Given if the OpenGL context has been lost, due to a graphics card reset";
#endif
    default: sprintf (last_error_buffer, "0x%X", x); return last_error_buffer;
  }
}

void print_workgroup_count() {
  int work_grp_cnt[3];

  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &work_grp_cnt[0]);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &work_grp_cnt[1]);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &work_grp_cnt[2]);

  std::cout << "max global (total) work group size x: "  << work_grp_cnt[0]
            << " y: " << work_grp_cnt[1]
            << " z: " << work_grp_cnt[2] << std::endl << std::endl;
}

void print_workgroup_size() {
  int work_grp_size[3];

  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &work_grp_size[0]);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &work_grp_size[1]);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &work_grp_size[2]);

  std::cout << "max local (in order) work group sizes x: "  << work_grp_size[0]
            << " y: " << work_grp_size[1]
            << " z: " << work_grp_size[2] << std::endl << std::endl;
}

int print_invocations() {
  int work_grp_inv;
  glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &work_grp_inv);

  std::cout << "max local work group invocations " << work_grp_inv << std::endl << std::endl;

  return work_grp_inv;
}

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

thrust::device_vector<float> compute_radii(const thrust::device_vector<glm::vec4> &vertices) {
  auto points = thrust::device_vector<float>(thrust::device_ptr<float>((float*)vertices.data().get()), thrust::device_ptr<float>((float*)vertices.data().get()) + 4 * vertices.size());  // Homogeneous one.
  auto query = thrust::device_vector<float>(points.begin(), points.end());

  thrust::device_vector<int> indices;
  thrust::device_vector<float> distances;
  kdtree::KDTreeSearchParams params(2);
  auto tree = kdtree::KDTreeFlann();
  tree.Build(points);
  tree.Search(query, params, indices, distances);

//  thrust::copy(distances.begin(), distances.end(), std::ostream_iterator<float>(std::cout, "\n"));
  for (size_t i = 1; i < indices.size(); i+=2) {
    distances[i / 2] = distances[i];
  }
  distances.resize(distances.size() / 2);
  thrust::transform(
    distances.begin(), distances.end(),
    distances.begin(),
    [] __device__ (const float &i){ return (i > 10.0f) ? 10.0f : i; }
  );

  return distances;
}
