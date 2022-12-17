#include <algorithm>
#include <limits>
#include <ostream>
#include <string>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <kdtree/kdtree_flann.h>
#include <opencv2/imgcodecs.hpp>
#include <stb_image_write.h>
#include <thrust/device_vector.h>

#include "npy.hpp"
#include "utils.h"

std::string glu_error_string(GLenum x) {
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

void init_glad(GLADloadproc pointer) {
  if (!gladLoadGLLoader(pointer)) {
    std::cerr << "Failed to initialize GLAD." << std::endl;
    throw std::exception();
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

thrust::device_vector<float> compute_radii(const thrust::device_vector<glm::vec4> &vertices) {
  auto points = thrust::device_vector<float>(thrust::device_ptr<float>((float*)vertices.data().get()), thrust::device_ptr<float>((float*)vertices.data().get()) + 4 * vertices.size());
  auto query = thrust::device_vector<float>(points.begin(), points.end());

  thrust::device_vector<int> indices;
  thrust::device_vector<float> distances;
  kdtree::KDTreeSearchParams params(2);
  auto tree = kdtree::KDTreeFlann();
  tree.Build(points);
//  tree.Build((float*)thrust::raw_pointer_cast(vertices.data()), vertices.size());
  tree.Search(query, params, indices, distances);

//  thrust::copy(distances.begin(), distances.end(), std::ostream_iterator<float>(std::cout, "\n"));
  for (size_t i = 1; i < indices.size(); i+=2) {
    distances[i / 2] = std::sqrt(distances[i]);  // KD tree returns squared distance
  }
  distances.resize(distances.size() / 2);
  thrust::transform(
    distances.begin(), distances.end(),
    distances.begin(),
    [] __device__ (const float &i){ return (i > 10.0f) ? 10.0f : i; }
  );

  return distances;
}

void FPSCounter::update() {
  ++counter;

  if (glfwGetTime() - lastTime >= 1.0) {
    std::cout << counter << " FPS" << std::endl;
    ++lastTime;
    counter = 0;
  }
}

std::ostream &glm::operator<<(std::ostream &out, const glm::vec3 &v) {
  out << v.x << " " << v.y << " " << v.z;
  return out;
}

std::ostream &glm::operator<<(std::ostream &out, const glm::vec4 &v) {
  out << v.x << " " << v.y << " " << v.z << " " << v.w;
  return out;
}

std::ostream &glm::operator<<(std::ostream &out, const glm::mat4 &m) {
  std::ios out_state(nullptr);
  out_state.copyfmt(out);
  out << std::setprecision(std::numeric_limits<float>::digits10)
      << std::fixed
      <<  "[[ " << m[0][0] << ", " << m[1][0] << ", " << m[2][0] << ", " << m[3][0]
      << " ][ " << m[0][1] << ", " << m[1][1] << ", " << m[2][1] << ", " << m[3][1]
      << " ][ " << m[0][2] << ", " << m[1][2] << ", " << m[2][2] << ", " << m[3][2]
      << " ][ " << m[0][3] << ", " << m[1][3] << ", " << m[2][3] << ", " << m[3][3] << " ]]";
  out.copyfmt(out_state);
  return out;
}

void glm::from_json(const nlohmann::json &j, glm::mat4 &P) {
  // Create matrix from row-major to column-major
  auto row0 = j[0].get<std::vector<float>>();
  auto row1 = j[1].get<std::vector<float>>();
  auto row2 = j[2].get<std::vector<float>>();
  auto row3 = j[3].get<std::vector<float>>();
  P = glm::mat4(
    row0[0], row1[0], row2[0], row3[0],
    row0[1], row1[1], row2[1], row3[1],
    row0[2], row1[2], row2[2], row3[2],
    row0[3], row1[3], row2[3], row3[3]
  );
}

void save_png(const std::string &filename, const Texture2D &texture) {
  glActiveTexture(GL_TEXTURE0);
  auto raw_data = texture.get_texture_data<float4>();
  std::transform(raw_data.begin(), raw_data.end(), raw_data.begin(), [](auto &val){ val.w = 1.0; return val; });
  auto png = std::vector<unsigned char>(4 * texture.width * texture.height);  // 4=RGBA
  auto begin = (const float*)raw_data.data();
  auto end = (const float*)(raw_data.data() + raw_data.size());
  std::transform(begin, end, png.begin(), [](const float &val){ return (unsigned char)(val * 255.0f); });
  // OpenGL expects the 0.0 coordinate on the y-axis to be on the bottom side of the image, but images usually
  // have 0.0 at the top of the y-axis. For now, this unifies output with the visualisation on the screen.
  stbi_flip_vertically_on_write(true);
  stbi_write_png(filename.c_str(), texture.width, texture.height, 4, png.data(), 4 * texture.width);  // 4=RGBA
}

void save_depth(const std::string &filename, const Texture2D &texture) {
  glActiveTexture(GL_TEXTURE0);
  auto raw_quartets = texture.get_texture_data<float4>();
  auto raw_data = std::vector<float>(texture.width * texture.height);
  std::transform(raw_quartets.begin(), raw_quartets.end(), raw_data.begin(), [](const auto &val){ return val.w; });
  // OpenGL expects the 0.0 coordinate on the y-axis to be on the bottom side of the image, but images usually
  // have 0.0 at the top of the y-axis. For now, this unifies output with the visualisation on the screen.
  for (int r = 0; r < (texture.height/2); ++r)
  {
    for (int c = 0; c != texture.width; ++c)
    {
      std::swap(raw_data[r * texture.width + c], raw_data[(texture.height - 1 - r) * texture.width + c]);
    }
  }
  auto png = std::vector<uint8_t>(texture.width * texture.height);
  auto begin = (const float*)raw_data.data();
  auto end = (const float*)(raw_data.data() + raw_data.size());
  std::transform(begin, end, png.begin(), [](const float &val){
      if (std::abs(val - 1.0f) < 0.00001f)  // To reflect texture of empty spaces handling in the original article.
        return (uint8_t)0;
      return (uint8_t)std::clamp((255.0f / 100.0f) * val, 0.0f, 255.0f); }
  );
  const std::vector<long unsigned> shape{(long unsigned)texture.height, (long unsigned)texture.width};
  const bool fortran_order{false};
  npy::SaveArrayAsNumpy(filename + ".npy", fortran_order, shape.size(), shape.data(), raw_data);
  auto img = cv::Mat(texture.height, texture.width, CV_8UC1, png.data());
  cv::imwrite(filename, img);
}
