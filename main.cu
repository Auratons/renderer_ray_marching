#include <cstdlib>
#include <exception>
#include <iostream>
#include <filesystem>
#include <string>

#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>

#include "camera.h"
#include "pointcloud.h"
#include "quad.h"
#include "ray_marching.h"
#include "shader.h"
#include "utils.h"

#define SCREEN_WIDTH 1024.0
#define SCREEN_HEIGHT 768.0

float lastX = SCREEN_WIDTH / 2.0f;
float lastY = SCREEN_HEIGHT / 2.0f;
float deltaTime = 0.0f;
float lastFrame = 0.0f;
bool firstMouse = true;
auto camera = Camera();

using namespace std;

const char* gluErrorString(GLenum);
char last_error_buffer[20];

const char* gluErrorString(GLenum x) {
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


//void CHECK_CUDA(cudaError_t err);
//void CHECK_CUDA(cudaError_t err) {
//  if(err != cudaSuccess) {
//    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
//    exit(-1);
//  }
//}

void CHECK_ERROR_GL();
void CHECK_ERROR_GL() {
  GLenum err = glGetError();
  if(err != GL_NO_ERROR) {
    std::cerr << "GL Error: " << gluErrorString(err) << std::endl;
    exit(-1);
  }
}


void printWorkGroupCount();
void printWorkGroupCount() {
  int work_grp_cnt[3];

  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &work_grp_cnt[0]);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &work_grp_cnt[1]);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &work_grp_cnt[2]);

  std::cout << "max global (total) work group size x: "  << work_grp_cnt[0]
            << " y: " << work_grp_cnt[1]
            << " z: " << work_grp_cnt[2] << std::endl << std::endl;
}

void printWorkGroupSize();
void printWorkGroupSize() {
  int work_grp_size[3];

  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &work_grp_size[0]);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &work_grp_size[1]);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &work_grp_size[2]);

  std::cout << "max local (in order) work group sizes x: "  << work_grp_size[0]
            << " y: " << work_grp_size[1]
            << " z: " << work_grp_size[2] << std::endl << std::endl;
}

int printInvocations();
int printInvocations() {
  int work_grp_inv;
  glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &work_grp_inv);

  std::cout << "max local work group invocations " << work_grp_inv << std::endl << std::endl;

  return work_grp_inv;
}

GLFWwindow* init_glfw();
void init_glad();
void framebuffer_size_callback(GLFWwindow *, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void process_input(GLFWwindow *window);


int main(int argc, char** argv) {
  CLI::App args{"Pointcloud Renderer"};
  string pcd_path;
  args.add_option("-f,--file", pcd_path, "A help string");
  CLI11_PARSE(args, argc, argv);

  auto [all_vertices, all_colors] = Pointcloud::load_ply(pcd_path);

  auto indicators = filter_view_frustrum(camera.GetViewMatrix(), all_vertices, SCREEN_WIDTH / SCREEN_HEIGHT, glm::radians(camera.Zoom));

  size_t cnt = count(indicators.begin(), indicators.end(), true);
  auto vertices = decltype(all_vertices)(); vertices.reserve(cnt);
  auto colors = decltype(all_colors)(); colors.reserve(cnt);
  for (size_t i = 0; i < indicators.size(); ++i) {
    if (indicators[i]) {
      vertices.push_back(all_vertices.at(i));
      colors.push_back(all_colors.at(i));
    }
  }
  vertices.resize(1000);
  colors.resize(1000);
  auto radii = compute_radii(vertices);
  auto vertices_d = thrust::device_vector<glm::vec3>(vertices.begin(), vertices.end());
  auto colors_d = thrust::device_vector<glm::vec3>(colors.begin(), colors.end());
  auto radii_d = thrust::device_vector<float>(radii.begin(), radii.end());

  try {
    auto window = init_glfw();
    init_glad();

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    auto shader_path_vertex = filesystem::current_path() / "shaders/vertex.glsl";
    auto shader_path_fragment = filesystem::current_path() / "shaders/fragment.glsl";
    auto graphical_shader = Shader({shader_path_vertex, shader_path_fragment}, {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER});
    if (!graphical_shader.good()) {
      cerr << "Shader failure, exiting." << endl;
      return -1;
    }
    auto quad = Quad();

    GLuint texOutput;
    glGenTextures(1, &texOutput);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texOutput);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindImageTexture(0, texOutput, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    glm::mat4 model(1.0f);

    double lastTime      = 0.0;
    unsigned int counter = 0;

    auto renderer = PointcloudRayMarcher::get_instance(
      reinterpret_cast<const float3 *>(vertices_d.data().get()),
      reinterpret_cast<const float3 *>(colors_d.data().get()),
      radii_d.data().get(),
      vertices_d.size(),
      texOutput
    );

    // render loop
    while (!glfwWindowShouldClose(window)) {
      auto currentFrame = static_cast<float>(glfwGetTime());
      deltaTime = currentFrame - lastFrame;
      lastFrame = currentFrame;

      process_input(window);

      renderer->render_to_texture(model, camera.GetViewMatrix(), glm::radians(camera.Zoom));

      graphical_shader.use();
      quad.bind();
      glClear(GL_COLOR_BUFFER_BIT);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, texOutput);
      glDrawArrays(GL_TRIANGLES, 0, 6);
      quad.unbind();

      ++counter;
      auto currentTime = glfwGetTime();

      if (currentTime - lastTime >= 1.0) {
        cout << counter << " FPS" << endl;
        ++lastTime;
        counter = 0;
      }

      glfwSwapBuffers(window);
      glfwPollEvents();
    }
  }
  catch (const std::exception &) {
    glfwTerminate();
    return EXIT_FAILURE;
  }
  glfwTerminate();
  return EXIT_SUCCESS;
}

GLFWwindow* init_glfw() {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
  auto window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Pointcloud Renderer", nullptr, nullptr);
  if (!window) {
    std::cerr << "Fatal error: Failed to create GLFW window." << std::endl;
    throw std::exception();
  }
  glfwMakeContextCurrent(window);
  return window;
}

void init_glad() {
  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD." << std::endl;
    throw std::exception();
  }
}

void framebuffer_size_callback(GLFWwindow *, int width, int height) {
  glViewport(0, 0, width, height);
}

void process_input(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera.ProcessKeyboard(FORWARD, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera.ProcessKeyboard(BACKWARD, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera.ProcessKeyboard(LEFT, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera.ProcessKeyboard(RIGHT, deltaTime);
}

void mouse_callback(GLFWwindow*, double xposIn, double yposIn) {
  auto xpos = static_cast<float>(xposIn);
  auto ypos = static_cast<float>(yposIn);

  if (firstMouse)
  {
    lastX = xpos;
    lastY = ypos;
    firstMouse = false;
  }

  float xoffset = xpos - lastX;
  float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

  lastX = xpos;
  lastY = ypos;

  camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow*, double, double yoffset) {
  camera.ProcessMouseScroll(static_cast<float>(yoffset));
}
