#include <cstdlib>
#include <exception>
#include <iostream>
#include <filesystem>
#include <string>

#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"
#include <cuda_runtime.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <thrust/device_vector.h>

#include "camera.h"
#include "common.h"
#include "pointcloud.h"
#include "quad.h"
#include "ray_marching.h"
#include "shader.h"
#include "texture.h"


float lastX = SCREEN_WIDTH / 2.0f;
float lastY = SCREEN_HEIGHT / 2.0f;
float deltaTime = 0.0f;
float lastFrame = 0.0f;
bool firstMouse = true;
auto camera = Camera();

using namespace std;

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

  auto [vertices, colors] = Pointcloud::load_ply(pcd_path);
  vertices.resize(10000);
  colors.resize(10000);
  auto vertices_d = thrust::device_vector<glm::vec4>(vertices.begin(), vertices.end());
  auto colors_d = thrust::device_vector<glm::vec4>(colors.begin(), colors.end());
  auto radii_d = compute_radii(vertices_d);
  cout << "Min radius: " << *std::min_element(radii_d.begin(), radii_d.end()) << endl;
  cout << "Max radius: " << *std::max_element(radii_d.begin(), radii_d.end()) << endl;
  cout << "Avg radius: " << std::accumulate(radii_d.begin(), radii_d.end(), 0.0f) / radii_d.size() << endl;

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
    auto quad = Quad(graphical_shader);
    auto texture = Texture2D(SCREEN_WIDTH, SCREEN_HEIGHT, nullptr, GL_RGBA32F, GL_RGBA, GL_CLAMP_TO_EDGE, GL_NEAREST);
    auto ray_marcher = PointcloudRayMarcher::get_instance(
      vertices_d,
      colors_d,
      radii_d,
      texture
    );

    double lastTime      = 0.0;
    unsigned int counter = 0;

    // render loop
    while (!glfwWindowShouldClose(window)) {
      auto currentFrame = static_cast<float>(glfwGetTime());
      deltaTime = currentFrame - lastFrame;
      lastFrame = currentFrame;

      process_input(window);

      ray_marcher->render_to_texture(camera.GetViewMatrix(), glm::radians(camera.Zoom));
      ray_marcher->save_png("test_image.png");
      quad.render(ray_marcher->get_texture().get_id());

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
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera.ProcessKeyboard(LEFT, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
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
