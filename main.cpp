#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <happly.h>

#include "camera.h"
#include "pointcloud.h"
#include "shader.h"
#include "utils.h"

#define SCREEN_WIDTH 1024.0
#define SCREEN_HEIGHT 768.0

float lastX = SCREEN_WIDTH / 2.0f;
float lastY = SCREEN_HEIGHT / 2.0f;
float deltaTime = 0.0f;
float lastFrame = 0.0f;
bool firstMouse = true;
auto camera = Camera(glm::vec3(0.0f, 0.0f, 3.0f));

using namespace std;

void init_glfw();
void framebuffer_size_callback(GLFWwindow *, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void process_input(GLFWwindow *window);


int main(int argc, char** argv) {
  CLI::App args{"Pointcloud Renderer"};
  string pcd_path;
  args.add_option("-f,--file", pcd_path, "A help string");
  CLI11_PARSE(args, argc, argv);

  happly::PLYData ply(pcd_path);
  auto vertices = ply.getVertexPositions();
  auto colors = ply.getVertexColors();
  auto radii = compute_radii(vertices, colors);

  try {
    init_glfw();
    auto window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Pointcloud Renderer", nullptr, nullptr);
    if (!window) {
      std::cerr << "Fatal error: Failed to create GLFW window." << std::endl;
      throw std::exception();
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
      std::cerr << "Failed to initialize GLAD." << std::endl;
      throw std::exception();
    }

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Global OpenGL state
    glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_DEPTH_TEST);

    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
    glDepthRange(0.0, 1.0);

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    auto pcd = Pointcloud(vertices, colors, radii);
    vector<string> paths{CMAKE_SOURCE_DIR "/shaders/vertex.vert", CMAKE_SOURCE_DIR "/shaders/fragment.frag"};
    vector<GLenum> types{GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
    auto shader = Shader(paths, types);
    if (!shader.good()) {
      cerr << "Shader failure, exiting." << endl;
      return -1;
    }
    shader.use();

    glm::mat4 model(1.0f);

    // render loop
    while (!glfwWindowShouldClose(window)) {
      auto currentFrame = static_cast<float>(glfwGetTime());
      deltaTime = currentFrame - lastFrame;
      lastFrame = currentFrame;

      process_input(window);

      shader.set_mat4("projective_", glm::perspective(glm::radians(camera.Zoom), (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT, 0.1f, 100.0f));
      shader.set_mat4("model_", model);
      shader.set_mat4("view_", camera.GetViewMatrix());

      pcd.bind();
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glDrawArrays(GL_POINTS, 0, (GLsizei)radii.size());
      pcd.unbind();

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

void init_glfw() {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
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
