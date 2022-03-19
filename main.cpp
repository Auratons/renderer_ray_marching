#include <cstdlib>
#include <iostream>
#include <exception>
#include <string>

#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "happly.h"
#include "shader.h"

#define SCREEN_WIDTH 1024.0
#define SCREEN_HEIGHT 768.0

using namespace std;

void init_glfw();

void framebuffer_size_callback(GLFWwindow *, int width, int height);

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);

int main() {
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

    glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    GLuint shader = link_shader_program("shaders/vertex.vert", "shaders/fragment.frag");

    // VAO
    GLuint vao;
    glGenVertexArrays(1, &vao);
    // VBO
    GLuint vbo;
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    const static GLfloat vertices[] = {
      -1.0f,  1.0f, 0.0f,
       1.0f,  1.0f, 0.0f,
       1.0f, -1.0f, 0.0f,
       1.0f, -1.0f, 0.0f,
      -1.0f, -1.0f, 0.0f,
      -1.0f,  1.0f, 0.0f
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) 0);
    glEnableVertexAttribArray(0);

    float resolution[2] = {SCREEN_WIDTH, SCREEN_HEIGHT};
    glUniform2fv(glGetUniformLocation(shader, "iResolution"), 1, &resolution[0]);

    // unbind
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // render loop
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      glClear(GL_COLOR_BUFFER_BIT);
      glBindVertexArray(vao);
      glUseProgram(shader);
      glUniform1f(glGetUniformLocation(shader, "iTime"), glfwGetTime());
      glUniform2fv(glGetUniformLocation(shader, "iResolution"), 1, &resolution[0]);
      glBindBuffer(GL_ARRAY_BUFFER, vbo);
      glDrawArrays(GL_TRIANGLES, 0, 6);
      glBindVertexArray(0);
      glfwSwapBuffers(window);
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
