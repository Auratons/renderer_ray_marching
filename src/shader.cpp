#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "shader.h"

Shader::Shader(const std::string &vertex_path, const std::string &fragment_path) {
  auto vertex_shader = compile_shader(vertex_path, GL_VERTEX_SHADER);
  auto fragment_shader = compile_shader(fragment_path, GL_FRAGMENT_SHADER);
  if (!vertex_shader or !fragment_shader) {
    if (vertex_shader) glDeleteShader(vertex_shader);
    if (fragment_shader) glDeleteShader(fragment_shader);
    ID = 0;
    return;
  }

  GLint shader_program = glCreateProgram();
  glAttachShader(shader_program, vertex_shader);
  glAttachShader(shader_program, fragment_shader);
  glBindFragDataLocation(shader_program, 0, "out_color");
  glLinkProgram(shader_program);

  auto status = GL_FALSE;
  glGetProgramiv(shader_program, GL_LINK_STATUS, &status);
  if (!status) {
    std::cerr << "Failed to link shaders." << std::endl;

    GLint log_length = 0;
    glGetProgramiv(shader_program, GL_INFO_LOG_LENGTH, &log_length);
    if (log_length >= 1) {
      std::cerr << "Log message from the linking process:" << std::endl;
      auto linking_error_log = new GLchar[log_length];
      glGetProgramInfoLog(shader_program, log_length, nullptr, linking_error_log);
      std::cerr << linking_error_log << std::endl;
    }

    glDeleteProgram(shader_program);
    shader_program = 0;
  }

  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);

  ID = shader_program;
}

std::string Shader::read_shader_file(const std::string &path) {
  std::stringstream fs_content;
  std::ifstream fs;
  fs.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  try {
    fs.open(path, std::ios::in);
    fs_content << fs.rdbuf();
  }
  catch (std::ifstream::failure& e) {
    std::cerr << "Could not read file " << path << ": " << e.what() << std::endl;
    return "";
  }

  return fs_content.str();
}

GLuint Shader::compile_shader(const std::string &shader_file_path, GLenum shader_type) {
  auto shader_source = read_shader_file(shader_file_path);
  auto shader_source_c = shader_source.c_str();
  auto shader = glCreateShader(shader_type);

  glShaderSource(shader, 1, &shader_source_c, nullptr);
  glCompileShader(shader);

  auto status = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
  if (!status) {
    std::cerr << "Failed to compile shader " << shader_file_path << "." << std::endl;

    GLint log_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
    if (log_length >= 1) {
      std::cerr << "Log message from the compilation:" << std::endl;
      auto shader_error_log = new GLchar[log_length];
      glGetShaderInfoLog(shader, log_length, nullptr, shader_error_log);
      std::cerr << shader_error_log << std::endl;
    }

    glDeleteShader(shader);
    return 0;
  }

  return shader;
}
