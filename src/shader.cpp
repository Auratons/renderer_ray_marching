#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "shader.h"

Shader::Shader(const std::vector<std::string> &paths, const std::vector<GLenum> &types) {
  if (paths.size() != types.size())
    throw std::runtime_error("Shaders paths and types do not have matching lengths.");
  std::vector<GLuint> shaders;
  std::transform(
      paths.begin(), paths.end(),
      types.begin(),
      std::back_inserter(shaders),
      [](const std::string &path, GLenum type){ return compile_shader(path, type); }
  );
  auto all_valid = std::accumulate(
      shaders.begin(), shaders.end(),
      true,
      [](GLuint a, GLuint b){ return bool(a) && bool(b); }
  );
  if (!all_valid) {
    for (auto &&shader : shaders) {
      if (shader) glDeleteShader(shader);
    }
    ID = 0;
    return;
  }

  GLint shader_program = glCreateProgram();
  for (auto &&shader : shaders)
    glAttachShader(shader_program, shader);
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

  for (auto &&shader : shaders)
    glDeleteShader(shader);

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
