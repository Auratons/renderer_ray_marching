#ifndef SHADER_H
#define SHADER_H

#include <string>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

std::string read_shader_file(const std::string &path);

GLuint compile_shader(const std::string &shader_src, GLenum shader_type);

GLuint link_shader_program(const std::string &vertex_path, const std::string &fragment_path);

#endif
