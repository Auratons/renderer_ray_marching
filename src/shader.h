#ifndef SHADER_H
#define SHADER_H

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <tuple>
#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>

class Shader
{
private:
    GLint ID = 0;

public:
    Shader(const std::vector<std::string> &paths, const std::vector<GLenum> &types);

    [[nodiscard]] auto get_id() const noexcept {
      return ID;
    }

    /*
     * Use the shader for rendering.
     */
    void use() const noexcept {
      glUseProgram(ID);
    }

    [[nodiscard]] bool good() const noexcept {
      return ID;
    }

    void set_bool(const char *name, bool value) const noexcept {
      glUniform1i(glGetUniformLocation(ID, name), (int)value);
    }
    void set_bool(const std::string &name, bool value) const noexcept {
      set_bool(name.c_str(), value);
    }

    void set_int(const char *name, int value) const noexcept {
      glUniform1i(glGetUniformLocation(ID, name), value);
    }
    void set_int(const std::string &name, int value) const noexcept {
      set_int(name.c_str(), value);
    }

    void set_uint(const char *name, unsigned int *value) const noexcept {
      glUniform1uiv(glGetUniformLocation(ID, name), 1, value);
    }
    void set_int(const std::string &name, unsigned int *value) const noexcept {
      set_uint(name.c_str(), value);
    }

    void set_float(const char *name, float value) const noexcept {
      glUniform1f(glGetUniformLocation(ID, name), value);
    }
    void set_float(const std::string &name, float value) const noexcept {
      set_float(name.c_str(), value);
    }

    void set_vec2(const char *name, const glm::vec2 &value) const noexcept {
      glUniform2fv(glGetUniformLocation(ID, name), 1, &value[0]);
    }
    void set_vec2(const std::string &name, const glm::vec2 &value) const noexcept {
      set_vec2(name.c_str(), value);
    }

    void set_vec2(const char *name, float x, float y) const noexcept {
      glUniform2f(glGetUniformLocation(ID, name), x, y);
    }
    void set_vec2(const std::string &name, float x, float y) const noexcept {
      set_vec2(name.c_str(), x, y);
    }

    void set_vec3(const char *name, const glm::vec3 &value) const noexcept {
      glUniform3fv(glGetUniformLocation(ID, name), 1, &value[0]);
    }
    void set_vec3(const std::string &name, const glm::vec3 &value) const noexcept {
      set_vec3(name.c_str(), value);
    }

    void set_vec3(const char *name, float x, float y, float z) const noexcept {
      glUniform3f(glGetUniformLocation(ID, name), x, y, z);
    }
    void set_vec3(const std::string &name, float x, float y, float z) const noexcept {
      set_vec3(name.c_str(), x, y, z);
    }

    void set_vec4(const char *name, const glm::vec4 &value) const noexcept {
      glUniform4fv(glGetUniformLocation(ID, name), 1, &value[0]);
    }
    void set_vec4(const std::string &name, const glm::vec4 &value) const noexcept {
      set_vec4(name.c_str(), value);
    }

    void set_vec4(const char *name, float x, float y, float z, float w) const noexcept {
      glUniform4f(glGetUniformLocation(ID, name), x, y, z, w);
    }
    void set_vec4(const std::string &name, float x, float y, float z, float w) const noexcept {
      set_vec4(name.c_str(), x, y, z, w);
    }

    void set_mat2(const char *name, const glm::mat2 &mat) const noexcept {
      glUniformMatrix2fv(glGetUniformLocation(ID, name), 1, GL_FALSE, &mat[0][0]);
    }
    void set_mat2(const std::string &name, const glm::mat2 &mat) const noexcept {
      set_mat2(name.c_str(), mat);
    }

    void set_mat3(const char *name, const glm::mat3 &mat) const noexcept {
      glUniformMatrix3fv(glGetUniformLocation(ID, name), 1, GL_FALSE, &mat[0][0]);
    }
    void set_mat3(const std::string &name, const glm::mat3 &mat) const noexcept {
      set_mat3(name.c_str(), mat);
    }

    void set_mat4(const char *name, const glm::mat4 &mat) const noexcept {
      glUniformMatrix4fv(glGetUniformLocation(ID, name), 1, GL_FALSE, &mat[0][0]);
    }
    void set_mat4(const std::string &name, const glm::mat4 &mat) const noexcept {
      set_mat4(name.c_str(), mat);
    }

private:
    static std::string read_shader_file(const std::string &path);

    static GLuint compile_shader(const std::string &shader_src, GLenum shader_type);
};

#endif
