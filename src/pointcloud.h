#ifndef POINTCLOUD_RENDERER_POINTCLOUD_H
#define POINTCLOUD_RENDERER_POINTCLOUD_H

#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>

class Pointcloud
{
private:
    GLuint ssbo = 0;
    GLuint index = 0;

public:
    Pointcloud(
        const std::vector<glm::vec3> &points,
        const std::vector<glm::vec3> &colors,
        const std::vector<float> &radii);

    ~Pointcloud() {
      glDeleteBuffers(1, &ssbo);
    }

    [[nodiscard]] auto get_id() const noexcept {
      return ssbo;
    }

    void bind(GLuint idx) noexcept {
      index = idx;
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, ssbo);
    }

    void unbind() noexcept {
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, 0);
      index = 0;
    }

    static std::tuple<std::vector<glm::vec4>, std::vector<glm::vec4>> load_ply(const std::string &path);
};


#endif //POINTCLOUD_RENDERER_POINTCLOUD_H
