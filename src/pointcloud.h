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
    GLuint vao = 0;
    GLuint vbo = 0;

public:
    Pointcloud(
        const std::vector<std::array<double, 3>> &points,
        const std::vector<std::array<unsigned char, 3>> &colors,
        const std::vector<float> &radii);

    ~Pointcloud();

    [[nodiscard]] auto get_id() const noexcept {
      return vao;
    }

    void bind() const noexcept {
      glBindVertexArray(vao);
    }

    void unbind() const noexcept {
      glBindVertexArray(0);
    }
};


#endif //POINTCLOUD_RENDERER_POINTCLOUD_H
