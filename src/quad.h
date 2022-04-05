#ifndef POINTCLOUD_RENDERER_QUAD_H
#define POINTCLOUD_RENDERER_QUAD_H

#include <glad/glad.h>

class Quad {
private:
    GLuint vao = 0;
    GLuint vbo = 0;

public:
    Quad();
    ~Quad();

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


#endif //POINTCLOUD_RENDERER_QUAD_H
