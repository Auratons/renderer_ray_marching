#ifndef POINTCLOUD_RENDERER_QUAD_H
#define POINTCLOUD_RENDERER_QUAD_H

#include <glad/glad.h>

#include "shader.h"


class Quad {
private:
    GLuint vao = 0;
    GLuint vbo = 0;
    Shader shader;

public:
    /*
     * Constructs 6 vertices for quad displaying  texture.
     * Shader is expected to use one sampler2D uniform under shader unit 0.
     */
    explicit Quad(const Shader &shader);
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

    void render(GLuint texture_id) const noexcept;
};


#endif //POINTCLOUD_RENDERER_QUAD_H
