#ifndef POINTCLOUD_RENDERER_TEXTURE_H
#define POINTCLOUD_RENDERER_TEXTURE_H

#include <stdexcept>
#include <vector>

#include <glad/glad.h>


class Texture2D {
private:
  GLuint id;
public:
  const GLsizei width, height;  // width and height of texture image
private:
  const void *data;  // pointer to the image data in host memory
  GLint internal_format;  // format of texture object
  GLenum format;  // format of the pixel data
  GLint wrap_s;  // wrapping mode on S axis
  GLint wrap_t;  // wrapping mode on T axis
  GLint min_filter;  // filtering mode if texture pixels < screen pixels
  GLint max_filter;  // filtering mode if texture pixels > screen pixels
  GLenum type;  // data type of the pixel data

public:
  Texture2D(
    GLsizei width,
    GLsizei height,
    const void *data,
    GLint internal_format = GL_RGB,
    GLenum format = GL_RGB,
    GLint wrap_s = GL_REPEAT,
    GLint wrap_t = GL_REPEAT,
    GLint min_filter = GL_LINEAR,
    GLint max_filter = GL_LINEAR,
    GLenum type = GL_FLOAT
  );

  Texture2D(
    GLsizei width,
    GLsizei height,
    const void *data,
    GLint internal_format,
    GLenum format,
    GLint wrap,
    GLint filter,
    GLenum type
  );

  ~Texture2D();

  [[nodiscard]] auto get_id() const noexcept {
    return id;
  }

  void bind() const noexcept {
    glBindTexture(GL_TEXTURE_2D, id);
  }

  void unbind() const noexcept {
    glBindTexture(GL_TEXTURE_2D, 0);
  }

  template<typename T>
  [[nodiscard]] auto get_texture_data() const {
    auto raw_data = std::vector<T>(width * height);
    glGetTexImage(GL_TEXTURE_2D, 0, format, type, (void*)raw_data.data());
    return raw_data;
  }
};

#endif //POINTCLOUD_RENDERER_TEXTURE_H

