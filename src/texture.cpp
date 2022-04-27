#include "texture.h"


Texture2D::Texture2D(
    GLsizei width, GLsizei height, const void *data, GLint internal_format, GLenum format,
    GLint wrap_s, GLint wrap_t, GLint min_filter, GLint max_filter, GLenum type) :
    width(width), height(height), data(data), internal_format(internal_format), format(format), wrap_s(wrap_s),
    wrap_t(wrap_t), min_filter(min_filter), max_filter(max_filter), type(type) {
  glGenTextures(1, &id);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, max_filter);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter);
  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, type, data);
  glBindImageTexture(0, id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
}

Texture2D::~Texture2D() {
  glDeleteTextures(1, &id);
}

Texture2D::Texture2D(
    GLsizei width, GLsizei height, const void *data, GLint internal_format, GLenum format,
    GLint wrap, GLint filter, GLenum type) : Texture2D(
    width, height, data, internal_format, format, wrap, wrap, filter, filter, type) {
}
