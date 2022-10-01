#ifndef RENDERER_SURFACE_SPLATTING_EGL_HPP
#define RENDERER_SURFACE_SPLATTING_EGL_HPP

#define EGL_EGLEXT_PROTOTYPES

#include <iostream>
#include <string>

#include <EGL/egl.h>
#include <EGL/eglext.h>

#define CHECK_ERROR_EGL(EXPRESSION) { \
  EXPRESSION; \
  EGLint err = eglGetError(); \
  if(err != EGL_SUCCESS) { \
    std::cerr << "EGL ERROR: " __FILE__ << " " << __LINE__ << " " << egl_error_string(err) << std::endl; \
    exit(1); \
  } \
}

// Supports all error codes listed here: https://www.khronos.org/registry/EGL/sdk/docs/man/html/eglGetError.xhtml
std::string egl_error_string(EGLint);

EGLDisplay  init_egl();

#endif //RENDERER_SURFACE_SPLATTING_EGL_HPP