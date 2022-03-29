#ifndef POINTCLOUD_RENDERER_UTILS_H
#define POINTCLOUD_RENDERER_UTILS_H

#include <array>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#undef glBufferData

template <class T>
inline void glBufferData(GLenum target, const std::vector<T>& v, GLenum usage) {
  glad_glBufferData(target, v.size() * sizeof(T), &v[0], usage);
}

struct Point {
    GLfloat x, y, z;
    GLfloat r, g, b;
    GLfloat radius;

    Point& operator=(Point&&) = default;
    Point() : x(0.0f), y(0.0f), z(0.0f), r(0.0f), g(0.0f), b(0.0f), radius(0.0f) {}
    Point(const std::array<double, 3> &xyz, const std::array<unsigned char, 3> &color, GLfloat radius) :
    x(xyz[0]), y(xyz[1]), z(xyz[2]), r(color[0] / 255.0f), g(color[1] / 255.0f), b(color[2] / 255.0f), radius(radius) {
    }
};

std::vector<Point> generate_vertex_buffer(
    std::vector<std::array<double, 3>> &vertices,
    std::vector<std::array<unsigned char, 3>> &colors
);

#endif //POINTCLOUD_RENDERER_UTILS_H
