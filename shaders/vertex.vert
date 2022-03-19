#version 330 core

#extension GL_ARB_explicit_attrib_location : require

layout (location = 0) in vec3 pos;

void main() {
	gl_Position = vec4(pos, 1.0);
}
