#version 330 core

layout (location = 0) in vec3 pos_;
layout (location = 1) in vec2 tex_;

out vec2 tex;

void main() {
	gl_Position = vec4(pos_, 1.0f);
	tex = tex_;
}
