#version 330 core

layout (location = 0) in vec3 pos_;
layout (location = 1) in vec3 color_;
layout (location = 2) in float radius_;

uniform mat4 model_;
uniform mat4 view_;
uniform mat4 projective_;

out vec4 color;
out float radius;

void main() {
	gl_Position = projective_ * view_ * model_  * vec4(pos_, 1.0f);
	gl_PointSize = radius_;
	color = vec4(color_, 1.0f);
}
