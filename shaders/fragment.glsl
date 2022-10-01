#version 330 core

in vec2 tex;

uniform sampler2D rendered_image;

out vec4 frag_color;

void main() {
    frag_color = texture(rendered_image, tex);
}
