#version 430 core
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable

#define ZFAR 100
#define MAX_STEPS 128
#define MIN_DIST 0.001
#define BACKGROUND_COLOR vec4(1)

struct RayHit {
    float hit_dist;
    int hit_index;
};

struct Point {
    vec4 pos;
    vec4 color_radius;
};

layout(local_size_x = 32, local_size_y = 32) in;
layout(rgba32f, binding = 0) uniform image2D rendered_image;
layout(std430, binding = 1) buffer Points {
    Point points[];
};

uniform mat4 model;
uniform mat4 view;
uniform float fov_radians;

RayHit sdf(vec4 pos);
int ray_march(vec4 rayOrigin, vec4 rayDir);

RayHit sdf(vec4 pos)
{
    int min_index = 0;
    float dist, radius, min_dist = 1.0 / 0.0;  // MAX_FLOAT
    vec4 position;
    for (int i = 0; i < points.length(); ++i) {
        position = points[i].pos;
        radius = points[i].color_radius.w;
        dist = length(position - pos) - radius;  // 4th elem is 1-1 for homogeneous coords
        if (dist < min_dist) {
            min_dist = dist;
            min_index = i;
        }
    }
    return RayHit(min_dist, min_index);
}

int ray_march(vec4 rayOrigin, vec4 rayDir)
{
    float total_distance_travelled = 0.0f;

    for (int i = 0; i < MAX_STEPS; i++) {
        vec4 current_position = vec4(rayOrigin.xyz + rayDir.xyz * total_distance_travelled, 1);
        RayHit res = sdf(current_position);
        total_distance_travelled += res.hit_dist;
        if (res.hit_dist < MIN_DIST) {
            return res.hit_index;
        }
        if (total_distance_travelled > ZFAR)
            break;
    }

    return -1;
}


void main()
{
    // Compute screen-centered x, y coordinates
    const ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 dims = imageSize(rendered_image);
    if (pixel_coords.x >= dims.x || pixel_coords.y >= dims.y) {
        return;  // For non-square textures
    }

    float x    = (float(pixel_coords.x * 2 - dims.x) / dims.x);
    float y    = (float(pixel_coords.y * 2 - dims.y) / dims.y);

    mat4 model_view = view * model;
    // We could look from origin of camera space and transform points'
    // positions, but this is just one matrix operation compared to many.
    int color_index = ray_march(
        model_view[3],
        vec4(normalize((model_view * vec4(x, y, fov_radians, 0)).xyz), 1)
    );
    vec4 finalColor = (color_index >= 0) ? vec4(points[color_index].color_radius.xyz, 1.0) : BACKGROUND_COLOR;

    imageStore(rendered_image, pixel_coords, finalColor);
}
