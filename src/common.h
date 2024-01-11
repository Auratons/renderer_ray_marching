#ifndef POINTCLOUD_RENDERER_COMMON_H
#define POINTCLOUD_RENDERER_COMMON_H

#define ZFAR 100.0f
#define ZNEAR 0.1f
#define MAX_STEPS 128
#define MIN_DIST 0.001f
// The missed pixels get depth 0, when reprojected
// to 3d, there's gonna be a set of points at origin,
// for InLoc, these should be exchanged for NaNs.
#define BACKGROUND_COLOR make_float4(0,0,0,0)

#define SCREEN_WIDTH 1024.0
#define SCREEN_HEIGHT 768.0

#define PI 3.141592653589793238f

#endif //POINTCLOUD_RENDERER_COMMON_H
