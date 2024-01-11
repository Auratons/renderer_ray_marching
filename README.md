# Pointcloud Renderer
OpenGL-based ray marching pointcloud renderer.

The renderer models points of the point cloud as spheres with
diameters defined through the nearest neighbor. For the k-NN
problem solution, used also for view frustum culling, a fixed
GPU implementation from the [FLANN](https://github.com/flann-lib/flann)
project is used. The radii get stored in a special file with
`.radii` suffix saved alongside of the input PLY file so that
it can be caches and later reused to shorten startup time.
The renderer features headless rendering mode, an option to
load a general point cloud, render specific views stored in
a file and also a windows FPS camera mode.

The repository contains git submodules, so either clone the repository
with `--recurse-submodules` option or inside of the folder run
`git submodule init && git subbmodule update --recursive`.

Pointcloud Renderer
Usage: marcher.bin [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -f,--file TEXT              Path to PLY pointcloud to render
  -m,--matrices TEXT          Path to view matrices json for which to render pointcloud in case of headless rendering
  -o,--output_path TEXT       Path where to store renders in case of headless rendering
  -s,--max_points INT         Take exact number of points from the PLY file
  -r,--max_radius FLOAT       Filter possible outliers in radii file by settings max radius
  -d,--headless               Run headlessly without a window
  -p,--precompute-radii       Precompute radii even if already precomputed
  -i,--ignore_existing        Ignore existing renders and forcefully rewrite them


For headless rendering on a multi-gpu machine, NVIDIA drivers may prevent
running the application on other than GPU0 with a cryptic EGL error. It is
a bug of the driver, not this application.

The format of camera paramater matrices json file:

    {
      "train": {
        "<relative path against --output_path or an absolute path to the output png>": {
          "calibration_mat": [
            [
              803.802316909057,
              0.0,
              533.5,
              0.0
            ],
            [
              0.0,
              803.802316909057,
              345.0,
              0.0
            ],
            [
              0.0,
              0.0,
              1.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0,
              1.0
            ]
          ],
          "camera_pose": [
            [
              0.999810651632215,
              0.005047574637627222,
              0.01879316027290225,
              0.16393394238855838
            ],
            [
              0.006089153405678126,
              -0.9984242270151509,
              -0.05578516935530967,
              0.25632920943896725
            ],
            [
              0.018481966712650667,
              0.055889040960424845,
              -0.9982659124737038,
              1.0211781008610987
            ],
            [
              0.0,
              0.0,
              0.0,
              1.0
            ]
          ]
        }
      },
      "val": {}
    }
