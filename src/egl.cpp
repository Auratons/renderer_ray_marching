#include <stdexcept>

#include "egl.hpp"

std::string egl_error_string(EGLint error) {
  switch(error){
    case EGL_NOT_INITIALIZED: return "EGL_NOT_INITIALIZED: EGL is not initialized, or could not be initialized, for the specified EGL display connection.";
    case EGL_BAD_ACCESS: return "EGL_BAD_ACCESS: EGL cannot access a requested resource (for example a context is bound in another thread).";
    case EGL_BAD_ALLOC: return "EGL_BAD_ALLOC: EGL failed to allocate resources for the requested operation.";
    case EGL_BAD_ATTRIBUTE: return "EGL_BAD_ATTRIBUTE: An unrecognized attribute or attribute value was passed in the attribute list.";
    case EGL_BAD_CONTEXT: return "EGL_BAD_CONTEXT: An EGLContext argument does not name a valid EGL rendering context.";
    case EGL_BAD_CONFIG: return "EGL_BAD_CONFIG: An EGLConfig argument does not name a valid EGL frame buffer configuration.";
    case EGL_BAD_CURRENT_SURFACE: return "EGL_BAD_CURRENT_SURFACE: The current surface of the calling thread is a window, pixel buffer or pixmap that is no longer valid.";
    case EGL_BAD_DISPLAY: return "EGL_BAD_DISPLAY: An EGLDisplay argument does not name a valid EGL display connection.";
    case EGL_BAD_SURFACE: return "EGL_BAD_SURFACE: An EGLSurface argument does not name a valid surface (window, pixel buffer or pixmap) configured for GL rendering.";
    case EGL_BAD_MATCH: return "EGL_BAD_MATCH: Arguments are inconsistent (for example, a valid context requires buffers not supplied by a valid surface).";
    case EGL_BAD_PARAMETER: return "EGL_BAD_PARAMETER: One or more argument values are invalid.";
    case EGL_BAD_NATIVE_PIXMAP: return "EGL_BAD_NATIVE_PIXMAP: A NativePixmapType argument does not refer to a valid native pixmap.";
    case EGL_BAD_NATIVE_WINDOW: return "EGL_BAD_NATIVE_WINDOW: A NativeWindowType argument does not refer to a valid native window.";
    case EGL_CONTEXT_LOST: return "EGL_CONTEXT_LOST: A power management event has occurred. The application must destroy all contexts and reinitialise OpenGL ES state and objects to continue rendering.";
    default:
      throw std::runtime_error("Failed to configure EGL Display: unknown error");
  }
}

EGLDisplay  init_egl() {
  static const EGLint configAttribs[] = {
          EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
          EGL_BLUE_SIZE, 8,
          EGL_GREEN_SIZE, 8,
          EGL_RED_SIZE, 8,
          EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
          EGL_NONE
  };

  EGLDisplay eglDpy;

  // 1. Initialize EGL
  EGLint major, minor, assigned_gpu_idx = 0;

  auto get_env = [](auto env_var_name) {
      const char *tmp = getenv(env_var_name);
      return std::string(tmp ? tmp : "");
  };
  auto slurm_step_gpus = get_env("SLURM_STEP_GPUS");  // Interactive Slurm job
  auto slurm_job_gpus = get_env("SLURM_JOB_GPUS");  // Batch Slurm mode

  // Running outside Slurm, not covering CUDA_VISIBLE_DEVICES
  if (!slurm_job_gpus.empty() || !slurm_step_gpus.empty()) {
    CHECK_ERROR_EGL(eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY));
  }
  else {  // Running inside Slurm with or without cgroups
    EGLint actualDeviceCount;
    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT;
    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT;

    // Running inside Slurm with or without cgroups. If running outside Slurm (not covering CUDA_VISIBLE_DEVICES),
    // assigned_gpu_idx is 0, casted properly below the same way as EGL_DEFAULT_DISPLAY is.
    if (!slurm_job_gpus.empty() || !slurm_step_gpus.empty()) {
      auto slurm_gpu = (slurm_step_gpus.empty()) ? slurm_job_gpus : slurm_step_gpus;
      assigned_gpu_idx = stoi(slurm_gpu);
    }
    std::cout << "Using GPU " << assigned_gpu_idx << "." << std::endl;
    CHECK_ERROR_EGL(eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC) eglGetProcAddress("eglQueryDevicesEXT"));
    CHECK_ERROR_EGL(eglQueryDevicesEXT(0, nullptr, &actualDeviceCount));
    EGLDeviceEXT actualEGLDevices[actualDeviceCount];
    CHECK_ERROR_EGL(eglQueryDevicesEXT(actualDeviceCount, actualEGLDevices, &actualDeviceCount));
    CHECK_ERROR_EGL(
            eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC) eglGetProcAddress("eglGetPlatformDisplayEXT")
    );
    CHECK_ERROR_EGL(
            eglDpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, actualEGLDevices[assigned_gpu_idx], nullptr)
    );
  }

  CHECK_ERROR_EGL(eglInitialize(eglDpy, &major, &minor));
  // 2. Select an appropriate configuration
  EGLint numConfigs;
  EGLConfig eglCfg;
  CHECK_ERROR_EGL(eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs));
  // 3. Create a surface
  EGLSurface eglSurf;
  CHECK_ERROR_EGL(eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, nullptr));
  // 4. Bind the API
  CHECK_ERROR_EGL(eglBindAPI(EGL_OPENGL_API));
  // 5. Create a context and make it current
  EGLContext eglCtx;
  CHECK_ERROR_EGL(eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, nullptr));
  CHECK_ERROR_EGL(eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx));
  return eglDpy;
}
