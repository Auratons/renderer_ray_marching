cmake_minimum_required(VERSION 3.16)
project(pointcloud_renderer LANGUAGES CUDA C CXX)

if(NOT "${CUDA_ARCH}" MATCHES "^[0-9][0-9]$")
  message(FATAL_ERROR "CUDA_ARCH not set, run e.g. `cmake -DCUDA_ARCH=61 ..` based on your GPU")
else()
  message(STATUS "Building for CUDA ${CUDA_ARCH}")
endif()

include(cmake/glad.cmake)
include(cmake/glm.cmake)
include(cmake/kdtree.cmake)
include(cmake/happly.cmake)

# set c++/cuda standards for all targets
# `target_compile_features({target} PUBLIC cxx_std_17 cuda_std_14)` does not work (https://gitlab.kitware.com/cmake/cmake/-/issues/21883)
# by removing `set(CMAKE_CUDA_STANDARD XX)` nvcc compiles with -std=c++CXX_STANDARD no matter the `cuda_std_14`
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(glfw3 3.3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL GLX)
find_package(CLI11 CONFIG REQUIRED)
find_package(nlohmann_json 3.10 REQUIRED)
find_package(Boost REQUIRED system serialization)
find_package(OpenCV REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for debugging, to generate build/compile_commands.json

add_executable(${PROJECT_NAME}
  main.cu
  src/egl.cpp
  src/quad.cpp
  src/pointcloud.cpp
  src/ray_marching.cu
  src/shader.cpp
  src/texture.cpp
  src/utils.cu
  src/npy.hpp
  include/stb_image_write.cpp
)

target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  GLAD
  HAPPLY
  KDTREE
  glm::glm
  CLI11::CLI11
  glfw
  OpenGL::GL
  OpenGL::OpenGL
  OpenGL::EGL
  OpenGL::GLX
  dl
  CUDA::toolkit
  nlohmann_json::nlohmann_json
  Boost::boost  # Header-only target for interprocess.
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
#  GLUT::GLUT
#  X11
#  Xrandr
#  Xinerama
#  Xi
#  Xxf86vm
#  Xcursor
#  pthread
)

# https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.htmlq
# https://cmake.org/cmake/help/latest/prop_sf/LANGUAGE.html
# https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html
target_compile_options(${PROJECT_NAME}
  PRIVATE
    $<$<AND:$<COMPILE_LANG_AND_ID:CXX,CUDA,GNU>,$<CONFIG:DEBUG>>:-Wall -Wextra -Wextra -Wunreachable-code -Wunused -Wunused-function -Wunused-label -Wunused-parameter -Wunused-value -Wunused-variable>
    $<$<AND:$<COMPILE_LANG_AND_ID:CXX,CUDA,GNU>,$<CONFIG:RELEASE>>:-O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --display_error_number,--diag_suppress="611,997,20012" --extended-lambda --relocatable-device-code=true --compile>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G -g>
)

set_source_files_properties(include/stb_image_write.cpp
    PROPERTIES
      COMPILE_FLAGS "-w"  # Ignore warnings
)

configure_file(shaders/vertex.glsl shaders/vertex.glsl COPYONLY)
configure_file(shaders/fragment.glsl shaders/fragment.glsl COPYONLY)
configure_file(shaders/compute.glsl shaders/compute.glsl COPYONLY)

if(NOT EXISTS ${PROJECT_BINARY_DIR}/.gitignore)
  file(WRITE ${PROJECT_BINARY_DIR}/.gitignore "*")
endif()
