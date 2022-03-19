cmake_minimum_required(VERSION 3.16)
project(pointcloud_renderer LANGUAGES CUDA C CXX)

if(NOT "${CUDA_ARCH}" MATCHES "^[0-9][0-9]$")
  message(FATAL_ERROR "CUDA_ARCH not set, run e.g. `cmake -DCUDA_ARCH=61 ..` based on your GPU")
else()
  message(STATUS "Building for CUDA ${CUDA_ARCH}")
endif()

# set c++/cuda standards for all targets
# `target_compile_features({target} PUBLIC cxx_std_17 cuda_std_14)` does not work (https://gitlab.kitware.com/cmake/cmake/-/issues/21883)
# by removing `set(CMAKE_CUDA_STANDARD XX)` nvcc compiles with -std=c++CXX_STANDARD no matter the `cuda_std_14`
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for debugging, to generate build/compile_commands.json

find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(X11 REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
#find_package(GLUT REQUIRED)

add_library(GLAD "${CMAKE_SOURCE_DIR}/src/glad.c")
target_include_directories(GLAD
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

add_library(HAPPLY INTERFACE)
target_include_directories(HAPPLY
    INTERFACE
        ${CMAKE_SOURCE_DIR}/include/happly
)

add_executable(${PROJECT_NAME} src/shader.cpp main.cpp)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        GLAD
        HAPPLY
        CLI11::CLI11
        glfw
        OpenGL::GL
        dl
        #GLUT::GLUT
        X11
        Xrandr
        Xinerama
        Xi
        Xxf86vm
        Xcursor
        pthread
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        "${PROJECT_BINARY_DIR}"
        ${CMAKE_SOURCE_DIR}/include
        #${OPENGL_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}
)

# https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.htmlq
# https://cmake.org/cmake/help/latest/prop_sf/LANGUAGE.html
# https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html
target_compile_options(${PROJECT_NAME}
    PRIVATE
        $<$<AND:$<COMPILE_LANG_AND_ID:CXX,CUDA,GNU>,$<CONFIG:DEBUG>>:-Werror -Wall -Wextra -Wextra -Wunreachable-code -Wunused -Wunused-function -Wunused-label -Wunused-parameter -Wunused-value -Wunused-variable>
        $<$<AND:$<COMPILE_LANG_AND_ID:CXX,CUDA,GNU>,$<CONFIG:RELEASE>>:-O2>
)

if(NOT EXISTS ${PROJECT_BINARY_DIR}/.gitignore)
  file(WRITE ${PROJECT_BINARY_DIR}/.gitignore "*")
endif()
