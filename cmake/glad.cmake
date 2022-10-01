add_library(GLAD
  ${CMAKE_SOURCE_DIR}/src/glad.c
)

target_include_directories(GLAD
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
