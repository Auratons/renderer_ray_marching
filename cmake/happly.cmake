add_library(HAPPLY INTERFACE)

target_include_directories(HAPPLY
  SYSTEM INTERFACE
    ${CMAKE_SOURCE_DIR}/include/happly
)
