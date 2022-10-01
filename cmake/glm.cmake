# Suppress warnings from glm, that's why there's not used:
# find_package(glm REQUIRED PATHS include/glm/cmake/glm)
add_subdirectory("include/glm")

target_compile_definitions(glm
  INTERFACE
    GLM_FORCE_SILENT_WARNINGS
    GLM_SILENT_WARNINGS=GLM_ENABLE
)
