# FLANN kd-tree on CUDA
# ---------------------
file(GLOB_RECURSE FLANN_CU_SOURCES third_party/flann/*.cu)

add_library(KDTREE STATIC
    src/kdtree/kdtree_flann.cu
    ${FLANN_CU_SOURCES}
)

# in cmake 3.19, theres no need for explicit
# $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_${CUDA_ARCH},code=[compute_${CUDA_ARCH},sm_${CUDA_ARCH}]>
# it is added there by set_target_properties({target} PROPERTIES CUDA_ARCHITECTURES)
set_target_properties(KDTREE
  PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH}
    # POSITION_INDEPENDENT_CODE ON
    # CXX_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(KDTREE
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>  # allows calling constexpr host functions from device code
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>  # allows __device__ annotations in lambda functions
    $<$<COMPILE_LANGUAGE:CUDA>:--relocatable-device-code=true --compile>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --display_error_number,--diag_suppress="611,997,68">  # Disable warnings from the library
  PUBLIC
    -DFLANN_USE_CUDA
    # eigen wants Matrix4f, Vector3f, ... to be aligned and throws an error otherwise
    # we use eigen on gpu and do not need to align the bytes
    # http://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html
    # also add_definitions(-DEIGEN_MAX_STATIC_ALIGN_BYTES=0) can be used
    # must be public, otherwise it can produce errors when passing eigen objects as arguments
    #    -DEIGEN_MAX_STATIC_ALIGN_BYTES=0
    #    -fPIC
)

target_include_directories(KDTREE
  PUBLIC # public exposes the dirs to a main project
    src
  SYSTEM INTERFACE
    ${CMAKE_SOURCE_DIR}/third_party
  PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party
)
