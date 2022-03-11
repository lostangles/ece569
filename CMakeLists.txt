### CMakeLists.txt for CUDA
project(Cuda+C++)
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
#set(CMAKE_VERBOSE_MAKEFILE ON)
# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_20,code=sm_20; --compiler-options -fPIE
    )
set(CUDA_HOST_COMPILER gcc-4.2)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

#string(REPLACE -m32 -m64 CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
# Specify include directories
include_directories(
    include
    #    utility
    )

# Specify library paths
link_directories(
	#/opt/foobar/lib
	#/opt/joestuff/lib
    )


# For compilation ...
# Specify target & source files to compile it from


cuda_add_executable(
    hellocuda
    src/hellocuda.cu
    #hellocuda.h
    #kernels/hellokernels.cu
    #kernels/hellokernels.h
    #utility/wrapper.cpp
    #utility/wrapper.h
        )

# For linking ...
# Specify target & libraries to link it with
#target_link_libraries(
#    hellocuda
#    #-lfoobar
#    #-ljoestuff
#    )
