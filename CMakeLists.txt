### CMakeLists.txt for CUDA
project(Cuda+C++)
cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
#set(CMAKE_VERBOSE_MAKEFILE ON)
# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 -g
    )
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

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
    canny
    src/gpu_canny.cu
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
