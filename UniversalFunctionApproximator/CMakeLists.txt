cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)
#set(CMAKE_MODULE_PATH "/ccs/home/vanstee/.conda/envs/powerai-ornl/lib/python3.6/site-packages/torch/share/cmake/Torch;${CMAKE_MODULE_PATH}")

find_package(Torch REQUIRED)
#find_package(MPI REQUIRED)

add_executable(ufa ufa.cpp)
#link_directories("/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/xl-16.1.1-5/spectrum-mpi-10.3.1.2-20200121-p6nrnt6vtvkn356wqg6f74n6jspnpjd2/lib")
target_link_libraries(ufa "${TORCH_LIBRARIES}" ) 
set_property(TARGET ufa PROPERTY CXX_STANDARD 14)
