cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(ufa ufa.cpp)
target_link_libraries(ufa "${TORCH_LIBRARIES}")
set_property(TARGET ufa PROPERTY CXX_STANDARD 14)
