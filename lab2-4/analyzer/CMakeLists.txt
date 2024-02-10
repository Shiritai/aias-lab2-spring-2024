
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(analyzer analyzer.cc)
target_link_libraries(analyzer "${TORCH_LIBRARIES}")
set_property(TARGET analyzer PROPERTY CXX_STANDARD 17)

