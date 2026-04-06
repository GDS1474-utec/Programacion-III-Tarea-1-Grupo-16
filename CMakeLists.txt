cmake_minimum_required(VERSION 4.2)
project(tarea_1)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(tarea_1
        main.cpp
        Tensor.cpp
        Tensor.H
        TensorTransform.cpp)
