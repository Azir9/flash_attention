cmake_minimum_required(VERSION 3.18)

project(flash_attention_naive LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)

# 默认使用 Release 优化
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 允许设置 GPU 架构，例如:
#   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80
if(NOT CMAKE_CUDA_ARCHITECTURES)
    # 你可以根据自己的 GPU 改成 75, 80, 86 等
    set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(flash_attention_naive_lib
    src/flash_attention_naive.cu
)

set_target_properties(flash_attention_naive_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_executable(flash_attention_demo
    src/main.cu
)

target_link_libraries(flash_attention_demo
    PRIVATE flash_attention_naive_lib
)



