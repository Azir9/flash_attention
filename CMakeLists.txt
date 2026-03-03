cmake_minimum_required(VERSION 3.18)

project(flash_attention_cutlass_demo LANGUAGES CXX CUDA)

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
    # CUTLASS cp.async / ldmatrix 常见目标架构
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# 查找 CUTLASS 头文件路径（支持 -DCUTLASS_DIR=... 或环境变量 CUTLASS_DIR）
set(_CUTLASS_HINTS
    "${CUTLASS_DIR}"
    "$ENV{CUTLASS_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include"
    "$ENV{HOME}/cutlass/include"
    "$ENV{HOME}/cutlass/build/include"
    "$ENV{HOME}/koopman_mpc/test_code/tvm/3rdparty/cutlass/include"
    "/usr/local/include"
    "/usr/include"
)

find_path(CUTLASS_INCLUDE_DIR
    NAMES cutlass/cutlass.h
    HINTS ${_CUTLASS_HINTS}
)

if(NOT CUTLASS_INCLUDE_DIR)
    message(FATAL_ERROR
        "未找到 CUTLASS 头文件。请设置 -DCUTLASS_DIR=/path/to/cutlass/include")
endif()

add_executable(flash_attention_cutlass_demo
    src/main.cu
)

target_include_directories(flash_attention_cutlass_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUTLASS_INCLUDE_DIR}
)

set_target_properties(flash_attention_cutlass_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

target_compile_options(flash_attention_cutlass_demo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
)
