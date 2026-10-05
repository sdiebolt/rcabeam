cmake_minimum_required(VERSION 3.24...3.27)
set(SKBUILD_PROJECT_NAME "rcabeam")
# Select the local GPU before project() initializes nvcc's default architecture.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(${SKBUILD_PROJECT_NAME} LANGUAGES CUDA CXX)

find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(nanobind CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
add_library(rcabeam_core STATIC src/${SKBUILD_PROJECT_NAME}/core.cu src/${SKBUILD_PROJECT_NAME}/ensemble.cu)
target_include_directories(rcabeam_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(rcabeam_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set_target_properties(rcabeam_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

nanobind_add_module(_cuda_impl NB_STATIC src/${SKBUILD_PROJECT_NAME}/bindings/python.cpp)
target_link_libraries(_cuda_impl PRIVATE rcabeam_core CUDA::cudart)
target_include_directories(_cuda_impl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDAToolkit_INCLUDE_DIRS})

install(TARGETS rcabeam_core ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(FILES include/rcabeam.h DESTINATION include)
install(TARGETS _cuda_impl LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
