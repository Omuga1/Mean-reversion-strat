cmake_minimum_required(VERSION 3.18)
project(microcore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -O3 + native tuning: the hot path is branch-light integer/FP arithmetic on
# contiguous arrays; let the compiler vectorize volume() and the CRC loop.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  # Fall back to the pip-installed pybind11 (pip install pybind11).
  execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
  find_package(pybind11 CONFIG REQUIRED)
endif()

pybind11_add_module(microcore src/bindings.cpp)
target_include_directories(microcore PRIVATE include)

# Standalone unit-test binary (no Python required).
add_executable(microcore_tests tests/test_core.cpp)
target_include_directories(microcore_tests PRIVATE include)

enable_testing()
add_test(NAME core COMMAND microcore_tests)
