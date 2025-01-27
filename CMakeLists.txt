cmake_minimum_required(VERSION 3.16)
project(tfhe++ CXX)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS
      "-march=native -O3 -g -funroll-loops -Wall -Wextra -pedantic -Wno-sign-compare"
  )
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)


option(USE_RANDEN "Use randen as CSPRNG" ON)
option(ENABLE_TEST "Build tests" ON)
option(USE_FFTW3 "Use FFTW3" ON)
option(USE_FPGA "Use FPGA" ON)

set(TFHEpp_DEFINITIONS
    ""
    PARENT_SCOPE)

if(USE_RANDEN)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_RANDEN"
      PARENT_SCOPE)
  add_compile_definitions(USE_RANDEN)
  add_subdirectory(thirdparties/randen)
endif()


if(USE_FFTW3)
  set(TFHEpp_DEFINITIONS
          "${TFHEpp_DEFINITIONS};USE_FFTW3"
          PARENT_SCOPE)
  add_compile_definitions(USE_FFTW3)
  add_subdirectory(thirdparties/fftw)
endif()

if(USE_FPGA)
  set(TFHEpp_DEFINITIONS
          "${TFHEpp_DEFINITIONS};USE_FPGA"
          PARENT_SCOPE)
  add_compile_definitions(USE_FPGA)
  # build external dependencies
  message("-- Building external dependencies")
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/extDep.cmake)
  find_package(IntelFPGAOpenCL REQUIRED)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)

  add_subdirectory(thirdparties/fpga)
  add_subdirectory(thirdparties/fftfpga)
endif()


add_subdirectory(src)

if(ENABLE_TEST)
  add_subdirectory(unit_test/fft)
  add_subdirectory(unit_test/batch)
  add_subdirectory(unit_test/externalproduct)
  add_subdirectory(unit_test/fpga)
  add_subdirectory(unit_test/encrypt)
  add_subdirectory(unit_test/gate)
  add_subdirectory(unit_test/nand)
  add_subdirectory(unit_test/decomposition)
  add_subdirectory(unit_test/bootstrapping)
endif()

install(TARGETS tfhe++ LIBRARY DESTINATION lib)
