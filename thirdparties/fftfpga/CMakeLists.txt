cmake_minimum_required(VERSION 3.10)
project(fftfpga VERSION 2.0 DESCRIPTION "APIs for FFT using FPGAs" LANGUAGES C CXX)

##
# Generate host executable that is required to call OpenCL kernel bitstreams
# Target: host
##
add_library(${PROJECT_NAME} STATIC fftfpga.c fft1d.c svm.c opencl_utils.c misc.c)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Werror)
target_include_directories(${PROJECT_NAME} PUBLIC ${IntelFPGAOpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${IntelFPGAOpenCL_LIBRARIES} m rt)