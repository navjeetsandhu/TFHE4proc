file(GLOB tfhe++_sources "*.cpp")

if(ENABLE_SHARED)
  add_library(tfhe++ SHARED ${tfhe++_sources})
else()
  add_library(tfhe++ STATIC ${tfhe++_sources})
endif()

target_include_directories(
  tfhe++
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/thirdparties/fftw
    ${PROJECT_SOURCE_DIR}/thirdparties/randen
    ${PROJECT_SOURCE_DIR}/thirdparties/cereal/include)
if(USE_RANDEN)
  target_link_libraries(tfhe++ INTERFACE randen)
endif()


if(USE_FFTW3)
  target_link_libraries(tfhe++ INTERFACE fftwproc)
  target_link_libraries(tfhe++ INTERFACE fftw3)
endif()


if(USE_FPGA)
  target_include_directories(tfhe++ PUBLIC ${PROJECT_SOURCE_DIR}/thirdparties/fpga)
  target_include_directories(tfhe++ PUBLIC ${PROJECT_SOURCE_DIR}/thirdparties/fftfpga)
  target_link_libraries(tfhe++ INTERFACE fftfpga rt)
  target_link_libraries(tfhe++ INTERFACE fpgaproc)
  target_link_libraries(tfhe++ PUBLIC ${IntelFPGAOpenCL_LIBRARIES} m)
endif()
