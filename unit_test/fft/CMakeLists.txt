file(GLOB test_sources RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "*.cpp")


foreach(test_source ${test_sources})
    string( REPLACE ".cpp" "" test_name ${test_source} )
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} tfhe++)
    target_link_libraries(nand Threads::Threads)
endforeach(test_source ${test_sources})

