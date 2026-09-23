if (EMSCRIPTEN)
    set(TARGET llama-decide-wasm)
    add_executable(${TARGET} kev-wasm.cpp)
    target_link_libraries(${TARGET} PRIVATE llama-common llama)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)

    set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME kev SUFFIX ".js")
    target_link_options(${TARGET} PRIVATE
        -sINITIAL_MEMORY=64MB
        -sSTACK_SIZE=8MB
        -sFORCE_FILESYSTEM=1
        -sEXIT_RUNTIME=0
        -sMODULARIZE=1
        -sEXPORT_ES6=1
        -sEXPORT_NAME=createKev
        -sENVIRONMENT=web,worker
        -sEXPORTED_FUNCTIONS=_kev_init,_kev_system_one,_malloc,_free
        -sEXPORTED_RUNTIME_METHODS=ccall,FS,UTF8ToString)
    if (NOT LLAMA_WASM_MEM64)
        target_link_options(${TARGET} PRIVATE -sMAXIMUM_MEMORY=4GB)
    endif()

    add_custom_command(TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/wasm/kev-wasm.js
            ${CMAKE_CURRENT_SOURCE_DIR}/wasm/index.html
            $<TARGET_FILE_DIR:${TARGET}>)

    return()
endif()

set(TARGET llama-decide)
add_executable(${TARGET} kev-decide.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
