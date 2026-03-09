set(TARGET whisper-quant-speed)
add_executable(${TARGET} quant-speed.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ${TARGET} RUNTIME)
