cmake_minimum_required(VERSION 3.14)
project(PieceTokenizer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LIB_SRCS
    src/ustr.cc
    src/sentence.cc
    src/sentencepiece_counter.cc
    src/sentencepiece_tokenizer.cc
    src/normalizer.cc
    src/bytepiece_tokenizer.cc
    src/bytepiece_counter.cc
    src/piece_tokenizer.cc
    src/piece_counter.cc
    src/naive_tokenizer.cc
    src/naive_counter.cc
)

add_library(piece_tokenizer_lib ${LIB_SRCS})
target_include_directories(piece_tokenizer_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(piece-tokenizer src/main.cc)
target_link_libraries(piece-tokenizer PRIVATE piece_tokenizer_lib)

add_executable(piece_tokenizer_test
    src/tokenizer_test.cc
    src/test.cc
    src/ustr_test.cc
)
target_link_libraries(piece_tokenizer_test PRIVATE piece_tokenizer_lib)

option(BUILD_PYTHON "Build Python bindings" OFF)
if(BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
    add_subdirectory(python)
endif()
