set(LIB_SRCS
    common.cc
    misc.cc
    ustr.cc
    sentence.cc
    new_counter.cc
    new_piecer.cc
    new_normalizer.cc
    piece_tokenizer.cc
    piece_counter.cc
    naive_tokenizer.cc
    naive_counter.cc
)

add_library(isma_tokenizer ${LIB_SRCS})
target_include_directories(isma_tokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(isma_tokenizer_test main.cc test.cc ustr_test.cc)
target_link_libraries(isma_tokenizer_test PRIVATE isma_tokenizer)
