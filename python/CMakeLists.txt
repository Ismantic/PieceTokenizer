pybind11_add_module(_core piece_tokenizer.cc)
target_link_libraries(_core PRIVATE piece_tokenizer_lib)

install(TARGETS _core
    LIBRARY DESTINATION piece_tokenizer
    RUNTIME DESTINATION piece_tokenizer
)

install(FILES
    ${CMAKE_SOURCE_DIR}/save/BERTc-Tokenizer.pt
    ${CMAKE_SOURCE_DIR}/save/Summer-Tokenizer.pt
    ${CMAKE_SOURCE_DIR}/save/Summer-Tokenizer.dict.txt
    DESTINATION piece_tokenizer/models
)
