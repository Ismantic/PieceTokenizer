# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Build C++ CLI and tests
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run all tests
./build/isma_tokenizer_test

# Build with Python bindings
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON
cmake --build build

# Install Python module
pip install .
```

## Architecture

All code lives in `src/` under the `piece` namespace. C++17 required.

**Counter/Tokenizer pairs** — each training method has a paired Counter (trainer) and Tokenizer (inference). They share the naming pattern `{method}_counter.h/cc` and `{method}_tokenizer.h/cc`:
- `naive` — basic byte-level BPE
- `piece` — index-linked-list optimized BPE
- `sentencepiece` — Symbol-cache BPE with Unicode normalization
- `bytepiece` — byte+fragment hybrid with Trie-based longest-match and byte fallback

Each Counter implements `Count()` + `Save()`. Each Tokenizer implements `Encode()` + `Decode()`. The `method` field stored in the model file auto-selects the right Tokenizer at load time (`main.cc`).

**Core types** (`piece_spec.h`):
- `Model` — vocabulary (pieces with scores/types) + `CounterSpec` + `NormalizerSpec`. Serialized as a text-format `.model` file with `[CounterSpec]`, `[NormalizerSpec]`, `[Pieces]` sections.
- `Model::Piece` — has type enum: NORMAL, UNKNOWN, CONTROL, USER_DEFINED, BYTE, UNUSED. The `u_` and `v_` fields store merge parents (u + v = piece).
- `piece::float_t` — aliased to `double` in `common.h`, used throughout for scores/weights.

**Key supporting modules**:
- `normalizer.h/cc` — NFKC Unicode normalization via precompiled Trie (`normalization_data.h`)
- `ustr.h/cc` — UTF-8 encoding/decoding, validation, text segmentation
- `darts.h` / `trie.h` — Double-Array Trie implementation used by BytePieceTokenizer and Normalizer
- `sentence.h/cc` — file I/O (ReadableFile, WritableFile, MultiFileSentenceIterator)

**Test framework** (`test.h/cc`) — lightweight custom framework mimicking gtest macros (TEST, EXPECT_EQ, ASSERT_EQ, etc.). Tests are in `tokenizer_test.cc` and `ustr_test.cc`, unified into one binary `isma_tokenizer_test`.

**Python bindings** (`python/isma_tokenizer.cc`) — pybind11 wrapper exposing a `Tokenizer` class with `load()`, `encode()`, `decode()`, `encode_as_ids()`, `encode_as_pieces()`.

## Language

Project documentation and code comments are in Chinese. The user communicates in Chinese.
