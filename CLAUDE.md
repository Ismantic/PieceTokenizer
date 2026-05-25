# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Build C++ CLI and tests
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run all tests (single binary, covers tokenizer_test.cc + ustr_test.cc)
./build/piece_tokenizer_test

# Build with Python bindings (requires pybind11)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON
cmake --build build

# Install Python module
pip install .
```

There is no way to run a single test — the custom test framework runs all registered tests sequentially in one binary.

## CLI Usage

The built binary is `./build/piece-tokenizer` with four subcommands:

```bash
# Train a model
./build/piece-tokenizer count --method bytepiece --input corpus.txt --model output/bp --vocab-size 8000

# Inference (all read stdin, write stdout)
echo "text" | ./build/piece-tokenizer tokenize --model output/bp.model
echo "text" | ./build/piece-tokenizer encode --model output/bp.model
echo "231 192" | ./build/piece-tokenizer decode --model output/bp.model
```

Training with `scripts/Makefile`: `cd scripts && make bytepiece` (or `make` for all methods). Configurable via `VOCAB_SIZE`, `CPU`, `MIN_COUNT` etc.

Data prep with `data/Makefile`: `cd data && make` downloads Chinese/English Wikipedia and produces `cn_sentences.txt` / `en_sentences.txt`.

## Architecture

All code lives in `src/` under the `piece` namespace. C++17 required.

**Counter/Tokenizer pairs** — each training method has a paired Counter (trainer) and Tokenizer (inference), named `{method}_counter.h/cc` and `{method}_tokenizer.h/cc`:
- `naive` — basic byte-level BPE (no normalizer, no byte tokens)
- `piece` — index-linked-list optimized BPE; supports CN mode (see below)
- `sentencepiece` — Symbol-cache BPE with Unicode normalization
- `bytepiece` — byte+fragment hybrid with Trie-based longest-match and byte fallback

Each Counter implements `Count()` + `Save()`. Each Tokenizer implements `Encode()` (returns piece+id pairs), `Tokenize()` (returns piece strings), and `Decode()`. The `method` field stored in the `.model` file auto-selects the right Tokenizer at load time in `main.cc`.

**Vocab size adjustment** — `main.cc` adds implicit tokens before passing to counters: `piece` adds +3 (control tokens), `sentencepiece`/`bytepiece` add +256+3 (byte tokens + control), `naive` adds nothing.

**Core types** (`piece_spec.h`):
- `Model` — vocabulary (pieces with scores/types) + `CounterSpec` + `NormalizerSpec`. Serialized as a human-readable text `.model` file with `[CounterSpec]`, `[NormalizerSpec]`, `[Pieces]` sections (tab-separated fields: index, piece, score, type, u, v).
- `Model::Piece` — type enum: NORMAL, UNKNOWN, CONTROL, USER_DEFINED, BYTE, UNUSED. The `u_` and `v_` fields store merge parents (u + v = piece).
- `piece::float_t` — aliased to `double` in `common.h`, used throughout for scores/weights.

**CN mode** (`--cn-dict`, supported by `piece` and `sentencepiece` methods) — three values:
- *(omitted/empty)* — disabled; Han runs go through normal BPE merging (may learn cross-word Chinese N-grams).
- `--cn-dict no` — **char mode**. Han runs are split per-codepoint so BPE cannot merge across them. **Implicitly forces `cut=1` + `split_digits=true`** in the persisted `NormalizerSpec`: digits also split per-codepoint, punctuation/spaces stand alone (no `▁(` prefix), only ASCII-letter runs go through BPE. Designed for char-level backbones / CWS / NER where you want maximum vocab budget for Chinese characters + clean English BPE.
- `--cn-dict path/to/dict.txt` — **dict mode**. Pre-segments Han runs via a Unigram dictionary (TSV `word\tfreq`), preventing BPE merges from crossing word boundaries. Internally wraps `BytePieceTokenizer` for the segmentation.

The `--cn-dict` flag must match between training and inference. `cut` and `split_digits` are persisted in the model so inference auto-applies them; old models lacking the `split_digits` field decode as `false` (backward-compatible).

**Key supporting modules**:
- `normalizer.h/cc` — NFKC Unicode normalization via precompiled Trie (`normalization_data.h`)
- `ustr.h/cc` — UTF-8 encoding/decoding, validation, `SplitText` (space/punct/word segmentation, `cut=0|1`), `SplitTextCn` (CN-mode variant with optional `split_digits` to per-codepoint-split digit runs), `IsHan` detection
- `darts.h` / `trie.h` — Double-Array Trie used by BytePieceTokenizer and Normalizer
- `sentence.h/cc` — file I/O (ReadableFile, WritableFile, MultiFileSentenceIterator)
- `piece_spec.h` — also contains `Escape`/`Unescape` functions for model serialization (hex encoding for invalid UTF-8)
- `common.h` — logging infrastructure (`LOG(INFO)`, `LOG(FATAL)` etc.)

**Space symbol** — `▁` (U+2581, `\xE2\x96\x81`) is the word-boundary marker, stored in `NormalizerSpec::space_`. SplitText attaches it as a prefix to following tokens.

**Test framework** (`test.h/cc`) — lightweight custom framework with auto-registration. Uses gtest-style macros (TEST, EXPECT_EQ, ASSERT_EQ, etc.) but ASSERT macros behave identically to EXPECT (they print and exit, no exception-based flow). Tests are in `tokenizer_test.cc` and `ustr_test.cc`.

**Python bindings** (`python/piece_tokenizer.cc`) — pybind11 wrapper exposing `Tokenizer` with `load()`, `encode()`, `decode()`, `encode_as_ids()`, `encode_as_pieces()`, `vocab_size()`, `method`.

## Language

Project documentation and code comments are in Chinese. The user communicates in Chinese.
