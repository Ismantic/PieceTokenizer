# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

Requires CMake 3.14+ and a C++17 compiler.

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

The built binary is `./build/piece-tokenizer` with seven subcommands:

```bash
# Train a model
./build/piece-tokenizer count --method bytepiece --input corpus.txt --model output/bp --vocab-size 8000

# Inference (all read stdin, write stdout)
echo "text" | ./build/piece-tokenizer tokenize --model output/bp.model
echo "text" | ./build/piece-tokenizer encode --model output/bp.model
echo "231 192" | ./build/piece-tokenizer decode --model output/bp.model

# Model-free utilities (no .model needed — just Normalize + PreTokenizer split)
echo "text" | ./build/piece-tokenizer pretokenize --split word --digit keep --dict no   # print pre-tokens
./build/piece-tokenizer raw-count --input corpus.txt --split isolate --output raw_count.txt   # pretokenize then emit word\tfreq (freq-desc)

# Append extra CONTROL tokens to a trained model (post-hoc, no retrain)
./build/piece-tokenizer insert-tokens --model output/bp.model --extra-tokens "<pad>,<user>,<assistant>" --output output/bp_v2.model
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

**Extra tokens** (`extra_tokens.h/cc`, `InsertExtraTokens(Model*, tokens, repoint_pad)`) — the single owner of "append extra CONTROL tokens (score 0) at the end of the vocab". Used by both the training counters (via `--extra-tokens`, which stores them in `CounterSpec.extra_tokens` as provenance) and the post-hoc `insert-tokens` subcommand. Semantics: dedup (skip tokens already present), sync `CounterSpec.vocab_size` to the final piece count, and repoint `pad_id` to `pad_piece` when pad was disabled (`pad_id<0`) and is now present. The `insert-tokens` subcommand replaced the old `scripts/add_extra_tokens.py` text-surgery script (it produces byte-identical output but shares the counters' code path, so the two can't drift).

**Core types** (`piece_spec.h`):
- `Model` — vocabulary (pieces with scores/types) + `CounterSpec` + `PreTokenizerSpec` (the config for the whole Normalize+Split stage: name/space/reconstruct + cut/split_digits). Serialized as a human-readable text `.model` file with `[CounterSpec]`, `[PreTokenizerSpec]`, `[Pieces]` sections (tab-separated fields: index, piece, score, type, u, v). The section header was renamed from `[NormalizerSpec]`; the parser still accepts the old header, so pre-rename models load unchanged.
- `Model::Piece` — type enum: NORMAL, UNKNOWN, CONTROL, USER_DEFINED, BYTE, UNUSED. The `u_` and `v_` fields store merge parents (u + v = piece).
- `piece::float_t` — aliased to `double` in `common.h`, used throughout for scores/weights.

**Pre-tokenization module** (`pretokenizer.h/cc`, class `piece::PreTokenizer`) — the single, non-trainable owner of the Normalize→Split stage, shared by the model-free CLI/Python entry points. Conceptually three orthogonal axes (any combination valid, none forces another), each mapping 1:1 to a persisted field:
- **Split** `{word, isolate}` = `PreTokenizerSpec.cut` 0/1. word = GPT-4-style (`▁` attaches to the following word/punct run; `don't`→`don`+`'t`); isolate = each `▁` and each punct char standalone (`don't` kept whole). `cut` only affects space/punct — letters/digits/Han stay as runs in both.
- **Digit** `{keep, split}` = `PreTokenizerSpec.split_digits`. split = digit runs → per-codepoint. Works independently of CN mode.
- **Cn** `{none, char, dict}` = `dict` `""`/`"no"`/path (the CN mode below).

`ustr::SplitTextCn` is the single split covering every combination (empty `cn_cut` = Cn none); `MakeCnCut` in `cut.h/cc` is the single builder of the CN cut function, reused by both `piece` and `sentencepiece` counters+tokenizers.

**CN mode** (`--dict`, supported by `piece` and `sentencepiece` methods) — three values:
- *(omitted/empty)* — disabled; Han runs go through normal BPE merging (may learn cross-word Chinese N-grams).
- `--dict no` — **char mode**. Han runs are split per-codepoint so BPE cannot merge across them. **Implicitly forces `cut=1 + split_digits=true`** in the persisted `PreTokenizerSpec` (i.e. Split=isolate + Digit=split): digits also split per-codepoint, punctuation/spaces stand alone, only ASCII-letter runs go through BPE. This forcing lives in a single place — `main.cc RunCount` — and applies uniformly to both `piece` and `sentencepiece`. Designed for char-level backbones / CWS / NER where you want maximum vocab budget for Chinese characters + clean English BPE.
- `--dict path/to/dict.txt` — **dict mode**. Pre-segments Han runs via a Unigram dictionary (TSV `word\tfreq`), preventing BPE merges from crossing word boundaries. Internally wraps `BytePieceTokenizer` for the segmentation.

The `--dict` flag must match between training and inference. `cut` and `split_digits` are persisted in the model so inference auto-applies them; old models lacking the `split_digits` field decode as `false` (backward-compatible).

**Key supporting modules**:
- `normalizer.h/cc` — NFKC Unicode normalization via precompiled Trie (`normalization_data.h`)
- `ustr.h/cc` — UTF-8 encoding/decoding, validation, `SplitText` (space/punct/word segmentation, `cut=0|1`), `SplitTextCn` (CN-mode variant with optional `split_digits` to per-codepoint-split digit runs), `IsHan` detection
- `darts.h` / `trie.h` — Double-Array Trie used by BytePieceTokenizer and Normalizer
- `sentence.h/cc` — file I/O (ReadableFile, WritableFile, MultiFileSentenceIterator)
- `piece_spec.h` — also contains `Escape`/`Unescape` functions for model serialization (hex encoding for invalid UTF-8)
- `common.h` — logging infrastructure (`LOG(INFO)`, `LOG(FATAL)` etc.)

**Space symbol** — `▁` (U+2581, `\xE2\x96\x81`) is the word-boundary marker, stored in `PreTokenizerSpec::space_`. SplitText attaches it as a prefix to following tokens.

**Test framework** (`test.h/cc`) — lightweight custom framework with auto-registration. Uses gtest-style macros (TEST, EXPECT_EQ, ASSERT_EQ, etc.) but ASSERT macros behave identically to EXPECT (they print and exit, no exception-based flow). Tests are in `tokenizer_test.cc` and `ustr_test.cc`.

**Python bindings** (`python/piece_tokenizer.cc`) — pybind11 wrapper exposing two classes:
- `Tokenizer` (loads a trained model): `load(model_file, dict="")`, `encode()`, `decode()`, `encode_as_ids()`, `encode_as_pieces()`, `piece_to_id()`, `id_to_piece()`, `id_to_piece_bytes()`, `vocab_size()`, `method`. Pass `dict=` to `load()` to match the training-time CN mode.
- `PreTokenizer(normalize="no", split="word", digit="keep", cn="", reconstruct=False)` — model-free Normalize + split, exposes `tokenize()`. Mirrors the `pretokenize` subcommand. See the three axes below.

## Language

Project documentation and code comments are in Chinese. The user communicates in Chinese.
