# Repository Guidelines

## Project Structure & Module Organization

Core C++17 code lives in `src/` under the `piece` namespace. Training and inference implementations are organized as `{method}_counter.{h,cc}` and `{method}_tokenizer.{h,cc}` for `naive`, `piece`, `sentencepiece`, and `bytepiece`. Shared normalization, pre-tokenization, model serialization, and UTF-8 utilities also live in `src/`. The CLI entry point is `src/main.cc`.

`README.md` contains project usage and links to the canonical conceptual chapters in the Text book. Do not add duplicate long-form tutorials to this repository.

Tests use the lightweight framework in `src/test.{h,cc}` and are currently collected in `src/tokenizer_test.cc` and `src/ustr_test.cc`. Python bindings are in `python/`, training recipes are in `scripts/`, and example dictionaries and saved tokenizer artifacts are at the repository root and in `save/`.

## Build, Test, and Development Commands

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/piece_tokenizer_test
uv pip install .
cd scripts && make bytepiece
```

The first two commands configure and build the CLI and tests. Run the test binary after every C++ change; it executes all registered tests. `uv pip install .` builds and installs the optional pybind11 module. The Make target trains one tokenizer; `make` trains all configured methods. It defaults to the corpora generated under `data/`; variables such as `CN_INPUT`, `EN_INPUT`, `VOCAB_SIZE`, `SPLIT`, `NUM`, and `DICT` can be overridden.

## Coding Style & Naming Conventions

Follow the existing C++ style: four-space indentation, braces on the same line, `snake_case` filenames, `PascalCase` classes/functions, and trailing underscores for data members (for example, `split_digits_`). Keep declarations in matching `.h`/`.cc` pairs and place code in namespace `piece`. Prefer existing `LOG(...)` utilities for diagnostics. No automatic formatter or linter is configured, so match adjacent code and keep diffs focused.

## Testing Guidelines

Add focused `TEST(Suite, Case)` cases to the existing `*_test.cc` files and use the provided `EXPECT_*`/`ASSERT_*` macros. Name tests by observable behavior, especially for UTF-8 boundaries, serialization compatibility, encode/decode round trips, and pre-tokenizer option combinations. There is no stated coverage threshold and no single-test filter; run the complete test binary.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects, often scoped with a prefix such as `sentencepiece:` or `binding:`. Keep each commit cohesive and explain compatibility-sensitive model-format changes in the body. Pull requests should summarize behavior changes, list verification commands, link relevant issues, and include before/after CLI examples for tokenization changes. Note any generated model or dictionary updates explicitly; avoid committing local build directories or large training outputs unless required.
