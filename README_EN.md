# PieceTokenizer

English | [中文](README.md)

[![PyPI](https://img.shields.io/pypi/v/piece-tokenizer)](https://pypi.org/project/piece-tokenizer/)
[![Python](https://img.shields.io/pypi/pyversions/piece-tokenizer)](https://pypi.org/project/piece-tokenizer/)
[![CI](https://github.com/Ismantic/PieceTokenizer/actions/workflows/ci.yml/badge.svg)](https://github.com/Ismantic/PieceTokenizer/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/Ismantic/PieceTokenizer)](LICENSE)

PieceTokenizer is a C++17 toolkit for training and running BPE, BBPE, and
BytePiece tokenizers. It provides both a command-line program and a Python API,
using a reproducible three-stage pipeline:

```text
Normalizer -> PreTokenizer -> Tokenizer
```

It supports NFKC normalization, configurable English, numeric, and Chinese
boundaries, an editable text-based model format, and UTF-8 byte fallback.
Ready-to-use tokenizers for BERTc and Summer are included.

## Quick Start

Prebuilt wheels support CPython 3.9-3.14 on Linux, Windows, Intel macOS, and
Apple Silicon macOS. Installation includes both vocabularies, with no separate
download or local compilation required:

```bash
pip install piece-tokenizer
```

```python
import piece_tokenizer as pt

bertc = pt.BERTcTokenizer()
ids = bertc.encode_as_ids("你好，PieceTokenizer")
print(bertc.decode(ids))
# 你好，PieceTokenizer

# The accompanying Chinese dictionary for Summer is loaded automatically
summer = pt.SummerTokenizer()
print(summer.encode("中华人民共和国"))

# Tokenizers can also be loaded by name
tok = pt.Tokenizer("BERTc")  # or "Summer"
```

| Built-in model | Method | Vocabulary size | Chinese pre-segmentation |
|---|---:|---:|---|
| BERTc | SentencePiece | 12,535 | Character mode |
| Summer | Piece BPE | 81,903 | Built-in 350,000-word dictionary |

## Building

Building from source requires CMake 3.14+ and C++17. Release mode is recommended:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/piece_tokenizer_test
uv pip install .          # Build the Python package from source
```

## Command Line and Training

```bash
# Data: download Chinese and English Wikipedia, then split it into sentences
# producing cn_sentences.txt and en_sentences.txt
cd data
make PYTHON=python

# Training: write scripts/output/{method}.model
cd scripts && make bytepiece            # Or use make for all methods; set VOCAB_SIZE=16000 to customize

# Tokenize, encode, and decode from stdin
echo "你好世界" | ./build/piece-tokenizer tokenize --model output/bytepiece.model
echo "你好世界" | ./build/piece-tokenizer encode   --model output/bytepiece.model
echo "897 411"  | ./build/piece-tokenizer decode   --model output/bytepiece.model
```

The data pipeline depends on `datasets` and `opencc-python-reimplemented`:

```bash
python -m pip install datasets opencc-python-reimplemented
```

`data/Makefile` downloads the complete Chinese and English FineWiki training
sets, which may require substantial network traffic, disk space, and processing
time. If corpora are already available, skip the download and override
`CN_INPUT` and `EN_INPUT` in the training scripts. Downloaded data, intermediate
text, and sentence-splitting output are ignored by Git.

## The Three PreTokenizer Parameters

The `Normalizer -> PreTokenizer` stages do not require a model. Three orthogonal
parameters can be combined freely. Each maps directly to a persistent field in
the model:

| Parameter | Values | Meaning |
|---|---|---|
| `--split` | `word` (default) / `isolate` | `word` uses GPT-4-style splitting (`▁` attaches to the following word and `don't` becomes `don` + `'t`); `isolate` separates spaces and each punctuation mark while preserving `don't` as a unit |
| `--num` | `keep` (default) / `split` | Preserve a full numeric sequence or split it by code point |
| `--dict` | empty / `no` / dictionary path | Keep consecutive Chinese characters intact, split by character, or segment with a dictionary; see CN Mode |

The legacy `--digit` option and Python `digit=` keyword remain supported as
compatibility aliases. New code should use `--num` and `num=`. Model files still
use the original `split_digits` field, so existing models require no conversion.

```bash
S="Hello, World! don't 你好，世界。123abc"
echo "$S" | ./build/piece-tokenizer pretokenize                  # Hello , ▁World ! ▁don 't ▁ 你好 ， 世界 。 123 abc
echo "$S" | ./build/piece-tokenizer pretokenize --split isolate  # Hello , ▁ World ! ▁ don't ▁ 你好 ， 世界 。 123 abc
echo "$S" | ./build/piece-tokenizer pretokenize --num split      # ... 世界 。 1 2 3 abc
echo "$S" | ./build/piece-tokenizer pretokenize --dict no        # ... ▁ 你 好 ， 世 界 。 123 abc
```

`--normalize <name>` (for example, `NFKC_CF`) and `--reconstruct` (preserve all
spaces) belong to the Normalizer stage. `raw-count` uses the same parameters for
pre-tokenization and outputs `word\tfreq` in descending frequency order.

## Training Methods

| Method | Description |
|---|---|
| `piece` | BPE optimized with indexed linked lists, similar to NanoChat RustBPE |
| `sentencepiece` | Symbol-cache BPE with `character_coverage` to retain characters |
| `bytepiece` | Trie longest-match encoding with byte fallback |
| `naive` | Basic byte-level BPE |

## Differences from SentencePiece

SentencePiece usually sends an entire normalized sentence directly to BPE or
Unigram and lets the statistical model learn subword boundaries. PieceTokenizer
uses an explicit three-stage pipeline:

```text
Normalizer -> PreTokenizer -> Tokenizer
```

Before the subword algorithm runs, `PreTokenizer` establishes boundaries that
cannot be crossed. All Counter and Tokenizer implementations share these
boundaries. It offers three independently configurable dimensions: `Split`
controls spaces, punctuation, and English contractions; `Num` controls whether
numeric sequences remain intact or are split by code point; and `Cn` controls
whether Chinese text remains intact, is split into characters, or is segmented
with a dictionary.

Chinese dictionary mode uses Trie + Viterbi Unigram to pre-segment consecutive
Chinese characters. Subsequent BPE merges remain within those segments, avoiding
pieces that cross Chinese word boundaries merely because of local co-occurrence
frequency. Character mode is suitable for CWS, NER, or models that use Chinese
characters as their basic units.

The `PreTokenizer` configuration is stored in the model, ensuring identical
boundary rules during training and inference. Unlike maintaining a separate
Chinese segmentation script outside SentencePiece, this design combines
normalization, pre-segmentation, and subword encoding into one reproducible
pipeline. Disabling Chinese pre-segmentation retains the traditional approach
without explicit Chinese word boundaries. SentencePiece itself supports Chinese;
the difference is that PieceTokenizer natively provides optional, explicit
Chinese boundary constraints.

## CN Mode (`piece` / `sentencepiece`)

BPE relies only on co-occurrence frequency and may learn cross-word sequences
such as `▁雨星朋友` from Chinese training data. CN mode pre-segments consecutive
Chinese characters before merging. Training and inference must use the same
`--dict` value:

- Empty: disabled; each complete sequence is passed to BPE.

- `no`: character mode. Every Chinese character becomes a separate unit, with
  an implicit `cut=1 + split_digits=true` (`Split=isolate + Num=split`, handled
  centrally by `main.cc`). Chinese characters, digits, and punctuation each
  occupy one token, while only English text uses BPE. This mode is suitable for
  character-level backbones, CWS, and NER.

- Dictionary path: use a TSV `word\tfreq` Unigram dictionary to segment Chinese
  characters into words. BPE cannot merge across word boundaries.

```bash
# Character mode; sentencepiece with character_coverage is recommended
./build/piece-tokenizer count --method sentencepiece --input cn.txt \
    --vocab-size 16000 --model output/sp_char --dict no
echo "2024年8月,GPT-4 model release 苹果公司" | \
    ./build/piece-tokenizer tokenize --model output/sp_char.model --dict no
# -> 2 0 2 4 年 8 月 , GPT - 4 ▁ model ▁ release ▁ 苹 果 公 司

# Dictionary mode
./build/piece-tokenizer count --method piece --input corpus.txt \
    --model output/pc --dict dict.txt
echo "Tom 他是英国人Bat" | \
    ./build/piece-tokenizer tokenize --model output/pc.model --dict dict.txt
# -> T om ▁ 他 是 英国 人 B at
```

The configuration is persisted in `PreTokenizerSpec`, and inference
automatically uses the training values. Older models remain backward compatible.

## Appending Special Tokens Without Retraining

Append CONTROL tokens to a trained model. The command automatically removes
duplicates, updates `vocab_size`, and assigns the next available `pad_id` to
`<pad>`:

```bash
./build/piece-tokenizer insert-tokens --model in.model \
    --extra-tokens "<pad>,<user>,<assistant>" --output out.model
```

Tokens can also be supplied during training with
`count --extra-tokens "<pad>,<user>"`; both paths produce the same result.

## Python

```python
import piece_tokenizer as pt

# Models bundled with the installed package
bertc = pt.Tokenizer("BERTc")
summer = pt.Tokenizer("Summer")  # Automatically loads its Chinese dictionary

# Model-independent pre-tokenization using the same three axes as the CLI
pt.PreTokenizer(split='isolate', num='split', cn='no').tokenize("你好123 hi")
# -> ['你', '好', '1', '2', '3', '▁', 'hi']

# Load a trained model; dict must match the training configuration
tok = pt.Tokenizer()
tok.load("output/bytepiece.model")
tok.encode("你好世界")          # -> [('你', 897), ('好', 411), ...]
tok.encode_as_ids("你好世界")   # -> [897, 411, ...]
tok.encode_bytes("😀")          # -> [(b'...', id), ...], preserving byte fragments exactly
tok.encode_as_piece_bytes("😀") # -> [b'...', ...]
tok.decode([897, 411])          # -> '你好'
tok.vocab_size(); tok.method
```

A single byte-level BPE piece may contain only part of a UTF-8 character, so it
cannot always be represented as a Python `str`. `encode()` and
`encode_as_pieces()` are appropriate for pieces that contain complete UTF-8.
For arbitrary text or vocabulary inspection, use `encode_bytes()`,
`encode_as_piece_bytes()`, and `id_to_piece_bytes()` to obtain lossless bytes.

See [API.md](API.md) for the complete API and custom-model examples. See
[TUTORIAL.md](TUTORIAL.md) for instructions on training BERTc and Summer from
scratch.

## Theory

Complete explanations of training and encoding are available in the
Chinese-language book "Low-Level Implementation: Text Processing":

- [Tokenizer: SentencePiece](https://ismantic.github.io/text/tokenizer-1.html)

- [Tokenizer: BytePiece](https://ismantic.github.io/text/tokenizer-2.html)

## License

MIT
