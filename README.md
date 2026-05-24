# PieceTokenizer

BPE/BBPE + BytePiece 的 C++ 实现，支持多种训练算法和推理策略。

## 特性

- NFKC Unicode 归一化，UTF-8 字节回退
- 文本格式模型文件，可读可编辑
- Python 绑定（Pybind11）
- 内置中英文维基百科语料下载与预处理流程
- CN 模式：对连续汉字做预切分
- 多线程流式数据读取和处理

## 构建

需要 CMake 3.14+，C++17。

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Python 模块（需要 Pybind11）：

```bash
uv pip install .
```

## 快速开始

### 准备数据

```bash
cd data
make          # 下载中英文维基百科，处理并分句
```

默认使用 hf-mirror.com 镜像加速下载，直连 HuggingFace 可用 `make download HF_ENDPOINT=`。

产出文件：`cn_sentences.txt`（中文）、`en_sentences.txt`（英文）。

### 训练模型

```bash
cd scripts
make bytepiece                    # 训练单个方法
make                              # 训练所有方法
make bytepiece VOCAB_SIZE=16000   # 自定义词表大小
```

模型输出到 `scripts/output/{method}.model`。

### 分词 / 编码 / 解码

```bash
echo "你好世界" | ./build/piece-tokenizer tokenize --model output/bytepiece.model
echo "你好世界" | ./build/piece-tokenizer encode --model output/bytepiece.model
echo "231 192 163 897" | ./build/piece-tokenizer decode --model output/bytepiece.model
```

### 预分词（PreTokenize）

不需要模型文件，直接对文本做 Normalize + SplitText：

```bash
# cut=0（默认，对齐 GPT-4 regex）
echo "Hello, World! don't 你好，世界。123abc" | ./build/piece-tokenizer pretokenize --cut 0
# Hello , ▁World ! ▁don 't ▁ 你好 ， 世界 。 123 abc

# cut=1（空格和标点全部独立，保留英文缩写）
echo "Hello, World! don't 你好，世界。123abc" | ./build/piece-tokenizer pretokenize --cut 1
# Hello , ▁ World ! ▁ don't ▁ 你好 ， 世界 。 123 abc

# 带归一化
echo "HELLO" | ./build/piece-tokenizer pretokenize --normalize NFKC_CF
# hello

# 保留所有空格（不合并、不去首尾）
echo "  Hello   World  " | ./build/piece-tokenizer pretokenize --reconstruct
# ▁ ▁Hello ▁ ▁ ▁World ▁ ▁
```

### 词频统计（raw-count）

对输入做预分词后统计词频，输出 `word\tfreq` 格式（按频率降序）：

```bash
./build/piece-tokenizer raw-count --input corpus.txt --cut 1 --output raw_count.txt
```

### Python 接口

```python
import piece_tokenizer as pt

# PreTokenizer：Normalize + SplitText，不需要模型文件
p = pt.PreTokenizer(normalize='no', cut=0)
p.tokenize("Hello, World!")        # → ['Hello', ',', '▁World', '!']

p1 = pt.PreTokenizer(cut=1)
p1.tokenize("Hello, World!")       # → ['Hello', ',', '▁', 'World', '!']

p2 = pt.PreTokenizer(reconstruct=True)
p2.tokenize("  Hello  ")           # → ['▁', '▁Hello', '▁', '▁']

# Tokenizer：加载训练好的模型做编码/解码
tok = pt.Tokenizer()
tok.load("output/bytepiece.model")

tok.encode("你好世界")             # → [('你', 897), ('好', 411), ...]
tok.encode_as_ids("你好世界")      # → [897, 411, ...]
tok.encode_as_pieces("你好世界")   # → ['你', '好', ...]
tok.decode([897, 411, 591])        # → '你好世界'

tok.vocab_size()                   # → 8259
tok.method                         # → 'bytepiece'
```

## Tokenizer 方法

| 方法 | 训练 | 推理 | 说明 |
|------|------|------|------|
| `piece` | PieceCounter | PieceTokenizer | 类似 NanoChat 的 RustBPE 实现 |
| `sentencepiece` | SentencePieceCounter | SentencePieceTokenizer | Google SentencePiece BPE 实现 |
| `bytepiece` | BytePieceCounter | BytePieceTokenizer | 科学空间 BytePiece 实现 |

## CN 模式（`piece` 和 `sentencepiece` 方法）

BPE 只看字节/字符共现频率，训练中文时经常把 `▁雨星朋友`、`及北部濒大西` 这种跨词串学进词表。CN 模式让你在 BPE/Unigram 合并之前，先对**连续汉字段**做一次预切，把汉字串切成更合理的单位（词或单字），合并就不会跨越这些边界。

`--cn-dict` 参数有 **3 种取值**：

| `--cn-dict` 取值 | 行为 | 用途 |
|---|---|---|
| 不传 / 空 | **不启用 CN 模式**，整段直接 BPE 合并 | 默认；可能学到中文 N-gram piece |
| `no`（字面值） | **Per-character 模式**：每个汉字独立成段，BPE 不跨段 → 中文最终都是单字 token | char-level 中文 + EN BPE 混合（适合做 BERT-style backbone / CWS teacher）|
| `path/to/dict.txt` | **Dict 模式**：用 TSV `word\tfreq` 词典构建 Unigram segmenter，把连续汉字段切成词 | 标准 BPE 训练，BPE 不跨越词边界 |

### 关键约束

- **训练和推理必须用同一份 `--cn-dict` 设置**（包括 `no` / 文件路径）。`piece` 和 `sentencepiece` tokenizer 都会根据 `--cn-dict` 在编码前预切，跟训练对齐
- 支持 `piece` 和 `sentencepiece` 两种方法；其它方法（`naive` / `bytepiece`）传了会给 warning
- 连续汉字段（CJK Unified/Ext 等）走 cn 预切，非汉字段（拉丁、数字、标点）按原 `SplitText` 处理，互不干扰
- 汉字开头**不带空格前缀**（`▁`），空格前缀只贴在紧随其后的非汉字段

### Per-character 模式（`--cn-dict no`）

想要"中文按字 + 英文 BPE"组合的最简方式。**词表里不会出现中文 N-gram piece**，所有中文 token 都是单字。

```bash
# 训练：sentencepiece 推荐，因为它带 character_coverage 保字（默认 0.9995）
./build/piece-tokenizer count \
    --method sentencepiece \
    --input cn_corpus.txt \
    --vocab-size 16000 \
    --model output/sp_char \
    --cn-dict no

# 推理：同样传 --cn-dict no
echo "GPT-4 model 苹果公司发布新款 iPhone" | ./build/piece-tokenizer tokenize \
    --model output/sp_char.model --cn-dict no
# → GPT - 4 ▁model 苹 果 公 司 发 布 新 款 ▁iPhone
```

`piece` 方法也支持 `--cn-dict no`（行为相同，但用 byte fallback 而非 UNK 兜底罕见字）。

### Dict 模式（`--cn-dict path/to/dict.txt`）

用外部词典做 Unigram 预切（如 [Iscut/dict.txt](https://github.com/tfbao/Iscut)）：

```
中国	2041237
人民	1156489
英国	892011
...
```

训练 + 推理：

```bash
./build/piece-tokenizer count \
    --method piece \
    --input corpus.txt \
    --model output/pc \
    --cn-dict path/to/dict.txt

echo "Tom 他是英国人Bat" | ./build/piece-tokenizer tokenize \
    --model output/pc.model \
    --cn-dict path/to/dict.txt
# → T om ▁ 他 是 英国 人 B at
```

### Python 接口

```python
import piece_tokenizer as pt
tok = pt.Tokenizer()

# Per-character 模式
tok.load("sp_char.model", cn_dict="no")

# Dict 模式
tok.load("pc.model", cn_dict="path/to/dict.txt")

# 不启用 CN 模式
tok.load("model.model")
```

### `piece` vs `sentencepiece` 在 CN 模式下的差异

| | `piece` + cn_dict | `sentencepiece` + cn_dict |
|---|---|---|
| 罕见字 / OOV 处理 | byte fallback（`<0xE5><0xAB><0xAE>` 3 token）| UNK 或 byte fallback（取决于 vocab）|
| 字符覆盖 | 按 `--min-count` 砍掉低频字 | `character_coverage` 默认 0.9995 自动保字 |
| 词表干净度 | vocab 含 256 个 byte piece | vocab 含较少 byte piece（受 coverage 控制）|
| 推荐场景 | 通用 BPE、需要 byte-fallback 重建 | char-level backbone、需要严格字数控制 |



## License

MIT
