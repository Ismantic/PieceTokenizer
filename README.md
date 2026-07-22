# PieceTokenizer

BPE/BBPE + BytePiece 的 C++ 实现，支持多种训练算法和推理策略。

## 特性

- NFKC Unicode 归一化，UTF-8 字节回退
- 文本格式模型文件，可读可编辑
- Python 绑定（Pybind11）
- 内置中英文维基百科语料下载与预处理流程
- 系统化的 PreTokenizer：Split / Digit / Cn 三个正交参数
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

不需要模型文件，直接对文本做 `Normalizer → PreTokenizer` 两段。PreTokenizer 有 **3 个正交参数**（任意组合，互不强制），各自 1:1 对应模型里持久化的一个字段：

| 参数 | 取值 | 含义 | 对应字段 |
|---|---|---|---|
| `--split` | `word`（默认）/ `isolate` | word=GPT-4 式（`▁` 依附后词，`don't`→`don`+`'t`）；isolate=空格与每个标点各自独立（`don't` 整体保留） | `cut` 0/1 |
| `--digit` | `keep`（默认）/ `split` | 数字串整段 / 逐码点切开 | `split_digits` |
| `--cn-dict` | 不传 / `no` / 词典路径 | 连续汉字段：不切 / 逐字 / 按词典（见下方 CN 模式） | `cn_dict` |

`--split` 只影响空格/标点；字母、数字、汉字在两种取值下都成连续串（数字逐字由 `--digit split`，汉字逐字由 `--cn-dict no` 控制）。

```bash
S="Hello, World! don't 你好，世界。123abc"

# split=word（默认，对齐 GPT-4 regex 骨架）
echo "$S" | ./build/piece-tokenizer pretokenize --split word
# Hello , ▁World ! ▁don 't ▁ 你好 ， 世界 。 123 abc

# split=isolate（空格和标点全部独立，英文缩写整体保留）
echo "$S" | ./build/piece-tokenizer pretokenize --split isolate
# Hello , ▁ World ! ▁ don't ▁ 你好 ， 世界 。 123 abc

# digit=split（数字逐字，独立于 CN 模式）
echo "$S" | ./build/piece-tokenizer pretokenize --digit split
# Hello , ▁World ! ▁don 't ▁ 你好 ， 世界 。 1 2 3 abc

# cn-dict=no（连续汉字逐字）
echo "$S" | ./build/piece-tokenizer pretokenize --cn-dict no
# Hello , ▁World ! ▁don 't ▁ 你 好 ， 世 界 。 123 abc

# 带归一化（Normalizer 阶段）
echo "HELLO" | ./build/piece-tokenizer pretokenize --normalize NFKC_CF
# hello

# 保留所有空格（不合并、不去首尾；Normalizer 阶段）
echo "  Hello   World  " | ./build/piece-tokenizer pretokenize --reconstruct
# ▁ ▁Hello ▁ ▁ ▁World ▁ ▁
```

> `--cut 0|1` 仍作为 `--split word|isolate` 的别名保留。

### 词频统计（raw-count）

对输入做预分词后统计词频，输出 `word\tfreq` 格式（按频率降序）：

```bash
./build/piece-tokenizer raw-count --input corpus.txt --split isolate --output raw_count.txt
```

`raw-count` 支持和 `pretokenize` 相同的 `--split / --digit / --cn-dict / --normalize / --reconstruct`。

### Python 接口

```python
import piece_tokenizer as pt

# PreTokenizer：Normalize + Split，不需要模型文件。3 个正交参数：
#   split='word'|'isolate'   digit='keep'|'split'   cn=''|'no'|'<词典路径>'
p = pt.PreTokenizer(normalize='no', split='word')
p.tokenize("Hello, World!")        # → ['Hello', ',', '▁World', '!']

pt.PreTokenizer(split='isolate').tokenize("Hello, World!")  # → ['Hello', ',', '▁', 'World', '!']
pt.PreTokenizer(digit='split').tokenize("abc123def")        # → ['abc', '1', '2', '3', 'def']
pt.PreTokenizer(cn='no').tokenize("你好世界hi")              # → ['你', '好', '世', '界', 'hi']
pt.PreTokenizer(reconstruct=True).tokenize("  Hello  ")     # → ['▁', '▁Hello', '▁', '▁']

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

想要"中文按字 + 英文 BPE"组合的最简方式。

**`--cn-dict no` 隐含 char mode：训练时会强制 `cut=1 + split_digits=true`**（即 Split=isolate + Digit=split，无需用户再传；该强制由 `main.cc` 单点处理，piece / sentencepiece 统一生效），所以词表里：
- **汉字** → 全部单字（不会有中文 N-gram piece）
- **数字** → 全部单 digit（不会有 `2024 / 200` 这类 N-gram）
- **标点 / 符号** → 单 codepoint，不带空格前缀（不会有 `▁( / ▁《`；空格 `▁` 单立成 token）
- **英文字母** → 唯一走 BPE 合并的部分（保留 `model / release / Google` 等高频整词）

效果：词表干净对齐 char-level backbone / CWS / NER 场景，词表预算最大化留给中文字 + 英文 BPE。

```bash
# 训练：sentencepiece 推荐，因为它带 character_coverage 保字（默认 0.9995）
./build/piece-tokenizer count \
    --method sentencepiece \
    --input cn_corpus.txt \
    --vocab-size 16000 \
    --model output/sp_char \
    --cn-dict no

# 推理：同样传 --cn-dict no
echo "2024年8月,GPT-4 model release 苹果公司新款 iPhone" | \
    ./build/piece-tokenizer tokenize \
    --model output/sp_char.model --cn-dict no
# → 2 0 2 4 年 8 月 , GPT - 4 ▁ model ▁ release ▁ 苹 果 公 司 新 款 ▁ iPhone
```

`piece` 方法也支持 `--cn-dict no`（行为相同，但用 byte fallback 而非 UNK 兜底罕见字）。

> **训推一致**：char mode 的 `cut=1 / split_digits=true` 持久化在模型的 `normalizer_spec` 里，推理时自动按训练值走。**老模型**（spec 里没 `split_digits` 字段）会安全降级为 `split_digits=false`，行为完全不变 — 向后兼容。

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
