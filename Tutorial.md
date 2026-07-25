# 训练 `save/` 中的两套 Tokenizer

本文说明如何使用 `scripts/Makefile` 训练与 `save/` 中两套示例模型配置相同的 Tokenizer：

- `save/Summer-Tokenizer.pt`：`piece`，最终词表大小为 81903，并使用中文词典。
- `save/BERTc-Tokenizer.pt`：`sentencepiece`，最终词表大小为 12535，中文逐字切分。

这里的“配置相同”不等于模型文件逐字节相同。词表内容和 token ID 由训练语料决定；要完全复现 `save/` 中的文件，还必须使用生成它们时完全相同的中英文语料、词典和代码版本。

## 1. 构建训练程序

在仓库根目录执行：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/piece_tokenizer_test
```

`scripts/Makefile` 会优先使用 `build/piece-tokenizer`；如果该文件不存在，则尝试 `build_venv/piece-tokenizer`。使用其他构建目录时，可以在运行 `make` 时显式传入 `BIN`。

## 2. 准备训练语料

### 使用项目的数据脚本

先安装数据处理依赖：

```bash
python3 -m pip install datasets opencc-python-reimplemented
```

数据处理分为 `download → process → sentences` 三个阶段。最简单的方式是一次执行全部阶段：

```bash
cd data
make PYTHON=python3
cd ..
```

`data/Makefile` 默认设置了 Hugging Face 镜像：

```text
HF_ENDPOINT=https://hf-mirror.com
```

完整流程及产物如下：

```text
下载数据集
  data/data/cnwiki.jsonl
  data/data/enwiki.jsonl
          │
          ▼
清洗、抽取文本（中文同时繁体转简体）
  data/cn.txt
  data/en.txt
          │
          ▼
切分句子
  data/cn_sentences.txt
  data/en_sentences.txt
```

各阶段也可以单独运行。

#### 下载原始数据

```bash
cd data
make download PYTHON=python3
```

该阶段分别调用 `fetch_finewiki.py` 和 `fetch_enwiki.py`，生成：

```text
data/data/cnwiki.jsonl
data/data/enwiki.jsonl
```

如果下载中断，重新运行同一命令即可让 Make 检查已有产物并继续处理缺失目标。

#### 清洗和抽取文本

```bash
make process PYTHON=python3
```

该阶段调用 `process.py`：

- 中文数据生成 `data/cn.txt`，并通过 OpenCC 将繁体中文转换为简体中文。
- 英文数据生成 `data/en.txt`。

#### 切分为训练句子

```bash
make sentences PYTHON=python3
```

该阶段调用 `split_sentences.py`，生成最终训练输入：

```text
data/cn_sentences.txt
data/en_sentences.txt
```

每个文件采用 UTF-8 编码，每行是一条训练样本。可以简单检查文件是否生成以及大致规模：

```bash
ls -lh cn_sentences.txt en_sentences.txt
wc -l cn_sentences.txt en_sentences.txt
head -n 3 cn_sentences.txt
head -n 3 en_sentences.txt
cd ..
```

`make` 会根据文件时间戳跳过已经完成的阶段，适合分阶段执行或断点续跑。若要删除下载文件及所有处理中间产物并从头开始：

```bash
cd data
make clean
cd ..
```

`make clean` 会删除原始下载和处理结果，无法用于恢复数据，请确认不再需要这些文件后再执行。

数据脚本会下载并处理完整的 FineWiki 中英文语料，可能消耗较多网络流量、磁盘空间和时间。若已有自己的语料，可以跳过整个下载流程，按下一节直接设置 `CN_INPUT` 和 `EN_INPUT`。

### 使用自己的语料

也可以跳过下载，直接准备两个 UTF-8 文本文件，每行一个训练样本。训练时通过 `CN_INPUT` 和 `EN_INPUT` 指定路径：

```bash
make -C scripts piece \
    CN_INPUT=/path/to/cn.txt \
    EN_INPUT=/path/to/en.txt
```

如果只有一种语言，可以把另一个变量置空，例如：

```bash
make -C scripts piece \
    CN_INPUT=/path/to/cn.txt \
    EN_INPUT=
```

## 3. 训练 Summer-Tokenizer

Summer 模型使用以下关键配置：

| 配置 | 值 |
|---|---|
| 方法 | `piece` |
| Make 参数 `VOCAB_SIZE` | `81896` |
| 中文模式 | 词典切分 |
| 规范化 | `no` |
| Split | `word` |
| Num | `keep` |
| Min count | `0` |
| 额外 token | `<pad>,<user>,<assistant>,<system>` |

仓库保存了与该模型配套的中文词典 `save/Summer-Tokenizer.dict.txt`。执行：

```bash
make -C scripts piece \
    MODEL_DIR="$PWD/scripts/output-summer" \
    VOCAB_SIZE=81896 \
    MIN_COUNT=0 \
    NORMALIZE=no \
    SPLIT=word \
    NUM=keep \
    DICT="$PWD/save/Summer-Tokenizer.dict.txt"
```

输出文件为：

```text
scripts/output-summer/piece.model
```

确认训练成功后，可以按示例模型的命名方式复制：

```bash
cp scripts/output-summer/piece.model save/Summer-Tokenizer.pt
```

`.model` 和 `.pt` 在本项目中使用相同的文本模型格式，修改扩展名不会转换模型内容。

训练和推理必须使用同一份中文词典。例如：

```bash
echo "你好世界" |
    ./build/piece-tokenizer tokenize \
        --model scripts/output-summer/piece.model \
        --dict save/Summer-Tokenizer.dict.txt
```

## 4. 训练 BERTc-Tokenizer

BERTc 模型使用以下关键配置：

| 配置 | 值 |
|---|---|
| 方法 | `sentencepiece` |
| Make 参数 `VOCAB_SIZE` | `12272` |
| 中文模式 | `no`，即中文逐字切分 |
| 规范化 | `no` |
| Split | `isolate`（由 `DICT=no` 强制） |
| Num | `split`（由 `DICT=no` 强制） |
| Min count | `16` |
| 额外 token | `<pad>,<user>,<assistant>,<system>` |

执行：

```bash
make -C scripts sentencepiece \
    MODEL_DIR="$PWD/scripts/output-bertc" \
    VOCAB_SIZE=12272 \
    MIN_COUNT=16 \
    NORMALIZE=no \
    DICT=no
```

输出文件为：

```text
scripts/output-bertc/sentencepiece.model
```

确认训练成功后，可以复制为：

```bash
cp scripts/output-bertc/sentencepiece.model save/BERTc-Tokenizer.pt
```

`DICT=no` 不是“不使用中文模式”，而是启用逐字模式。程序会强制使用 `Split=isolate` 和 `Num=split`，让中文、数字和标点以单个码点作为预分词边界。因此，不需要再显式传入 `SPLIT=isolate NUM=split`。

推理配置已经保存在模型中：

```bash
echo "你好2026" |
    ./build/piece-tokenizer tokenize \
        --model scripts/output-bertc/sentencepiece.model
```

## 5. 为什么 Make 参数和最终词表大小不同

`VOCAB_SIZE` 表示训练内容词表的目标大小。CLI 会根据算法补入基础 token，训练结束时又会追加 `EXTRA_TOKENS`。

Summer：

```text
81896
+ 3   (<unk>, <s>, </s>)
+ 4   (<pad>, <user>, <assistant>, <system>)
= 81903
```

BERTc：

```text
12272
+ 256 (byte fallback)
+ 3   (<unk>, <s>, </s>)
+ 4   (<pad>, <user>, <assistant>, <system>)
= 12535
```

因此，若目标是复现 `save/` 中显示的最终词表大小，不应直接把 `VOCAB_SIZE` 设置为 81903 或 12535。

## 6. 检查训练结果

模型是文本文件，可以直接查看开头：

```bash
sed -n '1,30p' scripts/output-summer/piece.model
sed -n '1,30p' scripts/output-bertc/sentencepiece.model
```

Summer 模型应包含：

```text
method=piece
vocab_size=81903
```

BERTc 模型应包含：

```text
method=sentencepiece
vocab_size=12535
min_count=16
```

随后做一次编码、解码往返测试：

```bash
echo "Hello，中国！2026" |
    ./build/piece-tokenizer encode \
        --model scripts/output-bertc/sentencepiece.model

echo "你好世界" |
    ./build/piece-tokenizer tokenize \
        --model scripts/output-summer/piece.model \
        --dict save/Summer-Tokenizer.dict.txt
```

也可以重新运行完整的项目测试：

```bash
./build/piece_tokenizer_test
```

## 7. 常见问题

### 修改参数后 `make` 没有重新训练

Make 根据输入和输出文件的时间戳判断目标是否需要更新，并不知道命令行变量发生了变化。推荐像本文一样为不同配置设置不同的 `MODEL_DIR`。也可以先清理默认输出：

```bash
make -C scripts clean
```

### 找不到训练程序

显式指定可执行文件：

```bash
make -C scripts piece \
    BIN="$PWD/build/piece-tokenizer" \
    MODEL_DIR="$PWD/scripts/output-summer" \
    VOCAB_SIZE=81896 \
    DICT="$PWD/save/Summer-Tokenizer.dict.txt"
```

### 训练出的 token ID 与 `save/` 不同

这是训练语料或词典不同造成的正常结果。参数相同只保证算法和预分词配置相同；语料的内容、顺序、清洗方式或词频发生变化，都会改变学习到的 pieces 及其 ID。

### Summer 模型推理时是否必须传词典

必须。词典决定中文预分词边界，推理应使用训练时的同一份 `save/Summer-Tokenizer.dict.txt`。BERTc 使用 `DICT=no` 的逐字模式，该配置已写入模型，不需要加载词典文件。
