# PieceTokenizer

BPE/BBPE + BytePiece 的 C++ 实现，多种训练/推理算法，带 Python 绑定。

三段式流水线：**Normalizer**（NFKC）→ **PreTokenizer**（Split / Digit / Cn 三个正交参数）→ **Tokenize**（可训练）。文本格式 `.pt`，可读可编辑，UTF-8 字节回退。

## 构建

需要 CMake 3.14+、C++17。

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
uv pip install .          # Python 模块（需 Pybind11）
```

## 快速开始

```bash
# 数据：下载中英文维基、分句 → cn_sentences.txt / en_sentences.txt
cd data && make

# 训练：输出 scripts/output/{method}.model
cd scripts && make bytepiece            # 或 make 训练全部；VOCAB_SIZE=16000 自定义

# 分词 / 编码 / 解码（读 stdin）
echo "你好世界" | ./build/piece-tokenizer tokenize --model output/bytepiece.model
echo "你好世界" | ./build/piece-tokenizer encode   --model output/bytepiece.model
echo "897 411"  | ./build/piece-tokenizer decode   --model output/bytepiece.model
```

## PreTokenizer 三轴

`Normalizer → PreTokenizer` 两段，不需要模型。三个正交参数（任意组合，互不强制），各 1:1 对应模型里持久化的字段：

| 参数 | 取值 | 含义 |
|---|---|---|
| `--split` | `word`（默认）/ `isolate` | word=GPT-4 式（`▁` 依附后词，`don't`→`don`+`'t`）；isolate=空格与每个标点各自独立（`don't` 整体保留）|
| `--digit` | `keep`（默认）/ `split` | 数字串整段 / 逐码点切开 |
| `--dict` | 空 / `no` / 词典路径 | 连续汉字：不切 / 逐字 / 按词典（见 CN 模式）|

```bash
S="Hello, World! don't 你好，世界。123abc"
echo "$S" | ./build/piece-tokenizer pretokenize                  # Hello , ▁World ! ▁don 't ▁ 你好 ， 世界 。 123 abc
echo "$S" | ./build/piece-tokenizer pretokenize --split isolate  # Hello , ▁ World ! ▁ don't ▁ 你好 ， 世界 。 123 abc
echo "$S" | ./build/piece-tokenizer pretokenize --digit split    # ... 世界 。 1 2 3 abc
echo "$S" | ./build/piece-tokenizer pretokenize --dict no        # ... ▁ 你 好 ， 世 界 。 123 abc
```

另有 `--normalize <name>`（如 `NFKC_CF`）、`--reconstruct`（保留所有空格）属 Normalizer 阶段。`raw-count` 用同一套参数做预分词并输出 `word\tfreq`（频率降序）。

## 训练方法

| 方法 | 说明 |
|---|---|
| `piece` | 索引链表优化 BPE（类 NanoChat RustBPE）|
| `sentencepiece` | Symbol-cache BPE + `character_coverage` 保字 |
| `bytepiece` | Trie 最长匹配 + byte fallback |
| `naive` | 基础字节级 BPE |

## CN 模式（`piece` / `sentencepiece`）

BPE 只看共现频率，训练中文时易学出 `▁雨星朋友` 这种跨词串。CN 模式在合并前先对连续汉字预切。**训练和推理必须传同一个 `--dict`**：

- **空** — 不启用，整段进 BPE。
- **`no`** — 逐字模式：汉字全单字，并隐含强制 `cut=1 + split_digits=true`（= Split=isolate + Digit=split，由 `main.cc` 统一处理），于是中文/数字/标点各占一个 token，只有英文走 BPE。适合 char-level backbone / CWS / NER。
- **词典路径** — 用 TSV `word\tfreq` 词典（Unigram）把汉字切成词，BPE 不跨词。

```bash
# 逐字模式（sentencepiece 带 character_coverage 保字，推荐）
./build/piece-tokenizer count --method sentencepiece --input cn.txt \
    --vocab-size 16000 --model output/sp_char --dict no
echo "2024年8月,GPT-4 model release 苹果公司" | \
    ./build/piece-tokenizer tokenize --model output/sp_char.model --dict no
# → 2 0 2 4 年 8 月 , GPT - 4 ▁ model ▁ release ▁ 苹 果 公 司

# 词典模式
./build/piece-tokenizer count --method piece --input corpus.txt \
    --model output/pc --dict dict.txt
echo "Tom 他是英国人Bat" | \
    ./build/piece-tokenizer tokenize --model output/pc.model --dict dict.txt
# → T om ▁ 他 是 英国 人 B at
```

配置持久化在 `PreTokenizerSpec` 里，推理自动按训练值走；老模型向后兼容。

## 追加特殊 token（无需重训）

给训练好的模型末尾追加 CONTROL token（自动去重、同步 `vocab_size`、`<pad>` 自动接上 `pad_id`）：

```bash
./build/piece-tokenizer insert-tokens --model in.model \
    --extra-tokens "<pad>,<user>,<assistant>" --output out.model
```

训练时也可直接 `count --extra-tokens "<pad>,<user>"`，两条路径产出一致。

## Python

```python
import piece_tokenizer as pt

# 模型无关的预分词（三轴同 CLI）
pt.PreTokenizer(split='isolate', digit='split', cn='no').tokenize("你好123 hi")
# → ['你', '好', '1', '2', '3', '▁', 'hi']

# 加载训练好的模型（dict 需与训练一致）
tok = pt.Tokenizer()
tok.load("output/bytepiece.model")
tok.encode("你好世界")          # → [('你', 897), ('好', 411), ...]
tok.encode_as_ids("你好世界")   # → [897, 411, ...]
tok.encode_bytes("😀")          # → [(b'...', id), ...]，精确保留 byte fragment
tok.encode_as_piece_bytes("😀") # → [b'...', ...]
tok.decode([897, 411])          # → '你好'
tok.vocab_size(); tok.method
```

Byte-level BPE 的单个 piece 可能只是 UTF-8 字符的一部分，因此不保证能表示为 Python `str`。`encode()`、`encode_as_pieces()` 适用于 UTF-8 完整的 piece；处理任意文本或检查词表时，使用 `encode_bytes()`、`encode_as_piece_bytes()` 和 `id_to_piece_bytes()` 获取无损字节。

## License

MIT
