# PieceTokenizer Python API

本文档描述 `piece_tokenizer` Python 模块的接口。模块由 `python/piece_tokenizer.cc` 通过 PyBind11 暴露，安装方式：

```bash
uv pip install .
```

模块包含两个类：

- `PreTokenizer`：无模型的文本规范化 + 预分词。
- `Tokenizer`：加载训练好的模型后进行编码 / 解码。

---

## `piece_tokenizer.PreTokenizer`

`Normalizer → PreTokenizer` 两段流水线的 Python 入口，对应 CLI 的 `pretokenize` 命令。

### 构造

```python
pt.PreTokenizer(
    normalize: str = "no",
    split: str = "word",
    num: str = "keep",
    cn: str = "",
    reconstruct: bool = False,
)
```

参数与 CLI 的三轴一一对应：

| 参数 | 类型 | 可选值 | 说明 |
|---|---|---|---|
| `normalize` | `str` | `"no"` 或规范化名称如 `"NFKC_CF"` | Normalizer 阶段名称。 |
| `split` | `str` | `"word"` / `"isolate"` | `"word"`：GPT-4 式，`▁` 依附后词，`don't` → `don` + `'t`；`"isolate"`：空格与每个标点各自独立，`don't` 整体保留。 |
| `num` | `str` | `"keep"` / `"split"` | `"keep"`：数字串整段保留；`"split"`：逐码点切开。 |
| `cn` | `str` | `""` / `"no"` / 词典路径 | 连续汉字的处理方式：`""` 不切；`"no"` 逐字切；路径按 TSV 词典切分。 |
| `reconstruct` | `bool` | `True` / `False` | 是否保留所有空格。 |

兼容别名：`digit=` 等价于 `num=`，旧代码可继续使用。

### 方法

#### `tokenize(text: str) -> list[str]`

对输入文本做规范化 + 预分词，返回字符串列表。

```python
import piece_tokenizer as pt

pt.PreTokenizer(split="isolate", num="split", cn="no").tokenize("你好123 hi")
# → ['你', '好', '1', '2', '3', '▁', 'hi']
```

---

## `piece_tokenizer.Tokenizer`

加载训练好的模型文件（`.pt` / `.model`），支持 `naive`、`piece`、`sentencepiece`、`bytepiece` 四种方法。

### 构造

```python
tok = pt.Tokenizer()
```

### 方法

#### `load(model_file: str, dict: str = "") -> bool`

加载模型文件。`piece` / `sentencepiece` 的 CN 模式下，`dict` 必须与训练时一致。
加载成功返回 `True`；以下情况返回 `False`：

- 模型文件不存在、无法读取或格式无效。
- 模型中的 `method` 不是 `naive`、`piece`、`sentencepiece`、`bytepiece` 之一。
- `piece` / `sentencepiece` 所需的中文词典无法加载或格式无效。

```python
tok = pt.Tokenizer()
ok = tok.load("output/bytepiece.model")
if not ok:
    raise RuntimeError("failed to load tokenizer")

# piece / sentencepiece 若训练时使用了 --dict，推理必须传同一个 dict
ok = tok.load("save/Summer-Tokenizer.pt", dict="save/Summer-Tokenizer.dict.txt")
```

#### `encode(text: str) -> list[tuple[str, int]]`

将文本编码为 `(piece, id)` 列表。

```python
tok.encode("你好世界")
# → [('你', 897), ('好', 411), ...]
```

#### `encode_as_ids(text: str) -> list[int]`

仅返回 token ids。

```python
tok.encode_as_ids("你好世界")
# → [897, 411, ...]
```

#### `encode_as_pieces(text: str) -> list[str]`

仅返回 piece 字符串。注意：byte-level piece 可能不是合法 UTF-8，此时可能抛出 `UnicodeDecodeError`。

```python
tok.encode_as_pieces("你好世界")
# → ['你', '好', ...]
```

#### `encode_bytes(text: str) -> list[tuple[bytes, int]]`

返回 `(模型 piece 的底层字节, id)` 对，不进行 UTF-8 解码。对于
`piece` / `bytepiece`，结果可能包含输入文本的原始字节片段；对于
`sentencepiece` 的 byte fallback，结果可能是 `b"<0xF0>"` 这样的模型
piece 表示，而不是单字节 `b"\xf0"`。

```python
tok.encode_bytes("😀")
# → [(b'...', id), ...]
```

#### `encode_as_piece_bytes(text: str) -> list[bytes]`

仅返回模型 piece 的底层字节列表，不进行 UTF-8 解码。与
`encode_bytes()` 一样，SentencePiece byte fallback 可能返回
`b"<0xNN>"` 形式的模型 piece。

```python
tok.encode_as_piece_bytes("😀")
# → [b'...', ...]
```

#### `decode(ids: list[int]) -> str`

将 token ids 解码回文本。

```python
tok.decode([897, 411])
# → '你好'
```

#### `piece_to_id(piece: str) -> int`

将 piece 字符串转换为对应 id，不存在时返回 `-1`。

```python
tok.piece_to_id("你")
# → 897
```

#### `id_to_piece(id: int) -> str`

将 id 转换为 piece 字符串。越界返回空串。

```python
tok.id_to_piece(897)
# → '你'
```

#### `id_to_piece_bytes(id: int) -> bytes`

将 id 转换为 piece 的原始字节，适合查看 byte-level 词表项。

```python
tok.id_to_piece_bytes(123)
# → b'...'
```

#### `vocab_size() -> int`

返回词表大小。

```python
tok.vocab_size()
# → 16000
```

### 只读属性

#### `method -> str`

返回模型对应的训练方法。

```python
tok.method
# → 'bytepiece'
```

---

## 完整示例

`save/` 目录下已提供示例模型：

- `BERTc-Tokenizer.pt`：方法为 `sentencepiece`，词表 12535
- `Summer-Tokenizer.pt`：方法为 `piece`，词表 81903，**训练时使用了 `Summer-Tokenizer.dict.txt`，推理必须一起加载**
- `Summer-Tokenizer.dict.txt`：`Summer-Tokenizer.pt` 配套的中文词典（350000 词）

```python
import piece_tokenizer as pt

# 1. 无模型的预分词
pretok = pt.PreTokenizer(split="word", num="keep", cn="")
print(pretok.tokenize("Hello, 世界！123"))
# → ['Hello', ',', '▁', '世界', '！', '123']

pretok2 = pt.PreTokenizer(split="isolate", num="split", cn="no")
print(pretok2.tokenize("你好123 hi"))
# → ['你', '好', '1', '2', '3', '▁', 'hi']

# 2. 加载 Summer-Tokenizer.pt（piece 方法，必须带配套 dict）
tok = pt.Tokenizer()
tok.load("save/Summer-Tokenizer.pt", dict="save/Summer-Tokenizer.dict.txt")

print(tok.method)        # → 'piece'
print(tok.vocab_size())  # → 81903
print(tok.encode_as_ids("你好世界"))
# → [3315, 1669, 2640]
print(tok.decode([3315, 1669, 2640]))
# → '你好世界'

# 3. 加载 BERTc-Tokenizer.pt（sentencepiece 方法）
tok2 = pt.Tokenizer()
tok2.load("save/BERTc-Tokenizer.pt")
print(tok2.method)        # → 'sentencepiece'
print(tok2.vocab_size())  # → 12535
print(tok2.encode_as_ids("你好世界"))
# → [6399, 5968, 5966, 6027]

# 4. byte-level 安全：emoji 可能被切成非 UTF-8 片段
print(tok.encode_bytes("😀"))
# → [(b'\xf0\x9f', 56525), (b'\x98', 155), (b'\x80', 131)]
print(tok.encode_as_piece_bytes("😀"))
# → [b'\xf0\x9f', b'\x98', b'\x80']
# 下面这行会抛 UnicodeDecodeError
# print(tok.encode_as_pieces("😀"))
```

---

## 注意事项

1. **UTF-8 安全性**：`encode()`、`encode_as_pieces()`、`id_to_piece()` 假设 piece 是合法 UTF-8。byte-level BPE / BytePiece 的单个 piece 可能只是 UTF-8 字符的一部分，处理这类文本时应使用带 `bytes` 后缀的接口。
2. **CN 模式一致性**：使用 `piece` / `sentencepiece` 且训练时传了 `--dict`，推理时 `load()` 必须传相同的 `dict`。
3. **模型格式**：支持项目自定义的 `.pt` 文本格式以及 `.model` 扩展名，模型方法字段必须是 `naive`、`piece`、`sentencepiece`、`bytepiece` 之一。
4. **加载前调用**：除 `load()` 外，`Tokenizer` 的编码、解码、词表查询和 `method` 等接口都要求先成功加载模型，否则抛出 `RuntimeError: tokenizer is not loaded`。
