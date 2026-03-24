# Tokenizer

BPE (Byte Pair Encoding) 分词器的 C++ 实现，支持多种训练算法和推理策略。

## 特性

- **多种训练算法**
  - NaiveCounter — 基础字节级 BPE
  - SimpleCounter — 基于索引链表的高效 BPE
  - SentencePieceCounter — 基于 Symbol 缓存的高级 BPE，支持大词表
  - BytePieceCounter — 字节 + 分片混合方案

- **多种推理方式**
  - NaiveTokenizer — 顺序合并规则
  - SimpleTokenizer — 基于合并规则映射的贪心扫描
  - SentencePieceTokenizer — 带 Unicode 归一化的编解码
  - BytePieceTokenizer — 基于 Trie 的最长匹配 + 字节回退

- **Unicode 支持**
  - NFKC 归一化（预编译 Trie）
  - UTF-8 编解码与校验
  - 字节回退机制处理未知字符

- **模型序列化**
  - 文本格式模型文件，支持保存和加载
  - 特殊 token 处理（`<unk>`, `<s>`, `</s>`, `<pad>`）

详细的算法原理见 [docs/](docs/) 目录：
- [BPE.md](docs/BPE.md) — BPE 算法原理
- [BytePieceCounter.md](docs/BytePieceCounter.md) — BytePiece 训练算法
- [BytePieceTokenizer.md](docs/BytePieceTokenizer.md) — BytePiece 推理算法
- [Label.md](docs/Label.md) — 标注规范

## 构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## 使用示例

### 训练 BPE 模型

```cpp
#include "piece_spec.h"
#include "bytepiece_counter.h"
#include "bytepiece_tokenizer.h"
#include "new_normalizer.h"

// 配置
piece::CounterSpec counter_spec;
counter_spec.add_input("corpus.txt");
counter_spec.set_vocab_size(8000 + 256 + 3);
counter_spec.set_model_prefix("my_model");

piece::NormalizerSpec normalizer_spec;

// 训练
piece::BytePieceCounter counter(counter_spec, normalizer_spec);
counter.Count();
counter.Save();
```

### 加载模型并分词

```cpp
piece::Model model;
model.Load("my_model.model");

piece::BytePieceTokenizer tokenizer(model);

auto tokens = tokenizer.Tokenize("hello world 中文");
for (const auto& t : tokens) {
    std::cout << t << " ";
}
```

## 项目结构

```
src/
  piece_spec.h          - 核心数据结构（Model, CounterSpec, NormalizerSpec）
  naive_counter.h/cc    - 基础 BPE 训练与推理
  naive_tokenizer.h/cc  - 链表优化的 BPE 训练与推理
  sentencepiece_counter.h/cc - Symbol 缓存 BPE 训练
  sentencepiece_tokenizer.h/cc - 高级编解码推理
  bytepiece_counter.h/cc - BytePiece 训练
  bytepiece_tokenizer.h/cc - BytePiece 推理（Trie 最长匹配）
  new_normalizer.h/cc   - NFKC Unicode 归一化
  ustr.h/cc             - UTF-8 编解码
  sentence.h/cc         - 文件 I/O
  darts.h / trie.h      - Double-Array Trie
  common.h/cc           - 日志
  misc.h/cc             - 工具函数
docs/
  BPE.md                - BPE 算法原理
  BytePieceCounter.md   - BytePiece 训练详解
  BytePieceTokenizer.md - BytePiece 推理详解
  Label.md              - 标注规范
```

## License

MIT
