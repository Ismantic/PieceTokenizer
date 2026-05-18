#!/usr/bin/env python3
r"""
给一个已经训练好的 piece .model 追加 extra CONTROL token —— 无需重新训练。

背景:
  `.model` 文件是纯文本,三段:[CounterSpec] / [NormalizerSpec] / [Pieces]。
  [Pieces] 里每行一个 piece: id<TAB>piece<TAB>score<TAB>type<TAB>u<TAB>v
  type: 1=NORMAL 2=UNKNOWN 3=CONTROL。piece 内的 \t \n \r \ 已被 \xHH 转义,
  所以整个文件按 \n 分行是安全的。

  `count --extra-tokens "..."` 的做法(piece_counter.cc:144-150)就是在词表末尾
  InsertPieces() 一个 CONTROL piece、score=0。本脚本直接在文件层面复刻这一步,
  产出与「带 --extra-tokens 重新 count」逐字节一致的结果,但跳过对语料的重新统计。

用法:
  python add_extra_tokens.py \
      --input  output/piece.model \
      --extra-tokens "<pad>,<user>,<assistant>,<system>" \
      --output ../../Summer/piece_v2.model
"""
import argparse
import sys


def patch_line(raw: bytes, old_line: bytes, new_line: bytes) -> bytes:
    """把整行 old_line 替换成 new_line,要求恰好命中一次(前后用 \\n 锚定)。"""
    pat = b"\n" + old_line + b"\n"
    rep = b"\n" + new_line + b"\n"
    n = raw.count(pat)
    if n != 1:
        sys.exit(f"ERROR: 期望整行 {old_line!r} 出现 1 次,实际 {n} 次")
    return raw.replace(pat, rep, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="已训练好的 piece .model")
    ap.add_argument("--output", required=True, help="输出的新 .model")
    ap.add_argument("--extra-tokens", required=True,
                    help="逗号分隔,如 '<pad>,<user>,<assistant>,<system>'")
    args = ap.parse_args()

    extras = [t for t in args.extra_tokens.split(",") if t]
    if not extras:
        sys.exit("ERROR: --extra-tokens 为空")

    with open(args.input, "rb") as f:
        raw = f.read()
    if not raw.endswith(b"\n"):
        raw += b"\n"

    # --- 从头部解析 vocab_size / pad_id / pad_piece / [Pieces].size ---
    head = raw[:raw.find(b"\n[Pieces]\n")].decode("latin-1")
    counter = {}
    for ln in head.splitlines():
        if "=" in ln and not ln.startswith("["):
            k, v = ln.split("=", 1)
            counter.setdefault(k, v)
    vocab_size = int(counter["vocab_size"])
    pad_piece = counter.get("pad_piece", "<pad>")

    # [Pieces] 段首的 size= 行
    after = raw[raw.find(b"\n[Pieces]\n") + len(b"\n[Pieces]\n"):]
    pieces_size = int(after.split(b"\n", 1)[0].decode("latin-1").split("=", 1)[1])
    if vocab_size != pieces_size:
        sys.exit(f"ERROR: vocab_size({vocab_size}) != [Pieces].size({pieces_size})")

    old_n = vocab_size
    new_n = old_n + len(extras)

    # --- 改三处计数 ---
    raw = patch_line(raw, f"vocab_size={old_n}".encode(),
                          f"vocab_size={new_n}".encode())
    raw = patch_line(raw, f"size={old_n}".encode(),
                          f"size={new_n}".encode())  # [Pieces] 段首

    # --- 若追加了 <pad>,顺带把 pad_id 指过去 ---
    pad_new_id = None
    for k, tok in enumerate(extras):
        if tok == pad_piece:
            pad_new_id = old_n + k
    if pad_new_id is not None:
        old_pad = counter.get("pad_id", "-1")
        raw = patch_line(raw, f"pad_id={old_pad}".encode(),
                              f"pad_id={pad_new_id}".encode())

    # --- 在文件末尾追加 CONTROL piece 行 ---
    tail = b"".join(
        f"{old_n + k}\t{tok}\t0\t3\t\t\n".encode()
        for k, tok in enumerate(extras)
    )
    raw += tail

    with open(args.output, "wb") as f:
        f.write(raw)

    print(f"输入: {args.input}  ({old_n} pieces)")
    print(f"输出: {args.output}  ({new_n} pieces, +{len(extras)} CONTROL)")
    for k, tok in enumerate(extras):
        print(f"  id {old_n + k:>6}  {tok}  (type=3 CONTROL)")
    if pad_new_id is not None:
        print(f"  pad_id -> {pad_new_id}")


if __name__ == "__main__":
    main()
