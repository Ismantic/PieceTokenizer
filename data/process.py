#!/usr/bin/env python3
"""Process downloaded datasets: extract plain text, optional t2s conversion."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path


def strip_markdown_headings(text: str) -> str:
    return re.sub(r"^#+\s+.*$", "", text, flags=re.MULTILINE)


def clean_text(text: str) -> str:
    text = strip_markdown_headings(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


_converter = None


def _init_worker(t2s: bool):
    global _converter
    if t2s:
        import opencc
        _converter = opencc.OpenCC("t2s")
    else:
        _converter = None


def _process_line(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None
    obj = json.loads(line)
    text = obj.get("text", "")
    if not text:
        return None
    text = clean_text(text)
    if _converter:
        text = _converter.convert(text)
    return text if text else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process jsonl dataset into plain text."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="input jsonl file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data.txt"),
        help="output file (default: data.txt)",
    )
    parser.add_argument(
        "--t2s",
        action="store_true",
        help="apply OpenCC traditional-to-simplified conversion",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=4,
        help="number of worker processes (default: 4)",
    )
    return parser.parse_args()


BATCH_SIZE = 1024


def main() -> None:
    args = parse_args()
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as fout, \
         Pool(args.nproc, initializer=_init_worker, initargs=(args.t2s,)) as pool:
        with args.input.open("r", encoding="utf-8") as fin:
            batch = []
            for line in fin:
                batch.append(line)
                if len(batch) >= BATCH_SIZE:
                    for result in pool.map(_process_line, batch):
                        if result:
                            fout.write(result + "\n")
                            count += 1
                    batch = []
            if batch:
                for result in pool.map(_process_line, batch):
                    if result:
                        fout.write(result + "\n")
                        count += 1

    print(f"wrote {count} lines to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
