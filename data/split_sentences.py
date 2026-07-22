#!/usr/bin/env python3
"""Split one-document-per-line text into one sentence per line."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


SENTENCE_END = set("。！？!?")
CLOSERS = set("\"'”’」』】）》〉〕］}）)")
NONTERMINAL_ABBREVIATIONS = {
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.",
    "e.g.", "i.e.", "no.", "vs.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split each input document into sentences, preserving punctuation."
    )
    parser.add_argument("input", type=Path, help="one-document-per-line UTF-8 text")
    parser.add_argument("output", type=Path, help="one-sentence-per-line output")
    return parser.parse_args()


def _after_closers(text: str, index: int) -> int:
    while index < len(text) and text[index] in CLOSERS:
        index += 1
    return index


def _period_ends_sentence(text: str, index: int, end: int) -> bool:
    if end < len(text) and not text[end].isspace():
        return False

    next_start = end
    while next_start < len(text) and text[next_start].isspace():
        next_start += 1
    if next_start == len(text):
        return True

    match = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*)$", text[:index])
    if not match:
        return True

    token = (match.group(1) + ".").lower()
    if token in NONTERMINAL_ABBREVIATIONS:
        return False
    return re.fullmatch(r"(?:[a-z]\.){2,}", token) is None


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    index = 0

    while index < len(text):
        char = text[index]
        if char not in SENTENCE_END and char != ".":
            index += 1
            continue

        end = index + 1
        while end < len(text) and text[end] in SENTENCE_END:
            end += 1
        end = _after_closers(text, end)

        if char == "." and not _period_ends_sentence(text, index, end):
            index += 1
            continue

        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
        index = end

    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    documents = 0
    sentences = 0

    with args.input.open("r", encoding="utf-8") as source, \
         args.output.open("w", encoding="utf-8") as destination:
        for line in source:
            text = line.strip()
            if not text:
                continue
            documents += 1
            for sentence in split_sentences(text):
                destination.write(sentence + "\n")
                sentences += 1

            if documents % 200_000 == 0:
                print(
                    f"{documents} documents, {sentences} sentences",
                    file=sys.stderr,
                )

    print(
        f"done: {documents} documents -> {sentences} sentences",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
