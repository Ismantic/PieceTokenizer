export HF_ENDPOINT := https://hf-mirror.com

PYTHON ?= python3

all: download process sentences

download: data/cnwiki.jsonl data/enwiki.jsonl

data/cnwiki.jsonl: | check-datasets
	$(PYTHON) fetch_finewiki.py

data/enwiki.jsonl: | check-datasets
	$(PYTHON) fetch_enwiki.py

process: cn.txt en.txt

cn.txt: data/cnwiki.jsonl | check-opencc
	$(PYTHON) process.py data/cnwiki.jsonl --output cn.txt --t2s

en.txt: data/enwiki.jsonl
	$(PYTHON) process.py data/enwiki.jsonl --output en.txt

sentences: cn_sentences.txt en_sentences.txt

cn_sentences.txt: cn.txt
	$(PYTHON) split_sentences.py cn.txt cn_sentences.txt

en_sentences.txt: en.txt
	$(PYTHON) split_sentences.py en.txt en_sentences.txt

check-datasets:
	@$(PYTHON) -c "import datasets" 2>/dev/null || { \
		echo "Missing Python package: datasets" >&2; \
		echo "Install it in the selected interpreter: $(PYTHON) -m pip install datasets" >&2; \
		exit 1; }

check-opencc:
	@$(PYTHON) -c "import opencc" 2>/dev/null || { \
		echo "Missing Python package: opencc" >&2; \
		echo "Install it in the selected interpreter: $(PYTHON) -m pip install opencc-python-reimplemented" >&2; \
		exit 1; }

clean:
	rm -rf data/ cn.txt en.txt cn_sentences.txt en_sentences.txt

.PHONY: all download process sentences clean check-datasets check-opencc
