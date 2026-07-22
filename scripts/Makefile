ROOT     := $(shell cd .. && pwd)
# Auto-detect cmake build dir: build_venv (when build/ is occupied by setuptools) or build
BIN := $(firstword $(wildcard $(ROOT)/build_venv/piece-tokenizer $(ROOT)/build/piece-tokenizer))
#CN_INPUT := $(ROOT)/data/cn_sentences.txt
CN_INPUT := /home/tfbao/Shiyu/Summer/tokenizer_corpus/cn.txt
#EN_INPUT := $(ROOT)/data/en_sentences.txt
EN_INPUT := /home/tfbao/Shiyu/Summer/tokenizer_corpus/en.txt
MODEL_DIR := $(CURDIR)/output

NORMALIZE       := no
VOCAB_SIZE      := 81896
CPU             := 8
MAX_SENTENCES   := 0
MIN_COUNT       := 0
CUT             := 0
RECONSTRUCT     := 0
MAX_PIECE_SIZE  := 0
DICT            := ../dict.txt
EXTRA_TOKENS    := "<pad>,<user>,<assistant>,<system>"

# When DICT is set (e.g. `make piece DICT=/path/to/dict.txt`),
# the `piece` method enters cn mode and pre-cuts Han runs with a
# Unigram segmenter built from the dictionary.
# When EXTRA_TOKENS is set (comma-separated, e.g. `EXTRA_TOKENS="<|bos|>,<|eos|>"`),
# those strings are appended to the vocabulary as CONTROL pieces at the end.
DICT_FLAG := $(if $(DICT),--dict $(DICT))
EXTRA_TOKENS_FLAG := $(if $(EXTRA_TOKENS),--extra-tokens '$(EXTRA_TOKENS)')
RECONSTRUCT_FLAG := $(if $(filter-out 0,$(RECONSTRUCT)),--reconstruct)
CN_INPUT_FLAG := $(if $(CN_INPUT),--input $(CN_INPUT))
EN_INPUT_FLAG := $(if $(EN_INPUT),--input $(EN_INPUT))

METHODS := bytepiece sentencepiece piece

# 默认训练所有方法
all: $(METHODS)

# 每种方法生成一个 .model 文件
define METHOD_RULE
$(1): $(MODEL_DIR)/$(1).model
$(MODEL_DIR)/$(1).model: $(wildcard $(EN_INPUT)) $(wildcard $(CN_INPUT)) | $(MODEL_DIR)
	$(BIN) count --method $(1) $(CN_INPUT_FLAG) $(EN_INPUT_FLAG) \
		--vocab-size $(VOCAB_SIZE) --model $(MODEL_DIR)/$(1) \
		--normalize $(NORMALIZE) --cpu $(CPU) \
		--max-sentences $(MAX_SENTENCES) --min-count $(MIN_COUNT) \
		--cut $(CUT) $(RECONSTRUCT_FLAG) \
		$(DICT_FLAG) $(EXTRA_TOKENS_FLAG)
endef

$(foreach m,$(METHODS),$(eval $(call METHOD_RULE,$(m))))

$(MODEL_DIR):
	mkdir -p $(MODEL_DIR)

MAX_PIECE_SIZE_FLAG := $(if $(filter-out 0,$(MAX_PIECE_SIZE)),--max-piece-size $(MAX_PIECE_SIZE))

raw_count: $(MODEL_DIR)/raw_count.txt
$(MODEL_DIR)/raw_count.txt: $(wildcard $(EN_INPUT)) $(wildcard $(CN_INPUT)) | $(MODEL_DIR)
	$(BIN) raw-count $(CN_INPUT_FLAG) $(EN_INPUT_FLAG) \
		--normalize $(NORMALIZE) --cut $(CUT) $(RECONSTRUCT_FLAG) \
		$(MAX_PIECE_SIZE_FLAG) \
		--output $@

clean:
	rm -rf $(MODEL_DIR)

.PHONY: all clean $(METHODS) raw_count
