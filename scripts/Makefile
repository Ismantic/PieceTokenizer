ROOT     := $(shell cd .. && pwd)
BIN      := $(ROOT)/build/piece-tokenizer
#CN_INPUT := $(ROOT)/data/cn_sentences.txt
CN_INPUT :=
EN_INPUT := $(ROOT)/data/en_sentences.txt
MODEL_DIR := $(CURDIR)/output

NORMALIZE       := no
VOCAB_SIZE      := 10000
CPU             := 8
MAX_SENTENCES   := 0
MIN_COUNT       := 0
CUT             := 1
#CN_DICT         := dict.txt
CN_DICT         :=

# When CN_DICT is set (e.g. `make piece CN_DICT=/path/to/dict.txt`),
# the `piece` method enters cn mode and pre-cuts Han runs with a
# Unigram segmenter built from the dictionary.
CN_DICT_FLAG := $(if $(CN_DICT),--cn-dict $(CN_DICT))
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
		--cut $(CUT) \
		$(CN_DICT_FLAG)
endef

$(foreach m,$(METHODS),$(eval $(call METHOD_RULE,$(m))))

$(MODEL_DIR):
	mkdir -p $(MODEL_DIR)

clean:
	rm -rf $(MODEL_DIR)

.PHONY: all clean $(METHODS)
