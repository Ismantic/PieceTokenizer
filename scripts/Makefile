ROOT := $(abspath ..)

# Override BIN when using a custom build directory.
BIN ?= $(if $(wildcard $(ROOT)/build/piece-tokenizer),$(ROOT)/build/piece-tokenizer,$(ROOT)/build_venv/piece-tokenizer)
CN_INPUT ?= $(ROOT)/data/cn_sentences.txt
EN_INPUT ?= $(ROOT)/data/en_sentences.txt
MODEL_DIR ?= $(CURDIR)/output

NORMALIZE ?= no
SPLIT ?= word
NUM ?= keep
VOCAB_SIZE ?= 81896
CPU ?= 8
MAX_SENTENCES ?= 0
MIN_COUNT ?= 0
MAX_PIECE_SIZE ?= 0
RECONSTRUCT ?= 0
DICT ?= $(ROOT)/dict.txt
EXTRA_TOKENS ?= <pad>,<user>,<assistant>,<system>

INPUTS := $(strip $(CN_INPUT) $(EN_INPUT))
INPUT_FLAGS := $(foreach input,$(INPUTS),--input $(input))
DICT_FLAG := $(if $(DICT),--dict $(DICT))
EXTRA_TOKENS_FLAG := $(if $(EXTRA_TOKENS),--extra-tokens '$(EXTRA_TOKENS)')
RECONSTRUCT_FLAG := $(if $(filter-out 0,$(RECONSTRUCT)),--reconstruct)
MAX_PIECE_SIZE_FLAG := $(if $(filter-out 0,$(MAX_PIECE_SIZE)),--max-piece-size $(MAX_PIECE_SIZE))

METHODS := bytepiece sentencepiece piece

all: $(METHODS)

define METHOD_RULE
$(1): $(MODEL_DIR)/$(1).model
$(MODEL_DIR)/$(1).model: $(INPUTS) | check-bin $(MODEL_DIR)
	$(BIN) count --method $(1) $(INPUT_FLAGS) \
		--vocab-size $(VOCAB_SIZE) --model $(MODEL_DIR)/$(1) \
		--normalize $(NORMALIZE) --split $(SPLIT) --num $(NUM) \
		--cpu $(CPU) --max-sentences $(MAX_SENTENCES) --min-count $(MIN_COUNT) \
		$(MAX_PIECE_SIZE_FLAG) $(RECONSTRUCT_FLAG) \
		$(if $(filter piece sentencepiece,$(1)),$(DICT_FLAG)) $(EXTRA_TOKENS_FLAG)
endef

$(foreach method,$(METHODS),$(eval $(call METHOD_RULE,$(method))))

raw_count: $(MODEL_DIR)/raw_count.txt
$(MODEL_DIR)/raw_count.txt: $(INPUTS) | check-bin $(MODEL_DIR)
	$(BIN) raw-count $(INPUT_FLAGS) \
		--normalize $(NORMALIZE) --split $(SPLIT) --num $(NUM) \
		$(MAX_PIECE_SIZE_FLAG) $(RECONSTRUCT_FLAG) $(DICT_FLAG) \
		--output $@

check-bin:
	@test -x "$(BIN)" || { echo "Tokenizer binary not found: $(BIN)" >&2; \
		echo "Build it first or run make BIN=/path/to/piece-tokenizer" >&2; exit 1; }

$(MODEL_DIR):
	mkdir -p $@

clean:
	rm -rf $(MODEL_DIR)

.PHONY: all clean check-bin $(METHODS) raw_count
