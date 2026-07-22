#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "cut.h"
#include "normalizer.h"
#include "piece_spec.h"
#include "ustr.h"

namespace piece {

// Deterministic, non-trainable pre-tokenization: Normalize -> Split.
// The single owner of the pre-tokenize stage, shared by every method's
// counter/tokenizer and by the model-free CLI/Python entry points.
//
// Three orthogonal axes, all carried by PreTokenizerSpec + dict:
//   Split (spec.cut):          word (0, GPT-4-style attach) / isolate (1)
//   Digit (spec.split_digits): keep / split (per-codepoint)
//   Cn    (dict):              none ("") / char ("no") / dict (path)
// Any combination is valid; the axes never force one another.
class PreTokenizer {
public:
  // `spec` supplies name/space/cut/split_digits/reconstruct; `dict`
  // selects the Cn axis ("" none, "no" char, path dict).
  explicit PreTokenizer(const PreTokenizerSpec& spec,
                        const std::string& dict = "");
  ~PreTokenizer();

  // Full pipeline: Normalize + Split.
  std::vector<std::string> PreTokenize(std::string_view text) const;

  // Just the split stage, on already-normalized text. Used by tokenizers
  // that normalize once and then split.
  std::vector<std::string> Split(std::string_view normalized) const;

  const Normalizer& normalizer() const { return normalizer_; }
  bool valid() const { return valid_; }

private:
  Normalizer normalizer_;
  std::string space_;
  int cut_;
  bool split_digits_;
  std::unique_ptr<CnCutter> cn_cutter_;  // owns the dict-mode segmenter
  ustr::CnCutFn cn_cut_fn_;              // empty when Cn=none
  bool valid_;
};

}  // namespace piece
