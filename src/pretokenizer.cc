#include "pretokenizer.h"

namespace piece {

PreTokenizer::PreTokenizer(const PreTokenizerSpec& spec,
                           const std::string& cn_dict)
    : normalizer_(spec),
      space_(spec.GetSpace()),
      cut_(spec.GetCut()),
      split_digits_(spec.GetSplitDigits()),
      cn_cut_fn_(MakeCnCut(cn_dict, &cn_cutter_)) {}

PreTokenizer::~PreTokenizer() = default;

std::vector<std::string> PreTokenizer::Split(std::string_view normalized) const {
  // SplitTextCn is the unified split: with an empty cn_cut_fn_ it keeps Han
  // runs whole (Cn=none) and still honors split_digits, so a single call
  // covers every (cut, split_digits, cn) combination.
  return ustr::SplitTextCn(normalized, space_, cn_cut_fn_, cut_, split_digits_);
}

std::vector<std::string> PreTokenizer::PreTokenize(std::string_view text) const {
  return Split(normalizer_.Normalize(text));
}

}  // namespace piece
