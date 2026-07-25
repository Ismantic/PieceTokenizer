#include "pretokenizer.h"

namespace piece {

PreTokenizer::PreTokenizer(const PreTokenizerSpec& spec,
                           const std::string& dict)
    : normalizer_(spec),
      space_(spec.GetSpace()),
      cut_(spec.GetCut()),
      split_digits_(spec.GetSplitDigits()),
      cn_cut_fn_(MakeCnCut(dict, &cn_cutter_)),
      valid_(dict.empty() || static_cast<bool>(cn_cut_fn_)) {}

PreTokenizer::~PreTokenizer() = default;

std::vector<std::string> PreTokenizer::Split(std::string_view normalized) const {
  return ustr::SplitTextCn(normalized, space_, cn_cut_fn_, cut_, split_digits_);
}

std::vector<std::string> PreTokenizer::PreTokenize(std::string_view text) const {
  return Split(normalizer_.Normalize(text));
}

}  // namespace piece
