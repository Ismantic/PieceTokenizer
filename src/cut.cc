#include "cut.h"

#include <algorithm>
#include <cmath>
#include <fstream>

#include "common.h"

namespace piece {

ustr::CnCutFn MakeCnCut(const std::string& cn_dict,
                        std::unique_ptr<CnCutter>* owner) {
  owner->reset();
  if (cn_dict.empty()) return {};  // Cn=none

  if (cn_dict == "no") {
    // Cn=char: split each Han run into individual codepoints.
    return [](std::string_view s) {
      std::vector<std::string> out;
      const char* p = s.data();
      const char* end = p + s.size();
      while (p < end) {
        const int n = std::min<int>(ustr::OneUTF8Size(p), end - p);
        out.emplace_back(p, n);
        p += n;
      }
      return out;
    };
  }

  // Cn=dict: Unigram dictionary segmentation.
  auto dict = LoadCnDict(cn_dict);
  if (dict.empty()) {
    LOG(ERROR) << "cn dict is empty: " << cn_dict;
    return {};
  }
  *owner = std::make_unique<CnCutter>(dict);
  CnCutter* cutter = owner->get();
  return [cutter](std::string_view s) { return cutter->Cut(s); };
}

std::unordered_map<std::string, float_t> LoadCnDict(const std::string& path) {
    std::unordered_map<std::string, float_t> dict;
    std::ifstream in(path);
    if (!in) {
        LOG(ERROR) << "Cannot open cn dict: " << path;
        return dict;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const size_t tab = line.find('\t');
        if (tab == std::string::npos || tab == 0 || tab + 1 >= line.size())
            continue;
        try {
            const float_t freq = std::stod(line.c_str() + tab + 1);
            if (freq > 0) dict.emplace(line.substr(0, tab), freq);
        } catch (...) {
            // skip malformed lines
        }
    }
    LOG(INFO) << "Loaded cn dict: " << path << " (" << dict.size() << " words)";
    return dict;
}

namespace {
// Compute log(1 / sum_of_freqs) — matches NaiveCutter's unknown-word
// penalty. Returns -10.0 as a safe default for empty dicts.
float_t ComputeFallbackWeight(
    const std::unordered_map<std::string, float_t>& dict) {
    float_t total = 0.0;
    for (const auto& [w, f] : dict) total += f;
    if (total <= 0.0) return -10.0;
    return std::log(1.0 / total);
}
} // namespace

CnCutter::CnCutter(const std::unordered_map<std::string, float_t>& dict)
    : tokenizer_(std::make_unique<BytePieceTokenizer>(
          dict, ComputeFallbackWeight(dict))) {}

CnCutter::~CnCutter() = default;

std::vector<std::string> CnCutter::Cut(std::string_view han_run) const {
    return tokenizer_->Tokenize(han_run);
}

} // namespace piece
