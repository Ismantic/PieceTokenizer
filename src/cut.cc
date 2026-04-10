#include "cut.h"

#include <cmath>
#include <fstream>

#include "common.h"

namespace piece {

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
