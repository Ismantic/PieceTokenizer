#include "cut.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

#include "common.h"

namespace piece {

ustr::CnCutFn MakeCnCut(const std::string& dict,
                        std::unique_ptr<CnCutter>* owner) {
  owner->reset();
  if (dict.empty()) return {};  // Cn=none

  if (dict == "no") {
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
  auto entries = LoadCnDict(dict);
  if (entries.empty()) {
    LOG(ERROR) << "dict is empty: " << dict;
    return {};
  }
  *owner = std::make_unique<CnCutter>(entries);
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
    : fallback_weight_(ComputeFallbackWeight(dict)) {
    float_t total = 0.0;
    for (const auto& [word, frequency] : dict) total += frequency;
    const float_t log_total = std::log(total);

    std::vector<std::pair<std::string, float_t>> sorted(dict.begin(), dict.end());
    std::sort(sorted.begin(), sorted.end());
    std::vector<const char*> keys;
    std::vector<int> values;
    int value = 1;
    for (const auto& [word, frequency] : sorted) {
        if (word.empty()) continue;
        keys.push_back(word.c_str());
        values.push_back(value);
        weights_[value++] = std::log(frequency) - log_total;
    }
    trie_.build(keys.size(), keys.data(), nullptr, values.data());
}

CnCutter::~CnCutter() = default;

std::vector<std::string> CnCutter::Cut(std::string_view han_run) const {
    const int size = static_cast<int>(han_run.size());
    std::vector<float_t> scores(size + 1,
        -std::numeric_limits<float_t>::infinity());
    std::vector<int> routes(size + 1);
    scores[0] = 0.0;
    for (int i = 0; i <= size; ++i) routes[i] = i;

    for (const auto& match : GetMatches(han_run)) {
        const int start = match.end - match.length + 1;
        const int end = match.end + 1;
        const float_t score = scores[start] + match.weight;
        if (score > scores[end]) {
            scores[end] = score;
            routes[end] = start;
        }
    }

    std::vector<std::string> result;
    for (int end = size; end > 0;) {
        const int start = routes[end];
        result.emplace_back(han_run.substr(start, end - start));
        end = start;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

std::vector<CnCutter::Match> CnCutter::GetMatches(std::string_view text) const {
    std::vector<Match> matches;
    const int size = static_cast<int>(text.size());
    for (int pos = 0; pos < size;) {
        const int length = static_cast<int>(
            ustr::UTF8CharLen(static_cast<unsigned char>(text[pos])));
        if (pos + length <= size) {
            matches.push_back({pos + length - 1, length, fallback_weight_});
        }

        using Result = new_darts::DoubleArray<int>::ResultPair;
        std::vector<Result> results(16);
        size_t count = trie_.commonPrefixSearch(
            text.data() + pos, results.data(), results.size(), size - pos);
        if (count > results.size()) {
            results.resize(count);
            count = trie_.commonPrefixSearch(
                text.data() + pos, results.data(), results.size(), size - pos);
        }
        for (size_t i = 0; i < count; ++i) {
            if (pos + results[i].length <= static_cast<size_t>(size)) {
                matches.push_back({pos + static_cast<int>(results[i].length) - 1,
                                   static_cast<int>(results[i].length),
                                   weights_.at(results[i].value)});
            }
        }
        pos += length;
    }
    return matches;
}

} // namespace piece
