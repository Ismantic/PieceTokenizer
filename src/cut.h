#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "bytepiece_tokenizer.h"
#include "common.h"

namespace piece {

// Loads a TSV dictionary file (`word\tfreq` per line) into a
// {word -> freq} map. Returns an empty map on failure.
std::unordered_map<std::string, float_t> LoadCnDict(const std::string& path);

// CnCutter applies Unigram-style word segmentation to runs of Han
// characters using a precomputed dictionary. The unknown-word penalty is
// log(1/sum_of_freqs), matching Iscut's NaiveCutter convention.
class CnCutter {
public:
    explicit CnCutter(const std::unordered_map<std::string, float_t>& dict);
    ~CnCutter();

    // Segment a run of Han characters into words. Caller guarantees the
    // input is a contiguous run of Han codepoints (no spaces or punct).
    std::vector<std::string> Cut(std::string_view han_run) const;

private:
    std::unique_ptr<BytePieceTokenizer> tokenizer_;
};

} // namespace piece
