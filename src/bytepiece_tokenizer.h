#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "piece_spec.h"
#include "pretokenizer.h"
#include "trie.h"
#include "ustr.h"

namespace piece {

class BytePieceTokenizer {
public:
    using EncodeResult = std::vector<std::pair<std::string, int>>;
    using StrToInt = std::unordered_map<std::string_view, int>;

    // `fallback_weight` is the log-prob assigned to the single-character
    // fallback edge in the Unigram lattice. Pass log(1/sum_of_freqs) to
    // align with a NaiveCutter-style "freq=1 unknown word" penalty.
    explicit BytePieceTokenizer(
        const std::unordered_map<std::string, float_t>& dict,
        float_t fallback_weight = -10.0);
    explicit BytePieceTokenizer(const Model& model);
    ~BytePieceTokenizer();

    EncodeResult Encode(std::string_view text) const;
    std::vector<int> EncodeAsIds(std::string_view text) const;
    std::string Decode(const std::vector<int>& ids) const;
    std::string Decode(const EncodeResult& encoded) const;
    std::vector<std::string> Tokenize(std::string_view text) const;
    int PieceID(std::string_view piece) const;

private:
    struct Match {
        int e;
        int n;
        float_t w;
        Match(int e, int n, float_t w) : e(e), n(n), w(w) {}
    };

    std::vector<Match> GetMatches(std::string_view text) const;
    std::vector<std::string> TokenizeSegment(std::string_view text) const;
    void InitTrie(const std::unordered_map<std::string, float_t>& dict);

    const Model* model_ = nullptr;
    std::unique_ptr<PreTokenizer> pretokenizer_;
    StrToInt pieces_;
    int unk_id_ = -1;
    trie::DoubleArray<int> trie_;
    std::unordered_map<int, float_t> value_map_;
    float_t fallback_weight_ = -10.0;
};

}  // namespace piece
