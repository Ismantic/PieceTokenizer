#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "darts.h"
#include "normalizer.h"
#include "piece_spec.h"
#include "ustr.h"

namespace piece {

class BytePieceTokenizer {
public:
    using EncodeResult = std::vector<std::pair<std::string, int>>;
    using StrToInt = std::unordered_map<std::string_view, int>;

    explicit BytePieceTokenizer(
        const std::unordered_map<std::string, float_t>& dict);
    explicit BytePieceTokenizer(const Model& model);
    ~BytePieceTokenizer();

    EncodeResult Encode(std::string_view text) const;
    std::string Decode(const std::vector<int>& ids) const;
    std::string Decode(const EncodeResult& encoded) const;
    std::vector<std::string> Tokenize(std::string_view text) const;

private:
    struct Match {
        int e;
        int n;
        float_t w;
        Match(int e, int n, float_t w) : e(e), n(n), w(w) {}
    };

    int PieceID(std::string_view piece) const;
    std::vector<Match> GetMatches(std::string_view text) const;
    void InitTrie(const std::unordered_map<std::string, float_t>& dict);

    const Model* model_ = nullptr;
    Normalizer normalizer_;
    StrToInt pieces_;
    int unk_id_ = -1;
    new_darts::DoubleArray<int> trie_;
    std::unordered_map<int, float_t> value_map_;
};

}  // namespace piece
