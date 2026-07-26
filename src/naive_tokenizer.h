#pragma once

#include "naive_counter.h"

namespace piece {

class NaiveTokenizer {
public:
    using EncodeResult = std::vector<std::pair<std::string, int>>;
    using StrToInt = std::unordered_map<std::string_view, int>;

    explicit NaiveTokenizer(const Model& model_proto);
    ~NaiveTokenizer();

    int PieceID(std::string_view piece) const;
    EncodeResult Encode(std::string_view text) const;
    std::vector<int> EncodeAsIds(std::string_view text) const;
    std::vector<std::string> Tokenize(std::string_view text) const;
    std::string Decode(const std::vector<int>& ids) const;
    std::string Decode(const EncodeResult& encoded) const;

private:
    void ApplyMerges(std::vector<int>& ids) const;

    const Model* model_proto_;
    std::vector<Merge> merges_;
    StrToInt pieces_;
};

}  // namespace piece
