#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "normalizer.h"
#include "piece_spec.h"
#include "ustr.h"

namespace piece {

class PieceTokenizer {
public:
  using EncodeResult = std::vector<std::pair<std::string, int>>;
  using StrToInt = std::unordered_map<std::string_view, int>;

  explicit PieceTokenizer(const Model& model);
  ~PieceTokenizer();

  EncodeResult Encode(std::string_view text) const;
  std::vector<std::string> Tokenize(std::string_view text) const;
  std::string Decode(const std::vector<int>& ids) const;
  std::string Decode(const EncodeResult& rs) const;
  int PieceID(std::string_view piece) const;

private:
  struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const {
      return std::hash<int>{}(p.first) ^ (std::hash<int>{}(p.second) << 1);
    }
  };

  std::string NormalizeText(std::string_view text) const;
  std::vector<int> BuildInitialTokenIds(const std::string& text) const;
  void GreedyMerge(std::vector<int>& ids) const;
  EncodeResult TokenIdsToResult(const std::vector<int>& ids) const;

  const Model* model_;
  std::unordered_map<std::pair<int, int>, int, PairHash> merge_ranks_;
  StrToInt pieces_;
  int unk_id_;
  int byte_to_id_[256];
};

}  // namespace piece
