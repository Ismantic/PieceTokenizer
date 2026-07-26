#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "piece_spec.h"
#include "pretokenizer.h"
#include "ustr.h"

namespace piece {

class PieceTokenizer {
public:
  using EncodeResult = std::vector<std::pair<std::string, int>>;
  using StrToInt = std::unordered_map<std::string_view, int>;

  explicit PieceTokenizer(const Model& model,
                          const std::string& dict = "");
  ~PieceTokenizer();

  EncodeResult Encode(std::string_view text) const;
  std::vector<int> EncodeAsIds(std::string_view text) const;
  std::vector<std::string> Tokenize(std::string_view text) const;
  std::string Decode(const std::vector<int>& ids) const;
  std::string Decode(const EncodeResult& rs) const;
  int PieceID(std::string_view piece) const;
  bool valid() const { return pretokenizer_.valid(); }

private:
  static uint64_t MergeKey(int left, int right) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32) |
           static_cast<uint32_t>(right);
  }

  void BuildInitialTokenIds(std::string_view text, std::vector<int>* ids) const;
  void GreedyMerge(std::vector<int>& ids) const;
  void AppendTokenIds(const std::vector<int>& ids, EncodeResult* result) const;

  const Model* model_;
  PreTokenizer pretokenizer_;
  std::unordered_map<uint64_t, int> merge_ranks_;
  StrToInt pieces_;
  int unk_id_;
  int byte_to_id_[256];
};

}  // namespace piece
