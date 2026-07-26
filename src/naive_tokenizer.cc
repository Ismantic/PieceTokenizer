#include "naive_tokenizer.h"



namespace piece {

NaiveTokenizer::NaiveTokenizer(const Model& model_proto)
    : model_proto_(&model_proto) {
  for (size_t i = 0; i < model_proto_->PiecesSize(); ++i) {
    const auto& p = model_proto_->GetPieces(i);
    pieces_[p.GetPiece()] = i;
    if (static_cast<int>(i) >= INITIAL_VOCAB_SIZE && !p.u().empty() && !p.v().empty()) {
      int u_id = PieceID(p.u());
      int v_id = PieceID(p.v());
      merges_.push_back({{u_id, v_id}, static_cast<int>(i)});
    }
  }
}

NaiveTokenizer::~NaiveTokenizer() = default;

int NaiveTokenizer::PieceID(std::string_view piece) const {
  const auto it = pieces_.find(piece);
  return it != pieces_.end() ? it->second : -1;
}

NaiveTokenizer::EncodeResult NaiveTokenizer::Encode(std::string_view text) const {
  std::vector<int> ids;
  ids.reserve(text.size());
  for (unsigned char c : text) {
    ids.push_back(c);
  }
  ApplyMerges(ids);

  EncodeResult result;
  for (int id : ids) {
    result.emplace_back(model_proto_->GetPieces(id).GetPiece(), id);
  }
  return result;
}

std::vector<int> NaiveTokenizer::EncodeAsIds(std::string_view text) const {
  std::vector<int> ids;
  ids.reserve(text.size());
  for (unsigned char c : text) ids.push_back(c);
  ApplyMerges(ids);
  return ids;
}

std::vector<std::string> NaiveTokenizer::Tokenize(std::string_view text) const {
  std::vector<std::string> tokens;
  auto encoded = Encode(text);
  tokens.reserve(encoded.size());
  for (const auto& [piece, id] : encoded) {
    tokens.push_back(piece);
  }
  return tokens;
}

std::string NaiveTokenizer::Decode(const std::vector<int>& ids) const {
  std::string result;
  for (int id : ids) {
    if (id >= 0 && id < static_cast<int>(model_proto_->PiecesSize())) {
      result += model_proto_->GetPieces(id).GetPiece();
    }
  }
  return result;
}

std::string NaiveTokenizer::Decode(const EncodeResult& encoded) const {
  std::vector<int> ids;
  ids.reserve(encoded.size());
  for (const auto& [piece, id] : encoded) {
    ids.push_back(id);
  }
  return Decode(ids);
}

void NaiveTokenizer::ApplyMerges(std::vector<int>& ids) const {
  for (const auto& merge : merges_) {
    if (ids.size() < 2) break;
    naive::MergePair(ids, merge.pair, merge.idx);
  }
}

}  // namespace piece
