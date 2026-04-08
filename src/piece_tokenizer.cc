#include "piece_tokenizer.h"

#include <climits>

namespace piece {

PieceTokenizer::PieceTokenizer(const Model& model) : model_(&model) {
  const auto& counter_spec = model_->GetCounterSpec();
  unk_id_ = counter_spec.unk_id();

  for (size_t i = 0; i < model_->PiecesSize(); ++i) {
    const auto& piece = model_->GetPieces(i);
    pieces_[piece.GetPiece()] = i;

    const std::string& u = piece.u();
    const std::string& v = piece.v();
    if (!u.empty() && !v.empty()) {
      int u_id = PieceID(u);
      int v_id = PieceID(v);
      if (u_id >= 0 && v_id >= 0) {
        merge_ranks_[{u_id, v_id}] = static_cast<int>(i);
      }
    }
  }

  // Pre-build byte → id lookup table.
  for (int b = 0; b < 256; ++b) {
    std::string byte_str(1, static_cast<char>(b));
    auto it = pieces_.find(byte_str);
    byte_to_id_[b] = it != pieces_.end() ? it->second : unk_id_;
  }

  LOG(INFO) << "Built merge ranks from model with "
            << model_->PiecesSize() << " pieces, "
            << merge_ranks_.size() << " merge rules";
}

PieceTokenizer::~PieceTokenizer() = default;

PieceTokenizer::EncodeResult PieceTokenizer::Encode(std::string_view text) const {
  std::string normalized = NormalizeText(text);
  std::vector<int> ids = BuildInitialTokenIds(normalized);
  GreedyMerge(ids);
  return TokenIdsToResult(ids);
}

std::vector<std::string> PieceTokenizer::Tokenize(std::string_view text) const {
  std::vector<std::string> tokens;
  auto encoded = Encode(text);
  tokens.reserve(encoded.size());
  for (const auto& [piece, id] : encoded) {
    tokens.push_back(piece);
  }
  return tokens;
}

std::string PieceTokenizer::Decode(const std::vector<int>& ids) const {
  std::string result;
  for (int id : ids) {
    if (id < 0 || id >= static_cast<int>(model_->PiecesSize())) {
      if (unk_id_ >= 0) {
        result += model_->GetPieces(unk_id_).GetPiece();
      }
      continue;
    }

    const auto& piece = model_->GetPieces(id);
    if (piece.GetType() == Model::Piece::BYTE) {
      auto byte_value = ustr::PieceToByte(piece.GetPiece());
      if (byte_value >= 0) {
        result.push_back(static_cast<char>(byte_value));
      } else {
        result += piece.GetPiece();
      }
    } else if (piece.GetType() != Model::Piece::UNKNOWN &&
               piece.GetType() != Model::Piece::CONTROL) {
      result += piece.GetPiece();
    }
  }

  if (model_->HasNormalizerSpec()) {
    Normalizer normalizer(model_->GetNormalizerSpec());
    return normalizer.ReplaceSpace(result);
  }

  return result;
}

std::string PieceTokenizer::Decode(const EncodeResult& rs) const {
  std::vector<int> ids;
  ids.reserve(rs.size());
  for (const auto& [piece, id] : rs) {
    ids.push_back(id);
  }
  return Decode(ids);
}

int PieceTokenizer::PieceID(std::string_view piece) const {
  const auto it = pieces_.find(piece);
  return it != pieces_.end() ? it->second : unk_id_;
}

std::string PieceTokenizer::NormalizeText(std::string_view text) const {
  if (model_->HasNormalizerSpec()) {
    Normalizer normalizer(model_->GetNormalizerSpec());
    return normalizer.Normalize(std::string(text));
  }
  return std::string(text);
}

std::vector<int> PieceTokenizer::BuildInitialTokenIds(
    const std::string& text) const {
  std::vector<int> ids;
  ids.reserve(text.size());
  for (unsigned char c : text) {
    ids.push_back(byte_to_id_[c]);
  }
  return ids;
}

// Greedy merge: repeatedly find the pair with the lowest merge rank
// (earliest learned merge) and apply it, until no more merges are possible.
void PieceTokenizer::GreedyMerge(std::vector<int>& ids) const {
  while (ids.size() >= 2) {
    // Find the pair with the smallest merge rank
    int best_rank = INT_MAX;
    size_t best_pos = 0;
    for (size_t i = 0; i + 1 < ids.size(); ++i) {
      auto it = merge_ranks_.find({ids[i], ids[i + 1]});
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_pos = i;
      }
    }
    if (best_rank == INT_MAX) break;

    // Apply: replace ids[best_pos] and ids[best_pos+1] with best_rank
    ids[best_pos] = best_rank;
    ids.erase(ids.begin() + best_pos + 1);
  }
}

PieceTokenizer::EncodeResult PieceTokenizer::TokenIdsToResult(
    const std::vector<int>& ids) const {
  EncodeResult result;
  for (int id : ids) {
    if (id >= 0 && id < static_cast<int>(model_->PiecesSize())) {
      const std::string& piece = model_->GetPieces(id).GetPiece();
      result.emplace_back(piece, id);
    } else if (unk_id_ >= 0) {
      const std::string& unk_piece = model_->GetPieces(unk_id_).GetPiece();
      result.emplace_back(unk_piece, unk_id_);
    }
  }
  return result;
}

}  // namespace piece
