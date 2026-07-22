#include "extra_tokens.h"

#include <unordered_set>

#include "common.h"

namespace piece {

int InsertExtraTokens(Model* model, const std::vector<std::string>& tokens,
                      bool repoint_pad) {
  // Existing piece strings, for dedup.
  std::unordered_set<std::string> existing;
  existing.reserve(model->PiecesSize() * 2);
  for (size_t i = 0; i < model->PiecesSize(); ++i)
    existing.insert(model->GetPieces(i).GetPiece());

  int added = 0;
  for (const auto& tok : tokens) {
    if (tok.empty()) continue;
    if (!existing.insert(tok).second) {
      LOG(INFO) << "InsertExtraTokens: skip existing token: " << tok;
      continue;
    }
    auto* p = model->InsertPieces();
    p->SetPiece(tok);
    p->SetType(Model::Piece::CONTROL);
    p->SetScore(0.0);
    ++added;
  }

  CounterSpec* cs = model->GetMutableCounterSpec();
  // Keep vocab_size == actual piece count.
  cs->set_vocab_size(static_cast<int32_t>(model->PiecesSize()));

  // Repoint pad_id to pad_piece if pad is currently disabled and now present.
  if (repoint_pad && cs->pad_id() < 0 && !cs->pad_piece().empty()) {
    for (size_t i = 0; i < model->PiecesSize(); ++i) {
      if (model->GetPieces(i).GetPiece() == cs->pad_piece()) {
        cs->set_pad_id(static_cast<int32_t>(i));
        LOG(INFO) << "InsertExtraTokens: pad_id -> " << i
                  << " (" << cs->pad_piece() << ")";
        break;
      }
    }
  }
  return added;
}

}  // namespace piece
