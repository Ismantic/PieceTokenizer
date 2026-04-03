#include "sentencepiece_tokenizer.h"

#include <queue>

#include "piece_spec.h"
#include "ustr.h"

namespace piece {

SentencePieceTokenizer::SentencePieceTokenizer(const Model& model)
    : model_(&model),
      normalizer_(model.GetNormalizerSpec()),
      unk_id_(-1) {
  for (size_t i = 0; i < model_->PiecesSize(); ++i) {
    const auto& p = model_->GetPieces(i);
    pieces_[p.GetPiece()] = i;
    if (p.GetType() == Model::Piece::UNKNOWN) {
      unk_id_ = i;
    }
  }
  LOG(INFO) << "PiecesSize=" << pieces_.size();
}

SentencePieceTokenizer::~SentencePieceTokenizer() = default;

std::vector<std::string> SentencePieceTokenizer::Tokenize(
    std::string_view text) const {
    std::vector<std::string> tokens;
    auto encoded = Encode(text);
    tokens.reserve(encoded.size());
    for (const auto& [piece, id] : encoded) {
        tokens.push_back(piece);
    }
    return tokens;
}

std::string SentencePieceTokenizer::Decode(const EncodeResult& rs) const {
    std::vector<int> ids;
    ids.reserve(rs.size());
    for (const auto& [piece, id] : rs) {
        ids.push_back(id);
    }
    return Decode(ids);
}

int SentencePieceTokenizer::PieceID(std::string_view piece) const {
    const auto it = pieces_.find(piece);
    return it != pieces_.end() ? it->second : unk_id_;
}

EncodeResult SentencePieceTokenizer::Encode(std::string_view str) const {
    std::string ns = normalizer_.Normalize(str);
    std::string_view text = ns;
    if (text.empty()) {
        return {};
    }

    struct SymbolPair {
        int left;
        int right;
        float score;
        size_t size;
    };

    class SymbolPairComparator {
      public:
        bool operator() (SymbolPair *h1, SymbolPair *h2) {
            return (h1->score < h2->score ||
                   (h1->score == h2->score && h1->left > h2->left));
        }
    };

    struct Symbol {
        int prev; // -1 for BOS
        int next; // -1 for EOS
        std::string_view piece;
    };

    using Agenda = std::priority_queue<SymbolPair*, std::vector<SymbolPair*>,
                                       SymbolPairComparator>;
    Agenda agenda;
    std::vector<Symbol> symbols;
    symbols.reserve(text.size());

    constexpr size_t kPreallocateSymbolPairSize = 256;
    FreeList<SymbolPair> symbol_pair_allocator(kPreallocateSymbolPairSize);

    auto MaybeAddNewSymbolPair = [this, &symbol_pair_allocator, &symbols,
                                  &agenda](int left, int right) {
        if (left == -1 || right == -1)
            return;
        const std::string_view piece(
            symbols[left].piece.data(),
            symbols[left].piece.size()+symbols[right].piece.size());
        const auto it = pieces_.find(piece);
        if (it == pieces_.end()) {
            return;
        }
        auto* h = symbol_pair_allocator.Allocate();
        h->left = left;
        h->right = right;
        h->score = model_->GetPieces(it->second).GetScore();
        h->size = piece.size();
        agenda.push(h);
    };

    // Splits the input into character sequence
    int index = 0;
    while (!text.empty()) {
        Symbol s;
        const int mblen = ustr::OneUTF8Size(text.data());
        s.piece = std::string_view(text.data(), mblen);
        s.prev = index == 0 ? -1 : index-1;
        text.remove_prefix(mblen);
        s.next = text.empty() ? -1 : index+1;
        ++index;
        symbols.emplace_back(s);
    }

    for (size_t i = 1; i < symbols.size(); ++i) {
        MaybeAddNewSymbolPair(i-1, i);
    }

    // Main loop.
    while (!agenda.empty()) {
        SymbolPair *top = agenda.top();
        agenda.pop();

        if (symbols[top->left].piece.empty() ||
            symbols[top->right].piece.empty() ||
            symbols[top->left].piece.size() + symbols[top->right].piece.size()
             != top->size) {
            continue;
        }

        symbols[top->left].piece = std::string_view(
            symbols[top->left].piece.data(),
            symbols[top->left].piece.size() + symbols[top->right].piece.size());

        symbols[top->left].next = symbols[top->right].next;
        if (symbols[top->right].next >= 0) {
            symbols[symbols[top->right].next].prev = top->left;
        }
        symbols[top->right].piece = std::string_view("");

        MaybeAddNewSymbolPair(symbols[top->left].prev, top->left);
        MaybeAddNewSymbolPair(top->left, symbols[top->left].next);
    }

    EncodeResult output;
    for (int index = 0; index != -1; index = symbols[index].next) {
        auto w = symbols[index].piece;
        int i = PieceID(w);

        if (i == unk_id_) {
            for (size_t j = 0; j < w.size(); ++j) {
                const uint8_t byte = static_cast<uint8_t>(w[j]);
                std::string byte_piece = ustr::ByteToPiece(byte);
                int byte_id = PieceID(byte_piece);
                output.emplace_back(byte_piece, byte_id);
            }
        } else {
            output.emplace_back(std::string(w), i);
        }
    }

    return output;
}

std::string SentencePieceTokenizer::Decode(const std::vector<int>& ids) const {
    std::string result;
    result.reserve(ids.size() * 3);

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

    return normalizer_.ReplaceSpace(result);
}

} // namespace piece
