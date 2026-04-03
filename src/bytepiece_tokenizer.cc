#include "bytepiece_tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace piece {

BytePieceTokenizer::BytePieceTokenizer(
    const std::unordered_map<std::string, float_t>& dict)
    : model_(nullptr), normalizer_(NormalizerSpec()) {
    InitTrie(dict);
}

BytePieceTokenizer::BytePieceTokenizer(const Model& model)
    : model_(&model),
      normalizer_(model.GetNormalizerSpec()) {
    const auto& counter_spec = model_->GetCounterSpec();
    unk_id_ = counter_spec.unk_id();

    std::unordered_map<std::string, float_t> dict;
    for (size_t i = 0; i < model_->PiecesSize(); ++i) {
        const auto& piece = model_->GetPieces(i);
        pieces_[piece.GetPiece()] = i;
        if (piece.GetType() == Model::Piece::NORMAL) {
            dict[piece.GetPiece()] = piece.GetScore();
        }
    }

    InitTrie(dict);
    LOG(INFO) << "PiecesSize=" << pieces_.size();
}

BytePieceTokenizer::~BytePieceTokenizer() = default;

BytePieceTokenizer::EncodeResult BytePieceTokenizer::Encode(
    std::string_view text) const {
    const std::string text_str = normalizer_.Normalize(text);
    const std::vector<std::string> tokens = Tokenize(text_str);

    EncodeResult output;
    for (const auto& token : tokens) {
        const int piece_id = PieceID(token);

        if (piece_id == unk_id_) {
            for (size_t i = 0; i < token.size(); ++i) {
                const uint8_t byte = static_cast<uint8_t>(token[i]);
                const std::string byte_piece = ustr::ByteToPiece(byte);
                const int byte_id = PieceID(byte_piece);
                output.emplace_back(byte_piece, byte_id);
            }
        } else {
            output.emplace_back(token, piece_id);
        }
    }

    return output;
}

std::string BytePieceTokenizer::Decode(const std::vector<int>& ids) const {
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
            const int byte_value = ustr::PieceToByte(piece.GetPiece());
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

std::string BytePieceTokenizer::Decode(const EncodeResult& encoded) const {
    std::vector<int> ids;
    ids.reserve(encoded.size());
    for (const auto& [piece, id] : encoded) {
        ids.push_back(id);
    }
    return Decode(ids);
}

std::vector<std::string> BytePieceTokenizer::Tokenize(
    std::string_view text) const {
    const float_t inf = std::numeric_limits<float_t>::infinity();
    const int num = text.length();
    std::vector<float_t> scores(num + 1, -inf);
    std::vector<int> routes(num + 1);
    scores[0] = 0;
    for (int i = 0; i <= num; i++) {
        routes[i] = i;
    }

    const auto matches = GetMatches(text);
    for (const auto& m : matches) {
        const int start = m.e - m.n + 1;
        const int end = m.e + 1;
        if (start < 0 || start >= static_cast<int>(scores.size()) ||
            end < 0 || end >= static_cast<int>(scores.size())) {
            continue;
        }

        const float_t score = scores[start] + m.w;
        if (score > scores[end]) {
            scores[end] = score;
            routes[end] = start;
        }
    }

    std::vector<std::string> tokens;
    int end = num;
    while (end > 0) {
        const int start = routes[end];
        tokens.emplace_back(text.substr(start, end - start));
        end = start;
    }
    std::reverse(tokens.begin(), tokens.end());
    return tokens;
}

int BytePieceTokenizer::PieceID(std::string_view piece) const {
    const auto it = pieces_.find(piece);
    return it != pieces_.end() ? it->second : unk_id_;
}

std::vector<BytePieceTokenizer::Match> BytePieceTokenizer::GetMatches(
    std::string_view text) const {
    std::vector<Match> matches;
    const int num = text.length();
    int pos = 0;
    while (pos < num) {
        const int n = static_cast<int>(
            ustr::UTF8CharLen(static_cast<unsigned char>(text[pos])));
        if (pos + n - 1 < num) {
            matches.emplace_back(pos + n - 1, n, -10.0);
        }

        const size_t kMaxNumResults = 16;
        new_darts::DoubleArray<int>::ResultPair results[kMaxNumResults];
        const size_t num_results = trie_.commonPrefixSearch(
            text.data() + pos, results, kMaxNumResults, num - pos);
        for (size_t i = 0; i < num_results; ++i) {
            if (pos + results[i].length - 1 < num) {
                matches.emplace_back(pos + results[i].length - 1,
                                     results[i].length,
                                     value_map_.at(results[i].value));
            }
        }
        pos += n;
    }
    return matches;
}

void BytePieceTokenizer::InitTrie(
    const std::unordered_map<std::string, float_t>& dict) {
    float_t total = 0.0;
    for (const auto& [piece, score] : dict) {
        total += score;
    }
    const float_t log_total = std::log(total);

    std::vector<std::pair<std::string, float_t>> sorted_dict(dict.begin(),
                                                             dict.end());
    std::sort(sorted_dict.begin(), sorted_dict.end());

    std::vector<const char*> strs;
    std::vector<int> values;
    int next_value = 1;
    for (const auto& [piece, score] : sorted_dict) {
        if (piece.empty()) {
            continue;
        }
        strs.push_back(piece.c_str());
        value_map_[next_value] = std::log(score) - log_total;
        values.push_back(next_value);
        next_value++;
    }

    trie_.build(strs.size(), strs.data(), nullptr, values.data());
}

}  // namespace piece
