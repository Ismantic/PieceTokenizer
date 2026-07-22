#pragma once
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "bytepiece_tokenizer.h"
#include "common.h"
#include "piece_spec.h"
#include "ustr.h"

namespace piece {
class BytePieceCounter {
public:
    BytePieceCounter(const CounterSpec& counter_spec,
                   const PreTokenizerSpec& pretokenizer_spec);
    ~BytePieceCounter();
    
    bool Count();
    bool Save() const;
    bool Serialize(Model* model) const;
    
private:
    using Str2Int = std::unordered_map<std::string, int>;

    size_t GetCpuCount(size_t work_items) const;

    template <typename Map>
    static void MergeCounts(Map* dest, const Map& src) {
        for (const auto& [key, value] : src) {
            (*dest)[key] += value;
        }
    }

    bool InitMetaPieces();
    std::unique_ptr<MultiFileSentenceIterator> MakeIterator() const;

    bool StreamCountRaw();
    void PruneRaw();
    std::vector<std::string> Tokenize(const std::string& text) const;
    Str2Int StreamCountPieces();
    Str2Int SplitPieces(const Str2Int& keep, const Str2Int& drop);
    Str2Int PrunePieces(Str2Int& pieces);
    
    std::map<int, std::pair<std::string, Model::Piece::Type>> meta_pieces_;
    std::vector<std::pair<std::string, float>> pieces_;
    CounterSpec counter_spec_;
    PreTokenizerSpec pretokenizer_spec_;
    
    static constexpr size_t max_piece_count_ = 6;
    static constexpr float_t INF = std::numeric_limits<float_t>::infinity();
    static const std::vector<std::vector<float_t>> T_;
    std::vector<std::unordered_map<std::string, float_t>> N_;
};

} // namespace piece
