#pragma once
#include <atomic>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
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
                   const NormalizerSpec& normalizer_spec);
    ~BytePieceCounter();
    
    bool Count();
    bool Save() const;
    bool Serialize(Model* model) const;
    
private:
    using Str2Int = std::unordered_map<std::string, int>;

    size_t GetWorkerCount(size_t work_items) const;

    template <typename Map>
    static void MergeCounts(Map* dest, const Map& src) {
        for (const auto& [key, value] : src) {
            (*dest)[key] += value;
        }
    }

    template <typename Map, typename CountFn>
    Map ParallelBatchCount(size_t total, size_t batch_size, CountFn&& count_fn) const {
        Map merged;
        if (total == 0) {
            return merged;
        }

        const size_t workers = GetWorkerCount((total + batch_size - 1) / batch_size);
        if (workers <= 1) {
            return count_fn(0, total);
        }

        std::atomic<size_t> next_begin{0};
        std::mutex merge_mu;
        std::vector<std::thread> threads;
        threads.reserve(workers);

        for (size_t worker = 0; worker < workers; ++worker) {
            threads.emplace_back([&, worker]() {
                while (true) {
                    const size_t begin = next_begin.fetch_add(batch_size);
                    if (begin >= total) {
                        break;
                    }
                    const size_t end = std::min(total, begin + batch_size);
                    Map batch_counts = count_fn(begin, end);
                    std::lock_guard<std::mutex> lock(merge_mu);
                    MergeCounts(&merged, batch_counts);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        return merged;
    }
    
    bool InitMetaPieces();
    std::unique_ptr<MultiFileSentenceIterator> MakeIterator() const;

    bool StreamCountRaw();
    void PruneRaw();
    std::vector<std::string> Tokenize(const std::string& text) const;
    Str2Int StreamCountPieces();
    Str2Int SplitPieces(const Str2Int& keep, const Str2Int& drop);
    Str2Int PrunePieces(Str2Int& pieces);
    void InitT();
    
    std::map<int, std::pair<std::string, Model::Piece::Type>> meta_pieces_;
    std::vector<std::pair<std::string, float>> pieces_;
    CounterSpec counter_spec_;
    NormalizerSpec normalizer_spec_;
    
    static constexpr size_t max_piece_count_ = 6;
    static constexpr size_t max_piece_size_ = 18;
    static constexpr size_t min_count_ = 2;
    static constexpr float_t INF = std::numeric_limits<float_t>::infinity();
    std::vector<std::vector<float_t>> T_;
    std::vector<std::unordered_map<std::string, float_t>> N_;
};

} // namespace piece
