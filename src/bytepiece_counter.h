#pragma once
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <queue>
#include <memory>
#include <algorithm>
#include <stdint.h>
#include <math.h>
#include <limits>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include "darts.h"
#include "piece_spec.h"
#include "common.h"
#include "ustr.h"
#include "misc.h"
#include "sentence.h"
#include "new_normalizer.h"

#include "trie.h"

namespace piece {
using float_t = double;
inline int SizeUTF8(uint8_t c) {
    if ((c & 0b10000000) == 0) return 1;
    else if ((c & 0b11100000) == 0b11000000) return 2;
    else if ((c & 0b11110000) == 0b11100000) return 3;
    else if ((c & 0b11111000) == 0b11110000) return 4;
    return 1;
}

inline bool IsValidUTF8(const std::string& str) {
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str.c_str());
    size_t len = str.length();
    
    for (size_t i = 0; i < len; i++) {
        if (bytes[i] <= 0x7F) {
            continue;
        } else if ((bytes[i] & 0xE0) == 0xC0) {
            if (i + 1 >= len || (bytes[i + 1] & 0xC0) != 0x80) {
                return false;
            }
            if ((bytes[i] & 0x1E) == 0) {
                return false;  
            }
            i += 1;
        } else if ((bytes[i] & 0xF0) == 0xE0) {
            if (i + 2 >= len || (bytes[i + 1] & 0xC0) != 0x80 || (bytes[i + 2] & 0xC0) != 0x80) {
                return false;
            }
            if ((bytes[i] == 0xE0 && (bytes[i + 1] & 0x20) == 0) ||
                (bytes[i] == 0xED && (bytes[i + 1] & 0x20) == 0x20)) {
                return false;
            }
            i += 2;
        } else if ((bytes[i] & 0xF8) == 0xF0) {
            if (i + 3 >= len || (bytes[i + 1] & 0xC0) != 0x80 || 
                (bytes[i + 2] & 0xC0) != 0x80 || (bytes[i + 3] & 0xC0) != 0x80) {
                return false;
            }
            if ((bytes[i] == 0xF0 && (bytes[i + 1] & 0x30) == 0) ||
                (bytes[i] > 0xF4) || (bytes[i] == 0xF4 && bytes[i + 1] > 0x8F)) {
                return false;
            }
            i += 3;
        } else {
            return false;
        }
    }
    return true;
}



class BytePieceTokenizer {
    public:
        using EncodeResult = std::vector<std::pair<std::string, int>>;
        using StrToInt = std::unordered_map<std::string_view, int>;
        
        
        BytePieceTokenizer(const std::unordered_map<std::string, float_t>& dict) 
            : model_(nullptr), normalizer_(nullptr) {
            InitFromDict(dict);
        }
        
        explicit BytePieceTokenizer(const Model& model) 
            : model_(&model),
            normalizer_(std::make_unique<Normalizer>(model.GetNormalizerSpec()))  {
            InitFromModel();
        }
        
        virtual ~BytePieceTokenizer() {}
        
        EncodeResult Encode(std::string_view text) const {
            if (trie_.size() == 0) {
                throw std::runtime_error("Tokenizer not initialized");
            }
            
            std::string text_str = normalizer_->Normalize(text);
            std::vector<std::string> tokens = Tokenize(text_str);
            
            EncodeResult output;
            for (const auto& token : tokens) {
                int i = PieceID(token);

                if (i == unk_id_) {
                    for (size_t i = 0; i < token.size(); ++i) {
                        const uint8_t byte = static_cast<uint8_t>(token[i]);
                        std::string byte_piece = ustr::ByteToPiece(byte);
                        int byte_id = PieceID(byte_piece);
                        output.emplace_back(byte_piece, byte_id);
                    }
                } else {
                    output.emplace_back(std::string(token),i);
                }
            }
            
            return output;
        }       
       
        std::string Decode(const std::vector<int>& ids) const {
            if (!model_) {
                throw std::runtime_error("Tokenizer not initialized with a model");
            }
            
            std::string result;
            result.reserve(ids.size() * 3); 
            
            for (int id : ids) {
                if (id < 0 || id >= static_cast<int>(model_->PiecesSize())) {
                    if (unk_id_ >= 0) {
                        const auto& counter_spec = model_->GetCounterSpec();
                        result += counter_spec.unk_piece();
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
            
            return normalizer_->ReplaceSpace(result);
        }    

        std::string Decode(const EncodeResult& encoded) const {
            std::vector<int> ids;
            ids.reserve(encoded.size());
            for (const auto& [piece, id] : encoded) {
                ids.push_back(id);
            }
            return Decode(ids);
        }
    
        std::vector<std::string> Tokenize(const std::string& sentence) const {
            const float_t INF = std::numeric_limits<float_t>::infinity();
            const int num = sentence.length();
            std::vector<float_t> scores(num + 1, -INF);
            std::vector<int> routes(num + 1);
            scores[0] = 0;
            for (int i = 0; i <= num; i++) {
                routes[i] = i;
            }
            auto matches = GetMatches(sentence);
            for (const auto& m : matches) {
                int s = m.e - m.n + 1;
                int e = m.e + 1;
    
                if (s < 0 || s >= scores.size() || e >= scores.size()) {
                    continue;
                }
    
                float_t score = scores[s] + m.w;
                if (score > scores[e]) {
                    scores[e] = score;
                    routes[e] = s;
                }
            }
            //std::cout << "MatchDone" << std::endl;
            std::vector<std::string> tokens;
            int e = num;
            // TODO: UNK
            while (e > 0) {
                int start = routes[e];
                tokens.push_back(sentence.substr(start, e - start));
                e = start;
            }
            std::reverse(tokens.begin(), tokens.end());
            return tokens;
        }
        
    private:
        int PieceID(std::string_view piece) const {
            auto it = reserve_.find(piece);
            if (it != reserve_.end()) {
                return it->second;
            }
            auto it2 = pieces_.find(piece);
            if (it2 != pieces_.end()) {
                return it2->second;
            }
            return unk_id_;
        }
       
        struct Match {
            int e;
            int n;
            float_t w;
            Match(int e, int n, float_t w)
                : e(e), n(n), w(w) {}
        };
        
        std::vector<Match> GetMatches(const std::string& sentence) const {
            std::vector<Match> matches;
            int num = sentence.length();
            int pos = 0;
            while (pos < num) {
                int n = SizeUTF8(sentence[pos]);
                if (pos + n - 1 < num) {
                    matches.emplace_back(pos + n - 1, n, -10.0);
                }
                const size_t kMaxNumResults = 16;
                //Darts::DoubleArray::result_pair_type results[kMaxNumResults];
                new_darts::DoubleArray<int>::ResultPair results[kMaxNumResults];
                size_t num_results = trie_.commonPrefixSearch(
                    sentence.c_str() + pos,
                    results,
                    kMaxNumResults,
                    num - pos
                );
                for (size_t i = 0; i < num_results; ++i) {
                    if (pos + results[i].length - 1 < num) {
                        matches.emplace_back(
                            pos + results[i].length - 1,
                            results[i].length,
                            value_map_.at(results[i].value)
                        );
                    }
                }
                pos += n;
            }
            return matches;
        }

        void InitFromModel() {
            if (!model_) {
                LOG(ERROR) << "Model is not initialized.";
                return;
            }

            pieces_.clear();
            reserve_.clear();
            value_map_.clear();

            const auto& counter_spec = model_->GetCounterSpec();
            unk_id_ = counter_spec.unk_id();

            std::unordered_map<std::string, float_t> dict;

            for (size_t i = 0; i < model_->PiecesSize(); ++i) {
                const auto& p = model_->GetPieces(i);
                const std::string& piece = p.GetPiece();
                pieces_[piece] = i;
                
                if (p.GetType() == Model::Piece::NORMAL) {
                    dict[piece] = p.GetScore();
                }
                
                if (p.GetType() != Model::Piece::NORMAL) {
                    reserve_[piece] = i;
                }
            }

            InitFromDict(dict);

            LOG(INFO) << "Initialized tokenizer from model with " << dict.size() << " pieces";
        }

        void InitFromDict(const std::unordered_map<std::string, float_t>& dict) {
            float_t total = 0.0;
            for (const auto& p : dict) {
                total += p.second;
            }
            float_t log_total = std::log(total);
        
            std::vector<std::pair<std::string, float_t>> dict_(dict.begin(), dict.end());
            std::sort(dict_.begin(), dict_.end());
        
            std::vector<const char*> strs;
            std::vector<int> values;
            int next_value = 1;
            for (const auto& p : dict_) {
                auto s = p.first.c_str();
                if (strlen(s) == 0) {
                    continue;
                }
                strs.push_back(p.first.c_str());
                
                value_map_[next_value] = std::log(p.second) - log_total;
                
                values.push_back(next_value);
                next_value++;
            }
        
            trie_.build(strs.size(), strs.data(), nullptr, values.data());
            LOG(INFO) << "Initialized tokenizer from dictionary with " << dict.size() << " entries";
        }
        
        const Model* model_;
        const std::unique_ptr<Normalizer> normalizer_;

        StrToInt pieces_;    
        StrToInt reserve_;  
        int unk_id_ = 0;    
        
        new_darts::DoubleArray<int> trie_;
        std::unordered_map<int, float_t> value_map_;
};

class BytePieceCounter {
public:
    BytePieceCounter(const CounterSpec& counter_spec,
                   const NormalizerSpec& normalizer_spec)
        : counter_spec_(counter_spec),
          normalizer_spec_(normalizer_spec),
          max_piece_count_(6),
          max_piece_size_(18),
          min_count_(2) {
        InitMetaPieces();
        InitT();
        N_.resize(max_piece_count_ + 1);
    }
    
    virtual ~BytePieceCounter() {}
    
    bool Count() {
        if (!LoadAndSplitSentences()) {
            LOG(ERROR) << "Failed to load sentences.";
            return false;
        }

        LOG(INFO) << "Starting BytePiece training with " << sentences_.size() << " unique tokens";

        CountRaw(sentences_);
        PruneRaw();

        auto pieces_count = CountPieces(sentences_);
        auto pruned_pieces = PrunePieces(pieces_count);

        std::vector<std::pair<std::string, int>> sorted_pieces(pruned_pieces.begin(), pruned_pieces.end());
        std::sort(sorted_pieces.begin(), sorted_pieces.end(), 
                  [](const auto& a, const auto& b) {
                      return a.second > b.second;  // 降序排序（大的在前）
                  });

        pieces_.clear();

        for (const auto& [piece, count] : sorted_pieces) {
            pieces_.emplace_back(piece, count);
            if (pieces_.size() % 100 == 0) {
                LOG(INFO) << "Added piece: " << piece << " count: " << count
                          << " total: " << pieces_.size();
            }
        }
        
        LOG(INFO) << "BytePiece training completed with " << pieces_.size() << " pieces";
        
        return true;
    }
    
    bool Save() const {
        std::string filename = counter_spec_.model_prefix()+".model";
        LOG(INFO) << "Saving model: " << filename;
        Model model;
        if (!Serialize(&model)) return false;
        auto output = NewWritableFile(filename);
        output->Write(model.AsStr());
        return true;
    }
    
    bool Serialize(Model* model) const {
        model->Clear();

        // Total pieces = meta_pieces + trained pieces
        size_t total = meta_pieces_.size() + pieces_.size();
        size_t fid = 0;
        for (size_t id = 0; id < total; ++id) {
            const auto it = meta_pieces_.find(id);
            if (it != meta_pieces_.end()) {
                auto *p = model->InsertPieces();
                p->SetPiece(it->second.first);
                p->SetType(it->second.second);
                p->SetScore(0.0);
            } else if (fid < pieces_.size()) {
                const auto &w = pieces_[fid++];
                auto *p = model->InsertPieces();
                p->SetPiece(w.first);
                p->SetScore(w.second);
            }
        }

        // Update vocab_size in counter_spec to match actual count
        CounterSpec spec = counter_spec_;
        spec.set_vocab_size(total);
        model->SetCounterSpec(spec);
        model->SetNormalizerSpec(normalizer_spec_);

        return true;
    }
    
private:
    using Sentence = std::pair<std::string, int64_t>;
    using Sentences = std::vector<Sentence>;
    using Str2Int = std::unordered_map<std::string, int>;
    using Str2Float = std::unordered_map<std::string, float_t>;

    size_t GetWorkerCount(size_t work_items) const {
        if (work_items <= 1) {
            return 1;
        }
        const unsigned int hw = std::thread::hardware_concurrency();
        const size_t max_workers = hw == 0 ? 4 : static_cast<size_t>(hw);
        return std::max<size_t>(1, std::min(work_items, max_workers));
    }

    template <typename Fn>
    void ParallelFor(size_t total, Fn&& fn) const {
        const size_t workers = GetWorkerCount(total);
        if (workers <= 1) {
            fn(0, total, 0);
            return;
        }

        std::vector<std::thread> threads;
        threads.reserve(workers);
        const size_t chunk = (total + workers - 1) / workers;
        for (size_t worker = 0; worker < workers; ++worker) {
            const size_t begin = worker * chunk;
            const size_t end = std::min(total, begin + chunk);
            if (begin >= end) {
                break;
            }
            threads.emplace_back([&, begin, end, worker]() {
                fn(begin, end, worker);
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }

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
    
    bool InitMetaPieces() {
        if (counter_spec_.unk_id() >= 0) {
            meta_pieces_[counter_spec_.unk_id()] = std::make_pair(
                counter_spec_.unk_piece(), Model::Piece::UNKNOWN);
        }
        
        if (counter_spec_.bos_id() >= 0) {
            meta_pieces_[counter_spec_.bos_id()] = std::make_pair(
                counter_spec_.bos_piece(), Model::Piece::CONTROL);
        }
        
        if (counter_spec_.eos_id() >= 0) {
            meta_pieces_[counter_spec_.eos_id()] = std::make_pair(
                counter_spec_.eos_piece(), Model::Piece::CONTROL);
        }
        
        if (counter_spec_.pad_id() >= 0) {
            meta_pieces_[counter_spec_.pad_id()] = std::make_pair(
                counter_spec_.pad_piece(), Model::Piece::CONTROL);
        }

        if (meta_pieces_.size() + 256 > counter_spec_.vocab_size()) {
            LOG(ERROR) << "Vocab size is too small for byte pieces. "
                       << "Need at least " << meta_pieces_.size() + 256 << " slots." << std::endl;
            return false;
        }

        int byte_id = meta_pieces_.size();
        for (int i = 0; i < 256; ++i) {
            std::string byte_piece = ustr::ByteToPiece(i);
            meta_pieces_[byte_id++] = std::make_pair(byte_piece, Model::Piece::BYTE);
        }

        return true;
    }
    
    bool LoadAndSplitSentences() {
        LOG(INFO) << "Loading and splitting sentences ...";

        const Normalizer normalizer(normalizer_spec_);
        const std::string_view space = normalizer_spec_.GetSpace();

        auto iter_ = std::make_unique<MultiFileSentenceIterator>(
            std::vector<std::string>(counter_spec_.input().begin(),
                                  counter_spec_.input().end()));
        SentenceIterator* sentence_iterator_ = iter_.get();

        std::unordered_map<std::string, int64_t> tokens;
        size_t line_count = 0;

        for (; !sentence_iterator_->done(); sentence_iterator_->Next()) {
            std::string sentence = sentence_iterator_->value();
            if (sentence.empty()) continue;

            std::string normalized = normalizer.Normalize(sentence);
            for (const auto& w : ustr::SplitText(normalized, space)) {
                tokens[std::string(w)] += 1;
            }

            if (++line_count % 1000000 == 0) {
                LOG(INFO) << "Processed " << line_count << " lines, "
                          << tokens.size() << " unique tokens";
            }
        }

        sentences_.clear();
        sentences_.reserve(tokens.size());
        for (auto& [word, count] : tokens) {
            sentences_.emplace_back(std::move(word), count);
        }

        LOG(INFO) << "Done! " << line_count << " lines, "
                  << sentences_.size() << " unique tokens";
        return true;
    }
    
    void CountRaw(const Sentences& sentences) {
        LOG(INFO) << "Counting raw substrings...";

        N_.clear();
        N_.resize(max_piece_count_ + 1);
        N_[0][""] = 0;

        for (const auto& [text, freq] : sentences) {
            for (size_t i = 0; i < text.length(); ++i) {
                N_[0][""] += freq;

                std::string piece;
                piece.reserve(std::min(max_piece_count_, text.length() - i));

                const size_t max_len = std::min(max_piece_count_, text.length() - i);
                for (size_t j = 1; j <= max_len; ++j) {
                    piece.push_back(text[i + j - 1]);
                    N_[j][piece] += freq;
                }
            }
        }

        size_t cnt = 0;
        for (size_t i = 0; i < N_.size(); ++i) {
            cnt += N_[i].size();
        }

        LOG(INFO) << "Done counting raw substrings " << cnt;
    }
    
    void PruneRaw() {
        LOG(INFO) << "Pruning raw counts...";
        
        // 确保所有字节值都在N_[1]中，与Python版本一致
        for (int i = 0; i < 256; ++i) {
            std::string byte_str(1, static_cast<char>(i));
            if (N_[1].find(byte_str) == N_[1].end()) {
                N_[1][byte_str] = 1;
                N_[0][""] += 1;  // 增加空字符串的计数
            }
        }
        
        // 从最长的n-gram开始向下迭代
        for (int i = N_.size() - 1; i >= 0; --i) {
            std::unordered_map<std::string, float_t> pruned;
            
            // 只保留长度正好为i且频率超过阈值的n-gram，并取对数
            for (const auto& [k, v] : N_[i]) {
                if (k.length() == i && v >= (i > 1 ? min_count_ : 0)) {
                    pruned[k] = std::log(v);
                }
            }
            
            // 对于长度i+1的n-gram，减去其前i个字符的对数频率
            if (i < N_.size() - 1) {
                std::unordered_map<std::string, float_t> next_pruned;
                
                for (const auto& [k, v] : N_[i + 1]) {
                    std::string prefix = k.substr(0, i);
                        
                    auto it = pruned[prefix];
                    // 减去前缀的对数频率
                    next_pruned[k] = v - it;
                }
                
                N_[i + 1] = std::move(next_pruned);
            }
            
            // 用修剪后的结果替换原始统计
            N_[i] = std::move(pruned);
        }
        
        int cnt = 0;
        for (size_t i = 0; i < N_.size(); ++i) {
            cnt += N_[i].size();
        }
        LOG(INFO) << "Done pruning raw counts " << cnt;
    }

    std::vector<std::string> Tokenize(const std::string& text) const {
        const int num = text.length();
        if (num == 0) return {};
        
        // 创建并初始化节点评分矩阵
        std::vector<std::vector<float_t>> nodes(num, std::vector<float_t>(max_piece_count_, -INF));
        
        // 预处理UTF-8字符结构
        std::vector<int> utf8_position(num, 0); // 0=首字节或ASCII, 1=第二字节, 2=第三字节, 3=第四字节
        
        int i = 0;
        while (i < num) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            int char_length = 1; // 默认为1个字节（ASCII字符）
            
            if (c < 128) {
                // ASCII字符，长度为1
                char_length = 1;
            } else if (c >= 192 && c < 224) {
                // 2字节UTF-8字符
                char_length = 2;
            } else if (c >= 224 && c < 240) {
                // 3字节UTF-8字符
                char_length = 3;
            } else if (c >= 240 && c < 248) {
                // 4字节UTF-8字符
                char_length = 4;
            } else if (c >= 128 && c < 192) {
                // 这是一个UTF-8后续字节，但没有前导首字节
                // 我们将其视为单个字节继续处理
                char_length = 1;
            }
            
            // 处理这个UTF-8字符的所有字节
            for (int j = 0; j < char_length && i + j < num; ++j) {
                utf8_position[i + j] = j;
            }
            
            // 移动到下一个字符
            i += char_length;
        }
        
        // 填充节点评分矩阵
        for (int j = 0; j < max_piece_count_; ++j) {
            for (int i = j; i < num; ++i) {
                // 对于j=0，确保位置i是UTF-8首字节
                if (j == 0 && utf8_position[i] > 0) {
                    continue;  // 跳过UTF-8后续字节
                }
                
                std::string piece = text.substr(i - j, j + 1);
                if (j + 1 < N_.size()) {
                    auto it = N_[j + 1].find(piece);
                    if (it != N_[j + 1].end()) {
                        nodes[i][j] = it->second;
                    }
                }
            }
        }
        
        // 创建路由矩阵
        std::vector<std::vector<int>> routes(num - 1, std::vector<int>(max_piece_count_, 0));
        
        // 执行动态规划算法，寻找最优路径
        for (i = 1; i < num; ++i) {
            for (int curr_j = 0; curr_j < max_piece_count_; ++curr_j) {
                // 如果当前状态对于此位置无效，跳过
                if (curr_j < utf8_position[i]) continue;
                
                int best_prev_j = -1; // 初始化为-1，表示尚未找到有效的前一个状态
                float_t best_score = -INF;
                
                for (int prev_j = 0; prev_j < max_piece_count_; ++prev_j) {
                    // 如果前一个状态对于前一个位置无效，跳过
                    if (prev_j < utf8_position[i-1]) continue;
                    
                    // 检查转移是否允许
                    if (T_[prev_j][curr_j] == -INF) continue;
                    
                   bool skip_ngram_check = (prev_j == max_piece_count_ - 1 && curr_j == max_piece_count_ - 1);
            
                   if (!skip_ngram_check) {
                       // 检查n-gram的起始位置
                       int ngram_start = i - curr_j;
                       if (ngram_start > 0 && utf8_position[ngram_start] > 0) {
                           // n-gram不能从UTF-8字符的中间开始
                           continue;
                       }
                   }
                   
                    
                    float_t score = nodes[i - 1][prev_j] + T_[prev_j][curr_j] + nodes[i][curr_j];
                    
                    if (score > best_score) {
                        best_score = score;
                        best_prev_j = prev_j;
                    }
                }
                
                if (best_prev_j != -1) {
                    routes[i - 1][curr_j] = best_prev_j;
                    nodes[i][curr_j] = best_score;
                } else {
                    // 没有找到有效的前一个状态，跳过这个状态
                    nodes[i][curr_j] = -INF;
                }
            }
        }
        
        // 找到最后一个位置的最佳状态
        int best_last_state = 0;
        float_t best_score = -INF;
        for (int j = 0; j < max_piece_count_; ++j) {
            if (j >= utf8_position[num - 1] && nodes[num - 1][j] > best_score) {
                best_score = nodes[num - 1][j];
                best_last_state = j;
            }
        }
        
        // 回溯构建最优路径
        std::vector<int> opt_route(num);
        int curr_pos = num - 1;
        int curr_state = best_last_state;
        
        while (curr_pos >= 0) {
            opt_route[curr_pos] = curr_state;
            
            if (curr_pos > 0) {
                curr_state = routes[curr_pos-1][curr_state];
                curr_pos--;
            } else {
                // Error
                break;
            }
        }
        
        // 找到所有状态为0的位置作为切分点
        std::vector<int> split_points;
        split_points.push_back(0); // 总是从文本开始处切分
        
        for (i = 1; i < opt_route.size(); ++i) {
            if (opt_route[i] == 0 && utf8_position[i] == 0) {
                // 确保只在UTF-8首字节处切分
                split_points.push_back(i);
            }
        }
        
        split_points.push_back(num); // 添加文本结束位置
        
        // 根据切分点提取标记
        std::vector<std::string> tokens;
        for (size_t i = 0; i < split_points.size() - 1; ++i) {
            tokens.push_back(text.substr(split_points[i], split_points[i + 1] - split_points[i]));
        }
        
        // 输出opt_route和分词结果（用于调试）
        //for (int i = 0; i < opt_route.size(); ++i) {
        //    std::cout << opt_route[i] << " ";
        //}
        //std::cout << "\n";
        
        //std::cout << text << "=>";
        //for (auto t : tokens) {
        //    std::cout << t << " ";
        //}
        //std::cout << "\n";
        
        return tokens;
    } 

   Str2Int CountPieces(const Sentences& sentences) {
        LOG(INFO) << "Counting pieces...";
        const size_t batch_size = 2048;
        Str2Int pieces = ParallelBatchCount<Str2Int>(
            sentences.size(), batch_size,
            [&](size_t begin, size_t end) {
                Str2Int batch_counts;
                for (size_t idx = begin; idx < end; ++idx) {
                    const auto& [text, freq] = sentences[idx];
                    auto tokens = Tokenize(text);
                    for (const auto& piece : tokens) {
                        if (piece.length() == 0) continue;
                        batch_counts[piece] += freq;
                    }
                }
                return batch_counts;
            });

        LOG(INFO) << "Found " << pieces.size() << " unique pieces";
        return pieces;
    }


    Str2Int SplitPieces(const Str2Int& keep, const Str2Int& drop) {
        std::unordered_map<std::string, float_t> dict;
        for (const auto& p : keep) {
            dict.emplace(p.first, static_cast<float_t>(p.second));
        }
        
        BytePieceTokenizer tokenizer(dict);

        std::vector<std::pair<std::string, int>> drop_entries(drop.begin(), drop.end());
        const size_t batch_size = 1024;
        Str2Int counter = ParallelBatchCount<Str2Int>(
            drop_entries.size(), batch_size,
            [&](size_t begin, size_t end) {
                Str2Int batch_counts;
                for (size_t idx = begin; idx < end; ++idx) {
                    const auto& [str, cnt] = drop_entries[idx];
                    for (const auto& token : tokenizer.Tokenize(str)) {
                        batch_counts[token] += cnt;
                    }
                }
                return batch_counts;
            });
        return counter;
    }

   
    Str2Int PrunePieces(Str2Int& pieces) {
        LOG(INFO) << "Pruning pieces...";
       
        Str2Int keep, drop;
        
        for (const auto& [str, cnt] : pieces) {
            if (str.length() == 1 ||
                (str.length() <= max_piece_size_ && cnt >= min_count_)) {
                keep[str] = cnt;
            } else {
                drop[str] = cnt;
            }
        }
        
        auto new_counter = SplitPieces(keep, drop);
        for (const auto& [str, cnt] : new_counter) {
            keep[str] += cnt;
        }
        
        while (true) {
            size_t n = keep.size();
            auto entire_keep_as_drop = keep;
            
            keep = SplitPieces(keep, entire_keep_as_drop);
            
            if (keep.size() == n) {
                break;
            }
        }
        
        int vocab_size = counter_spec_.vocab_size();
        if (keep.size() <= vocab_size - meta_pieces_.size()) {
            LOG(INFO) << "Final pieces count: " << keep.size();
            return keep;
        }
        
        std::vector<std::pair<std::string, int>> pieces_vec(
            keep.begin(), keep.end());
        
        std::sort(pieces_vec.begin(), pieces_vec.end(),
               [](const auto& a, const auto& b) {
                   bool a_num = a.first.length() > 1;
                   bool b_num = b.first.length() > 1;
                   if (a_num != b_num) return a_num < b_num;
                   if (a.second != b.second) return a.second > b.second;
                   if (a.first.length() != b.first.length())
                       return a.first.length() > b.first.length();
                   return a.first < b.first;
               });
        
        Str2Int new_pieces;
        size_t limit = vocab_size - meta_pieces_.size();
        for (size_t i = 0; i < limit && i < pieces_vec.size(); ++i) {
            new_pieces[pieces_vec[i].first] = pieces_vec[i].second;
        }
        
        LOG(INFO) << "Final pieces count after pruning: " << new_pieces.size();
        return new_pieces;
    }
    
    void InitTX() {
        int num_ = max_piece_count_;
        T_.resize(num_, std::vector<float_t>(num_, -INF));
        for (int i = 0; i < num_; ++i) {
            T_[i][0] = 0;
            T_[i][std::min(i + 1, num_ - 1)] = 0;
        }
    }
    void InitT() {
        int num_ = max_piece_count_;
        T_.resize(num_, std::vector<float_t>(num_, -INF));
        
        // 修改：只允许特定的状态转移
        for (int i = 0; i < num_; ++i) {
            // 从任何状态可以转移到状态0
            T_[i][0] = 0;
            
            // 从状态i只能转移到状态i+1
            if (i + 1 < num_) {
                T_[i][i + 1] = 0;
            }
            
            // 最高状态可以自环
            if (i == num_ - 1) {
                T_[i][i] = 0;
            }
        }
    }
    
    std::map<int, std::pair<std::string, Model::Piece::Type>> meta_pieces_;
    std::vector<std::pair<std::string, float>> byte_pieces_;
    std::vector<std::pair<std::string, float>> pieces_;
    Sentences sentences_;
    CounterSpec counter_spec_;
    NormalizerSpec normalizer_spec_;
    
    size_t max_piece_count_;
    size_t max_piece_size_; // TODO
    size_t min_count_;
    std::vector<std::vector<float_t>> T_;
    std::vector<std::unordered_map<std::string, float_t>> N_;
    const float_t INF = std::numeric_limits<float_t>::infinity();
};

} // namespace piece
