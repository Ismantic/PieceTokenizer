#pragma once 

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <limits>
#include <fstream>

#include <stdint.h>

#include "common.h"
#include "piece_spec.h"
#include "ustr.h"
#include "misc.h"
#include "sentence.h"
#include "new_normalizer.h"


namespace piece {
constexpr size_t INITAL_VOCAB_SIZE = 256;
constexpr size_t MAX_TEXT_SIZE = 1024;

struct IntPair {
    int first;
    int second;

    bool operator==(const IntPair& other) const {
        return first == other.first && second == other.second;
    }
};

struct Merge {
    IntPair pair;
    int idx;
};

struct NaiveModel {
    std::vector<Merge> merges;
    std::vector<std::vector<uint8_t>> vocab;
};

namespace naive {
    inline size_t GetPairIndexInCounts(const std::vector<size_t>& pair_counts,
                                      size_t pair_counts_size,
                                      const IntPair& pair) {
        for (size_t i = 0; i < pair_counts_size; ++i) {
            if (pair_counts[i * 3] == static_cast<size_t>(pair.first) && 
                pair_counts[i * 3 + 1] == static_cast<size_t>(pair.second)) {
                return i;
            }
        }
        return pair_counts_size;
    }

    inline void TokenCounts(const std::vector<int>& ids, 
                           std::vector<size_t>& pair_counts,
                           size_t& pair_counts_size) {
        pair_counts_size = 0;
        for (size_t i = 0; i < ids.size()-1; ++i) {
            IntPair pair{ids[i], ids[i+1]};
            size_t index = GetPairIndexInCounts(pair_counts, pair_counts_size, pair);

            if (index == pair_counts_size) {
                pair_counts[pair_counts_size*3] = pair.first;
                pair_counts[pair_counts_size*3 + 1] = pair.second;
                pair_counts[pair_counts_size*3 + 2] = 1;
                pair_counts_size++;
            } else {
                pair_counts[index*3 + 2]++;
            }
        }
    }

    inline void MergePair(std::vector<int>& ids, const IntPair& pair, int idx) {
        std::vector<int> new_ids;
        new_ids.reserve(ids.size());

        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] == pair.first && i < ids.size() -1 && 
                ids[i+1] == pair.second) {
                new_ids.push_back(idx);
                ++i;
            } else {
                new_ids.push_back(ids[i]);
            }
        }

        ids = std::move(new_ids);
    }

    inline size_t GetPairIndex(const std::vector<Merge>& merges, const IntPair& pair) {
        auto it = std::find_if(merges.begin(), merges.end(),
            [&pair](const Merge& m) {return m.pair == pair;});
        
        return it != merges.end() ?
            std::distance(merges.begin(), it) :
            merges.size();
    }
} // namespace naive


class NaiveCounter {
    public:
        NaiveCounter(const CounterSpec& counter_spec,
                     const NormalizerSpec& normalizer_spec)
            : counter_spec_(counter_spec),
              normalizer_spec_(normalizer_spec) {
            InitMetaPieces();
        }
        
        virtual ~NaiveCounter() {}

        bool Count() {
            // 加载句子
            if (!LoadSentences()) {
                LOG(ERROR) << "Failed to load sentences.";
                return false;
            }
            
            // 如果需要，按空格分割句子
            SplitSentencesByWhitespace();
            
            // 合并所有文本进行处理
            std::string combined_text;
            for (const auto& sentence : sentences_) {
                combined_text += sentence.first;
                // 可以在句子之间添加分隔符，如空格
                combined_text += " ";
            }
            
            // 初始化词汇表
            InitializeVocab();
            
            // 计算需要进行的合并次数（词汇表大小减去初始词汇表大小和特殊标记数量）
            int num_merges = counter_spec_.vocab_size() - INITAL_VOCAB_SIZE - meta_pieces_.size();
            
            if (num_merges <= 0) {
                LOG(ERROR) << "Vocabulary size too small for BPE training.";
                return false;
            }
            
            LOG(INFO) << "Starting BPE training with " << num_merges << " merges";
            
            // 将文本转换为id序列
            std::vector<int> ids(combined_text.begin(), combined_text.end());
            
            merges_.reserve(num_merges);
            
            // 追踪有效的UTF-8合并数量
            int valid_utf8_merges = 0;
            int max_iterations = num_merges * 3;  // 允许更多迭代以找到足够的有效UTF-8
            
            // 开始BPE合并过程
            for (int i = 0; i < max_iterations && ids.size() >= 2 && valid_utf8_merges < num_merges; ++i) {
                std::vector<size_t> pair_counts(MAX_TEXT_SIZE*3);
                size_t pair_counts_size;
                naive::TokenCounts(ids, pair_counts, pair_counts_size);
                
                size_t max_count = 0;
                IntPair next{0, 0};
                
                for (size_t j = 0; j < pair_counts_size*3; j += 3) {
                    IntPair pair{static_cast<int>(pair_counts[j]),
                               static_cast<int>(pair_counts[j+1])};
                    size_t count = pair_counts[j+2];
                    if (count > max_count) {
                        max_count = count;
                        next = pair;
                    }
                }
                
                if (max_count <= 1) break;
                
                // 创建合并后的piece
                std::string piece;
                int first_id = next.first;
                int second_id = next.second;
                
                // 获取第一个token的字符串表示
                std::string first_piece;
                if (first_id < INITAL_VOCAB_SIZE) {
                    first_piece.push_back(static_cast<char>(first_id));
                } else {
                    for (const auto& merge : merges_) {
                        if (merge.idx == first_id) {
                            std::string temp;
                            DecodeToken(merge.pair.first, temp);
                            DecodeToken(merge.pair.second, temp);
                            first_piece = temp;
                            break;
                        }
                    }
                }
                
                // 获取第二个token的字符串表示
                std::string second_piece;
                if (second_id < INITAL_VOCAB_SIZE) {
                    second_piece.push_back(static_cast<char>(second_id));
                } else {
                    for (const auto& merge : merges_) {
                        if (merge.idx == second_id) {
                            std::string temp;
                            DecodeToken(merge.pair.first, temp);
                            DecodeToken(merge.pair.second, temp);
                            second_piece = temp;
                            break;
                        }
                    }
                }
                
                // 合并两个piece
                piece = first_piece + second_piece;
                
                // 检查合并结果是否是有效的UTF-8
                bool is_valid_utf8 = true;
                {
                    const uint8_t* data = reinterpret_cast<const uint8_t*>(piece.data());
                    size_t len = piece.size();
                    size_t pos = 0;
                    
                    while (pos < len) {
                        if (data[pos] < 0x80) {
                            // 单字节ASCII字符
                            pos += 1;
                        } else if ((data[pos] & 0xE0) == 0xC0) {
                            // 2字节序列: 110xxxxx 10xxxxxx
                            if (pos + 1 >= len || (data[pos+1] & 0xC0) != 0x80) {
                                is_valid_utf8 = false;
                                break;
                            }
                            pos += 2;
                        } else if ((data[pos] & 0xF0) == 0xE0) {
                            // 3字节序列: 1110xxxx 10xxxxxx 10xxxxxx
                            if (pos + 2 >= len || 
                                (data[pos+1] & 0xC0) != 0x80 || 
                                (data[pos+2] & 0xC0) != 0x80) {
                                is_valid_utf8 = false;
                                break;
                            }
                            pos += 3;
                        } else if ((data[pos] & 0xF8) == 0xF0) {
                            // 4字节序列: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
                            if (pos + 3 >= len || 
                                (data[pos+1] & 0xC0) != 0x80 || 
                                (data[pos+2] & 0xC0) != 0x80 ||
                                (data[pos+3] & 0xC0) != 0x80) {
                                is_valid_utf8 = false;
                                break;
                            }
                            pos += 4;
                        } else {
                            // 无效的UTF-8起始字节
                            is_valid_utf8 = false;
                            break;
                        }
                    }
                }
                
                // 如果不是有效UTF-8，跳过这次合并
                if (!is_valid_utf8) {
                    // 应用合并规则（但不计入有效合并数量）
                    int idx = INITAL_VOCAB_SIZE + merges_.size();
                    naive::MergePair(ids, next, idx);
                    merges_.push_back({next, idx});
                    vocab_.push_back({static_cast<uint8_t>(next.first),
                                     static_cast<uint8_t>(next.second)});
                    continue;
                }
                
                // 如果是有效UTF-8，计数并加入词表
                valid_utf8_merges++;
                int idx = INITAL_VOCAB_SIZE + merges_.size();
                naive::MergePair(ids, next, idx);
                
                merges_.push_back({next, idx});
                
                vocab_.push_back({static_cast<uint8_t>(next.first),
                                 static_cast<uint8_t>(next.second)});
                
                // 将合并结果添加到pieces_
                pieces_.emplace_back(piece, -static_cast<float>(pieces_.size()));
                
                // 只打印有效UTF-8合并结果
                if (i % 1 == 0) {
                    LOG(INFO) << "Merge " << valid_utf8_merges << "/" << num_merges
                             << " with count " << max_count
                             << " piece: " << piece
                             << " (valid UTF-8)";
                }
            }
            
            LOG(INFO) << "BPE training completed with " << valid_utf8_merges << " valid UTF-8 merges";
            
            return true;
        }        

        bool CountX() {
            // 加载句子
            if (!LoadSentences()) {
                LOG(ERROR) << "Failed to load sentences.";
                return false;
            }
            
            // 如果需要，按空格分割句子
            SplitSentencesByWhitespace();
            
            // 合并所有文本进行处理
            std::string combined_text;
            for (const auto& sentence : sentences_) {
                combined_text += sentence.first;
                // 可以在句子之间添加分隔符，如空格
                combined_text += " ";
            }
            
            // 初始化词汇表
            InitializeVocab();
            
            // 计算需要进行的合并次数（词汇表大小减去初始词汇表大小和特殊标记数量）
            int num_merges = counter_spec_.vocab_size() - INITAL_VOCAB_SIZE - meta_pieces_.size();
            
            if (num_merges <= 0) {
                LOG(ERROR) << "Vocabulary size too small for BPE training.";
                return false;
            }
            
            LOG(INFO) << "Starting BPE training with " << num_merges << " merges";
            
            // 将文本转换为id序列
            std::vector<int> ids(combined_text.begin(), combined_text.end());
            
            merges_.reserve(num_merges);
            
            // 开始BPE合并过程
            for (int i = 0; i < num_merges && ids.size() >= 2; ++i) {
                std::vector<size_t> pair_counts(MAX_TEXT_SIZE*3);
                size_t pair_counts_size;
                naive::TokenCounts(ids, pair_counts, pair_counts_size);
                
                size_t max_count = 0;
                IntPair next{0, 0};
                
                for (size_t j = 0; j < pair_counts_size*3; j += 3) {
                    IntPair pair{static_cast<int>(pair_counts[j]),
                                static_cast<int>(pair_counts[j+1])};
                    size_t count = pair_counts[j+2];
                    if (count > max_count) {
                        max_count = count;
                        next = pair;
                    }
                }
                
                if (max_count <= 1) break;
                
                int idx = INITAL_VOCAB_SIZE + i;
                naive::MergePair(ids, next, idx);
                
                merges_.push_back({next, idx});
                
                vocab_.push_back({static_cast<uint8_t>(next.first),
                                static_cast<uint8_t>(next.second)});
                
                // 将合并结果添加到pieces_
                std::string piece;
                DecodeToken(idx, piece);
                pieces_.emplace_back(piece, -static_cast<float>(pieces_.size()));
                
                if (i % 1 == 0) {
                    LOG(INFO) << "Merge " << i + 1 << "/" << num_merges
                            << " with count " << max_count
                            << " piece: " << piece;
                }
            }
            
            return true;
        }
        
        bool Save() const {
            SaveModel(counter_spec_.model_prefix() + ".model");
            SaveVocab(counter_spec_.model_prefix() + ".vocab");
            return true;
        }
        
        bool Serialize(Model* model_proto) const {
            model_proto->Clear();
            
            // 首先添加特殊标记
            for (int id = 0; id < counter_spec_.vocab_size(); ++id) {
                const auto it = meta_pieces_.find(id);
                if (it != meta_pieces_.end()) {
                    auto* p = model_proto->InsertPieces();
                    p->SetPiece(it->second.first);
                    p->SetType(it->second.second);
                    p->SetScore(0.0);
                } else if (id < INITAL_VOCAB_SIZE) {
                    // 添加基础字节字符
                    std::string byte_piece(1, static_cast<char>(id));
                    auto* p = model_proto->InsertPieces();
                    p->SetPiece(byte_piece);
                    p->SetType(Model::Piece::BYTE);
                    p->SetScore(0.0);
                } else if (id - INITAL_VOCAB_SIZE < static_cast<int>(pieces_.size())) {
                    // 添加BPE学习到的词汇
                    const auto& piece = pieces_[id - INITAL_VOCAB_SIZE];
                    auto* p = model_proto->InsertPieces();
                    p->SetPiece(piece.first);
                    p->SetScore(piece.second);
                }
            }
            
            // 设置CounterSpec和NormalizerSpec
            model_proto->SetCounterSpec(counter_spec_);
            model_proto->SetNormalizerSpec(normalizer_spec_);
            
            return true;
        }
        
    private:
        void InitializeVocab() {
            vocab_.resize(INITAL_VOCAB_SIZE);
            for (int i = 0; i < INITAL_VOCAB_SIZE; ++i) {
                vocab_[i] = std::vector<uint8_t>{static_cast<unsigned char>(i)};
            }
            merges_.clear();
            pieces_.clear();
        }
        
        void DecodeToken(int id, std::string& text) const {
            if (id < INITAL_VOCAB_SIZE) {
                text.push_back(static_cast<char>(id));
                return;
            }
            
            const auto& merge = merges_[id - INITAL_VOCAB_SIZE];
            DecodeToken(merge.pair.first, text);
            DecodeToken(merge.pair.second, text);
        }
        
        bool InitMetaPieces() {
            // 添加特殊标记
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
            
            return true;
        }
        
        bool LoadSentences() {
            LOG(INFO) << "Loading sentences ...";
            
            auto iter_ = std::make_unique<MultiFileSentenceIterator>(
                std::vector<std::string>(counter_spec_.input().begin(),
                                        counter_spec_.input().end()));
            SentenceIterator* sentence_iterator_ = iter_.get();
            
            for (; !sentence_iterator_->done(); sentence_iterator_->Next()) {
                std::string sentence = sentence_iterator_->value();
                if (sentence.empty()) {
                    continue;
                }
                sentences_.emplace_back(std::make_pair(sentence, 1));
            }
            
            LOG(INFO) << "Normalizing sentences ...";
            const Normalizer normalizer(normalizer_spec_);
            for (size_t i = 0; i < sentences_.size(); ++i) {
                auto* s = &sentences_[i].first;
                *s = normalizer.Normalize(*s);
            }
            
            LOG(INFO) << "Done! preprocessed " << sentences_.size() << " sentences.";
            return true;
        }
        
        void SplitSentencesByWhitespace() {
            LOG(INFO) << "Tokenizing input sentences with whitespace: "
                      << sentences_.size();
            const std::string_view space = normalizer_spec_.GetSpace();
            std::unordered_map<std::string, int64_t> tokens;
            for (const auto& s : sentences_) {
                for (const auto& w : ustr::SplitText(s.first, space)) {
                    tokens[std::string(w)] += s.second;
                }
            }
            sentences_ = misc::Sorted(tokens);
            LOG(INFO) << "Done! " << sentences_.size();
        }
        
        bool SaveModel(std::string_view filename) const {
            LOG(INFO) << "Saving model: " << filename;
            Model model_proto;
            
            if (!Serialize(&model_proto)) return false;
            
            // 这里可以实现模型序列化的具体逻辑
            // 但这需要根据您的实际需求来定义
            // TODO: 实现模型保存逻辑
            
            return true;
        }
        
        bool SaveVocab(std::string_view filename) const {
            LOG(INFO) << "Saving vocabs: " << filename;
            Model model_proto;
            
            if (!Serialize(&model_proto)) return false;
            
            auto output = NewWritableFile(filename);
            
            for (const auto& piece : model_proto.GetPieces()) {
                std::ostringstream os;
                os << piece.GetPiece() << "\t" << piece.GetScore();
                output->WriteLine(os.str());
            }
            
            return true;
        }
        
        using Sentence = std::pair<std::string, int64_t>;
        using Sentences = std::vector<Sentence>;
        
        std::map<int, std::pair<std::string, Model::Piece::Type>> meta_pieces_;
        std::vector<std::pair<std::string, float>> pieces_;
        Sentences sentences_;
        CounterSpec counter_spec_;
        NormalizerSpec normalizer_spec_;
        
        std::vector<Merge> merges_;
        std::vector<std::vector<uint8_t>> vocab_;
        
        static constexpr int INITAL_VOCAB_SIZE = 256;
        static constexpr size_t MAX_TEXT_SIZE = 1024 * 1024;
    };
    
    class NaiveTokenizer {
    public:
        using EncodeResult = std::vector<std::pair<std::string, int>>;
        using StrToInt = std::unordered_map<std::string_view, int>;
        
        NaiveTokenizer() : initialized_(false), model_proto_(nullptr) {}
        
        explicit NaiveTokenizer(const Model& model_proto) 
            : model_proto_(&model_proto), initialized_(true) {
            InitFromModel();
        }
        
        virtual ~NaiveTokenizer() {}
        
        void InitFromModel() {
            if (!model_proto_) return;
            
            // 清除之前的数据
            merges_.clear();
            vocab_.clear();
            pieces_.clear();
            reserved_.clear();
            
            // 获取特殊标记
            const auto& counter_spec = model_proto_->GetCounterSpec();
            unk_id_ = counter_spec.unk_id();
            
            // 加载所有pieces
            for (size_t i = 0; i < model_proto_->PiecesSize(); ++i) {
                const auto& p = model_proto_->GetPieces(i);
                const std::string& piece = p.GetPiece();
                pieces_[piece] = i;
                
                // 初始化字节vocab
                if (i < INITAL_VOCAB_SIZE) {
                    if (piece.size() == 1) {
                        std::vector<uint8_t> byte_vec = {static_cast<uint8_t>(piece[0])};
                        vocab_.push_back(byte_vec);
                    } else {
                        vocab_.push_back({static_cast<uint8_t>(i)}); // 默认值
                    }
                }
                
                // 处理特殊标记
                if (p.GetType() != Model::Piece::NORMAL) {
                    reserved_[piece] = i;
                }
            }
            
            // 重建merges
            RebuildMerges();
            
            initialized_ = true;
        }
        
        bool IsInitialized() const {
            return initialized_;
        }
        
        float GetScore(int id) const {
            if (!model_proto_ || id < 0 || id >= static_cast<int>(model_proto_->PiecesSize())) {
                return 0.0f;
            }
            return model_proto_->GetPieces(id).GetScore();
        }
        
        int PieceID(std::string_view piece) const {
            const auto it = pieces_.find(piece);
            if (it != pieces_.end()) {
                return it->second;
            }
            
            const auto it2 = reserved_.find(piece);
            if (it2 != reserved_.end()) {
                return it2->second;
            }
            
            return unk_id_;
        }
        
        EncodeResult Encode(std::string_view text) const {
            if (!initialized_) {
                throw std::runtime_error("Tokenizer not initialized with a model");
            }
            
            std::string text_str(text);
            std::vector<int> ids;
            ids.assign(text_str.begin(), text_str.end());
            
            // 应用所有合并规则
            ApplyMerges(ids);
            
            // 构建编码结果
            EncodeResult result;
            for (int id : ids) {
                std::string piece;
                DecodeToken(id, piece);
                result.emplace_back(piece, id);
            }
            
            return result;
        }
        
        std::string Detokenize(const std::vector<int>& ids) const {
            if (!initialized_) {
                throw std::runtime_error("Tokenizer not initialized with a model");
            }
            
            std::string result;
            for (int id : ids) {
                DecodeToken(id, result);
            }
            
            return result;
        }
        
    private:
        void ApplyMerges(std::vector<int>& ids) const {
            while (ids.size() >= 2) {
                std::vector<size_t> pair_counts(MAX_TEXT_SIZE*3);
                size_t pair_counts_size;
                naive::TokenCounts(ids, pair_counts, pair_counts_size);
                
                IntPair next{-1, -1};
                size_t next_idx = 0;
                size_t min_merge_idx = std::numeric_limits<size_t>::max();
                
                for (size_t i = 0; i < pair_counts_size*3; i+= 3) {
                    IntPair pair{static_cast<int>(pair_counts[i]),
                               static_cast<int>(pair_counts[i+1])};
                    size_t idx = naive::GetPairIndex(merges_, pair);
                    if (idx < merges_.size() && idx < min_merge_idx) {
                        min_merge_idx = idx;
                        next = pair;
                        next_idx = idx;
                    }
                }
                
                if (next.first == -1) break;
                naive::MergePair(ids, next, merges_[next_idx].idx);
            }
        }
        
        void DecodeToken(int id, std::string& text) const {
            if (id < INITAL_VOCAB_SIZE) {
                text.push_back(static_cast<char>(id));
                return;
            }
            
            for (const auto& merge : merges_) {
                if (merge.idx == id) {
                    DecodeToken(merge.pair.first, text);
                    DecodeToken(merge.pair.second, text);
                    return;
                }
            }
            
            // 如果找不到对应的merge，可能是特殊标记
            const auto& counter_spec = model_proto_->GetCounterSpec();
            if (id == counter_spec.unk_id()) {
                text += counter_spec.unk_piece();
            } else if (id == counter_spec.bos_id()) {
                text += counter_spec.bos_piece();
            } else if (id == counter_spec.eos_id()) {
                text += counter_spec.eos_piece();
            } else if (id == counter_spec.pad_id()) {
                text += counter_spec.pad_piece();
            }
        }
        
        void RebuildMerges() {
            // 从model_proto中重建merges
            // 先构建一个完整的piece表示形式到ID的映射
            std::unordered_map<std::string, int> piece_to_id;
            for (size_t i = 0; i < model_proto_->PiecesSize(); ++i) {
                const auto& piece = model_proto_->GetPieces(i).GetPiece();
                piece_to_id[piece] = i;
            }
            
            // 遍历所有非基本字符piece，尝试确定其合并来源
            for (const auto& [piece, id] : piece_to_id) {
                if (id >= INITAL_VOCAB_SIZE && piece.length() > 1) {
                    // 尝试找出可能的前缀和后缀组合
                    for (size_t split = 1; split < piece.length(); ++split) {
                        std::string prefix = piece.substr(0, split);
                        std::string suffix = piece.substr(split);
                        
                        auto prefix_it = piece_to_id.find(prefix);
                        auto suffix_it = piece_to_id.find(suffix);
                        
                        if (prefix_it != piece_to_id.end() && suffix_it != piece_to_id.end()) {
                            // 找到可能的合并源
                            IntPair pair{prefix_it->second, suffix_it->second};
                            merges_.push_back({pair, id});
                            
                            // 为这个合并ID创建vocab_条目（如果超出初始范围）
                            if (id >= vocab_.size()) {
                                // 填充可能缺失的条目
                                while (vocab_.size() < id) {
                                    vocab_.push_back({});
                                }
                                // 追加新条目
                                vocab_.push_back({});
                            }
                            
                            break;
                        }
                    }
                }
            }
            
            // 按merge index排序
            std::sort(merges_.begin(), merges_.end(), 
                     [](const Merge& a, const Merge& b) {
                         return a.idx < b.idx;
                     });
        }
        
        const Model* model_proto_;
        std::vector<Merge> merges_;
        std::vector<std::vector<uint8_t>> vocab_;
        StrToInt pieces_;    // piece到id的映射
        StrToInt reserved_;  // 特殊标记到id的映射
        int unk_id_;         // 未知标记的id
        bool initialized_;
        
        static constexpr int INITAL_VOCAB_SIZE = 256;
        static constexpr size_t MAX_TEXT_SIZE = 1024 * 1024;
    };
    

} // namespace piece