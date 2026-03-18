#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <utility>
#include <string>
#include <algorithm>
#include <iostream>

#include <stdint.h>

#include "common.h"
#include "piece_spec.h"
#include "ustr.h"
#include "misc.h"
#include "sentence.h"
#include "new_normalizer.h"

namespace piece {

template<typename T>
class Node {
public:
  int count;
  T value;
  int pos;

  Node(int count, const T& value, int pos)
    : count(count), value(value), pos(pos) {}
  
  int GetCount() const { return count; }

  bool operator<(const Node& o) const {
    return this->GetCount() < o.GetCount();
  }
};

template<typename T>
class Multiset {
public:
  Multiset() = default;
  ~Multiset() {
    for (auto node : vec_) {
        delete node;
    }
  }

  void Insert(const T& item, int count = 1) {
    to_insert_[item] += count;
  }

  void Remove(const T& item, int count = 1) {
    to_remove_[item] += count;
  }

  int GetCount(const T& item) {
    _Commit();
    auto it = map_.find(item);
    return it == map_.end() ? 0 : it->second->count;
  }

  T Top() {
    _Commit();
    return vec_.empty() ? T() : vec_[0]->value;
  }

  std::vector<std::pair<T, int>> TopK(int k) {
    _Commit();
    std::vector<std::pair<T, int>> res;
    if (vec_.empty()) return res;
        
    using HeapItem = std::tuple<int, T, int, int>;  // -key, value, count, pos
    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> heap;
        
    heap.push(std::make_tuple(-vec_[0]->GetCount(), 
                              vec_[0]->value, 
                              vec_[0]->count, 
                              vec_[0]->pos));
        
    for (int i = 0; i < k && !heap.empty(); ++i) {
        auto [_key, val, count, pos] = heap.top();
        heap.pop();
        res.emplace_back(val, count);
        
        for (int child_pos : {pos * 2 + 1, pos * 2 + 2}) {
            if (child_pos < vec_.size()) {
                heap.push(std::make_tuple(
                    -vec_[child_pos]->GetCount(),
                     vec_[child_pos]->value,
                     vec_[child_pos]->count,
                     child_pos
                ));
            }
        }
    }
    return res;
  }
  
  explicit operator bool() const {
    const_cast<Multiset*>(this)->_Commit();
    return !vec_.empty();
  }

private:
  void _Insert(const T& item, int count = 1) {
    auto it = map_.find(item);
    if (it == map_.end()) {
        Node<T>* node = new Node<T>(0, item, vec_.size());
        map_[item] = node;
        vec_.push_back(node);
        it = map_.find(item);
    }
    it->second->count += count;
    _ItemIncrease(it->second->pos);
  }
  
  void _Remove(const T& item, int count = 1) {
    auto it = map_.find(item);
    if (it != map_.end()) {
        it->second->count -= count;
        _ItemDecrease(it->second->pos);
    }
  }
  
  void _Commit() {
    for (const auto& pair : to_insert_) {
        _Insert(pair.first, pair.second);
    }
    for (const auto& pair : to_remove_) {
        _Remove(pair.first, pair.second);
    }
    to_insert_.clear();
    to_remove_.clear();
  }
  
  void _ItemIncrease(int pos) {
    Node<T>* node = vec_[pos];
    while (pos > 0) {
        int uppos = (pos - 1) >> 1;
        Node<T>* up = vec_[uppos];
        if (*up < *node) {
            vec_[pos] = up;
            up->pos = pos;
            pos = uppos;
            continue;
        }
        break;
    }
    vec_[pos] = node;
    node->pos = pos;
  }
  
  void _ItemDecrease(int pos) {
    int endpos = vec_.size();
    Node<T>* node = vec_[pos]; 
    int downpos = 2*pos + 1;
    while (downpos < endpos) {
        int rightpos = downpos + 1;
        if (rightpos < endpos && !(*vec_[rightpos] < *vec_[downpos])) {
            downpos = rightpos;
        }
        Node<T>* downnode = vec_[downpos];
        if (*node < *downnode) {
            vec_[pos] = downnode;
            downnode->pos = pos;
            pos = downpos;
            downpos = 2*pos + 1;
        } else {
            break;
        }
    }
    vec_[pos] = node;
    node->pos = pos;
  }

  struct Hash {
      template<typename U>
      size_t operator()(const U& x) const {
          return std::hash<U>()(x);
      }

      size_t operator()(const std::pair<int,int>& p) const {
          auto h1 = std::hash<int>{}(p.first);
          auto h2 = std::hash<int>{}(p.second);
          return h1 ^ (h2 << 1);
      }
  };

  std::vector<Node<T>*> vec_;
  std::unordered_map<T, Node<T>*, Hash> map_; 
  std::unordered_map<T, int, Hash> to_insert_;
  std::unordered_map<T, int, Hash> to_remove_;
};

template<typename T>
class IndexedList {
public:
  class Node {
  public:
    T value;
    Node* prev;
    Node* next;

    Node(const T& value, Node* prev, Node* next) 
        : value(value), prev(prev), next(next) {
        }
    ~Node() {}
    
    void Delete() {
        if (prev) prev->next = next;
        if (next) next->prev = prev;
        next = prev = nullptr;
    };
  };

  struct PairHash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1,T2>& p) const {
        auto s1 = std::hash<T1>{}(p.first);
        auto s2 = std::hash<T2>{}(p.second);
        return s1 ^ s2;
    }
  };

  Node* start_;
  std::unordered_map<std::pair<T, T>, 
                     std::vector<Node*>,
                     PairHash> index_;

public:
  template<typename Iterator>
  IndexedList(Iterator begin, Iterator end) {
    if (begin == end) {
        start_ = nullptr;
        return;
    }

    auto it = begin;
    T a = *it;
    start_ = new Node(a, nullptr, nullptr);
    Node* prev_node = start_;

    ++it;
    while (it != end) {
        T b = *it;
        Node* node = new Node(b, prev_node, nullptr);
        prev_node->next = node;
        InsertToIndex(std::make_pair(a, b), prev_node);
        a = b;
        prev_node = node;
        ++it;
    }
  }

  ~IndexedList() {
    Node* current = start_;
    while (current) {
        Node* next = current->next;
        delete current;
        current = next;
    }
  }

  IndexedList(const IndexedList& other) : start_(nullptr) {
    if (!other.start_) {
        return;
    }

    start_ = new Node(other.start_->value, nullptr, nullptr);
    Node* curr = start_;
    Node* other_curr = other.start_->next;
    
    while (other_curr) {
        curr->next = new Node(other_curr->value, curr, nullptr);
        curr = curr->next;
        other_curr = other_curr->next;
    }

    curr = start_;
    while (curr->next) {
        InsertToIndex(std::make_pair(curr->value, curr->next->value), curr);
        curr = curr->next;
    }
}

const std::unordered_map<std::pair<T, T>, std::vector<Node*>, PairHash>& GetIndexMap() const {
    return index_;
}


IndexedList(IndexedList&& other) noexcept 
    : start_(other.start_), index_(std::move(other.index_)) {
    other.start_ = nullptr;
}

IndexedList& operator=(const IndexedList& other) {
    if (this != &other) {
        IndexedList temp(other);  // copy-and-swap idiom
        std::swap(start_, temp.start_);
        std::swap(index_, temp.index_);
    }
    return *this;
}

IndexedList& operator=(IndexedList&& other) noexcept {
    if (this != &other) {
        Node* current = start_;
        while (current) {
            Node* next = current->next;
            delete current;
            current = next;
        }
        
        start_ = other.start_;
        index_ = std::move(other.index_);
        
        other.start_ = nullptr;
    }
    return *this;
}

  std::vector<Node*>& GetIndex(const std::pair<T,T>& pair) {
    return index_[pair];
  }
  const std::vector<Node*>& GetIndex(const std::pair<T,T>& pair) const {
    return index_.at(pair);
  }


  void InsertToIndex(const std::pair<T, T>& pair, Node* node) {
    index_[pair].push_back(node);
  }

void RemoveIndex(Node* node) {
    // 1. 删除 (prev->value, node->value) 对
    if (node->prev) {
        auto pair = std::make_pair(node->prev->value, node->value);
        auto it = index_.find(pair);
        if (it != index_.end()) {
            auto& nodes = it->second;
            nodes.erase(
                std::remove(nodes.begin(), nodes.end(), node->prev),
                nodes.end()
            );
            if (nodes.empty()) {
                index_.erase(it);
            }
        }
    }
    
    // 2. 删除 (node->value, next->value) 对
    if (node->next) {
        auto pair = std::make_pair(node->value, node->next->value);
        auto it = index_.find(pair);
        if (it != index_.end()) {
            auto& nodes = it->second;
            nodes.erase(
                std::remove(nodes.begin(), nodes.end(), node),
                nodes.end()
            );
            if (nodes.empty()) {
                index_.erase(it);
            }
        }
    }
}

  void UpdateIndex(Node* node) {
    if (node->prev) {
      InsertToIndex(std::make_pair(node->prev->value, node->value), node->prev);
    }
    if (node->next) {
      InsertToIndex(std::make_pair(node->value, node->next->value), node);
    }
  }

  class Iterator {
  private:
    Node* current;
  
  public:
    explicit Iterator(Node* node) : current(node) {}

    Iterator& operator++() {
        if (current) current = current->next;
        return *this;
    }

    Node* operator*() const { return current; }

    bool operator!=(const Iterator& o) const {
        return current != o.current;
    }
  };

  Iterator begin() { return Iterator(start_); }
  Iterator end() { return Iterator(nullptr); }

  Iterator begin() const { return Iterator(start_); }
  Iterator end() const { return Iterator(nullptr); }

};

inline bool IsUTF8Start(uint8_t c) {
  return (c & 0x80) == 0 || (c & 0xC0) == 0xC0;
}

inline bool IsUTF8Continuation(uint8_t c) {
  return (c & 0xC0) == 0x80;
}

inline bool IsCrossUTF8Boundary(unsigned char prev, unsigned char curr) {
  return IsUTF8Continuation(prev) && IsUTF8Start(curr);
}


class SimpleCounter {
  public:
      SimpleCounter(const CounterSpec& counter_spec,
                    const NormalizerSpec& normalizer_spec)
          : counter_spec_(counter_spec),
            normalizer_spec_(normalizer_spec),
            vocab_size_(counter_spec.vocab_size()) {
          InitMetaPieces();
      }
      
      virtual ~SimpleCounter() {}
      
      bool Count() {
          // 加载句子
          if (!LoadSentences()) {
              LOG(ERROR) << "Failed to load sentences.";
              return false;
          }
          
          // 如果需要，按空格分割句子
          SplitSentencesByWhitespace();
          
          // 抽取文本内容进行BBPE训练
          std::vector<std::string> texts;
          for (const auto& sentence : sentences_) {
              texts.push_back(sentence.first);
          }
          
          // 计算合并操作
          merge_tree_.clear();
          auto stats = InitPairsStats(texts);
          std::vector<IndexedList<int>> indexed_lists;
          indexed_lists.reserve(texts.size());
          for (const auto& text : texts) {
              indexed_lists.push_back(BuildIndexedList(text));
          }
          
          int num_merges = vocab_size_ - meta_pieces_.size();
          LOG(INFO) << "Starting BBPE training with " << num_merges << " merges";

          int cnt = 0;
          for (int i = 0; i < 256; i++) {
            std::string t(1, i);
            vocab_[i] = t;
            std::vector<std::string> vec = {t, "", ""};
            pieces_.emplace_back(std::make_pair(vec, 1));
            cnt += 1;
          }
          
          while (cnt < num_merges && stats) {
              auto top = stats.Top();
              int n = stats.GetCount(top);
              int new_id = vocab_.size();
              merge_tree_.emplace_back(top, new_id);
              vocab_[new_id] = vocab_[top.first] + vocab_[top.second];

              std::string p = vocab_[new_id];
              std::string u = vocab_[top.first];
              std::string v = vocab_[top.second];

              std::vector<std::string> vec = {p,u,v};
              pieces_.emplace_back(std::make_pair(vec,n));
             
              for (auto& list : indexed_lists) {
                  Merge(top, new_id, list, &stats);
              }

             
              //if (!ustr::IsStructurallyValid(vocab_[new_id]))
              //  continue;

              if (cnt % 10 == 0) {
                  LOG(INFO) << "Merge " << cnt + 1 << "/" << num_merges
                           << ": (" << top.first << "," << top.second
                           << ") -> " << new_id << " (" << vocab_[new_id]
                           << ") had " << n
                           << " occurrences";
              }
 
              //pieces_.emplace_back(vocab_[new_id], -static_cast<float>(pieces_.size()));
              cnt += 1;
          }
          LOG(INFO) << "Done! " << cnt << " merges";
          
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

        size_t fid = 0;
        std::cout << counter_spec_.vocab_size() << " "
                  << meta_pieces_.size() << " "
                  << pieces_.size() << std::endl;
        for (int id = 0; id < counter_spec_.vocab_size(); ++id) {
            const auto it = meta_pieces_.find(id);
            if (it != meta_pieces_.end()) {
                auto *p = model->InsertPieces();
                p->SetPiece(it->second.first);
                p->SetType(it->second.second);
                p->SetScore(0.0);
            } else if (fid < pieces_.size()) {
                const auto x = pieces_[fid++];
                auto w = x.first;
                auto s = x.second;
                auto p = w[0];
                auto u = w[1];
                auto v = w[2];
                auto *piece = model->InsertPieces();
                piece->SetPiece(p, u, v);
                piece->SetScore(s);
            } else {
                LOG(ERROR) << "Invalid piece id: " << id;
                return false;
            }
        }
       
        model->SetCounterSpec(counter_spec_);
        model->SetNormalizerSpec(normalizer_spec_);
        
        return true;
    }
     
  private:
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
    
        int byte_id = meta_pieces_.size();
        if (byte_id + 256 > counter_spec_.vocab_size()) {
            std::cerr << "Error: Vocabulary size too small for byte_fallback. "
                      << "Need at least " << (byte_id + 256) << " slots." << std::endl; 
            return false;
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
          sentences_ = misc::Sorted(tokens); // TODO
          LOG(INFO) << "Done! " << sentences_.size();
      }
      
      static IndexedList<int> BuildIndexedList(const std::string& text) {
          std::vector<int> bytes;
          for (unsigned char c : text) {
              bytes.push_back(static_cast<int>(c));
          }
          return IndexedList<int>(bytes.begin(), bytes.end());
      }
      
      static Multiset<std::pair<int,int>> InitPairsStats(
              const std::vector<std::string>& texts) {
          Multiset<std::pair<int,int>> stats;
          std::vector<int> bytes;
          for (const auto& text : texts) {
              bytes.clear();
              for (uint8_t c : text) {
                  bytes.push_back(static_cast<int>(c));
              }
              for (size_t i = 0; i < bytes.size()-1; i++) {
                  uint8_t prev = static_cast<uint8_t>(bytes[i]);
                  uint8_t curr = static_cast<uint8_t>(bytes[i+1]);
                  //if (IsCrossUTF8Boundary(prev, curr)) {
                  //    continue;
                  //}
                  stats.Insert({bytes[i],bytes[i+1]});
              }
          }
          return stats;
      }
      
      static void Merge(const std::pair<int, int>& pair, int new_id,
                       IndexedList<int>& indexed_list,
                       Multiset<std::pair<int, int>>* stats = nullptr) {

          auto& nodes = indexed_list.GetIndex(pair);
          for (auto* node : nodes) {
              if (node->value != pair.first ||
                  node->next == nullptr ||
                  node->next->value != pair.second) {
                  continue;
              }

              indexed_list.RemoveIndex(node);
              indexed_list.RemoveIndex(node->next);

              if (stats != nullptr) {
                  stats->Remove(pair);
                  if (node->next->next != nullptr) {
                      stats->Remove({node->next->value, node->next->next->value});
                      stats->Insert({new_id, node->next->next->value});
                  }
                  if (node->prev != nullptr) {
                      stats->Remove({node->prev->value, pair.first});
                      stats->Insert({node->prev->value, new_id});
                  }
              }
              auto* remove = node->next;
              node->next->Delete();
              delete remove;
              node->value = new_id;
              indexed_list.UpdateIndex(node);
          }

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
      //std::vector<std::pair<std::string, float>> pieces_;
      std::vector<std::pair<std::vector<std::string>, float>> pieces_;
      Sentences sentences_;
      CounterSpec counter_spec_;
      NormalizerSpec normalizer_spec_;
      std::vector<std::pair<std::pair<int,int>, int>> merge_tree_;
      std::unordered_map<int, std::string> vocab_;
      int vocab_size_;
  };

  class SimpleTokenizer {
    public:
        using EncodeResult = std::vector<std::pair<std::string, int>>;
        using StrToInt = std::unordered_map<std::string_view, int>;
        
        explicit SimpleTokenizer(const Model& model) 
            : model_(&model) {
            InitFromModel();
        }
        
        virtual ~SimpleTokenizer() {}
    
        EncodeResult Encode(std::string_view text) const {
            if (!model_) {
                throw std::runtime_error("Tokenizer not initialized with a model");
            }
            
            // 1. 文本标准化
            std::string normalized_text = NormalizeText(text);
            
            // 2. 构建初始token ID链表
            auto token_list = BuildInitialTokenList(normalized_text);
            
            // 3. 应用合并规则
            ApplyMergeRules(token_list);
            
            // 4. 将token ID转换为结果格式
            return TokenIdsToResult(token_list);
        }
        
        // 分词函数：只返回tokens，不返回ID
        std::vector<std::string> Tokenize(std::string_view text) const {
            std::vector<std::string> tokens;
            auto encoded = Encode(text);
            tokens.reserve(encoded.size());
            for (const auto& [piece, id] : encoded) {
                tokens.push_back(piece);
            }
            return tokens;
        }
        
        // 解码函数：从ID序列还原文本
        std::string Decode(const std::vector<int>& ids) const {
            if (!model_) {
                throw std::runtime_error("Tokenizer not initialized with a model");
            }
            
            std::string result;
            for (int id : ids) {
                if (id < 0 || id >= model_->PiecesSize()) {
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
            
            // 应用反标准化处理（如果需要）
            if (model_->HasNormalizerSpec()) {
                Normalizer normalizer(model_->GetNormalizerSpec());
                return normalizer.ReplaceSpace(result);
            }
            
            return result;
        }
        
        // 解码函数重载：从EncodeResult还原文本
        std::string Decode(const EncodeResult& rs) const {
            std::vector<int> ids;
            ids.reserve(rs.size());
            for (const auto & [piece, id] : rs) {
                ids.push_back(id);
            }
            return Decode(ids);
        }
        
        int PieceID(std::string_view piece) const {
            // 先检查是否为保留标记
            const auto it2 = reserve_.find(piece);
            if (it2 != reserve_.end()) {
                return it2->second;
            }
            
            // 再检查是否为普通piece
            const auto it = pieces_.find(piece);
            if (it != pieces_.end()) {
                return it->second;
            }
            
            // 未找到，返回未知标记ID
            return unk_id_;
        }
    
    private:
        // 用于规则索引的哈希函数
        struct PairHash {
            size_t operator()(const std::pair<int, int>& p) const {
                auto h1 = std::hash<int>{}(p.first);
                auto h2 = std::hash<int>{}(p.second);
                return h1 ^ (h2 << 1);
            }
        };
        
        // 合并规则类型
        using MergeRule = std::pair<std::pair<int, int>, int>;  // ((first_id, second_id), merged_id)
        
        // 规则信息类型
        using RuleInfo = std::pair<size_t, int>;  // (优先级, 合并后ID)
        
        // 成员变量
        const Model* model_;
        std::vector<MergeRule> merge_rules_;  // 基于ID的合并规则
        std::unordered_map<std::pair<int, int>, RuleInfo, PairHash> pair_to_rule_;  // 规则索引
        std::unordered_map<int, std::string> vocab_;  // id -> piece
        StrToInt pieces_;  // piece -> id
        StrToInt reserve_;  // 特殊标记映射
        int unk_id_;  // 未知标记ID
        
        // 文本标准化
        std::string NormalizeText(std::string_view text) const {
            if (model_->HasNormalizerSpec()) {
                Normalizer normalizer(model_->GetNormalizerSpec());
                return normalizer.Normalize(std::string(text));
            }
            return std::string(text);
        }
        
        // 构建初始token ID链表
        IndexedList<int> BuildInitialTokenList(const std::string& text) const {
            std::vector<int> initial_token_ids;
            initial_token_ids.reserve(text.size());
            
            for (unsigned char c : text) {
                std::string byte_str(1, c);
                auto it = pieces_.find(byte_str);
                if (it != pieces_.end()) {
                    initial_token_ids.push_back(it->second);
                } else {
                    std::cerr << "Error" << std::endl;
                }
            }
            
            return IndexedList<int>(initial_token_ids.begin(), initial_token_ids.end());
        }

        void ApplyMergeRules(IndexedList<int>& token_list) const {
            bool found_merge;
            
            // 不断扫描和合并，直到没有可合并的token对
            do {
                found_merge = false;
                
                // 遍历文本中的所有相邻token对
                for (auto it = token_list.begin(); it != token_list.end(); ++it) {
                    auto* node = *it;
                    if (!node || !node->next) continue;
                    
                    // 当前token对
                    std::pair<int, int> pair(node->value, node->next->value);
                    
                    // 查找是否有对应的合并规则
                    auto rule_it = pair_to_rule_.find(pair);
                    if (rule_it != pair_to_rule_.end()) {
                        // 找到了可合并的token对
                        int merged_id = rule_it->second.second;
                        
                        // 执行合并
                        Merge(pair, merged_id, token_list);
                        found_merge = true;
                        
                        // 找到一个可合并的token对后立即退出循环，重新从头扫描
                        // 这确保了总是优先处理文本前面的token对
                        break;
                    }
                }
            } while (found_merge);
        }

        // 应用合并规则
        void ApplyMergeRulesX(IndexedList<int>& token_list) const {
            // 合并周期
            bool changed = true;
            while (changed) {
                changed = false;
                
                // 每轮找出优先级最高的有效合并点
                size_t best_priority = std::numeric_limits<size_t>::max();
                std::pair<int, int> best_pair;
                int best_merged_id = -1;
                
                // 遍历索引中的所有规则，检查每个规则对应的token对是否存在于文本中
                for (const auto& [pair, rule_info] : pair_to_rule_) {
                    const auto& [priority, merged_id] = rule_info;
                    
                    // 如果已找到更高优先级的规则，跳过较低优先级的规则
                    if (priority >= best_priority) {
                        continue;
                    }
                    
                    // 检查该token对是否存在于文本中
                    auto& nodes = token_list.GetIndex(pair);
                    if (!nodes.empty()) {
                        // 找到一个更高优先级的有效合并点
                        best_priority = priority;
                        best_pair = pair;
                        best_merged_id = merged_id;
                    }
                }
                
                // 如果找到有效的合并点，执行合并
                if (best_merged_id >= 0) {
                    Merge(best_pair, best_merged_id, token_list);
                    changed = true;
                }
            }
        }
        
        // 合并操作
        void Merge(const std::pair<int, int>& pair, int new_id, 
                   IndexedList<int>& indexed_list) const {
            auto& nodes = indexed_list.GetIndex(pair);
            for (auto* node : nodes) {
                // 验证节点是否有效且符合预期
                if (node->value != pair.first || 
                    node->next == nullptr || 
                    node->next->value != pair.second) {
                    continue;
                }
                
                // 从索引中移除当前节点和下一个节点
                indexed_list.RemoveIndex(node);
                indexed_list.RemoveIndex(node->next);
                
                // 执行合并
                auto* to_remove = node->next;
                node->value = new_id;
                node->next = to_remove->next;
                if (to_remove->next) to_remove->next->prev = node;
                delete to_remove;
                
                // 更新索引
                indexed_list.UpdateIndex(node);
            }
        }
        
        // 将token ID链表转换为EncodeResult
        EncodeResult TokenIdsToResult(const IndexedList<int>& token_list) const {
            EncodeResult result;
            
            for (auto it = token_list.begin(); it != token_list.end(); ++it) {
                auto node = *it;
                if (!node) continue;
                
                int token_id = node->value;
                if (token_id >= 0 && token_id < model_->PiecesSize()) {
                    const std::string& piece = model_->GetPieces(token_id).GetPiece();
                    result.emplace_back(piece, token_id);
                } else if (unk_id_ >= 0) {
                    const std::string& unk_piece = model_->GetPieces(unk_id_).GetPiece();
                    result.emplace_back(unk_piece, unk_id_);
                }
            }
            
            return result;
        }
        
        // 初始化模型数据
        void InitFromModel() {
            if (!model_) {
                throw std::runtime_error("No model provided for initialization");
            }
            
            // 清除之前的数据
            merge_rules_.clear();
            pair_to_rule_.clear();
            vocab_.clear();
            pieces_.clear();
            reserve_.clear();
            
            // 获取特殊标记
            const auto& counter_spec = model_->GetCounterSpec();
            unk_id_ = counter_spec.unk_id();
            
            // 初始化所有pieces映射和词汇表
            for (size_t i = 0; i < model_->PiecesSize(); ++i) {
                const auto& p = model_->GetPieces(i);
                const std::string& piece = p.GetPiece();
                
                pieces_[piece] = i;
                vocab_[i] = piece;
                
                if (p.GetType() != Model::Piece::NORMAL) {
                    reserve_[piece] = i;
                }
            }
            
            // 构建合并规则和规则索引
            BuildMergeRules();
        }
        
        // 构建合并规则和规则索引
        void BuildMergeRules() {
            merge_rules_.clear();
            pair_to_rule_.clear();
            
            LOG(INFO) << "Building merge rules from model with " 
                      << model_->PiecesSize() << " pieces";
            
            for (size_t i = 0; i < model_->PiecesSize(); ++i) {
                const auto& piece = model_->GetPieces(i);
                
                if (piece.GetType() != Model::Piece::NORMAL) {
                    continue;
                }
                
                // 获取u和v，这是合并前的两个piece
                const std::string& u = piece.u();
                const std::string& v = piece.v();
                
                if (!u.empty() && !v.empty()) {
                    int u_id = PieceID(u);
                    int v_id = PieceID(v);
                    
                    if (u_id >= 0 && v_id >= 0) {
                        // 将这对ID加入到合并规则中
                        std::pair<int, int> token_pair(u_id, v_id);
                        size_t rule_idx = merge_rules_.size();
                        merge_rules_.emplace_back(token_pair, static_cast<int>(i));
                        
                        // 同时更新规则索引 - 存储每个token对的规则优先级和合并后ID
                        pair_to_rule_[token_pair] = {rule_idx, static_cast<int>(i)};
                        
                        if (rule_idx % 10 == 0) {
                            LOG(INFO) << "Added merge rule " << rule_idx << ": (" 
                                      << u << "," << v << ") -> " << piece.GetPiece() 
                                      << " (ID: " << i << ")";
                        }
                    }
                }
            }
            
            LOG(INFO) << "Total merge rules: " << merge_rules_.size();
        }
    };
  
} // namespace piece