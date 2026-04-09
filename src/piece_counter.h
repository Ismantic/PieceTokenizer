#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "misc.h"
#include "normalizer.h"
#include "piece_spec.h"
#include "sentence.h"
#include "ustr.h"

namespace piece {

// ---------------------------------------------------------------------------
// Multiset — max-heap with O(1) top, O(log n) insert/remove, keyed lookup
// ---------------------------------------------------------------------------

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
    int downpos = 2 * pos + 1;
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
        downpos = 2 * pos + 1;
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

    size_t operator()(const std::pair<int, int>& p) const {
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

// ---------------------------------------------------------------------------
// PieceCounter — byte-level BPE trainer
// ---------------------------------------------------------------------------

class PieceCounter {
public:
  PieceCounter(const CounterSpec& counter_spec,
               const NormalizerSpec& normalizer_spec);
  ~PieceCounter();

  bool Count();
  bool Save() const;
  bool Serialize(Model* model) const;

  struct Token {
    int value;
    Token* prev;
    Token* next;
  };

private:
  using Sentence = std::pair<std::string, int64_t>;
  using Sentences = std::vector<Sentence>;

  struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const {
      return std::hash<int>{}(p.first) ^ (std::hash<int>{}(p.second) << 1);
    }
  };

  using PairIndex = std::unordered_map<std::pair<int, int>,
                                       std::vector<size_t>, PairHash>;
  using DeltaMap = std::unordered_map<std::pair<int, int>, int64_t, PairHash>;
  using IndexEntry = std::pair<std::pair<int, int>, size_t>;

  bool InitMetaPieces();
  bool LoadSentences();

  static Token* BuildTokenList(const std::string& text);
  static void FreeTokenList(Token* head);

  void InitPairsStatsAndIndex(Multiset<std::pair<int, int>>& stats,
                              PairIndex& pair_index);

  static void MergeSentence(const std::pair<int, int>& pair, int new_id,
                            Token* head, int64_t freq,
                            Multiset<std::pair<int, int>>& stats,
                            size_t sentence_idx, PairIndex& pair_index);

  std::map<int, std::pair<std::string, Model::Piece::Type>> meta_pieces_;
  std::vector<std::pair<std::vector<std::string>, float>> pieces_;
  Sentences sentences_;
  std::vector<int64_t> freqs_;
  std::vector<Token*> token_lists_;
  CounterSpec counter_spec_;
  NormalizerSpec normalizer_spec_;
  std::unordered_map<int, std::string> vocab_;
};

}  // namespace piece
