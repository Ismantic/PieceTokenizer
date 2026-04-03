#pragma once

#include <algorithm>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "normalizer.h"

namespace piece {

template <class T>
class FreeList {
 public:
  FreeList() = delete;
  explicit FreeList(size_t chunk_size) : chunk_size_(chunk_size) {}
  virtual ~FreeList() {
    for (auto& chunk : freelist_) delete[] chunk;
  }

  size_t size() const { return chunk_size_ * chunk_index_ + element_index_; }

  T* Allocate() {
    if (element_index_ >= chunk_size_) {
      ++chunk_index_;
      element_index_ = 0;
    }

    if (chunk_index_ == freelist_.size()) {
      T* chunk = new T[chunk_size_];
      memset(static_cast<void*>(chunk), 0, sizeof(*chunk) * chunk_size_);
      freelist_.push_back(chunk);
    }

    T* result = freelist_[chunk_index_] + element_index_;
    ++element_index_;

    return result;
  }

 private:
  std::vector<T*> freelist_;
  size_t element_index_ = 0;
  size_t chunk_index_ = 0;
  size_t chunk_size_ = 0;
};

using EncodeResult = std::vector<std::pair<std::string, int>>;

class Model;

class SentencePieceTokenizer {
public:
  using StrToInt = std::unordered_map<std::string_view, int>;

  explicit SentencePieceTokenizer(const Model& model);
  ~SentencePieceTokenizer();

  int PieceID(std::string_view piece) const;
  EncodeResult Encode(std::string_view text) const;
  std::vector<std::string> Tokenize(std::string_view text) const;
  std::string Decode(const std::vector<int>& ids) const;
  std::string Decode(const EncodeResult& rs) const;

private:
  const Model* model_;
  Normalizer normalizer_;
  StrToInt pieces_;
  int unk_id_;
};

} // namespace piece
