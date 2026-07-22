#pragma once

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "pretokenizer.h"
#include "ustr.h"

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

  explicit SentencePieceTokenizer(const Model& model,
                                  const std::string& dict = "");
  ~SentencePieceTokenizer();

  int PieceID(std::string_view piece) const;
  EncodeResult Encode(std::string_view text) const;
  std::vector<std::string> Tokenize(std::string_view text) const;
  std::string Decode(const std::vector<int>& ids) const;
  std::string Decode(const EncodeResult& rs) const;
  bool valid() const { return pretokenizer_.valid(); }

private:
  // Core Viterbi-BPE encoding on a single segment (no normalize, no split).
  EncodeResult EncodeSegment(std::string_view text) const;

  const Model* model_;
  PreTokenizer pretokenizer_;
  StrToInt pieces_;
  int unk_id_;
};

} // namespace piece
