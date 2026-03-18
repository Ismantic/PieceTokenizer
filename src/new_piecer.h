#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>
#include <memory>

#include "new_normalizer.h"
#include "common.h"


namespace piece {


template <class T>
class FreeList {
 public:
  FreeList() = delete;
  explicit FreeList(size_t chunk_size) : chunk_size_(chunk_size) {}
  virtual ~FreeList() {
    for (auto& chunk : freelist_) delete[] chunk;
  }

  // `Free` doesn't free the object but reuse the allocated memory chunks.
  void Free() {
    const int size = std::min<int>(chunk_index_ + 1, freelist_.size());
    for (int i = 0; i < size; ++i) {
      T* chunk = freelist_[i];
      memset(static_cast<void*>(chunk), 0, sizeof(*chunk) * chunk_size_);
    }
    chunk_index_ = 0;
    element_index_ = 0;
  }

  // Returns the number of allocated elements.
  size_t size() const { return chunk_size_ * chunk_index_ + element_index_; }

  void swap(FreeList<T>& other) {
    std::swap(freelist_, other.freelist_);
    std::swap(element_index_, other.element_index_);
    std::swap(chunk_index_, other.chunk_index_);
    std::swap(chunk_size_, other.chunk_size_);
  }

  // Returns the element as an array.
  T* operator[](size_t index) const {
    return freelist_[index / chunk_size_] + index % chunk_size_;
  }

  // Allocates new element.
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

  // The last element is stored at freelist_[chunk_index_][element_index_]
  size_t element_index_ = 0;
  size_t chunk_index_ = 0;
  size_t chunk_size_ = 0;  // Do not modify except in swap()
};

using EncodeResult = std::vector<std::pair<std::string,int>>;

class Model;

class NewModel {

public:
  using StrToInt = std::unordered_map<std::string_view, int>;

  explicit NewModel(const Model& model);

  virtual ~NewModel();

  int PieceID(std::string_view piece) const;

  EncodeResult Encode(std::string_view text) const;

  std::vector<std::string> Tokenize(std::string_view text) const {
    std::vector<std::string> tokens;
    auto encoded = Encode(text);
    tokens.reserve(encoded.size());
    
    for (const auto& [piece, id] : encoded) {
        tokens.push_back(piece);
    }
    
    return tokens;  
  }

  std::string Decode(const std::vector<int>& ids) const;

  std::string Decode(const EncodeResult& rs) const {
    std::vector<int> ids;
    ids.reserve(rs.size());

    for (const auto & [piece, id] : rs) {
      ids.push_back(id);
    }

    return Decode(ids);
  }

private:
  void InitPieces();
  float GetScore(int id) const {
    return model_->GetPieces(id).GetScore();
  }
 
  const Model* model_ = nullptr;
  const std::unique_ptr<Normalizer> normalizer_;

  StrToInt pieces_;
  StrToInt reserve_;
  int unk_id_ = 0;

};

} // namespace piece
