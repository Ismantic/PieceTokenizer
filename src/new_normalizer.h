#pragma once

#include <vector>
#include <memory>
#include <set>
#include <map>
#include <string>
#include <functional>

#include "darts.h"

#include "trie.h"

#include "piece_spec.h"

namespace piece {

class MapBuilder {
public:
  MapBuilder() = delete;
  ~MapBuilder() = delete;

  using Ustr = std::vector<uint32_t>;
  using UstrMap = std::map<Ustr, Ustr>;

  static bool CompileUstrMap(const UstrMap& ustr_map,
                             std::string* output);
  static bool DecompileUstrMap(std::string_view blob, 
                               UstrMap* ustr_map);
  static bool GetPrecompiledUstrMap(std::string_view name, 
                                    std::string *output);
  static std::string EncodePrecompiledMap(std::string_view trie_blob, 
                                                 std::string_view normalized);
  static bool DecodePrecompiledMap(std::string_view blob, 
                                          std::string_view *trie_blob, 
                                          std::string_view *normalized);  
};

class Normalizer {
public:
  explicit Normalizer(const NormalizerSpec &spec);
  virtual ~Normalizer();

  bool Normalize(std::string_view input,
                         std::string *output, 
                         std::vector<size_t> *n2o) const;
    
  std::string Normalize(std::string_view input) const;
  std::string ReplaceSpace(std::string_view input) const;

  friend class MapBuilder;  

private:
  void Init();

  std::pair<std::string_view, int> ProcessTrie(std::string_view input) const;

  static constexpr int kMaxTrieResultSize = 32;
  //std::unique_ptr<Darts::DoubleArray> trie_;
  std::unique_ptr<new_darts::DoubleArray<int>> trie_;
  const char *normalized_ = nullptr; 

  const NormalizerSpec* spec_;
  std::string map_data_;

};

} // namepsace piece