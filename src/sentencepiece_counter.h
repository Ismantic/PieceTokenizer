#pragma once 

#include <algorithm>
#include <vector>
#include <string> 
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>

#include <stdint.h>

#include "sentence.h"
#include "piece_spec.h"
#include "ustr.h"
#include "misc.h"

namespace piece {

class SentencePieceCounter {
public:
  using Sentence = std::pair<std::string, int64_t>;
  using Sentences = std::vector<Sentence>;

  SentencePieceCounter(const CounterSpec& counter_spec, 
          const NormalizerSpec& normalizer_spec);
  virtual ~SentencePieceCounter();


  bool Count();

  bool Save() const;

  bool Serialize(Model* model) const;

private:
  bool LoadSentences();
  void SplitSentencesByWhitespace();

  std::map<int, std::pair<std::string, Model::Piece::Type>> meta_pieces_;
  std::unordered_map<uint32_t, uint64_t> required_chars_;
  std::vector<std::pair<std::string, float>> pieces_;

  Sentences sentences_;

  CounterSpec counter_spec_;
  NormalizerSpec normalizer_spec_;

  bool InitMetaPieces();


  // Symbol represents a character or symbol bigram.
  struct Symbol {
    const Symbol *left;              // left symbol in bigram
    const Symbol *right;             // right symbol in bigram
    ustr::UnicodeText chars;  // all flattend chracter sequence
    bool is_unk;                     // true if this symbol is unknown.
    uint64_t fp;                     // fingerprint of this symbol.
    uint64_t freq;                   // frequency of this symbol.

    // Position list. Use set so that we can keep the order of occurrence.
    // See EncodePos/DecodePos.
    //absl::btree_set<uint64_t> positions;
    std::set<uint64_t> positions;

    bool IsBigram() const { return left != nullptr && right != nullptr; }
    std::string ToString() const;
    Symbol() : left(nullptr), right(nullptr), is_unk(false), fp(0), freq(0) {}
  };

  struct Position {
    int sid;    // sentence id
    int left;   // left symbol index
    int right;  // right symbol index
  };

  // Encodes sid, left and right bigram index into uint64_t.
  // Encoded value keeps the order of sid, left and right.
  static uint64_t EncodePos(int sid, int l, int r) {
    const uint64_t n = (static_cast<uint64_t>(sid) << 32) |
                       (static_cast<uint64_t>(l) << 16) | r;
    return n;
  }

  // Decodes sid, left and right bigram index from uint64_t.
  static Position DecodePos(uint64_t n) {
    Position p;
    p.sid = n >> 32;
    p.left = (n >> 16) & 0xffff;
    p.right = n & 0xffff;
    return p;
  }

  Symbol *GetCharSymbol(uint32_t c);
  Symbol *GetPairSymbol(const Symbol *left, const Symbol *right);
  void ComputeFreq(Symbol *symbol) const;
  int GetNextIndex(int sid, int index) const;
  int GetPrevIndex(int sid, int index) const;
  void AddNewPair(int sid, int left, int right);
  void ResetFreq(int sid, int left, int right, const Symbol *best);
  void UpdateActiveSymbols();

  
  std::unordered_map<uint64_t, Symbol*> symbols_cache_;
  std::set<Symbol*> active_symbols_;
  std::vector<Symbol*> allocated_;
  std::vector<std::vector<Symbol*>> symbols_;

};

} // namespace piece
