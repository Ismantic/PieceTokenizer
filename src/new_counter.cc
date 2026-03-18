#include "new_counter.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <memory>

#include <math.h>

#include "common.h"
#include "new_normalizer.h"
#include "ustr.h"
#include "sentence.h"
#include "misc.h"

namespace piece {

NewCounter::NewCounter(const CounterSpec &counter_spec, 
                 const NormalizerSpec &normalizer_spec)
    : counter_spec_(counter_spec),
      normalizer_spec_(normalizer_spec) {
    InitMetaPieces();
}

NewCounter::~NewCounter() {}

bool NewCounter::InitMetaPieces() {
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

    for (int i = 0; i < 256; ++i) {
        std::string byte_piece = ustr::ByteToPiece(i);
        meta_pieces_[byte_id++] = std::make_pair(byte_piece, Model::Piece::BYTE);
    }

    return true;
}

void NewCounter::SplitSentencesByWhitespace() {
    LOG(INFO) << "Tokenizing input sentences with whitespace: "
              << sentences_.size();
    const std::string_view space = normalizer_spec_.GetSpace();
    std::unordered_map<std::string, int64_t> tokens;
    for (const auto& s : sentences_) {
        for (const auto &w :
            ustr::SplitText(s.first, space)) {
            tokens[std::string(w)] += s.second;
        }
    }
    sentences_ = misc::Sorted(tokens);
    LOG(INFO) << "Done! " << sentences_.size(); 
}

bool NewCounter::LoadSentences() {

    uint32_t UNK = counter_spec_.GetUnkUnicode();

    auto iter_ = std::make_unique<MultiFileSentenceIterator>(
        std::vector<std::string>(counter_spec_.input().begin(),
                                 counter_spec_.input().end()));
    SentenceIterator* sentence_iterator_ = iter_.get();

    LOG(INFO) << "Loading sentences ...";
    for (; !sentence_iterator_->done(); sentence_iterator_->Next()) {
        std::string sentence = sentence_iterator_->value();
        if (sentence.empty()) {
            continue;
        }
        sentences_.emplace_back(std::make_pair(sentence, 1));
    }

    LOG(INFO) << "Normalizing sentences ...";
    //const normalizer::Normalizer normalizer(normalizer_spec_);
    const Normalizer normalizer(normalizer_spec_);
    for (size_t i = 0; i < sentences_.size(); ++i) {
        auto *s = &sentences_[i].first;
        *s = normalizer.Normalize(*s);
    }

    // Count
    int64_t all_chars_count = 0;
    std::unordered_map<uint32_t, int64_t> chars_count;
    for (const auto &w : sentences_) {
        for (const uint32_t c : ustr::UTF8ToUnicodeText(w.first)) {
            if (!ustr::IsValidCodepoint(c)) continue;    
            chars_count[c] += w.second;
            all_chars_count += w.second;
        }
    }
    LOG(INFO) << "all chars count=" << all_chars_count;

    // Determines required_chars which must be included in the vocabulary.
    int64_t accumulated_chars_count = 0;
    for (const auto &w : misc::Sorted(chars_count)) {
        const float coverage = 1.0*accumulated_chars_count/all_chars_count;
        if (coverage >= counter_spec_.character_coverage()) {
            LOG(INFO) << "Done: " << 100.0*coverage << "%s characters are covered.";
            break;
        }
        accumulated_chars_count += w.second;
        required_chars_.emplace(w.first, w.second);
    }
    LOG(INFO) << "Alphabase size=" << required_chars_.size();
    LOG(INFO) << "Final character coverage="
              << 1.0*accumulated_chars_count/all_chars_count;
    if (misc::ContainsKey(required_chars_, UNK)) {
        return false;
    }
    
    // Replaces rare characters (characters not included in required_chars_)
    for (auto &w : sentences_) {
        ustr::UnicodeText uw2;
        for (const uint32_t c : ustr::UTF8ToUnicodeText(w.first)) {
            if (misc::ContainsKey(required_chars_, c)) {
                uw2.push_back(c);
            } else {
                uw2.push_back(UNK);
            }
        }
        w.first = ustr::UnicodeTextToUTF8(uw2);
    }

    LOG(INFO) << "Done! preprocessed " << sentences_.size() << " sentences.";

    return true;
}

NewCounter::Symbol* NewCounter::GetCharSymbol(uint32_t c) {
  const uint64_t freq = misc::FindWithDefault(required_chars_, c, 1);
  const auto it = symbols_cache_.find(c);
  if (it != symbols_cache_.end()) {
    return it->second;
  }
  Symbol *s = new Symbol;
  allocated_.push_back(s);
  s->is_unk = (counter_spec_.GetUnkUnicode() == c);
  s->fp = c;
  s->chars.push_back(c);
  s->freq = freq;
  misc::InsertOrDie(&symbols_cache_, s->fp, s);
  return s;    
}

NewCounter::Symbol* NewCounter::GetPairSymbol(const Symbol* left, 
                                              const Symbol* right) {
  if (left == nullptr || right == nullptr || left->is_unk || right->is_unk) {
    return nullptr;
  }

  const uint64_t fp = misc::FingerprintCat(left->fp, right->fp);    
  const auto it = symbols_cache_.find(fp);
  if (it != symbols_cache_.end()) {
    return it->second; 
  }

  ustr::UnicodeText ut;
  for (const uint32_t c : left->chars) ut.push_back(c);
  for (const uint32_t c : right->chars) ut.push_back(c);

  Symbol* s = new Symbol;
  allocated_.push_back(s);
  s->fp = fp;
  s->left = left;
  s->right = right;
  s->chars = ut;
  misc::InsertOrDie(&symbols_cache_, s->fp, s);
  return s;
}

void NewCounter::AddNewPair(int sid, int left, int right) {
  if (left == -1 || right == -1) return;
  auto *symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
  if (symbol != nullptr) {
    active_symbols_.insert(symbol);
    symbol->positions.insert(EncodePos(sid, left, right));
  }
}

void NewCounter::ComputeFreq(Symbol *symbol) const {
    if (symbol->freq > 0) {
        return;
    }
    for (auto it = symbol->positions.begin(); it != symbol->positions.end();) {
        const Position pos = DecodePos(*it);
        if (symbol->left != symbols_[pos.sid][pos.left] || 
            symbol->right != symbols_[pos.sid][pos.right]) {
                it = symbol->positions.erase(it);
        } else {
            symbol->freq += sentences_[pos.sid].second;
            ++it;
        }
    }
}

void NewCounter::UpdateActiveSymbols() {
    std::vector<Symbol*> symbols;
    for (auto &it : symbols_cache_) {
        Symbol* symbol = it.second;
        if (symbol->IsBigram()) {
            ComputeFreq(symbol);
            symbols.push_back(symbol);
        }
    }
  // At least kMinActiveSymbolsSize symbols must be in |active_symbols_|.
  constexpr int kMinActiveSymbolsSize = 1000;

  // Keeps top 5% frequent symbols.
  constexpr float kTopFrequentRatio = 0.05;
  const int size =
      std::min<int>(std::max<int>(kMinActiveSymbolsSize,
                                  symbols_cache_.size() * kTopFrequentRatio),
                    symbols.size());

  std::partial_sort(symbols.begin(), symbols.begin() + size, symbols.end(),
                    [](Symbol *s1, Symbol *s2) { return s1->freq > s2->freq; });
  LOG(INFO) << "Updating active symbols. max_freq=" << symbols[0]->freq
            << " min_freq=" << symbols[size - 1]->freq;

  active_symbols_.clear();
  active_symbols_.insert(symbols.begin(), symbols.begin() + size);
}
int NewCounter::GetPrevIndex(int sid, int index) const {
    for (int i = index-1; i >= 0; --i) {
        if (symbols_[sid][i] == nullptr) continue;
        return i;
    }
    return -1;
}
int NewCounter::GetNextIndex(int sid, int index) const {
    for (size_t i = index + 1; i < symbols_[sid].size(); ++i) {
        if (symbols_[sid][i] == nullptr) continue;
        return i;
    }
    return -1;
}

void NewCounter::ResetFreq(int sid, int left, int right, const Symbol* best) {
    if (left == -1 || right == -1)  return;
    auto* symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
    if (symbol != nullptr && symbol != best) {
        symbol->freq = 0;
    }
}

std::string NewCounter::Symbol::ToString() const {
    return ustr::UnicodeTextToUTF8(chars);
}

bool NewCounter::Count() {
    symbols_.clear();
    allocated_.clear();
    symbols_cache_.clear();
    active_symbols_.clear();

    LoadSentences();

    SplitSentencesByWhitespace();

    // Initializes symbols_. symbols_[s][i] stores an unary symbol.
    symbols_.resize(sentences_.size());
    for (size_t i = 0; i < sentences_.size(); ++i) {
        for (const uint32_t c : ustr::UTF8ToUnicodeText(sentences_[i].first)) {
            symbols_[i].push_back(GetCharSymbol(c));
        }
    }

    // Makes all bigram symbols.
    for (size_t s = 0; s < symbols_.size(); ++s) {
        for (size_t i = 1; i < symbols_[s].size(); ++i) {
            AddNewPair(s, i-1, i);
        }
    }

    const int vocab_size = 
        counter_spec_.vocab_size() - meta_pieces_.size() - required_chars_.size();
    

    // We may see duplicated pieces that are extracted with different path.
    // In real segmentation phase, we can consider them as one symbol.
    // e.g., "aaa" => "aa" 
    std::unordered_set<std::string> dup;

    // Main loop
    while (vocab_size > 0 && pieces_.size() < static_cast<size_t>(vocab_size)) {
        constexpr int kUpdataActiveSymbolsInterval = 100;
        if (pieces_.size() % kUpdataActiveSymbolsInterval == 0) {
            UpdateActiveSymbols();
        }

        // Scanning active symbols, finds the best_symbol with highest freq.
        Symbol *best_symbol = nullptr;
        for (auto& it : active_symbols_) {
            Symbol* symbol = it;
            ComputeFreq(symbol);
            // If the frequency is the same, take shorter symbol.
            // if the length is the same, use lexicographical comparison
            if (best_symbol == nullptr ||
                (symbol->freq > best_symbol->freq ||
                (symbol->freq == best_symbol->freq &&
                (symbol->chars.size() < best_symbol->chars.size() ||
                (symbol->chars.size() == best_symbol->chars.size() &&
                 symbol->ToString() < best_symbol->ToString()))))) {
                best_symbol = symbol;
            }
        }

        if (best_symbol == nullptr) {
            LOG(WARNING) << "No valid symbol found";
            break;
        }
        
        if (!dup.insert(best_symbol->ToString()).second) {
            // Removes best_symbol so it is not selected again.
            symbols_cache_.erase(best_symbol->fp);
            active_symbols_.erase(best_symbol);
            continue;
        }

        // Stores the best_symbol in the final output.
        pieces_.emplace_back(best_symbol->ToString(),
                                   -static_cast<float>(pieces_.size()));                            
        
        if (pieces_.size() % 20 == 0) {
            LOG(INFO) << "Added: freq=" << best_symbol->freq
                << " size=" << pieces_.size()
                << " all=" << symbols_cache_.size()
                << " active=" << active_symbols_.size()
                << " piece=" << best_symbol->ToString()
                << " target=" << vocab_size;
        }

        // Add new bigrams which are created after symbol replacement.
        // We do not need to scan all characters, but scan the neighbors 
        // in best_symbol.
        for (const uint64_t &encoded_pos : best_symbol->positions) {
            const Position pos = DecodePos(encoded_pos);

            if (symbols_[pos.sid][pos.left] == nullptr) {
                // left index might be NULL (set in the previous iteration)
                // when left_symbol == right_symbol
                continue;
            }

            // We have three bigrams [prev, left], [left, right], [right, next],
            // which are affected with this symbol replacement.
            const int prev = GetPrevIndex(pos.sid, pos.left);
            const int next = GetNextIndex(pos.sid, pos.right);

            // Resets the frequencies of bigrams [prev, left] and [right, next].
            ResetFreq(pos.sid, prev, pos.left, best_symbol);
            ResetFreq(pos.sid, pos.right, next, best_symbol);

            // Merges two symbols
            symbols_[pos.sid][pos.left] = best_symbol;
            symbols_[pos.sid][pos.right] = nullptr;

            // Makes new symbol bigrams [prev, left] and [left, next].
            AddNewPair(pos.sid, prev, pos.left);
            AddNewPair(pos.sid, pos.left, next);
        }

        // Removes best_symbol so it is not selected again.
        symbols_cache_.erase(best_symbol->fp);
        active_symbols_.erase(best_symbol);
    } // end of main loop
    
    // Adds required_chars_
    for (const auto &w : misc::Sorted(required_chars_)) {
        const Symbol *symbol = GetCharSymbol(w.first);
        pieces_.emplace_back(symbol->ToString(),
                               -static_cast<float>(pieces_.size()));
    }    

    misc::STLDeleteElements(&allocated_);

    return true;
}

bool NewCounter::Serialize(Model* model) const {

    model->Clear();

    size_t fid = 0;
    for (int id = 0; id < counter_spec_.vocab_size(); ++id) {
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

    model->SetCounterSpec(counter_spec_);
    model->SetNormalizerSpec(normalizer_spec_);

    return true;
}

bool NewCounter::Save() const {
    std::string filename = counter_spec_.model_prefix()+".model";
    LOG(INFO) << "Saving model: " << filename;
    Model model;
    if (!Serialize(&model)) return false;
    auto output = NewWritableFile(filename);
    output->Write(model.AsStr());
    return true;
}


} // namespace piece
