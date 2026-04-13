#include "sentencepiece_counter.h"

#include <memory>
#include <thread>

#include "common.h"
#include "normalizer.h"

namespace piece {

SentencePieceCounter::SentencePieceCounter(const CounterSpec &counter_spec,
                 const NormalizerSpec &normalizer_spec)
    : counter_spec_(counter_spec),
      normalizer_spec_(normalizer_spec) {
    InitMetaPieces();
}

SentencePieceCounter::~SentencePieceCounter() = default;

bool SentencePieceCounter::InitMetaPieces() {
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
        LOG(ERROR) << "Vocabulary size too small for byte_fallback. Need at least "
                   << (byte_id + 256) << " slots.";
        return false;
    }

    for (int i = 0; i < 256; ++i) {
        std::string byte_piece = ustr::ByteToPiece(i);
        meta_pieces_[byte_id++] = std::make_pair(byte_piece, Model::Piece::BYTE);
    }

    return true;
}

bool SentencePieceCounter::LoadSentences() {
    const uint32_t UNK = counter_spec_.GetUnkUnicode();
    const Normalizer normalizer(normalizer_spec_);
    const std::string_view space = normalizer_spec_.GetSpace();
    const int cut = normalizer_spec_.GetCut();

    // Batch-read + parallel normalize/split + merge into global map.
    LOG(INFO) << "Loading and tokenizing sentences ...";
    const int num_threads = counter_spec_.cpu_count();
    constexpr size_t kBatchSize = 1000000;
    std::unordered_map<std::string, int64_t> tokens;
    std::vector<std::string> batch;
    batch.reserve(kBatchSize);
    int64_t line_count = 0;

    auto iter = std::make_unique<MultiFileSentenceIterator>(
        std::vector<std::string>(counter_spec_.input().begin(),
                                 counter_spec_.input().end()));

    auto process_batch = [&]() {
        if (batch.empty()) return;
        if (num_threads <= 1 || batch.size() < 256) {
            for (const auto& line : batch) {
                std::string normalized = normalizer.Normalize(line);
                for (const auto& w : ustr::SplitText(normalized, space, cut))
                    tokens[std::string(w)] += 1;
            }
        } else {
            std::vector<std::unordered_map<std::string, int64_t>>
                local_maps(num_threads);
            std::vector<std::thread> threads;
            for (int t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    for (size_t i = t; i < batch.size(); i += num_threads) {
                        std::string normalized =
                            normalizer.Normalize(batch[i]);
                        for (const auto& w :
                             ustr::SplitText(normalized, space, cut))
                            local_maps[t][std::string(w)] += 1;
                    }
                });
            }
            for (auto& t : threads) t.join();
            for (const auto& lm : local_maps)
                for (const auto& [k, v] : lm) tokens[k] += v;
        }
        batch.clear();
    };

    const int32_t max_s = counter_spec_.max_sentences();
    for (; !iter->done(); iter->Next()) {
        if (max_s > 0 && line_count >= max_s) break;
        const std::string& sentence = iter->value();
        if (sentence.empty()) continue;
        batch.push_back(sentence);
        ++line_count;
        if (batch.size() >= kBatchSize) {
            process_batch();
            LOG(INFO) << "  " << line_count << " lines, "
                      << tokens.size() << " unique tokens";
        }
    }
    process_batch();

    sentences_ = misc::Sorted(tokens);
    { decltype(tokens) tmp; tokens.swap(tmp); }  // free map memory

    // Filter out tokens below min_count.
    const int32_t min_count = counter_spec_.min_count();
    if (min_count > 1) {
        size_t old_size = sentences_.size();
        sentences_.erase(
            std::remove_if(sentences_.begin(), sentences_.end(),
                           [min_count](const Sentence& s) {
                             return s.second < min_count;
                           }),
            sentences_.end());
        sentences_.shrink_to_fit();
        if (sentences_.size() < old_size)
            LOG(INFO) << "Filtered by min_count=" << min_count << ": "
                      << old_size << " -> " << sentences_.size();
    }

    LOG(INFO) << "Done! " << line_count << " lines -> "
              << sentences_.size() << " unique tokens";

    // Count character frequencies (weighted by token frequency).
    int64_t all_chars_count = 0;
    std::unordered_map<uint32_t, int64_t> chars_count;
    for (const auto& [text, freq] : sentences_) {
        for (const uint32_t c : ustr::UTF8ToUnicodeText(text)) {
            if (!ustr::IsValidCodepoint(c)) continue;
            chars_count[c] += freq;
            all_chars_count += freq;
        }
    }
    LOG(INFO) << "all chars count=" << all_chars_count;

    // Determine required_chars by character coverage.
    int64_t accumulated_chars_count = 0;
    for (const auto& w : misc::Sorted(chars_count)) {
        const float coverage = 1.0 * accumulated_chars_count / all_chars_count;
        if (coverage >= counter_spec_.character_coverage()) {
            LOG(INFO) << "Done: " << 100.0 * coverage
                      << "% characters are covered.";
            break;
        }
        accumulated_chars_count += w.second;
        required_chars_.emplace(w.first, w.second);
    }
    LOG(INFO) << "Alphabase size=" << required_chars_.size();
    LOG(INFO) << "Final character coverage="
              << 1.0 * accumulated_chars_count / all_chars_count;
    if (misc::ContainsKey(required_chars_, UNK))
        return false;

    // Replace rare characters with UNK.
    for (auto& [text, freq] : sentences_) {
        ustr::UnicodeText uw2;
        for (const uint32_t c : ustr::UTF8ToUnicodeText(text)) {
            uw2.push_back(misc::ContainsKey(required_chars_, c) ? c : UNK);
        }
        text = ustr::UnicodeTextToUTF8(uw2);
    }

    // EncodePos uses 16 bits for symbol index; drop tokens exceeding this.
    constexpr size_t kMaxTokenBytes = 65535;
    {
        size_t old_size = sentences_.size();
        sentences_.erase(
            std::remove_if(sentences_.begin(), sentences_.end(),
                           [](const Sentence& s) {
                             return s.first.size() > kMaxTokenBytes;
                           }),
            sentences_.end());
        if (sentences_.size() < old_size)
            LOG(INFO) << "Dropped " << (old_size - sentences_.size())
                      << " tokens exceeding " << kMaxTokenBytes << " bytes";
    }

    LOG(INFO) << "Done! preprocessed " << sentences_.size() << " sentences.";
    return true;
}

SentencePieceCounter::Symbol* SentencePieceCounter::GetCharSymbol(uint32_t c) {
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
  s->byte_size = c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
  s->freq = freq;
  misc::InsertOrDie(&symbols_cache_, s->fp, s);
  return s;
}

SentencePieceCounter::Symbol* SentencePieceCounter::GetPairSymbol(const Symbol* left,
                                                                  const Symbol* right) {
  if (left == nullptr || right == nullptr || left->is_unk || right->is_unk) {
    return nullptr;
  }

  // Enforce max_piece_size (in UTF-8 bytes) to avoid learning huge pieces
  // from noisy repeated-punctuation lines.
  if (left->byte_size + right->byte_size >
      static_cast<size_t>(counter_spec_.max_piece_size())) {
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
  s->byte_size = left->byte_size + right->byte_size;
  misc::InsertOrDie(&symbols_cache_, s->fp, s);
  return s;
}

void SentencePieceCounter::AddNewPair(int sid, int left, int right) {
  if (left == -1 || right == -1) return;
  auto *symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
  if (symbol != nullptr) {
    active_symbols_.insert(symbol);
    symbol->positions.push_back(EncodePos(sid, left, right));
  }
}

void SentencePieceCounter::ComputeFreq(Symbol *symbol) const {
    if (symbol->freq > 0) {
        return;
    }
    size_t write = 0;
    for (size_t read = 0; read < symbol->positions.size(); ++read) {
        const Position pos = DecodePos(symbol->positions[read]);
        if (symbol->left == symbols_[pos.sid][pos.left] &&
            symbol->right == symbols_[pos.sid][pos.right]) {
            symbol->freq += freqs_[pos.sid];
            symbol->positions[write++] = symbol->positions[read];
        }
    }
    symbol->positions.resize(write);
}

void SentencePieceCounter::UpdateActiveSymbols() {
    std::vector<Symbol*> symbols;
    for (auto &it : symbols_cache_) {
        Symbol* symbol = it.second;
        if (symbol->IsBigram()) {
            symbols.push_back(symbol);
        }
    }

    // Parallel ComputeFreq: each symbol is independent
    const int num_threads = counter_spec_.cpu_count();
    if (num_threads > 1 && symbols.size() > 256) {
        std::vector<std::thread> threads;
        for (int n = 0; n < num_threads; ++n) {
            threads.emplace_back([&, n]() {
                for (size_t i = n; i < symbols.size();
                     i += num_threads) {
                    ComputeFreq(symbols[i]);
                }
            });
        }
        for (auto& t : threads) t.join();
    } else {
        for (auto* s : symbols) ComputeFreq(s);
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
int SentencePieceCounter::GetPrevIndex(int sid, int index) const {
    for (int i = index-1; i >= 0; --i) {
        if (symbols_[sid][i] == nullptr) continue;
        return i;
    }
    return -1;
}
int SentencePieceCounter::GetNextIndex(int sid, int index) const {
    for (size_t i = index + 1; i < symbols_[sid].size(); ++i) {
        if (symbols_[sid][i] == nullptr) continue;
        return i;
    }
    return -1;
}

void SentencePieceCounter::ResetFreq(int sid, int left, int right, const Symbol* best) {
    if (left == -1 || right == -1)  return;
    auto* symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
    if (symbol != nullptr && symbol != best) {
        symbol->freq = 0;
    }
}

std::string SentencePieceCounter::Symbol::ToString() const {
    return ustr::UnicodeTextToUTF8(chars);
}

bool SentencePieceCounter::Count() {
    symbols_.clear();
    allocated_.clear();
    symbols_cache_.clear();
    active_symbols_.clear();

    LoadSentences();

    const int num_threads = counter_spec_.cpu_count();

    // Initializes symbols_. symbols_[s][i] stores an unary symbol.
    symbols_.resize(sentences_.size());
    for (size_t i = 0; i < sentences_.size(); ++i) {
        for (const uint32_t c : ustr::UTF8ToUnicodeText(sentences_[i].first)) {
            symbols_[i].push_back(GetCharSymbol(c));
        }
    }

    // Extract frequencies and free sentence strings (no longer needed).
    freqs_.resize(sentences_.size());
    for (size_t i = 0; i < sentences_.size(); ++i)
        freqs_[i] = sentences_[i].second;
    { Sentences tmp; sentences_.swap(tmp); }
    LOG(INFO) << "Freed sentence strings, kept " << freqs_.size() << " frequencies";

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

        // Parallel frequency computation, then serial best-symbol selection.
        std::vector<Symbol*> active_vec(active_symbols_.begin(),
                                        active_symbols_.end());
        if (num_threads > 1 && active_vec.size() > 256) {
            std::vector<std::thread> threads;
            for (int n = 0; n < num_threads; ++n) {
                threads.emplace_back([&, n]() {
                    for (size_t i = n; i < active_vec.size();
                         i += num_threads) {
                        ComputeFreq(active_vec[i]);
                    }
                });
            }
            for (auto& t : threads) t.join();
        } else {
            for (auto* s : active_vec) ComputeFreq(s);
        }

        // Serial: find best symbol
        Symbol *best_symbol = nullptr;
        for (auto* symbol : active_vec) {
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

        // Serial merge step.
        for (const uint64_t &encoded_pos : best_symbol->positions) {
            const Position pos = DecodePos(encoded_pos);

            if (symbols_[pos.sid][pos.left] == nullptr) {
                continue;
            }

            const int prev = GetPrevIndex(pos.sid, pos.left);
            const int next = GetNextIndex(pos.sid, pos.right);

            ResetFreq(pos.sid, prev, pos.left, best_symbol);
            ResetFreq(pos.sid, pos.right, next, best_symbol);

            symbols_[pos.sid][pos.left] = best_symbol;
            symbols_[pos.sid][pos.right] = nullptr;

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

bool SentencePieceCounter::Serialize(Model* model) const {

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

bool SentencePieceCounter::Save() const {
    std::string filename = counter_spec_.model_prefix()+".model";
    LOG(INFO) << "Saving model: " << filename;
    Model model;
    if (!Serialize(&model)) return false;
    auto output = NewWritableFile(filename);
    output->Write(model.AsStr());
    return true;
}


} // namespace piece
