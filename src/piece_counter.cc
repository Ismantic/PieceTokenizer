#include "piece_counter.h"

#include <algorithm>

#include "cut.h"

namespace piece {

PieceCounter::PieceCounter(const CounterSpec& counter_spec,
                           const NormalizerSpec& normalizer_spec)
    : counter_spec_(counter_spec),
      normalizer_spec_(normalizer_spec) {
  InitMetaPieces();
}

PieceCounter::~PieceCounter() = default;

// ---------------------------------------------------------------------------
// Count — main BPE training loop
// ---------------------------------------------------------------------------

bool PieceCounter::Count() {
  if (!LoadSentences()) {
    LOG(ERROR) << "Failed to load sentences.";
    return false;
  }

  // Build per-sentence token linked lists.
  token_lists_.reserve(sentences_.size());
  for (const auto& s : sentences_)
    token_lists_.push_back(BuildTokenList(s.first));

  // Extract frequencies and free sentence strings.
  freqs_.resize(sentences_.size());
  for (size_t i = 0; i < sentences_.size(); ++i)
    freqs_[i] = sentences_[i].second;
  { Sentences tmp; sentences_.swap(tmp); }
  LOG(INFO) << "Freed sentence strings, kept " << freqs_.size() << " frequencies";

  // Build initial pair statistics and global pair→sentence index.
  Multiset<std::pair<int, int>> stats;
  PairIndex pair_index;
  InitPairsStatsAndIndex(stats, pair_index);

  const int num_merges = counter_spec_.vocab_size() - meta_pieces_.size();
  LOG(INFO) << "Starting BBPE training with " << num_merges << " merges";

  // Seed byte-level vocabulary (0–255).
  int cnt = 0;
  for (int i = 0; i < 256; i++) {
    std::string t(1, i);
    vocab_[i] = t;
    pieces_.emplace_back(std::vector<std::string>{t, "", ""},
                         -static_cast<float>(i));
    ++cnt;
  }

  while (cnt < num_merges && stats) {
    const auto top = stats.Top();
    const int n = stats.GetCount(top);
    const int new_id = vocab_.size();
    vocab_[new_id] = vocab_[top.first] + vocab_[top.second];
    pieces_.emplace_back(
        std::vector<std::string>{vocab_[new_id],
                                 vocab_[top.first], vocab_[top.second]},
        -static_cast<float>(pieces_.size()));

    // Look up which sentences contain this pair.
    auto it = pair_index.find(top);
    if (it == pair_index.end() || it->second.empty()) {
      stats.Remove(top, n);
      continue;
    }

    // Move out indices and erase — this pair is consumed.
    // Indices may contain duplicates; MergeImpl naturally skips
    // sentences where the pair no longer exists.
    std::vector<size_t> indices = std::move(it->second);
    pair_index.erase(it);

    for (size_t j : indices)
      MergeSentence(top, new_id, token_lists_[j],
                    freqs_[j], stats, j, pair_index);

    if (cnt % 10 == 0)
      LOG(INFO) << "Merge " << cnt + 1 << "/" << num_merges
                << ": (" << top.first << "," << top.second
                << ") -> " << new_id << " (" << Escape(vocab_[new_id])
                << ") had " << n << " occurrences"
                << " in " << indices.size() << " sentences";
    ++cnt;
  }
  LOG(INFO) << "Done! " << cnt << " merges";

  for (auto* head : token_lists_) FreeTokenList(head);
  token_lists_.clear();
  return true;
}

// ---------------------------------------------------------------------------
// Save / Serialize
// ---------------------------------------------------------------------------

bool PieceCounter::Save() const {
  const std::string filename = counter_spec_.model_prefix() + ".model";
  LOG(INFO) << "Saving model: " << filename;
  Model model;
  if (!Serialize(&model)) return false;
  auto output = NewWritableFile(filename);
  output->Write(model.AsStr());
  return true;
}

bool PieceCounter::Serialize(Model* model) const {
  model->Clear();
  size_t fid = 0;
  for (int id = 0; id < counter_spec_.vocab_size(); ++id) {
    const auto it = meta_pieces_.find(id);
    if (it != meta_pieces_.end()) {
      auto* p = model->InsertPieces();
      p->SetPiece(it->second.first);
      p->SetType(it->second.second);
      p->SetScore(0.0);
    } else if (fid < pieces_.size()) {
      const auto& [w, s] = pieces_[fid++];
      auto* piece = model->InsertPieces();
      piece->SetPiece(w[0], w[1], w[2]);
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

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

bool PieceCounter::InitMetaPieces() {
  auto add = [&](int id, const std::string& piece, Model::Piece::Type type) {
    if (id >= 0) meta_pieces_[id] = {piece, type};
  };
  add(counter_spec_.unk_id(), counter_spec_.unk_piece(), Model::Piece::UNKNOWN);
  add(counter_spec_.bos_id(), counter_spec_.bos_piece(), Model::Piece::CONTROL);
  add(counter_spec_.eos_id(), counter_spec_.eos_piece(), Model::Piece::CONTROL);
  add(counter_spec_.pad_id(), counter_spec_.pad_piece(), Model::Piece::CONTROL);

  if (static_cast<int>(meta_pieces_.size()) + 256 > counter_spec_.vocab_size()) {
    LOG(ERROR) << "Vocabulary size too small for byte_fallback. Need at least "
               << (meta_pieces_.size() + 256) << " slots.";
    return false;
  }
  return true;
}

bool PieceCounter::LoadSentences() {
  const Normalizer normalizer(normalizer_spec_);
  const std::string_view space = normalizer_spec_.GetSpace();
  const int num_threads = counter_spec_.cpu_count();
  constexpr size_t kBatchSize = 1000000;

  // Optional cn-mode Unigram cutter for Han runs.
  std::unique_ptr<CnCutter> cn_cutter;
  ustr::CnCutFn cn_cut_fn;
  if (!counter_spec_.cn_dict().empty()) {
    auto dict = LoadCnDict(counter_spec_.cn_dict());
    if (dict.empty()) {
      LOG(ERROR) << "cn dict is empty: " << counter_spec_.cn_dict();
      return false;
    }
    cn_cutter = std::make_unique<CnCutter>(dict);
    cn_cut_fn = [cutter = cn_cutter.get()](std::string_view s) {
      return cutter->Cut(s);
    };
    LOG(INFO) << "cn mode enabled";
  }

  LOG(INFO) << "Loading and tokenizing sentences ...";
  std::unordered_map<std::string, int64_t> tokens;
  std::vector<std::string> batch;
  batch.reserve(kBatchSize);
  int64_t line_count = 0;

  auto iter = std::make_unique<MultiFileSentenceIterator>(
      std::vector<std::string>(counter_spec_.input().begin(),
                               counter_spec_.input().end()));

  auto split_one = [&](const std::string& line,
                       std::unordered_map<std::string, int64_t>& sink) {
    const std::string normalized = normalizer.Normalize(line);
    if (cn_cutter) {
      for (auto& w : ustr::SplitTextCn(normalized, space, cn_cut_fn))
        sink[std::move(w)] += 1;
    } else {
      for (const auto& w : ustr::SplitText(normalized, space))
        sink[std::string(w)] += 1;
    }
  };

  auto process_batch = [&]() {
    if (batch.empty()) return;
    if (num_threads <= 1 || batch.size() < 256) {
      for (const auto& line : batch) split_one(line, tokens);
    } else {
      std::vector<std::unordered_map<std::string, int64_t>>
          local_maps(num_threads);
      std::vector<std::thread> threads;
      for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
          for (size_t i = t; i < batch.size(); i += num_threads)
            split_one(batch[i], local_maps[t]);
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
  return true;
}

// ---------------------------------------------------------------------------
// Token linked list helpers
// ---------------------------------------------------------------------------

PieceCounter::Token* PieceCounter::BuildTokenList(const std::string& text) {
  if (text.empty()) return nullptr;
  auto* head = new Token{static_cast<int>(static_cast<uint8_t>(text[0])),
                         nullptr, nullptr};
  Token* prev = head;
  for (size_t i = 1; i < text.size(); ++i) {
    auto* node = new Token{static_cast<int>(static_cast<uint8_t>(text[i])),
                           prev, nullptr};
    prev->next = node;
    prev = node;
  }
  return head;
}

void PieceCounter::FreeTokenList(Token* head) {
  while (head) {
    Token* next = head->next;
    delete head;
    head = next;
  }
}

// ---------------------------------------------------------------------------
// InitPairsStatsAndIndex
// ---------------------------------------------------------------------------

void PieceCounter::InitPairsStatsAndIndex(
    Multiset<std::pair<int, int>>& stats,
    PairIndex& pair_index) {
  for (size_t j = 0; j < token_lists_.size(); ++j) {
    const int64_t freq = freqs_[j];
    for (Token* node = token_lists_[j]; node && node->next; node = node->next) {
      std::pair<int, int> pair = {node->value, node->next->value};
      stats.Insert(pair, freq);
      pair_index[pair].push_back(j);
    }
  }
}

// ---------------------------------------------------------------------------
// Merge helpers
// ---------------------------------------------------------------------------

// Helper: apply one merge in a sentence, updating neighboring pairs.
// `on_remove(old_pair)` and `on_add(new_pair)` are called for stats/index.
template <typename OnRemove, typename OnAdd>
static void MergeImpl(const std::pair<int, int>& pair, int new_id,
                      PieceCounter::Token* head, int64_t freq,
                      OnRemove on_remove, OnAdd on_add) {
  using Token = PieceCounter::Token;
  std::vector<Token*> pending_delete;

  for (auto* node = head; node && node->next; node = node->next) {
    if (node->value != pair.first || node->next->value != pair.second)
      continue;

    on_remove(pair, freq);
    if (node->next->next) {
      on_remove({node->next->value, node->next->next->value}, freq);
      on_add({new_id, node->next->next->value}, freq);
    }
    if (node->prev) {
      on_remove({node->prev->value, pair.first}, freq);
      on_add({node->prev->value, new_id}, freq);
    }

    auto* remove = node->next;
    if (remove->next) remove->next->prev = node;
    node->next = remove->next;
    pending_delete.push_back(remove);
    node->value = new_id;
  }

  for (auto* p : pending_delete) delete p;
}

void PieceCounter::MergeSentence(
    const std::pair<int, int>& pair, int new_id,
    Token* head, int64_t freq,
    Multiset<std::pair<int, int>>& stats,
    size_t sentence_idx, PairIndex& pair_index) {
  MergeImpl(pair, new_id, head, freq,
    [&](const std::pair<int, int>& p, int64_t f) {
      stats.Remove(p, f);
    },
    [&](const std::pair<int, int>& p, int64_t f) {
      stats.Insert(p, f);
      pair_index[p].push_back(sentence_idx);
    });
}


}  // namespace piece
