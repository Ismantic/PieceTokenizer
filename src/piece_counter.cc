#include "piece_counter.h"

#include <algorithm>

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
  SplitSentencesByWhitespace();

  // Build per-sentence token linked lists.
  token_lists_.reserve(sentences_.size());
  for (const auto& s : sentences_)
    token_lists_.push_back(BuildTokenList(s.first));

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
    std::vector<size_t> indices = std::move(it->second);
    pair_index.erase(it);

    // Deduplicate (index may accumulate duplicates).
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

    for (size_t j : indices)
      MergeSentence(top, new_id, token_lists_[j],
                    sentences_[j].second, stats, j, pair_index);

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
  LOG(INFO) << "Loading sentences ...";
  auto iter = std::make_unique<MultiFileSentenceIterator>(
      std::vector<std::string>(counter_spec_.input().begin(),
                               counter_spec_.input().end()));
  for (; !iter->done(); iter->Next()) {
    if (!iter->value().empty())
      sentences_.emplace_back(iter->value(), 1);
  }

  LOG(INFO) << "Normalizing sentences ...";
  const Normalizer normalizer(normalizer_spec_);
  const int num_threads = counter_spec_.cpu_count();
  if (num_threads > 1 && sentences_.size() > 256) {
    std::vector<std::thread> threads;
    for (int n = 0; n < num_threads; ++n) {
      threads.emplace_back([&, n]() {
        for (size_t i = n; i < sentences_.size(); i += num_threads)
          sentences_[i].first = normalizer.Normalize(sentences_[i].first);
      });
    }
    for (auto& t : threads) t.join();
  } else {
    for (auto& [text, freq] : sentences_)
      text = normalizer.Normalize(text);
  }

  LOG(INFO) << "Done! preprocessed " << sentences_.size() << " sentences.";
  return true;
}

void PieceCounter::SplitSentencesByWhitespace() {
  LOG(INFO) << "Tokenizing input sentences with whitespace: "
            << sentences_.size();
  const std::string_view space = normalizer_spec_.GetSpace();
  std::unordered_map<std::string, int64_t> tokens;
  for (const auto& [text, freq] : sentences_)
    for (const auto& w : ustr::SplitText(text, space))
      tokens[std::string(w)] += freq;
  sentences_ = misc::Sorted(tokens);
  LOG(INFO) << "Done! " << sentences_.size();
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
  const int num_threads = counter_spec_.cpu_count();
  if (num_threads > 1 && sentences_.size() > 256) {
    const int nt = std::min(num_threads,
                            static_cast<int>(sentences_.size()));
    std::vector<DeltaMap> thread_counts(nt);
    std::vector<std::vector<IndexEntry>> thread_indices(nt);
    std::vector<std::thread> threads;
    const size_t chunk = (sentences_.size() + nt - 1) / nt;

    for (int t = 0; t < nt; ++t) {
      const size_t start = t * chunk;
      const size_t end = std::min(start + chunk, sentences_.size());
      if (start >= end) break;
      threads.emplace_back([&, t, start, end]() {
        for (size_t j = start; j < end; ++j) {
          const auto& text = sentences_[j].first;
          const int64_t freq = sentences_[j].second;
          for (size_t i = 0; i + 1 < text.size(); ++i) {
            std::pair<int, int> pair = {
                static_cast<int>(static_cast<uint8_t>(text[i])),
                static_cast<int>(static_cast<uint8_t>(text[i + 1]))};
            thread_counts[t][pair] += freq;
            thread_indices[t].push_back({pair, j});
          }
        }
      });
    }
    for (auto& t : threads) t.join();

    for (const auto& counts : thread_counts)
      for (const auto& [pair, count] : counts)
        stats.Insert(pair, count);
    for (const auto& indices : thread_indices)
      for (const auto& [pair, sid] : indices)
        pair_index[pair].push_back(sid);
  } else {
    for (size_t j = 0; j < sentences_.size(); ++j) {
      const auto& text = sentences_[j].first;
      const int freq = sentences_[j].second;
      for (size_t i = 0; i + 1 < text.size(); ++i) {
        std::pair<int, int> pair = {
            static_cast<int>(static_cast<uint8_t>(text[i])),
            static_cast<int>(static_cast<uint8_t>(text[i + 1]))};
        stats.Insert(pair, freq);
        pair_index[pair].push_back(j);
      }
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
