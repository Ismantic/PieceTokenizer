#include "bytepiece_counter.h"

#include <cmath>
#include <future>

namespace piece {

BytePieceCounter::BytePieceCounter(const CounterSpec& counter_spec,
                                   const NormalizerSpec& normalizer_spec)
    : counter_spec_(counter_spec),
      normalizer_spec_(normalizer_spec) {
  InitMetaPieces();
  InitT();
  N_.resize(max_piece_count_ + 1);
}

BytePieceCounter::~BytePieceCounter() = default;

bool BytePieceCounter::Count() {
  if (!StreamCountRaw()) {
    LOG(ERROR) << "Failed to count raw substrings.";
    return false;
  }
  PruneRaw();

  auto pieces_count = StreamCountPieces();
  auto pruned_pieces = PrunePieces(pieces_count);

  std::vector<std::pair<std::string, int>> sorted_pieces(
      pruned_pieces.begin(), pruned_pieces.end());
  std::sort(sorted_pieces.begin(),
            sorted_pieces.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  pieces_.clear();
  for (const auto& [piece, count] : sorted_pieces) {
    pieces_.emplace_back(piece, count);
    if (pieces_.size() % 100 == 0) {
      LOG(INFO) << "Added piece: " << piece << " count: " << count
                << " total: " << pieces_.size();
    }
  }

  LOG(INFO) << "BytePiece training completed with " << pieces_.size()
            << " pieces";
  return true;
}

bool BytePieceCounter::Save() const {
  std::string filename = counter_spec_.model_prefix() + ".model";
  LOG(INFO) << "Saving model: " << filename;
  Model model;
  if (!Serialize(&model)) return false;
  auto output = NewWritableFile(filename);
  output->Write(model.AsStr());
  return true;
}

bool BytePieceCounter::Serialize(Model* model) const {
  model->Clear();

  size_t total = meta_pieces_.size() + pieces_.size();
  size_t fid = 0;
  for (size_t id = 0; id < total; ++id) {
    const auto it = meta_pieces_.find(id);
    if (it != meta_pieces_.end()) {
      auto* p = model->InsertPieces();
      p->SetPiece(it->second.first);
      p->SetType(it->second.second);
      p->SetScore(0.0);
    } else if (fid < pieces_.size()) {
      const auto& w = pieces_[fid++];
      auto* p = model->InsertPieces();
      p->SetPiece(w.first);
      p->SetScore(w.second);
    }
  }

  CounterSpec spec = counter_spec_;
  spec.set_vocab_size(total);
  model->SetCounterSpec(spec);
  model->SetNormalizerSpec(normalizer_spec_);
  return true;
}

size_t BytePieceCounter::GetWorkerCount(size_t work_items) const {
  if (work_items <= 1) {
    return 1;
  }
  const unsigned int hw = std::thread::hardware_concurrency();
  const size_t max_workers = hw == 0 ? 4 : static_cast<size_t>(hw);
  return std::max<size_t>(1, std::min(work_items, max_workers));
}

bool BytePieceCounter::InitMetaPieces() {
  if (counter_spec_.unk_id() >= 0) {
    meta_pieces_[counter_spec_.unk_id()] =
        std::make_pair(counter_spec_.unk_piece(), Model::Piece::UNKNOWN);
  }
  if (counter_spec_.bos_id() >= 0) {
    meta_pieces_[counter_spec_.bos_id()] =
        std::make_pair(counter_spec_.bos_piece(), Model::Piece::CONTROL);
  }
  if (counter_spec_.eos_id() >= 0) {
    meta_pieces_[counter_spec_.eos_id()] =
        std::make_pair(counter_spec_.eos_piece(), Model::Piece::CONTROL);
  }
  if (counter_spec_.pad_id() >= 0) {
    meta_pieces_[counter_spec_.pad_id()] =
        std::make_pair(counter_spec_.pad_piece(), Model::Piece::CONTROL);
  }

  if (meta_pieces_.size() + 256 > counter_spec_.vocab_size()) {
    LOG(ERROR) << "Vocab size is too small for byte pieces. Need at least "
               << meta_pieces_.size() + 256 << " slots.";
    return false;
  }

  int byte_id = meta_pieces_.size();
  for (int i = 0; i < 256; ++i) {
    std::string byte_piece = ustr::ByteToPiece(i);
    meta_pieces_[byte_id++] = std::make_pair(byte_piece, Model::Piece::BYTE);
  }
  return true;
}

std::unique_ptr<MultiFileSentenceIterator> BytePieceCounter::MakeIterator() const {
  return std::make_unique<MultiFileSentenceIterator>(
      std::vector<std::string>(counter_spec_.input().begin(),
                               counter_spec_.input().end()));
}

bool BytePieceCounter::StreamCountRaw() {
  const size_t workers = std::min<size_t>(std::thread::hardware_concurrency(), 8);
  LOG(INFO) << "Pass 1: counting substrings with " << workers << " workers...";

  using NMaps = std::vector<std::unordered_map<std::string, float_t>>;

  N_.clear();
  N_.resize(max_piece_count_ + 1);

  auto iter = MakeIterator();
  size_t line_count = 0;
  constexpr size_t kBatchLines = 1000000;

  auto fill_batch = [&](std::vector<std::string>& batch) -> size_t {
    batch.clear();
    size_t lines = 0;
    for (; !iter->done() && lines < kBatchLines; iter->Next(), ++lines) {
      const std::string& line = iter->value();
      if (line.empty()) continue;
      for (std::string_view word : ustr::SplitWords(line)) {
        batch.emplace_back(word);
      }
    }
    return lines;
  };

  auto process_batch = [&](std::vector<std::string>& batch) {
    if (batch.empty()) return;
    std::vector<NMaps> per_thread(workers);
    for (auto& nm : per_thread) nm.resize(max_piece_count_ + 1);

    std::vector<std::thread> threads;
    const size_t chunk = (batch.size() + workers - 1) / workers;
    for (size_t w = 0; w < workers; ++w) {
      const size_t begin = w * chunk;
      const size_t end = std::min(batch.size(), begin + chunk);
      if (begin >= end) break;
      threads.emplace_back([&, begin, end, w]() {
        auto& local_N = per_thread[w];
        for (size_t idx = begin; idx < end; ++idx) {
          const auto& text = batch[idx];
          for (size_t i = 0; i < text.length(); ++i) {
            local_N[0][""] += 1;
            std::string piece;
            const size_t max_len = std::min(max_piece_count_, text.length() - i);
            for (size_t j = 1; j <= max_len; ++j) {
              piece.push_back(text[i + j - 1]);
              local_N[j][piece] += 1;
            }
          }
        }
      });
    }
    for (auto& t : threads) t.join();

    for (auto& local_N : per_thread) {
      for (size_t i = 0; i <= max_piece_count_; ++i) {
        for (auto& [k, v] : local_N[i]) {
          N_[i][k] += v;
        }
      }
    }
  };

  std::vector<std::string> work_batch, read_batch;
  line_count += fill_batch(work_batch);

  while (!work_batch.empty()) {
    auto read_future =
        std::async(std::launch::async, [&]() { return fill_batch(read_batch); });
    process_batch(work_batch);
    line_count += read_future.get();
    LOG(INFO) << "Pass 1: " << line_count << " lines";
    std::swap(work_batch, read_batch);
  }

  size_t cnt = 0;
  for (size_t i = 0; i < N_.size(); ++i) cnt += N_[i].size();
  LOG(INFO) << "Pass 1 done: " << cnt << " entries";
  return true;
}

void BytePieceCounter::PruneRaw() {
  LOG(INFO) << "Pruning raw counts...";

  for (int i = 0; i < 256; ++i) {
    std::string byte_str(1, static_cast<char>(i));
    if (N_[1].find(byte_str) == N_[1].end()) {
      N_[1][byte_str] = 1;
      N_[0][""] += 1;
    }
  }

  for (int i = N_.size() - 1; i >= 0; --i) {
    std::unordered_map<std::string, float_t> pruned;
    for (const auto& [k, v] : N_[i]) {
      if (k.length() == i && v >= (i > 1 ? min_count_ : 0)) {
        pruned[k] = std::log(v);
      }
    }

    if (i < static_cast<int>(N_.size()) - 1) {
      std::unordered_map<std::string, float_t> next_pruned;
      for (const auto& [k, v] : N_[i + 1]) {
        std::string prefix = k.substr(0, i);
        auto it = pruned[prefix];
        next_pruned[k] = v - it;
      }
      N_[i + 1] = std::move(next_pruned);
    }

    N_[i] = std::move(pruned);
  }

  int cnt = 0;
  for (size_t i = 0; i < N_.size(); ++i) cnt += N_[i].size();
  LOG(INFO) << "Done pruning raw counts " << cnt;
}

std::vector<std::string> BytePieceCounter::Tokenize(const std::string& text) const {
  const int num = text.length();
  if (num == 0) return {};

  std::vector<std::vector<float_t>> nodes(
      num, std::vector<float_t>(max_piece_count_, -INF));
  std::vector<int> utf8_position(num, 0);

  int i = 0;
  while (i < num) {
    const int char_length = ustr::OneUTF8Size(text.data() + i);
    for (int j = 0; j < char_length && i + j < num; ++j) {
      utf8_position[i + j] = j;
    }
    i += char_length;
  }

  for (int j = 0; j < static_cast<int>(max_piece_count_); ++j) {
    for (int i = j; i < num; ++i) {
      if (j == 0 && utf8_position[i] > 0) {
        continue;
      }
      std::string piece = text.substr(i - j, j + 1);
      if (j + 1 < static_cast<int>(N_.size())) {
        auto it = N_[j + 1].find(piece);
        if (it != N_[j + 1].end()) {
          nodes[i][j] = it->second;
        }
      }
    }
  }

  std::vector<std::vector<int>> routes(
      num - 1, std::vector<int>(max_piece_count_, 0));

  for (i = 1; i < num; ++i) {
    for (int curr_j = 0; curr_j < static_cast<int>(max_piece_count_); ++curr_j) {
      if (curr_j < utf8_position[i]) continue;

      int best_prev_j = -1;
      float_t best_score = -INF;
      for (int prev_j = 0; prev_j < static_cast<int>(max_piece_count_); ++prev_j) {
        if (prev_j < utf8_position[i - 1]) continue;
        if (T_[prev_j][curr_j] == -INF) continue;

        bool skip_ngram_check =
            (prev_j == static_cast<int>(max_piece_count_) - 1 &&
             curr_j == static_cast<int>(max_piece_count_) - 1);
        if (!skip_ngram_check) {
          int ngram_start = i - curr_j;
          if (ngram_start > 0 && utf8_position[ngram_start] > 0) {
            continue;
          }
        }

        float_t score =
            nodes[i - 1][prev_j] + T_[prev_j][curr_j] + nodes[i][curr_j];
        if (score > best_score) {
          best_score = score;
          best_prev_j = prev_j;
        }
      }

      if (best_prev_j != -1) {
        routes[i - 1][curr_j] = best_prev_j;
        nodes[i][curr_j] = best_score;
      } else {
        nodes[i][curr_j] = -INF;
      }
    }
  }

  int best_last_state = 0;
  float_t best_score = -INF;
  for (int j = 0; j < static_cast<int>(max_piece_count_); ++j) {
    if (j >= utf8_position[num - 1] && nodes[num - 1][j] > best_score) {
      best_score = nodes[num - 1][j];
      best_last_state = j;
    }
  }

  std::vector<int> opt_route(num);
  int curr_pos = num - 1;
  int curr_state = best_last_state;
  while (curr_pos >= 0) {
    opt_route[curr_pos] = curr_state;
    if (curr_pos > 0) {
      curr_state = routes[curr_pos - 1][curr_state];
      curr_pos--;
    } else {
      break;
    }
  }

  std::vector<int> split_points;
  split_points.push_back(0);
  for (i = 1; i < static_cast<int>(opt_route.size()); ++i) {
    if (opt_route[i] == 0 && utf8_position[i] == 0) {
      split_points.push_back(i);
    }
  }
  split_points.push_back(num);

  std::vector<std::string> tokens;
  for (size_t i = 0; i + 1 < split_points.size(); ++i) {
    tokens.push_back(
        text.substr(split_points[i], split_points[i + 1] - split_points[i]));
  }
  return tokens;
}

BytePieceCounter::Str2Int BytePieceCounter::StreamCountPieces() {
  LOG(INFO) << "Pass 2: counting pieces...";
  Str2Int total_pieces;

  auto iter = MakeIterator();
  size_t line_count = 0;
  std::vector<std::string> buffer;
  const size_t flush_size = 50000;

  auto flush = [&]() {
    if (buffer.empty()) return;
    Str2Int batch = ParallelBatchCount<Str2Int>(
        buffer.size(), 2048, [&](size_t begin, size_t end) {
          Str2Int counts;
          for (size_t i = begin; i < end; ++i) {
            for (const auto& piece : Tokenize(buffer[i])) {
              if (!piece.empty()) counts[piece] += 1;
            }
          }
          return counts;
        });
    MergeCounts(&total_pieces, batch);
    buffer.clear();
  };

  for (; !iter->done(); iter->Next()) {
    const std::string& line = iter->value();
    if (line.empty()) continue;
    for (std::string_view word : ustr::SplitWords(line)) {
      buffer.emplace_back(word);
    }
    if (buffer.size() >= flush_size) {
      flush();
    }
    if (++line_count % 1000000 == 0) {
      LOG(INFO) << "Pass 2: " << line_count << " lines, "
                << total_pieces.size() << " pieces";
    }
  }
  flush();

  LOG(INFO) << "Pass 2 done: " << total_pieces.size() << " unique pieces";
  return total_pieces;
}

BytePieceCounter::Str2Int BytePieceCounter::SplitPieces(const Str2Int& keep,
                                                        const Str2Int& drop) {
  std::unordered_map<std::string, float_t> dict;
  for (const auto& p : keep) {
    dict.emplace(p.first, static_cast<float_t>(p.second));
  }

  BytePieceTokenizer tokenizer(dict);
  std::vector<std::pair<std::string, int>> drop_entries(drop.begin(), drop.end());
  const size_t batch_size = 1024;
  Str2Int counter = ParallelBatchCount<Str2Int>(
      drop_entries.size(), batch_size, [&](size_t begin, size_t end) {
        Str2Int batch_counts;
        for (size_t idx = begin; idx < end; ++idx) {
          const auto& [str, cnt] = drop_entries[idx];
          for (const auto& token : tokenizer.Tokenize(str)) {
            batch_counts[token] += cnt;
          }
        }
        return batch_counts;
      });
  return counter;
}

BytePieceCounter::Str2Int BytePieceCounter::PrunePieces(Str2Int& pieces) {
  LOG(INFO) << "Pruning pieces...";

  Str2Int keep, drop;
  for (const auto& [str, cnt] : pieces) {
    if (str.length() == 1 ||
        (str.length() <= max_piece_size_ && cnt >= min_count_)) {
      keep[str] = cnt;
    } else {
      drop[str] = cnt;
    }
  }

  auto new_counter = SplitPieces(keep, drop);
  for (const auto& [str, cnt] : new_counter) {
    keep[str] += cnt;
  }

  while (true) {
    size_t n = keep.size();
    auto entire_keep_as_drop = keep;
    keep = SplitPieces(keep, entire_keep_as_drop);
    if (keep.size() == n) {
      break;
    }
  }

  int vocab_size = counter_spec_.vocab_size();
  if (keep.size() <= vocab_size - meta_pieces_.size()) {
    LOG(INFO) << "Final pieces count: " << keep.size();
    return keep;
  }

  std::vector<std::pair<std::string, int>> pieces_vec(keep.begin(), keep.end());
  std::sort(pieces_vec.begin(), pieces_vec.end(), [](const auto& a, const auto& b) {
    bool a_num = a.first.length() > 1;
    bool b_num = b.first.length() > 1;
    if (a_num != b_num) return a_num < b_num;
    if (a.second != b.second) return a.second > b.second;
    if (a.first.length() != b.first.length()) {
      return a.first.length() > b.first.length();
    }
    return a.first < b.first;
  });

  Str2Int new_pieces;
  size_t limit = vocab_size - meta_pieces_.size();
  for (size_t i = 0; i < limit && i < pieces_vec.size(); ++i) {
    new_pieces[pieces_vec[i].first] = pieces_vec[i].second;
  }

  LOG(INFO) << "Final pieces count after pruning: " << new_pieces.size();
  return new_pieces;
}

void BytePieceCounter::InitT() {
  int num_ = max_piece_count_;
  T_.resize(num_, std::vector<float_t>(num_, -INF));
  for (int i = 0; i < num_; ++i) {
    T_[i][0] = 0;
    if (i + 1 < num_) {
      T_[i][i + 1] = 0;
    }
    if (i == num_ - 1) {
      T_[i][i] = 0;
    }
  }
}

}  // namespace piece
