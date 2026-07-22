#pragma once

#include <string>
#include <vector>

#include "piece_spec.h"

namespace piece {

// Inserts `tokens` into `model` as CONTROL pieces (score 0) at the end of the
// vocabulary — the single owner of "extra token" semantics, shared by the
// training counters (via extra_tokens in the spec) and the `insert-tokens`
// subcommand.
//
// - Skips any token whose string already exists in the vocab (dedup).
// - Syncs CounterSpec.vocab_size to the final piece count.
// - When `repoint_pad` is true, pad_id < 0, and the CounterSpec's pad_piece is
//   present in the vocab (just-added or pre-existing), repoints pad_id to it.
//
// Returns the number of pieces actually inserted.
int InsertExtraTokens(Model* model, const std::vector<std::string>& tokens,
                      bool repoint_pad = true);

}  // namespace piece
