#include <cstring>
#include <charconv>
#include <climits>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "piece_spec.h"
#include "naive_counter.h"
#include "naive_tokenizer.h"
#include "piece_counter.h"
#include "piece_tokenizer.h"
#include "sentencepiece_counter.h"
#include "sentencepiece_tokenizer.h"
#include "bytepiece_counter.h"
#include "bytepiece_tokenizer.h"
#include "normalizer.h"
#include "pretokenizer.h"
#include "extra_tokens.h"

namespace piece {

// Parse a comma-separated --extra-tokens value. Tolerates a stray pair of
// surrounding shell quotes around the whole value (e.g. Makefile double-quoting
// like --extra-tokens '"<pad>,<user>"') so the quote chars don't leak into the
// token strings. Skips empty entries.
std::vector<std::string> ParseExtraTokens(const char* arg) {
    std::string val(arg);
    if (val.size() >= 2 && (val.front() == '"' || val.front() == '\'') &&
        val.back() == val.front()) {
        val = val.substr(1, val.size() - 2);
    }
    std::vector<std::string> out;
    std::istringstream ts(val);
    std::string token;
    while (std::getline(ts, token, ',')) {
        if (!token.empty()) out.push_back(token);
    }
    return out;
}

struct PreTokenizerOptions {
    std::string normalizer = "no";
    int cut = 0;
    bool split_digits = false;
    bool reconstruct = false;
    std::string dict;
};

enum class ParseResult { kHandled, kUnknown, kError };

bool ParseIntegerOption(int argc, char* argv[], int* index, int minimum,
                        int maximum, int* output) {
    const char* option = argv[*index];
    if (*index + 1 >= argc) {
        std::cerr << "Error: " << option << " requires a value\n";
        return false;
    }
    const std::string_view value = argv[++*index];
    int parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc() || result.ptr != value.data() + value.size() ||
        parsed < minimum || parsed > maximum) {
        std::cerr << "Error: " << option << " must be an integer in ["
                  << minimum << ", " << maximum << "]\n";
        return false;
    }
    *output = parsed;
    return true;
}

ParseResult ParsePreTokenizerOption(int argc, char* argv[], int* index,
                                    PreTokenizerOptions* options) {
    const std::string option = argv[*index];
    auto next_value = [&]() -> const char* {
        if (*index + 1 >= argc) {
            std::cerr << "Error: " << option << " requires a value\n";
            return nullptr;
        }
        return argv[++*index];
    };

    if (option == "--reconstruct") {
        options->reconstruct = true;
        return ParseResult::kHandled;
    }
    if (option != "--normalize" && option != "--split" &&
        option != "--digit" && option != "--cut" && option != "--dict" &&
        option != "--cn-dict") {
        return ParseResult::kUnknown;
    }

    const char* value = next_value();
    if (!value) return ParseResult::kError;
    if (option == "--normalize") {
        options->normalizer = value;
    } else if (option == "--split") {
        if (std::strcmp(value, "word") != 0 &&
            std::strcmp(value, "isolate") != 0) {
            std::cerr << "Error: --split must be word or isolate\n";
            return ParseResult::kError;
        }
        options->cut = std::strcmp(value, "isolate") == 0 ? 1 : 0;
    } else if (option == "--digit") {
        if (std::strcmp(value, "keep") != 0 &&
            std::strcmp(value, "split") != 0) {
            std::cerr << "Error: --digit must be keep or split\n";
            return ParseResult::kError;
        }
        options->split_digits = std::strcmp(value, "split") == 0;
    } else if (option == "--cut") {
        if (std::strcmp(value, "0") != 0 && std::strcmp(value, "1") != 0) {
            std::cerr << "Error: --cut must be 0 or 1\n";
            return ParseResult::kError;
        }
        options->cut = value[0] - '0';
    } else {
        options->dict = value;
    }
    return ParseResult::kHandled;
}

PreTokenizerSpec MakePreTokenizerSpec(const PreTokenizerOptions& options) {
    PreTokenizerSpec spec;
    spec.SetName(options.normalizer);
    spec.SetCut(options.cut);
    spec.SetSplitDigits(options.split_digits);
    spec.SetReconstruct(options.reconstruct);
    return spec;
}

template <typename Counter, typename... Args>
bool CountAndSave(Args&&... args) {
    Counter counter(std::forward<Args>(args)...);
    return counter.Count() && counter.Save();
}

void PrintUsage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " count [options]\n"
              << "  " << prog << " pretokenize [options]\n"
              << "  " << prog << " raw-count [options]\n"
              << "  " << prog << " tokenize --model <file>\n"
              << "  " << prog << " encode --model <file>\n"
              << "  " << prog << " decode --model <file>\n"
              << "  " << prog << " insert-tokens --model <file> --extra-tokens <a,b,c> --output <file>\n"
              << "\nCount options:\n"
              << "  --method <naive|piece|sentencepiece|bytepiece>  (default: bytepiece)\n"
              << "  --input <file>         Input corpus file\n"
              << "  --model <prefix>       Model output prefix (default: tokenizer)\n"
              << "  --vocab-size <int>     Vocabulary size (default: 8000)\n"
              << "  --normalize <name>     Normalizer: no|NMT_NFKC (default: no)\n"
              << "  --cpu <int>            Number of threads (default: 4)\n"
              << "  --max-sentences <int>  Max input lines to load (default: 0=unlimited)\n"
              << "  --min-count <int>      Discard tokens with freq < this (default: 32)\n"
              << "  --cut <0|1>            Pre-tokenize mode: 0=default, 1=split spaces/punct independently\n"
              << "  --reconstruct          Preserve all spaces (no stripping/merging)\n"
              << "  --max-piece-size <int> Max bytes per learned piece (default: 18, ~6 CJK chars)\n"
              << "  --dict <file>          Enable CN mode for `piece`/`sentencepiece` using\n"
              << "                         a TSV (word\\tfreq) Unigram dictionary\n"
              << "\nPretokenize/Raw-count options (PreTokenizer 3 axes):\n"
              << "  --normalize <name>     Normalizer: no|NMT_NFKC|NFKC_CF (default: no)\n"
              << "  --split <word|isolate> Split axis: word=GPT-4-style attach, isolate=spaces/punct standalone (default: word)\n"
              << "  --digit <keep|split>   Digit axis: keep runs, or split per-codepoint (default: keep)\n"
              << "  --dict <no|file>       Cn axis: no=per-char, TSV(word\\tfreq)=dict, omitted=none\n"
              << "  --cut <0|1>            Alias for --split (0=word, 1=isolate)\n"
              << "  --reconstruct          Preserve all spaces (no stripping/merging)\n"
              << "  --input <file>         Input file (repeatable for raw-count)\n"
              << "  --output <file>        Output file (raw-count only, default: stdout)\n"
              << "  --max-piece-size <int> Skip tokens exceeding this many bytes (raw-count, default: 0=unlimited)\n"
              << "\nTokenize/Encode options:\n"
              << "  --model <file>         Model file to load\n"
              << "  --input <file>         Read input from file instead of stdin\n"
              << "  --dict <file>          Enable CN mode (must match training)\n"
              << "\nTokenize/Encode/Decode read from stdin, write to stdout.\n"
              << "Tokenize outputs space-separated pieces per line.\n"
              << "Encode outputs one token per line (piece TAB id).\n"
              << "Decode reads token ids (space-separated) and outputs text.\n";
}

bool RunCount(const std::string& method,
              const std::vector<std::string>& inputs,
              const std::string& model_prefix, int vocab_size,
              const std::string& normalizer_name, int cpu_count,
              int max_sentences, int min_count, int max_piece_size,
              const std::string& dict, int cut, bool reconstruct,
              const std::vector<std::string>& extra_tokens = {}) {
    CounterSpec counter_spec;
    for (const auto& f : inputs) counter_spec.add_input(f);
    counter_spec.set_model_prefix(model_prefix);
    counter_spec.set_method(method);
    counter_spec.set_cpu_count(cpu_count);
    counter_spec.set_max_sentences(max_sentences);
    counter_spec.set_min_count(min_count);
    counter_spec.set_max_piece_size(max_piece_size);
    counter_spec.set_dict(dict);

    for (const auto& t : extra_tokens) counter_spec.add_extra_token(t);

    if (!dict.empty() && method != "piece" && method != "sentencepiece") {
        std::cerr << "Warning: --dict is only supported for --method piece "
                  << "or --method sentencepiece; ignoring for method=" << method << "\n";
        counter_spec.set_dict("");
    }

    PreTokenizerOptions options;
    options.normalizer = normalizer_name;
    options.cut = cut;
    options.reconstruct = reconstruct;
    options.dict = dict;
    PreTokenizerSpec pretokenizer_spec = MakePreTokenizerSpec(options);

    // Single owner of the char-mode policy: dict="no" forces cut=1 +
    // split_digits=true (= Split isolate + Digit split) for every method
    // that supports cn mode, so digits/punct/symbols stay single codepoints
    // and only ASCII-letter runs go through BPE. Persisted to the model so
    // inference matches training.
    if (counter_spec.dict() == "no") {
        pretokenizer_spec.SetCut(1);
        pretokenizer_spec.SetSplitDigits(true);
    }

    // Adjust vocab_size for byte tokens and control tokens
    int size = vocab_size;
    if (method == "bytepiece" || method == "sentencepiece") {
        size = vocab_size + 256 + 3;  // +256 byte tokens +3 control tokens
    } else if (method == "piece") {
        size = vocab_size + 3;
    } else if (method == "naive") {
        size = vocab_size;
    }
    counter_spec.set_vocab_size(size);

    for (const auto& f : inputs)
        std::cerr << "Counting: method=" << method << " input=" << f
                  << " vocab_size=" << vocab_size << " model=" << model_prefix << "\n";

    if (method == "naive") {
        if (!CountAndSave<NaiveCounter>(counter_spec)) return false;
    } else if (method == "piece") {
        if (!CountAndSave<PieceCounter>(counter_spec, pretokenizer_spec)) return false;
    } else if (method == "sentencepiece") {
        if (!CountAndSave<SentencePieceCounter>(counter_spec, pretokenizer_spec))
            return false;
    } else if (method == "bytepiece") {
        if (!CountAndSave<BytePieceCounter>(counter_spec, pretokenizer_spec))
            return false;
    } else {
        std::cerr << "Unknown method: " << method << "\n";
        return false;
    }

    std::cerr << "Model saved to " << model_prefix << ".model\n";
    return true;
}

template <typename Operation>
bool WithTokenizer(const std::string& model_file, const std::string& dict,
                   Operation operation) {
    Model model;
    if (!model.Load(model_file)) {
        std::cerr << "Error: cannot load model: " << model_file << "\n";
        return false;
    }

    const std::string& method = model.GetCounterSpec().method();
    if (!dict.empty() && method != "piece" && method != "sentencepiece") {
        std::cerr << "Warning: --dict is only supported for --method piece "
                  << "or --method sentencepiece; ignoring for method=" << method << "\n";
    }
    if (method == "naive") {
        NaiveTokenizer tokenizer(model);
        operation(tokenizer);
    } else if (method == "piece") {
        PieceTokenizer tokenizer(model, dict);
        if (!tokenizer.valid()) {
            std::cerr << "Error: cannot load dictionary: " << dict << "\n";
            return false;
        }
        operation(tokenizer);
    } else if (method == "sentencepiece") {
        SentencePieceTokenizer tokenizer(model, dict);
        if (!tokenizer.valid()) {
            std::cerr << "Error: cannot load dictionary: " << dict << "\n";
            return false;
        }
        operation(tokenizer);
    } else if (method == "bytepiece") {
        BytePieceTokenizer tokenizer(model);
        operation(tokenizer);
    } else {
        std::cerr << "Unknown method in model: " << method << "\n";
        return false;
    }
    return true;
}

bool RunEncode(const std::string& model_file, const std::string& dict) {
    return WithTokenizer(model_file, dict, [](auto& tokenizer) {
        std::string line;
        while (std::getline(std::cin, line)) {
            for (const auto& token : tokenizer.Encode(line)) {
                std::cout << token.first << "\t" << token.second << "\n";
            }
            std::cout << "\n";
        }
    });
}

bool RunTokenize(const std::string& model_file, const std::string& dict) {
    return WithTokenizer(model_file, dict, [](auto& tokenizer) {
        std::string line;
        while (std::getline(std::cin, line)) {
            auto tokens = tokenizer.Tokenize(line);
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i > 0) std::cout << " ";
                std::cout << Escape(tokens[i]);
            }
            std::cout << "\n";
        }
    });
}

bool RunDecode(const std::string& model_file) {
    return WithTokenizer(model_file, "", [](auto& tokenizer) {
        std::string line;
        while (std::getline(std::cin, line)) {
            std::vector<int> ids;
            std::istringstream iss(line);
            int id;
            while (iss >> id) {
                ids.push_back(id);
            }
            std::cout << tokenizer.Decode(ids) << "\n";
        }
    });
}

} // namespace piece

int main(int argc, char* argv[]) {
    if (argc < 2) {
        piece::PrintUsage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    if (command == "count") {
        std::string method = "bytepiece";
        std::vector<std::string> inputs;
        std::string model_prefix = "tokenizer";
        int vocab_size = 8000;
        std::string normalizer = "no";
        int cpu_count = 4;
        int max_sentences = 0;
        int min_count = 32;
        int max_piece_size = 18;
        int cut = 0;
        bool reconstruct = false;
        std::string dict;
        std::vector<std::string> extra_tokens;

        for (int i = 2; i < argc; i++) {
            if (std::strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
                method = argv[++i];
            } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
                inputs.push_back(argv[++i]);
            } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
                model_prefix = argv[++i];
            } else if (std::strcmp(argv[i], "--vocab-size") == 0) {
                if (!piece::ParseIntegerOption(argc, argv, &i, 1, INT_MAX,
                                               &vocab_size)) return 1;
            } else if (std::strcmp(argv[i], "--normalize") == 0 && i + 1 < argc) {
                normalizer = argv[++i];
            } else if (std::strcmp(argv[i], "--cpu") == 0) {
                if (!piece::ParseIntegerOption(argc, argv, &i, 1, INT_MAX,
                                               &cpu_count)) return 1;
            } else if (std::strcmp(argv[i], "--max-sentences") == 0) {
                if (!piece::ParseIntegerOption(argc, argv, &i, 0, INT_MAX,
                                               &max_sentences)) return 1;
            } else if (std::strcmp(argv[i], "--min-count") == 0) {
                if (!piece::ParseIntegerOption(argc, argv, &i, 0, INT_MAX,
                                               &min_count)) return 1;
            } else if (std::strcmp(argv[i], "--max-piece-size") == 0) {
                if (!piece::ParseIntegerOption(argc, argv, &i, 0, INT_MAX,
                                               &max_piece_size)) return 1;
            } else if ((std::strcmp(argv[i], "--dict") == 0 ||
                        std::strcmp(argv[i], "--cn-dict") == 0) && i + 1 < argc) {
                dict = argv[++i];
            } else if (std::strcmp(argv[i], "--cut") == 0) {
                if (!piece::ParseIntegerOption(argc, argv, &i, 0, 1, &cut)) return 1;
            } else if (std::strcmp(argv[i], "--reconstruct") == 0) {
                reconstruct = true;
            } else if (std::strcmp(argv[i], "--extra-tokens") == 0 && i + 1 < argc) {
                extra_tokens = piece::ParseExtraTokens(argv[++i]);
            } else {
                std::cerr << "Unknown option: " << argv[i] << "\n";
                piece::PrintUsage(argv[0]);
                return 1;
            }
        }

        if (inputs.empty()) {
            std::cerr << "Error: --input is required for count\n";
            return 1;
        }

        if (!piece::RunCount(method, inputs, model_prefix, vocab_size, normalizer,
                             cpu_count, max_sentences, min_count, max_piece_size,
                             dict, cut, reconstruct, extra_tokens)) return 1;

    } else if (command == "pretokenize") {
        piece::PreTokenizerOptions options;
        std::string input_file;

        for (int i = 2; i < argc; i++) {
            const auto parsed = piece::ParsePreTokenizerOption(argc, argv, &i, &options);
            if (parsed == piece::ParseResult::kHandled) {
                continue;
            } else if (parsed == piece::ParseResult::kError) {
                return 1;
            } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
                input_file = argv[++i];
            } else {
                std::cerr << "Unknown option: " << argv[i] << "\n";
                piece::PrintUsage(argv[0]);
                return 1;
            }
        }

        std::ifstream file_in;
        if (!input_file.empty()) {
            file_in.open(input_file);
            if (!file_in) {
                std::cerr << "Error: cannot open input file: " << input_file << "\n";
                return 1;
            }
            std::cin.rdbuf(file_in.rdbuf());
        }

        piece::PreTokenizerSpec spec = piece::MakePreTokenizerSpec(options);
        piece::PreTokenizer tokenizer(spec, options.dict);

        std::string line;
        while (std::getline(std::cin, line)) {
            auto tokens = tokenizer.PreTokenize(line);
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i > 0) std::cout << ' ';
                std::cout << tokens[i];
            }
            std::cout << '\n';
        }

    } else if (command == "raw-count") {
        piece::PreTokenizerOptions options;
        int max_piece_size = 0;
        std::vector<std::string> inputs;
        std::string output_file;

        for (int i = 2; i < argc; i++) {
            const auto parsed = piece::ParsePreTokenizerOption(argc, argv, &i, &options);
            if (parsed == piece::ParseResult::kHandled) {
                continue;
            } else if (parsed == piece::ParseResult::kError) {
                return 1;
            } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
                inputs.push_back(argv[++i]);
            } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
                output_file = argv[++i];
            } else if (std::strcmp(argv[i], "--max-piece-size") == 0) {
                if (!piece::ParseIntegerOption(argc, argv, &i, 0, INT_MAX,
                                               &max_piece_size)) return 1;
            } else {
                std::cerr << "Unknown option: " << argv[i] << "\n";
                piece::PrintUsage(argv[0]);
                return 1;
            }
        }

        piece::PreTokenizerSpec spec = piece::MakePreTokenizerSpec(options);
        piece::PreTokenizer tokenizer(spec, options.dict);

        std::unordered_map<std::string, int64_t> counts;
        int64_t line_count = 0;

        auto process_stream = [&](std::istream& in) {
            std::string line;
            while (std::getline(in, line)) {
                for (const auto& token : tokenizer.PreTokenize(line)) {
                    if (max_piece_size > 0 &&
                        static_cast<int>(token.size()) > max_piece_size)
                        continue;
                    counts[token] += 1;
                }
                if (++line_count % 5000000 == 0)
                    std::cerr << "  " << line_count << " lines, "
                              << counts.size() << " unique tokens\n";
            }
        };

        if (inputs.empty()) {
            process_stream(std::cin);
        } else {
            for (const auto& f : inputs) {
                std::cerr << "Reading: " << f << "\n";
                std::ifstream fin(f);
                if (!fin) {
                    std::cerr << "Error: cannot open " << f << "\n";
                    return 1;
                }
                process_stream(fin);
            }
        }

        // Sort by frequency descending.
        std::vector<std::pair<std::string, int64_t>> sorted(
            counts.begin(), counts.end());
        counts.clear();
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) {
                    if (a.second != b.second) return a.second > b.second;
                    return a.first < b.first;
                  });

        std::ofstream fout;
        if (!output_file.empty()) {
            fout.open(output_file);
            if (!fout) {
                std::cerr << "Error: cannot open output file: " << output_file << "\n";
                return 1;
            }
        }
        std::ostream& out = output_file.empty() ? std::cout : fout;

        for (const auto& [token, freq] : sorted)
            out << token << '\t' << freq << '\n';

        std::cerr << "Done! " << line_count << " lines, "
                  << sorted.size() << " unique tokens\n";

    } else if (command == "insert-tokens") {
        std::string model_file;
        std::string output_file;
        std::vector<std::string> tokens;

        for (int i = 2; i < argc; i++) {
            if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
                model_file = argv[++i];
            } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
                output_file = argv[++i];
            } else if (std::strcmp(argv[i], "--extra-tokens") == 0 && i + 1 < argc) {
                tokens = piece::ParseExtraTokens(argv[++i]);
            } else {
                std::cerr << "Unknown option: " << argv[i] << "\n";
                piece::PrintUsage(argv[0]);
                return 1;
            }
        }

        if (model_file.empty() || output_file.empty() || tokens.empty()) {
            std::cerr << "insert-tokens requires --model, --output, and --extra-tokens\n";
            return 1;
        }

        piece::Model model;
        if (!model.Load(model_file)) {
            std::cerr << "Error: cannot load model: " << model_file << "\n";
            return 1;
        }
        const size_t before = model.PiecesSize();
        const int added = piece::InsertExtraTokens(&model, tokens, /*repoint_pad=*/true);
        if (!model.Save(output_file)) {
            std::cerr << "Error: cannot save model: " << output_file << "\n";
            return 1;
        }
        std::cerr << "insert-tokens: " << model_file << " (" << before
                  << " pieces) -> " << output_file << " (" << model.PiecesSize()
                  << " pieces, +" << added << " CONTROL)\n";

    } else if (command == "tokenize" || command == "encode" || command == "decode") {
        std::string model_file;
        std::string input_file;
        std::string dict;

        for (int i = 2; i < argc; i++) {
            if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
                model_file = argv[++i];
            } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
                input_file = argv[++i];
            } else if ((std::strcmp(argv[i], "--dict") == 0 ||
                        std::strcmp(argv[i], "--cn-dict") == 0) && i + 1 < argc) {
                dict = argv[++i];
            } else {
                std::cerr << "Unknown option: " << argv[i] << "\n";
                piece::PrintUsage(argv[0]);
                return 1;
            }
        }

        if (model_file.empty()) {
            std::cerr << "Error: --model is required\n";
            return 1;
        }

        // Redirect stdin from file if --input is specified
        std::ifstream file_in;
        if (!input_file.empty()) {
            file_in.open(input_file);
            if (!file_in) {
                std::cerr << "Error: cannot open input file: " << input_file << "\n";
                return 1;
            }
            std::cin.rdbuf(file_in.rdbuf());
        }

        if (command == "tokenize") {
            if (!piece::RunTokenize(model_file, dict)) return 1;
        } else if (command == "encode") {
            if (!piece::RunEncode(model_file, dict)) return 1;
        } else {
            if (!piece::RunDecode(model_file)) return 1;
        }

    } else {
        std::cerr << "Unknown command: " << command << "\n";
        piece::PrintUsage(argv[0]);
        return 1;
    }

    return 0;
}
