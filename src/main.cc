#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
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
              << "  --cn-dict <file>       Enable CN mode for `piece` method using\n"
              << "                         a TSV (word\\tfreq) Unigram dictionary\n"
              << "\nPretokenize/Raw-count options (PreTokenizer 3 axes):\n"
              << "  --normalize <name>     Normalizer: no|NMT_NFKC|NFKC_CF (default: no)\n"
              << "  --split <word|isolate> Split axis: word=GPT-4-style attach, isolate=spaces/punct standalone (default: word)\n"
              << "  --digit <keep|split>   Digit axis: keep runs, or split per-codepoint (default: keep)\n"
              << "  --cn-dict <no|file>    Cn axis: no=per-char, TSV(word\\tfreq)=dict, omitted=none\n"
              << "  --cut <0|1>            Alias for --split (0=word, 1=isolate)\n"
              << "  --reconstruct          Preserve all spaces (no stripping/merging)\n"
              << "  --input <file>         Input file (repeatable for raw-count)\n"
              << "  --output <file>        Output file (raw-count only, default: stdout)\n"
              << "  --max-piece-size <int> Skip tokens exceeding this many bytes (raw-count, default: 0=unlimited)\n"
              << "\nTokenize/Encode options:\n"
              << "  --model <file>         Model file to load\n"
              << "  --input <file>         Read input from file instead of stdin\n"
              << "  --cn-dict <file>       Enable CN mode for `piece` model (must match training)\n"
              << "\nTokenize/Encode/Decode read from stdin, write to stdout.\n"
              << "Tokenize outputs space-separated pieces per line.\n"
              << "Encode outputs one token per line (piece TAB id).\n"
              << "Decode reads token ids (space-separated) and outputs text.\n";
}

void RunCount(const std::string& method,
              const std::vector<std::string>& inputs,
              const std::string& model_prefix, int vocab_size,
              const std::string& normalizer_name, int cpu_count,
              int max_sentences, int min_count, int max_piece_size,
              const std::string& cn_dict, int cut, bool reconstruct,
              const std::vector<std::string>& extra_tokens = {}) {
    CounterSpec counter_spec;
    for (const auto& f : inputs) counter_spec.add_input(f);
    counter_spec.set_model_prefix(model_prefix);
    counter_spec.set_method(method);
    counter_spec.set_cpu_count(cpu_count);
    counter_spec.set_max_sentences(max_sentences);
    counter_spec.set_min_count(min_count);
    counter_spec.set_max_piece_size(max_piece_size);
    counter_spec.set_cn_dict(cn_dict);

    for (const auto& t : extra_tokens) counter_spec.add_extra_token(t);

    if (!cn_dict.empty() && method != "piece" && method != "sentencepiece") {
        std::cerr << "Warning: --cn-dict is only supported for --method piece "
                  << "or --method sentencepiece; ignoring for method=" << method << "\n";
        counter_spec.set_cn_dict("");
    }

    PreTokenizerSpec pretokenizer_spec;
    pretokenizer_spec.SetName(normalizer_name);
    pretokenizer_spec.SetCut(cut);
    pretokenizer_spec.SetReconstruct(reconstruct);

    // Single owner of the char-mode policy: cn_dict="no" forces cut=1 +
    // split_digits=true (= Split isolate + Digit split) for every method
    // that supports cn mode, so digits/punct/symbols stay single codepoints
    // and only ASCII-letter runs go through BPE. Persisted to the model so
    // inference matches training.
    if (counter_spec.cn_dict() == "no") {
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
        NaiveCounter counter(counter_spec);
        counter.Count();
        counter.Save();
    } else if (method == "piece") {
        PieceCounter counter(counter_spec, pretokenizer_spec);
        counter.Count();
        counter.Save();
    } else if (method == "sentencepiece") {
        SentencePieceCounter counter(counter_spec, pretokenizer_spec);
        counter.Count();
        counter.Save();
    } else if (method == "bytepiece") {
        BytePieceCounter counter(counter_spec, pretokenizer_spec);
        counter.Count();
        counter.Save();
    } else {
        std::cerr << "Unknown method: " << method << "\n";
        return;
    }

    std::cerr << "Model saved to " << model_prefix << ".model\n";
}

void RunEncode(const std::string& model_file, const std::string& cn_dict) {
    Model model;
    if (!model.Load(model_file)) {
        std::cerr << "Error: cannot load model: " << model_file << "\n";
        return;
    }

    const std::string& method = model.GetCounterSpec().method();
    if (!cn_dict.empty() && method != "piece" && method != "sentencepiece") {
        std::cerr << "Warning: --cn-dict is only supported for --method piece "
                  << "or --method sentencepiece; ignoring for method=" << method << "\n";
    }
    std::string line;
    if (method == "naive") {
        NaiveTokenizer tokenizer(model);
        while (std::getline(std::cin, line)) {
            auto tokens = tokenizer.Encode(line);
            for (const auto& t : tokens) {
                std::cout << t.first << "\t" << t.second << "\n";
            }
            std::cout << "\n";
        }
    } else if (method == "piece") {
        PieceTokenizer tokenizer(model, cn_dict);
        while (std::getline(std::cin, line)) {
            auto tokens = tokenizer.Encode(line);
            for (const auto& t : tokens) {
                std::cout << t.first << "\t" << t.second << "\n";
            }
            std::cout << "\n";
        }
    } else if (method == "sentencepiece") {
        SentencePieceTokenizer tokenizer(model);
        while (std::getline(std::cin, line)) {
            auto tokens = tokenizer.Encode(line);
            for (const auto& t : tokens) {
                std::cout << t.first << "\t" << t.second << "\n";
            }
            std::cout << "\n";
        }
    } else if (method == "bytepiece") {
        BytePieceTokenizer tokenizer(model);
        while (std::getline(std::cin, line)) {
            auto tokens = tokenizer.Encode(line);
            for (const auto& t : tokens) {
                std::cout << t.first << "\t" << t.second << "\n";
            }
            std::cout << "\n";
        }
    } else {
        std::cerr << "Unknown method in model: " << method << "\n";
    }
}

void RunTokenize(const std::string& model_file, const std::string& cn_dict) {
    Model model;
    if (!model.Load(model_file)) {
        std::cerr << "Error: cannot load model: " << model_file << "\n";
        return;
    }

    const std::string& method = model.GetCounterSpec().method();
    if (!cn_dict.empty() && method != "piece" && method != "sentencepiece") {
        std::cerr << "Warning: --cn-dict is only supported for --method piece "
                  << "or --method sentencepiece; ignoring for method=" << method << "\n";
    }
    std::string line;
    if (method == "naive") {
        NaiveTokenizer tokenizer(model);
        while (std::getline(std::cin, line)) {
            auto tokens = tokenizer.Tokenize(line);
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i > 0) std::cout << " ";
                std::cout << Escape(tokens[i]);
            }
            std::cout << "\n";
        }
    } else if (method == "piece") {
        PieceTokenizer tokenizer(model, cn_dict);
        while (std::getline(std::cin, line)) {
            auto tokens = tokenizer.Tokenize(line);
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i > 0) std::cout << " ";
                std::cout << Escape(tokens[i]);
            }
            std::cout << "\n";
        }
    } else if (method == "sentencepiece") {
        SentencePieceTokenizer tokenizer(model);
        while (std::getline(std::cin, line)) {
            auto tokens = tokenizer.Tokenize(line);
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i > 0) std::cout << " ";
                std::cout << Escape(tokens[i]);
            }
            std::cout << "\n";
        }
    } else if (method == "bytepiece") {
        BytePieceTokenizer tokenizer(model);
        while (std::getline(std::cin, line)) {
            auto tokens = tokenizer.Tokenize(line);
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i > 0) std::cout << " ";
                std::cout << Escape(tokens[i]);
            }
            std::cout << "\n";
        }
    } else {
        std::cerr << "Unknown method in model: " << method << "\n";
    }
}

void RunDecode(const std::string& model_file) {
    Model model;
    if (!model.Load(model_file)) {
        std::cerr << "Error: cannot load model: " << model_file << "\n";
        return;
    }

    const std::string& method = model.GetCounterSpec().method();

    std::string line;
    if (method == "naive") {
        NaiveTokenizer tokenizer(model);
        while (std::getline(std::cin, line)) {
            std::vector<int> ids;
            std::istringstream iss(line);
            int id;
            while (iss >> id) {
                ids.push_back(id);
            }
            std::cout << tokenizer.Decode(ids) << "\n";
        }
    } else if (method == "piece") {
        PieceTokenizer tokenizer(model, "");
        while (std::getline(std::cin, line)) {
            std::vector<int> ids;
            std::istringstream iss(line);
            int id;
            while (iss >> id) {
                ids.push_back(id);
            }
            std::cout << tokenizer.Decode(ids) << "\n";
        }
    } else if (method == "bytepiece") {
        BytePieceTokenizer tokenizer(model);
        while (std::getline(std::cin, line)) {
            std::vector<int> ids;
            std::istringstream iss(line);
            int id;
            while (iss >> id) {
                ids.push_back(id);
            }
            std::cout << tokenizer.Decode(ids) << "\n";
        }
    } else if (method == "sentencepiece") {
        SentencePieceTokenizer tokenizer(model);
        while (std::getline(std::cin, line)) {
            std::vector<int> ids;
            std::istringstream iss(line);
            int id;
            while (iss >> id) {
                ids.push_back(id);
            }
            std::cout << tokenizer.Decode(ids) << "\n";
        }
    } else {
        std::cerr << "Unknown method in model: " << method << "\n";
    }
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
        std::string cn_dict;
        std::vector<std::string> extra_tokens;

        for (int i = 2; i < argc; i++) {
            if (std::strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
                method = argv[++i];
            } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
                inputs.push_back(argv[++i]);
            } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
                model_prefix = argv[++i];
            } else if (std::strcmp(argv[i], "--vocab-size") == 0 && i + 1 < argc) {
                vocab_size = std::atoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--normalize") == 0 && i + 1 < argc) {
                normalizer = argv[++i];
            } else if (std::strcmp(argv[i], "--cpu") == 0 && i + 1 < argc) {
                cpu_count = std::atoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--max-sentences") == 0 && i + 1 < argc) {
                max_sentences = std::atoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--min-count") == 0 && i + 1 < argc) {
                min_count = std::atoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--max-piece-size") == 0 && i + 1 < argc) {
                max_piece_size = std::atoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--cn-dict") == 0 && i + 1 < argc) {
                cn_dict = argv[++i];
            } else if (std::strcmp(argv[i], "--cut") == 0 && i + 1 < argc) {
                cut = std::atoi(argv[++i]);
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

        piece::RunCount(method, inputs, model_prefix, vocab_size, normalizer, cpu_count, max_sentences, min_count, max_piece_size, cn_dict, cut, reconstruct, extra_tokens);

    } else if (command == "pretokenize") {
        std::string normalizer = "no";
        int cut = 0;
        bool split_digits = false;
        bool reconstruct = false;
        std::string cn_dict;
        std::string input_file;

        for (int i = 2; i < argc; i++) {
            if (std::strcmp(argv[i], "--normalize") == 0 && i + 1 < argc) {
                normalizer = argv[++i];
            } else if (std::strcmp(argv[i], "--split") == 0 && i + 1 < argc) {
                cut = std::strcmp(argv[++i], "isolate") == 0 ? 1 : 0;
            } else if (std::strcmp(argv[i], "--digit") == 0 && i + 1 < argc) {
                split_digits = std::strcmp(argv[++i], "split") == 0;
            } else if (std::strcmp(argv[i], "--cut") == 0 && i + 1 < argc) {
                cut = std::atoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--cn-dict") == 0 && i + 1 < argc) {
                cn_dict = argv[++i];
            } else if (std::strcmp(argv[i], "--reconstruct") == 0) {
                reconstruct = true;
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

        piece::PreTokenizerSpec spec;
        spec.SetName(normalizer);
        spec.SetCut(cut);
        spec.SetSplitDigits(split_digits);
        spec.SetReconstruct(reconstruct);
        piece::PreTokenizer tokenizer(spec, cn_dict);

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
        std::string normalizer = "no";
        int cut = 0;
        bool split_digits = false;
        bool reconstruct = false;
        std::string cn_dict;
        int max_piece_size = 0;
        std::vector<std::string> inputs;
        std::string output_file;

        for (int i = 2; i < argc; i++) {
            if (std::strcmp(argv[i], "--normalize") == 0 && i + 1 < argc) {
                normalizer = argv[++i];
            } else if (std::strcmp(argv[i], "--split") == 0 && i + 1 < argc) {
                cut = std::strcmp(argv[++i], "isolate") == 0 ? 1 : 0;
            } else if (std::strcmp(argv[i], "--digit") == 0 && i + 1 < argc) {
                split_digits = std::strcmp(argv[++i], "split") == 0;
            } else if (std::strcmp(argv[i], "--cut") == 0 && i + 1 < argc) {
                cut = std::atoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--cn-dict") == 0 && i + 1 < argc) {
                cn_dict = argv[++i];
            } else if (std::strcmp(argv[i], "--reconstruct") == 0) {
                reconstruct = true;
            } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
                inputs.push_back(argv[++i]);
            } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
                output_file = argv[++i];
            } else if (std::strcmp(argv[i], "--max-piece-size") == 0 && i + 1 < argc) {
                max_piece_size = std::atoi(argv[++i]);
            } else {
                std::cerr << "Unknown option: " << argv[i] << "\n";
                piece::PrintUsage(argv[0]);
                return 1;
            }
        }

        piece::PreTokenizerSpec spec;
        spec.SetName(normalizer);
        spec.SetCut(cut);
        spec.SetSplitDigits(split_digits);
        spec.SetReconstruct(reconstruct);
        piece::PreTokenizer tokenizer(spec, cn_dict);

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
                    return a.second > b.second;
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
        std::string cn_dict;

        for (int i = 2; i < argc; i++) {
            if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
                model_file = argv[++i];
            } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
                input_file = argv[++i];
            } else if (std::strcmp(argv[i], "--cn-dict") == 0 && i + 1 < argc) {
                cn_dict = argv[++i];
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
            piece::RunTokenize(model_file, cn_dict);
        } else if (command == "encode") {
            piece::RunEncode(model_file, cn_dict);
        } else {
            piece::RunDecode(model_file);
        }

    } else {
        std::cerr << "Unknown command: " << command << "\n";
        piece::PrintUsage(argv[0]);
        return 1;
    }

    return 0;
}
