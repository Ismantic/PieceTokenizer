#include <cstring>
#include <fstream>
#include <iostream>
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

namespace piece {

void PrintUsage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " count [options]\n"
              << "  " << prog << " tokenize --model <file>\n"
              << "  " << prog << " encode --model <file>\n"
              << "  " << prog << " decode --model <file>\n"
              << "\nCount options:\n"
              << "  --method <naive|piece|sentencepiece|bytepiece>  (default: bytepiece)\n"
              << "  --input <file>         Input corpus file\n"
              << "  --model <prefix>       Model output prefix (default: tokenizer)\n"
              << "  --vocab-size <int>     Vocabulary size (default: 8000)\n"
              << "  --normalize <name>     Normalizer: identity|NMT_NFKC (default: identity)\n"
              << "  --cpu <int>            Number of threads (default: 4)\n"
              << "\nTokenize/Encode/Decode read from stdin, write to stdout.\n"
              << "Tokenize outputs space-separated pieces per line.\n"
              << "Encode outputs one token per line (piece TAB id).\n"
              << "Decode reads token ids (space-separated) and outputs text.\n";
}

void RunTrain(const std::string& method,
              const std::vector<std::string>& inputs,
              const std::string& model_prefix, int vocab_size,
              const std::string& normalizer_name, int cpu_count) {
    CounterSpec counter_spec;
    for (const auto& f : inputs) counter_spec.add_input(f);
    counter_spec.set_model_prefix(model_prefix);
    counter_spec.set_method(method);
    counter_spec.set_cpu_count(cpu_count);

    NormalizerSpec normalizer_spec;
    normalizer_spec.SetName(normalizer_name);

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
        PieceCounter counter(counter_spec, normalizer_spec);
        counter.Count();
        counter.Save();
    } else if (method == "sentencepiece") {
        SentencePieceCounter counter(counter_spec, normalizer_spec);
        counter.Count();
        counter.Save();
    } else if (method == "bytepiece") {
        BytePieceCounter counter(counter_spec, normalizer_spec);
        counter.Count();
        counter.Save();
    } else {
        std::cerr << "Unknown method: " << method << "\n";
        return;
    }

    std::cerr << "Model saved to " << model_prefix << ".model\n";
}

void RunEncode(const std::string& model_file) {
    Model model;
    if (!model.Load(model_file)) {
        std::cerr << "Error: cannot load model: " << model_file << "\n";
        return;
    }

    const std::string& method = model.GetCounterSpec().method();
    const auto& normalizer_spec = model.GetNormalizerSpec();
    Normalizer normalizer(normalizer_spec);

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
        PieceTokenizer tokenizer(model);
        while (std::getline(std::cin, line)) {
            auto tokens = tokenizer.Encode(normalizer.Normalize(line));
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

void RunTokenize(const std::string& model_file) {
    Model model;
    if (!model.Load(model_file)) {
        std::cerr << "Error: cannot load model: " << model_file << "\n";
        return;
    }

    const std::string& method = model.GetCounterSpec().method();
    const auto& normalizer_spec = model.GetNormalizerSpec();
    Normalizer normalizer(normalizer_spec);

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
        PieceTokenizer tokenizer(model);
        while (std::getline(std::cin, line)) {
            auto tokens = tokenizer.Tokenize(normalizer.Normalize(line));
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
        PieceTokenizer tokenizer(model);
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
        std::string normalizer = "identity";
        int cpu_count = 4;

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

        piece::RunTrain(method, inputs, model_prefix, vocab_size, normalizer, cpu_count);

    } else if (command == "tokenize" || command == "encode" || command == "decode") {
        std::string model_file;
        std::string input_file;

        for (int i = 2; i < argc; i++) {
            if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
                model_file = argv[++i];
            } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
                input_file = argv[++i];
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
            piece::RunTokenize(model_file);
        } else if (command == "encode") {
            piece::RunEncode(model_file);
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
