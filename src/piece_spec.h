#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <cstdint>

#include "sentence.h"
#include "ustr.h"

namespace piece {
// Escape: unsafe bytes (\t \n \r \) always use \xHH format.
// Valid UTF-8 multi-byte sequences pass through as-is.
// Invalid UTF-8 single bytes also use \xHH.
inline std::string Escape(const std::string& str) {
    if (str.empty()) return "";
    std::string result;
    result.reserve(str.size() * 2);
    const char* p = str.data();
    const char* end = p + str.size();
    while (p < end) {
        unsigned char c = *p;
        // Unsafe single bytes: must escape to avoid breaking TSV format.
        if (c == '\t' || c == '\n' || c == '\r' || c == '\\') {
            char buf[5];
            snprintf(buf, sizeof(buf), "\\x%02X", c);
            result += buf;
            ++p;
        } else if (c < 0x80) {
            // Safe ASCII: pass through.
            result += c;
            ++p;
        } else {
            // Multi-byte: check if valid UTF-8 sequence.
            size_t len = ustr::OneUTF8Size(p);
            if (p + len <= end && ustr::IsStructurallyValid(std::string_view(p, len))) {
                result.append(p, len);
                p += len;
            } else {
                // Invalid byte: hex escape.
                char buf[5];
                snprintf(buf, sizeof(buf), "\\x%02X", c);
                result += buf;
                ++p;
            }
        }
    }
    return result;
}

// Unescape: handles \xHH sequences, everything else is literal.
inline std::string Unescape(const std::string& str) {
    if (str.empty()) return "";
    std::string result;
    result.reserve(str.size());
    for (size_t i = 0; i < str.size(); ++i) {
        if (str[i] == '\\' && i + 3 < str.size() && str[i + 1] == 'x') {
            std::string hex = str.substr(i + 2, 2);
            result += static_cast<char>(strtol(hex.c_str(), nullptr, 16));
            i += 3;
        } else {
            result += str[i];
        }
    }
    return result;
}



class CounterSpec {
public:
    std::vector<std::string> input_;
    std::string model_prefix_;
    std::string method_ = "bytepiece";
    int32_t vocab_size_ = 8000;
    float character_coverage_ = 0.9995f;
    int32_t min_count_ = 32;
    int32_t cpu_count_ = 4;
    int32_t max_sentences_ = 0;  // 0 = unlimited
    int32_t max_piece_size_ = 18;  // max bytes per piece (~6 CJK chars)

    // Path to a TSV (`word\tfreq`) Chinese-segmenter dictionary. When
    // non-empty, PieceCounter enters cn mode and runs Han runs through
    // a Unigram cutter. Not serialized to the .model file.
    std::string cn_dict_;

    int32_t unk_id_ = 0;
    int32_t bos_id_ = 1;
    int32_t eos_id_ = 2;
    int32_t pad_id_ = -1;
    uint32_t unk_unicode_ = 0x2585; // ▅
    std::string unk_piece_ = "<unk>";
    std::string bos_piece_ = "<s>";
    std::string eos_piece_ = "</s>";
    std::string pad_piece_ = "<pad>";

    // Extra CONTROL tokens appended after the normal vocabulary.
    // Each string becomes a CONTROL-type piece at the end of the vocab.
    std::vector<std::string> extra_tokens_;

    const std::vector<std::string>& input() const { return input_; }
    void add_input(const std::string& input) { input_.push_back(input); }
    
    const std::string& model_prefix() const { return model_prefix_; }
    void set_model_prefix(const std::string& prefix) { model_prefix_ = prefix; }

    const std::string& method() const { return method_; }
    void set_method(const std::string& m) { method_ = m; }
    
    int32_t vocab_size() const { return vocab_size_; }
    void set_vocab_size(int32_t size) { vocab_size_ = size; }
    
    float character_coverage() const { return character_coverage_; }
    void set_character_coverage(float coverage) { character_coverage_ = coverage; }

    int32_t min_count() const { return min_count_; }
    void set_min_count(int32_t count) { min_count_ = count; }

    int32_t cpu_count() const { return cpu_count_; }
    void set_cpu_count(int32_t count) { cpu_count_ = count; }

    int32_t max_sentences() const { return max_sentences_; }
    void set_max_sentences(int32_t n) { max_sentences_ = n; }

    int32_t max_piece_size() const { return max_piece_size_; }
    void set_max_piece_size(int32_t n) { max_piece_size_ = n; }

    const std::string& cn_dict() const { return cn_dict_; }
    void set_cn_dict(const std::string& path) { cn_dict_ = path; }
    
    int32_t unk_id() const { return unk_id_; }
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t pad_id() const { return pad_id_; }
    
    const std::string& unk_piece() const { return unk_piece_; }
    const std::string& bos_piece() const { return bos_piece_; }
    const std::string& eos_piece() const { return eos_piece_; }
    const std::string& pad_piece() const { return pad_piece_; }
    
    uint32_t GetUnkUnicode() const { return unk_unicode_; }

    const std::vector<std::string>& extra_tokens() const { return extra_tokens_; }
    void add_extra_token(const std::string& token) { extra_tokens_.push_back(token); }
    
    void Clear() {
        input_.clear();
        model_prefix_.clear();
        vocab_size_ = 8000;
        character_coverage_ = 0.9995f;
    }

    std::string AsStr() const {
        std::ostringstream oss;

        oss << "method=" << method_ << "\n";
        oss << "vocab_size=" << vocab_size_ << "\n";
        oss << "character_coverage=" << character_coverage_ << "\n";
        oss << "min_count=" << min_count_ << "\n";
        oss << "max_piece_size=" << max_piece_size_ << "\n";

        oss << "unk_id=" << unk_id_ << "\n";
        oss << "bos_id=" << bos_id_ << "\n";
        oss << "eos_id=" << eos_id_ << "\n";
        oss << "pad_id=" << pad_id_ << "\n";
        oss << "unk_unicode=" << unk_unicode_ << "\n";
        oss << "unk_piece=" << unk_piece_ << "\n";
        oss << "bos_piece=" << bos_piece_ << "\n";
        oss << "eos_piece=" << eos_piece_ << "\n";
        oss << "pad_piece=" << pad_piece_ << "\n";

        if (!extra_tokens_.empty()) {
            oss << "extra_tokens=";
            for (size_t i = 0; i < extra_tokens_.size(); ++i) {
                if (i > 0) oss << ",";
                oss << extra_tokens_[i];
            }
            oss << "\n";
        }

        return oss.str();
    }

    bool FromStr(const std::string& data) {
        std::istringstream iss(data);
        std::string line;

        Clear();

        while (std::getline(iss, line)) {
            std::string key, value;
            size_t pos = line.find('=');
            if (pos == std::string::npos) continue;
            
            key = line.substr(0, pos);
            value = line.substr(pos + 1);
            
            if (key == "method") {
                method_ = value;
            } else if (key == "vocab_size") {
                vocab_size_ = std::stoi(value);
            } else if (key == "character_coverage") {
                character_coverage_ = std::stof(value);
            } else if (key == "min_count") {
                min_count_ = std::stoi(value);
            } else if (key == "max_piece_size") {
                max_piece_size_ = std::stoi(value);
            } else if (key == "unk_unicode") {
                unk_unicode_ = std::stoul(value);
            } else if (key == "unk_id") {
                unk_id_ = std::stoi(value);
            } else if (key == "bos_id") {
                bos_id_ = std::stoi(value);
            } else if (key == "eos_id") {
                eos_id_ = std::stoi(value);
            } else if (key == "pad_id") {
                pad_id_ = std::stoi(value);
            } else if (key == "unk_piece") {
                unk_piece_ = value;
            } else if (key == "bos_piece") {
                bos_piece_ = value;
            } else if (key == "eos_piece") {
                eos_piece_ = value;
            } else if (key == "pad_piece") {
                pad_piece_ = value;
            } else if (key == "extra_tokens") {
                extra_tokens_.clear();
                std::istringstream ts(value);
                std::string token;
                while (std::getline(ts, token, ',')) {
                    if (!token.empty()) extra_tokens_.push_back(token);
                }
            }
        }
        
        return true;
    }
};

class NormalizerSpec {
public:
    NormalizerSpec() = default;

    void SetName(const std::string& name) { name_ = name; }

    const std::string& GetName() const { return name_; }

    void SetSpace(const std::string& space_str) { space_ = space_str; }
    const std::string& GetSpace() const { return space_; }

    void SetCut(int cut) { cut_ = cut; }
    int GetCut() const { return cut_; }

    void SetReconstruct(bool r) { reconstruct_ = r; }
    bool GetReconstruct() const { return reconstruct_; }

    void Clear() {
        name_.clear();
        space_.clear();
        cut_ = 0;
        reconstruct_ = false;
    }

    std::string AsStr() const {
        std::ostringstream oss;
        oss << "name=" << name_ << "\n";
        oss << "space=" << Escape(space_) << "\n";
        oss << "cut=" << cut_ << "\n";
        oss << "reconstruct=" << (reconstruct_ ? 1 : 0) << "\n";
        return oss.str();
    }

    bool FromStr(const std::string& data) {
        std::istringstream iss(data);
        std::string line;

        Clear();

        while (std::getline(iss, line)) {
            std::string key, value;
            size_t pos = line.find('=');
            if (pos == std::string::npos) continue;
            
            key = line.substr(0, pos);
            value = line.substr(pos + 1);
            
            if (key == "name") {
                name_ = value;
            } else if (key == "space") {
                space_ = Unescape(value);
            } else if (key == "cut") {
                cut_ = std::stoi(value);
            } else if (key == "reconstruct") {
                reconstruct_ = std::stoi(value) != 0;
            }
        }

        return true;
    }

private:
    std::string name_;
    std::string space_ = "\xe2\x96\x81";
    int cut_ = 0;           // 0: default, 1: split spaces and punctuation independently
    bool reconstruct_ = false;  // true: preserve all spaces (no stripping/merging)
};

class Model {
public:
    class Piece {
    public:
        enum Type {
            NORMAL = 1,
            UNKNOWN = 2,
            CONTROL = 3,
            USER_DEFINED = 4,
            BYTE = 6,
            UNUSED = 5
        };
        
        void SetPiece(const std::string& piece,
                      const std::string& u = "",
                      const std::string& v = "") { 
            piece_ = piece; 
            u_ = u;
            v_ = v;
        }
        const std::string& GetPiece() const { return piece_; }
        const std::string& u() const { return u_; }
        const std::string& v() const { return v_; }
        bool HasPiece() const { return !piece_.empty(); }
        
        void SetScore(float score) { score_ = score; }
        float GetScore() const { return score_; }
        bool HasScore() const { return score_ != 0.0f; }
        
        void SetType(Type type) { type_ = type; }
        Type GetType() const { return type_; }
        bool HasType() const { return type_ != NORMAL; }

        void Clear() {
            piece_.clear();
            score_ = 0.0f;
            type_ = NORMAL;
        }

        static const Piece& DefaultInstance() {
            static const Piece instance;
            return instance;
        }

    private:
        std::string piece_;  // piece must not be empty
        std::string u_; // (u_ + v_) == piece_;
        std::string v_;
        float score_ = 0.0f;
        Type type_ = NORMAL;
    };

    // Pieces management
    Piece* InsertPieces() {
        pieces_.emplace_back();
        return &pieces_.back();
    }
    const std::vector<Piece>& GetPieces() const { return pieces_; }
    std::vector<Piece>* GetMutablePieces() { return &pieces_; }
    const Piece& GetPieces(int index) const { return pieces_[index]; }
    Piece* GetMutablePieces(int index) { return &pieces_[index]; }
    size_t PiecesSize() const { return pieces_.size(); }
    void ClearPieces() { pieces_.clear(); }
    
    // Counter spec
    void SetCounterSpec(const CounterSpec& spec) { counter_spec_ = spec; }
    const CounterSpec& GetCounterSpec() const { return counter_spec_; }
    CounterSpec* GetMutableCounterSpec() { return &counter_spec_; }
    bool HasCounterSpec() const { return true; }
    
    // Normalizer spec
    void SetNormalizerSpec(const NormalizerSpec& spec) { normalizer_spec_ = spec; }
    const NormalizerSpec& GetNormalizerSpec() const { return normalizer_spec_; }
    NormalizerSpec* GetMutableNormalizerSpec() { return &normalizer_spec_; }
    bool HasNormalizerSpec() const { return true; }
    
    void Clear() {
        pieces_.clear();
        counter_spec_.Clear();
        normalizer_spec_.Clear();
    }


    std::string AsStr() const {
        std::ostringstream oss;
    
        oss << "[CounterSpec]\n";
        oss << counter_spec_.AsStr();
        oss << "\n";
    
        oss << "[NormalizerSpec]\n";
        oss << normalizer_spec_.AsStr();
        oss << "\n";
    
        oss << "[Pieces]\n";
        oss << "size=" << pieces_.size() << "\n";
        for (size_t i = 0; i < pieces_.size(); ++i) {
            const auto& piece = pieces_[i];
            // 确保即使u和v为空，也输出制表符
            oss << i << "\t" << Escape(piece.GetPiece()) << "\t"
                << piece.GetScore() << "\t"
                << static_cast<int>(piece.GetType());
            
            // 始终输出u和v字段，即使它们为空
            oss << "\t" << Escape(piece.u()) << "\t" << Escape(piece.v()) << "\n";
        }
    
        return oss.str();
    }
    
    bool FromStr(const std::string& data) {
        std::istringstream iss(data);
        std::string line;
        std::string section;
        std::string counter_spec_str;
        std::string normalizer_spec_str;
    
        Clear();
    
        size_t pieces_size = 0;
    
        while (std::getline(iss, line)) {
            if (line.empty()) continue;
    
            if (line[0] == '[' && line[line.size()-1] == ']') {
                section = line.substr(1, line.size()-2);
                continue;
            }
    
            if (section == "CounterSpec") {
                counter_spec_str += line + "\n";
            } else if (section == "NormalizerSpec") {
                normalizer_spec_str += line + "\n";
            } else if (section == "Pieces") {
                if (line.find("size=") == 0) {
                    pieces_size = std::stoul(line.substr(5));
                    continue;
                }
    
                // 使用制表符分割行，但确保正确处理空字段
                std::vector<std::string> parts;
                size_t start = 0;
                size_t end = 0;
                
                // 寻找所有制表符，包括连续的制表符（表示空字段）
                while ((end = line.find('\t', start)) != std::string::npos) {
                    parts.push_back(line.substr(start, end - start));
                    start = end + 1;
                }
                
                // 添加最后一个字段（如果有的话）
                if (start < line.size()) {
                    parts.push_back(line.substr(start));
                } else if (start == line.size()) {
                    // 如果最后一个字符是制表符，添加一个空字段
                    parts.push_back("");
                }
    
                // 确保至少有基本的4个字段
                if (parts.size() >= 4) {
                    try {
                        int i = std::stoi(parts[0]);
                        std::string p = parts[1];
                        float s = std::stof(parts[2]);
                        int t = std::stoi(parts[3]);
                        
                        std::string u = "";
                        std::string v = "";
                        
                        // 安全地获取u和v，如果它们存在的话
                        if (parts.size() >= 5) {
                            u = parts[4];
                        }
                        
                        if (parts.size() >= 6) {
                            v = parts[5];
                        }
                        
                        auto* piece = InsertPieces();
                        piece->SetPiece(Unescape(p), Unescape(u), Unescape(v));
                        piece->SetScore(s);
                        piece->SetType(static_cast<Piece::Type>(t));
                    } catch (const std::exception& e) {
                        std::cerr << "Error: failed to parse piece line: " << line << " (" << e.what() << ")" << std::endl;
                    }
                } else {
                    std::cerr << "Error: invalid piece line format (not enough fields): " << line << std::endl;
                }
            }
        }
    
        if (!counter_spec_.FromStr(counter_spec_str) ||
            !normalizer_spec_.FromStr(normalizer_spec_str)) {
                return false;
        }
    
        if (pieces_.size() != pieces_size) {
            std::cerr << "Error: pieces_size not match " << pieces_.size()
                      << " != " << pieces_size << std::endl;
            return false;
        }
    
        return true;
    }    

    static const Model& DefaultInstance() {
        static const Model instance;
        return instance;
    }

    bool Save(const std::string& filename) {
        auto output = NewWritableFile(filename);
        if (!output) {
            std::cerr << "Error: cannot open file: " << filename << std::endl;
            return false;
        }

        std::string data = AsStr();
        output->Write(data);
        return true;
    }

    bool Load(const std::string& filename) {
        auto input = NewReadableFile(filename);
        if (!input) {
            std::cerr << "Error: cannot open file: " << filename << std::endl;
            return false;
        }

        std::string data;
        input->ReadAll(&data);
        return FromStr(data);
    }

private:
    std::vector<Piece> pieces_;
    CounterSpec counter_spec_;
    NormalizerSpec normalizer_spec_;
};


} // namespace piece