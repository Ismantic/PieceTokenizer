#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <cstdint>

#include "sentence.h"
#include "ustr.h"

namespace piece {
inline std::string EscapeSimple(const std::string& str) {
    std::string result;
    for (char c : str) {
        if (c == '\n') result += "\\n";
        else if (c == '\t') result += "\\t";
        else if (c == '\r') result += "\\r";
        else if (c == '\\') result += "\\\\";
        else result += c;
    }
    return result;
}

inline std::string UnescapeSimple(const std::string& str) {
    std::string result;
    for (size_t i = 0; i < str.size(); ++i) {
        if (str[i] == '\\' && i + 1 < str.size()) {
            if (str[i + 1] == 'n') result += '\n';
            else if (str[i + 1] == 't') result += '\t';
            else if (str[i + 1] == 'r') result += '\r';
            else if (str[i + 1] == '\\') result += '\\';
            else result += str[i + 1];
            ++i;
        } else {
            result += str[i];
        }
    }
    return result;
}

inline std::string Escape(const std::string& str) {
    // 处理空字符串
    if (str.empty()) {
        return "";
    }

    // 首先检查是否为合法的UTF-8字符串
    if (ustr::IsStructurallyValid(str)) {
        // 如果是合法UTF-8，使用简单转义
        return EscapeSimple(str);
    } else {
        // 如果不是合法UTF-8，所有字符都使用十六进制表示
        std::string result;
        result.reserve(str.size() * 4); // 每个字符最多需要4个字符表示(\xHH)
        for (unsigned char c : str) {
            char buf[5];
            snprintf(buf, sizeof(buf), "\\x%02X", c);
            result += buf;
        }
        return result;
    }
}

inline std::string Unescape(const std::string& str) {
    // 处理空字符串
    if (str.empty()) {
        return "";
    }

    // 检查是否包含十六进制转义序列
    bool has_hex_escape = false;
    for (size_t i = 0; i < str.size() - 1; ++i) {
        if (str[i] == '\\' && i + 1 < str.size() && str[i + 1] == 'x') {
            has_hex_escape = true;
            break;
        }
    }
    
    if (!has_hex_escape) {
        // 如果没有十六进制转义序列，使用简单解转义
        return UnescapeSimple(str);
    } else {
        // 如果有十六进制转义序列，只处理十六进制转义
        std::string result;
        result.reserve(str.size());
        
        for (size_t i = 0; i < str.size(); ++i) {
            if (str[i] == '\\' && i + 3 < str.size() && str[i + 1] == 'x') {
                // 处理十六进制转义序列 \xHH
                if (i + 3 >= str.size()) {
                    // 安全检查：如果不足以形成一个完整的十六进制转义序列，直接添加当前字符
                    result += str[i];
                    continue;
                }
                std::string hex = str.substr(i + 2, 2);
                // 直接解析十六进制值，不需要检查错误
                unsigned char c = static_cast<unsigned char>(strtol(hex.c_str(), nullptr, 16));
                result += c;
                i += 3; // 跳过 \x 和两个十六进制字符
            } else {
                result += str[i];
            }
        }
        return result;
    }
}



class CounterSpec {
public:
    std::vector<std::string> input_;
    std::string model_prefix_;
    int32_t vocab_size_ = 8000;
    float character_coverage_ = 0.9995f;
    
    int32_t unk_id_ = 0;
    int32_t bos_id_ = 1;
    int32_t eos_id_ = 2;
    int32_t pad_id_ = -1;
    uint32_t unk_unicode_ = 0x2585; // ▅
    std::string unk_piece_ = "<unk>";
    std::string bos_piece_ = "<s>";
    std::string eos_piece_ = "</s>";
    std::string pad_piece_ = "<pad>";
    
    const std::vector<std::string>& input() const { return input_; }
    void add_input(const std::string& input) { input_.push_back(input); }
    
    const std::string& model_prefix() const { return model_prefix_; }
    void set_model_prefix(const std::string& prefix) { model_prefix_ = prefix; }
    
    int32_t vocab_size() const { return vocab_size_; }
    void set_vocab_size(int32_t size) { vocab_size_ = size; }
    
    float character_coverage() const { return character_coverage_; }
    void set_character_coverage(float coverage) { character_coverage_ = coverage; }
    
    int32_t unk_id() const { return unk_id_; }
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t pad_id() const { return pad_id_; }
    
    const std::string& unk_piece() const { return unk_piece_; }
    const std::string& bos_piece() const { return bos_piece_; }
    const std::string& eos_piece() const { return eos_piece_; }
    const std::string& pad_piece() const { return pad_piece_; }
    
    uint32_t GetUnkUnicode() const { return unk_unicode_; }
    
    void Clear() {
        input_.clear();
        model_prefix_.clear();
        vocab_size_ = 8000;
        character_coverage_ = 0.9995f;
    }

    std::string AsStr() const {
        std::ostringstream oss;

        oss << "vocab_size=" << vocab_size_ << "\n";
        oss << "character_coverage=" << character_coverage_ << "\n";

        oss << "unk_id=" << unk_id_ << "\n";
        oss << "bos_id=" << bos_id_ << "\n";
        oss << "eos_id=" << eos_id_ << "\n";
        oss << "pad_id=" << pad_id_ << "\n";
        oss << "unk_unicode=" << unk_unicode_ << "\n";
        oss << "unk_piece=" << unk_piece_ << "\n";
        oss << "bos_piece=" << bos_piece_ << "\n";
        oss << "eos_piece=" << eos_piece_ << "\n";
        oss << "pad_piece=" << pad_piece_ << "\n";

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
            
            if (key == "vocab_size") {
                vocab_size_ = std::stoi(value);
            } else if (key == "character_coverage") {
                character_coverage_ = std::stof(value);
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

    void Clear() {
        name_.clear();
        space_.clear();
    }

    std::string AsStr() const {
        std::ostringstream oss;
        oss << "name=" << name_ << "\n";
        oss << "space=" << Escape(space_) << "\n";
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
            }
        }

        return true;
    }

private:
    std::string name_;
    std::string space_ = "\xe2\x96\x81";
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