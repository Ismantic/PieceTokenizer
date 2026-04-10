#pragma once 

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <memory>
#include <sstream>
#include <string>
#include <iomanip>
#include <unordered_map>

#include <string_view>
#include <algorithm>
#include <functional>
#include <vector>

namespace ustr {

using UnicodeText = std::vector<uint32_t>;
static constexpr uint32_t UnicodeError = 0xFFFD;

inline size_t UTF8CharLen(unsigned char lead_byte) {
  return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[lead_byte >> 4];
}

// Return length of a single UTF-8 source character
inline size_t OneUTF8Size(const char *src) {
  return UTF8CharLen(static_cast<unsigned char>(*src));
}

// Return (x & 0xC0) == 0x80;
// Since trail bytes are always in [0x80, 0xBF], we can optimize
inline bool IsTrailByte(char x) {
    return static_cast<signed char>(x) < -0x40;
}

inline bool IsValidCodepoint(uint32_t c) {
    return (static_cast<uint32_t>(c) < 0xD800) || (c >= 0xE000 && c <= 0x10FFFF);
}

bool IsStructurallyValid(std::string_view str);

uint32_t DecodeUTF8(const char *begin, const char *end, size_t *mblen);

inline uint32_t DecodeUTF8(std::string_view input, size_t *mblen) {
    return DecodeUTF8(input.data(), input.data()+input.size(), mblen);
}

size_t EncodeUTF8(uint32_t c, char *output);

UnicodeText UTF8ToUnicodeText(std::string_view utf8);

std::string UnicodeTextToUTF8(const UnicodeText &utext);

inline bool IsValidDecodeUTF8(std::string_view input, size_t *mblen) {
    const uint32_t c = DecodeUTF8(input, mblen);
    return c != UnicodeError || *mblen == 3;
}

template <typename T>
inline std::string EncodePOD(const T &value) {
    std::string s; 
    s.resize(sizeof(T));
    memcpy(const_cast<char *>(s.data()), &value, sizeof(T));
    return s;
}

template <typename T> 
inline bool DecodePOD(std::string_view str, T *result) {
    if (sizeof(*result) != str.size()) {
        return false;
    } 
    memcpy(result, str.data(), sizeof(T));
    return true;
}


inline std::string ByteToPiece(unsigned char c) {
    std::stringstream ss;
    ss << "<0x" << std::setfill('0') << std::setw(2) << std::uppercase << std::hex << static_cast<int>(c) << ">";
    return ss.str();
}
  
inline int PieceToByte(std::string_view piece) {
    using PieceToByteMap = std::unordered_map<std::string, unsigned char>;
    static const auto *const kMap = []() -> PieceToByteMap * {
      auto *m = new PieceToByteMap();
      for (int i = 0; i < 256; ++i) {
        (*m)[ByteToPiece(i)] = i;
      }
      return m;
    }();
    const auto it = kMap->find(std::string(piece));
    if (it == kMap->end()) {
      return -1;
    } else {
      return it->second;
    }
}

bool IsDigitToken(std::string_view text);

// Returns true if the codepoint is a "word" character: a letter from any
// script (Latin, Greek, Cyrillic, Arabic, CJK, Hangul, ...), a digit, a
// CJK ideograph, or a combining mark that continues a word. Used by
// SplitText to group contiguous word characters into runs.
bool IsWordChar(uint32_t cp);

// Returns true if `cp` is a Han (CJK) ideograph. Covers CJK Unified
// Ideographs, Extension A, Extensions B-H in the supplementary plane,
// and CJK Compatibility Ideographs. Used by cn-mode post-processing
// to detect runs of Chinese characters.
bool IsHan(uint32_t cp);

bool IsPunctuationToken(std::string_view text);

// A "separator" is any codepoint that breaks a word run: punctuation or
// symbol. Digits are NOT separators -- they belong to the word class and
// are grouped with adjacent letters by SplitText.
inline bool IsSeparatorToken(std::string_view text) {
  return IsPunctuationToken(text);
}

std::vector<std::string_view> SplitText(std::string_view text, const std::string_view space);

// CN-mode variant of SplitText. After running normal SplitText, each
// resulting piece is further split at Han / non-Han boundaries: Han
// runs are passed through `cn_cut` (a Unigram word segmenter), non-Han
// runs are kept verbatim. A leading space sentinel inside a piece is
// peeled off as a standalone token if (and only if) the piece would
// otherwise attach the space to a Han character — Han words never
// carry an attached space prefix.
//
// Returns owned strings (not views into `text`) because cutter outputs
// are owned strings.
using CnCutFn = std::function<std::vector<std::string>(std::string_view)>;
std::vector<std::string> SplitTextCn(std::string_view text,
                                     std::string_view space,
                                     const CnCutFn& cn_cut);

} // namespace ustr
