#include "ustr.h"

namespace ustr {

// mblen stores the number of bytes consumed after decoding.
uint32_t DecodeUTF8(const char *begin, const char *end, size_t *mblen) {
    const size_t len = end - begin;
    if (static_cast<unsigned char>(begin[0]) < 0x80) {
        *mblen = 1;
        return static_cast<unsigned char>(begin[0]);
    } else if (len >= 2 && (begin[0] & 0xE0) == 0xC0) {
        const uint32_t cp = (((begin[0] & 0x1F) << 6) | ((begin[1] & 0x3F)));
        if (IsTrailByte(begin[1]) && cp >= 0x0080 && IsValidCodepoint(cp)) {
            *mblen = 2;
            return cp;
        }
    } else if (len >= 3 && (begin[0] & 0xF0) == 0xE0) {
        const uint32_t cp = (((begin[0] & 0x0F) << 12) | ((begin[1] & 0x3F) << 6) |
                       ((begin[2] & 0x3F)));
        if (IsTrailByte(begin[1]) && IsTrailByte(begin[2]) && cp >= 0x0800 &&
                IsValidCodepoint(cp)) {
            *mblen = 3;
            return cp;
        }
    } else if (len >= 4 && (begin[0] & 0xf8) == 0xF0) {
        const uint32_t cp = (((begin[0] & 0x07) << 18) | ((begin[1] & 0x3F) << 12) |
                       ((begin[2] & 0x3F) << 6) | ((begin[3] & 0x3F)));
        if (IsTrailByte(begin[1]) && IsTrailByte(begin[2]) &&
                IsTrailByte(begin[3]) && cp >= 0x10000 && IsValidCodepoint(cp)) {
            *mblen = 4;
            return cp;
        }
    }

    // Invalid UTF-8
    *mblen = 1;
    return UnicodeError;
}

bool IsStructurallyValid(std::string_view str) {
    const char *begin = str.data();
    const char *end = str.data() + str.size();
    size_t mblen = 0;
    while (begin < end) {
        const uint32_t c = DecodeUTF8(begin, end, &mblen);
        if (c == UnicodeError && mblen != 3) return false;
        if (!IsValidCodepoint(c)) return false;
        begin += mblen;
    }
    return true;
}

size_t EncodeUTF8(uint32_t c, char *output) {
  if (c <= 0x7F) {
    *output = static_cast<char>(c);
    return 1;
  }

  if (c <= 0x7FF) {
    output[1] = 0x80 | (c & 0x3F);
    c >>= 6;
    output[0] = 0xC0 | c;
    return 2;
  }

  // if `c` is out-of-range, convert it to REPLACEMENT CHARACTER (U+FFFD).
  // This treatment is the same as the original runetochar.
  if (c > 0x10FFFF) c = UnicodeError;

  if (c <= 0xFFFF) {
    output[2] = 0x80 | (c & 0x3F);
    c >>= 6;
    output[1] = 0x80 | (c & 0x3F);
    c >>= 6;
    output[0] = 0xE0 | c;
    return 3;
  }

  output[3] = 0x80 | (c & 0x3F);
  c >>= 6;
  output[2] = 0x80 | (c & 0x3F);
  c >>= 6;
  output[1] = 0x80 | (c & 0x3F);
  c >>= 6;
  output[0] = 0xF0 | c;

  return 4;
}

UnicodeText UTF8ToUnicodeText(std::string_view utf8) {
    UnicodeText uc;
    const char *begin = utf8.data();
    const char *end = utf8.data() + utf8.size();
    while (begin < end) {
        size_t mblen;
        const uint32_t c = DecodeUTF8(begin, end, &mblen);
        uc.push_back(c);
        begin += mblen;
    }
    return uc;
}

std::string UnicodeTextToUTF8(const UnicodeText &utext) {
    char buf[8];
    std::string result;
    for (const uint32_t c : utext) {
        const size_t mblen = EncodeUTF8(c, buf);
        result.append(buf, mblen);
    }
    return result;
}

bool IsDigitToken(std::string_view text) {
    if (text.size() == 1 && text[0] >= '0' && text[0] <= '9') return true;

    if (text.size() == 3 &&
        static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBC &&
        static_cast<unsigned char>(text[2]) >= 0x90 &&
        static_cast<unsigned char>(text[2]) <= 0x99) {
        return true;
    }

    return false;
}

bool IsPunctuationToken(std::string_view text) {
    if (text.empty()) return false;
    size_t mblen;
    const uint32_t cp = DecodeUTF8(text, &mblen);
    if (mblen == 0 || mblen != text.size()) return false;

    // ASCII punctuation: ! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~
    if ((cp >= 0x21 && cp <= 0x2F) ||
        (cp >= 0x3A && cp <= 0x40) ||
        (cp >= 0x5B && cp <= 0x60) ||
        (cp >= 0x7B && cp <= 0x7E)) return true;

    // General Punctuation: U+2000-U+206F (includes — – … ' ' " " ‹ › etc.)
    if (cp >= 0x2000 && cp <= 0x206F) return true;

    // Supplemental Punctuation: U+2E00-U+2E7F
    if (cp >= 0x2E00 && cp <= 0x2E7F) return true;

    // CJK Symbols and Punctuation: U+3000-U+303F (includes 。 「 」 〈 〉 《 》 【 】 etc.)
    if (cp >= 0x3000 && cp <= 0x303F) return true;

    // CJK Compatibility Forms: U+FE30-U+FE4F
    if (cp >= 0xFE30 && cp <= 0xFE4F) return true;

    // Small Form Variants: U+FE50-U+FE6F
    if (cp >= 0xFE50 && cp <= 0xFE6F) return true;

    // Halfwidth and Fullwidth Forms (punctuation part): U+FF01-U+FF0F, U+FF1A-U+FF20,
    // U+FF3B-U+FF40, U+FF5B-U+FF65
    if ((cp >= 0xFF01 && cp <= 0xFF0F) ||
        (cp >= 0xFF1A && cp <= 0xFF20) ||
        (cp >= 0xFF3B && cp <= 0xFF40) ||
        (cp >= 0xFF5B && cp <= 0xFF65)) return true;

    // Latin-1 Supplement punctuation: ¡ ¢ £ ¤ ¥ ¦ § ¨ © « ¬ ® ° ± ´ · » ¿ etc.
    if (cp >= 0x00A1 && cp <= 0x00BF) return true;

    return false;
}

std::vector<std::string_view> SplitText(std::string_view text, const std::string_view space) {
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    std::vector<std::string_view> result;

    if (begin >= end) return result;

    result.emplace_back(begin, 0);
    
    while (begin < end) {
        const int mblen = std::min<int>(ustr::OneUTF8Size(begin), end-begin);
        std::string_view current_char(begin, mblen);
        
        if (current_char == space) {
            if (result.back().size() == 0) {
                result.back() = std::string_view(begin, mblen);
            } else {
                result.emplace_back(begin, mblen);
            }
            result.emplace_back(begin + mblen, 0);
            begin += mblen;
            continue;
        }
        
        if (IsSeparatorToken(current_char)) {
            if (result.back().size() == 0) {
                result.back() = std::string_view(begin, mblen);
            } else {
                result.emplace_back(begin, mblen);
            }
            
            result.emplace_back(begin + mblen, 0);
            begin += mblen;
            continue;
        } 

        result.back() = std::string_view(
            result.back().data(),
            result.back().size() + mblen
        );
        
        begin += mblen;
    }
    
    result.erase(
        std::remove_if(result.begin(), result.end(), 
            [](const std::string_view& s) { return s.empty(); }),
        result.end()
    );
    
    return result;
}

} // namespace
