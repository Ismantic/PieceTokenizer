#include "ustr.h"

#include "regex_tokenizer.h"

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

static bool IsWhitespaceCodepoint(uint32_t cp) {
    if (cp >= 0x09 && cp <= 0x0D) return true;   // \t \n \v \f \r
    if (cp == 0x20) return true;                 // space
    if (cp == 0x85) return true;                 // NEL
    if (cp == 0xA0) return true;                 // NBSP
    if (cp == 0x1680) return true;               // Ogham space
    if (cp >= 0x2000 && cp <= 0x200A) return true; // en/em spaces
    if (cp == 0x2028 || cp == 0x2029) return true; // line/para sep
    if (cp == 0x202F) return true;               // narrow NBSP
    if (cp == 0x205F) return true;               // medium math space
    if (cp == 0x3000) return true;               // ideographic space
    return false;
}

bool IsWordChar(uint32_t cp) {
    // ASCII letters and digits.
    if (cp >= '0' && cp <= '9') return true;
    if (cp >= 'A' && cp <= 'Z') return true;
    if (cp >= 'a' && cp <= 'z') return true;

    // Latin-1 Supplement letters (excluding × U+00D7 and ÷ U+00F7).
    if (cp >= 0x00C0 && cp <= 0x00FF && cp != 0x00D7 && cp != 0x00F7) return true;

    // Latin Extended-A / B, IPA Extensions, Spacing Modifier Letters.
    if (cp >= 0x0100 && cp <= 0x02FF) return true;

    // Combining Diacritical Marks (continue a word).
    if (cp >= 0x0300 && cp <= 0x036F) return true;

    // Greek / Cyrillic / Armenian / Hebrew / Arabic / Syriac / Thaana / NKo.
    if (cp >= 0x0370 && cp <= 0x07FF) return true;

    // Samaritan / Mandaic / Syriac Supplement.
    if (cp >= 0x0800 && cp <= 0x085F) return true;

    // Devanagari..Khmer, Mongolian, Limbu, Tai Le, New Tai Lue, Khmer Symbols,
    // Buginese, Tai Tham, Combining Marks Extended, Balinese, Sundanese,
    // Batak, Lepcha, Ol Chiki, Cyrillic Ext-C, Georgian Ext, Sundanese Sup,
    // Vedic Ext.
    if (cp >= 0x0900 && cp <= 0x1CFF) return true;

    // Phonetic Extensions, Combining Marks Supplement, Latin Extended Additional,
    // Greek Extended.
    if (cp >= 0x1D00 && cp <= 0x1FFF) return true;

    // CJK Radicals Supplement, Kangxi Radicals, Ideographic Description Chars.
    if (cp >= 0x2E80 && cp <= 0x2FFF) return true;

    // Hiragana, Katakana, Bopomofo, Hangul Compatibility Jamo, Kanbun,
    // Bopomofo Extended, CJK Strokes, Katakana Phonetic Extensions.
    if (cp >= 0x3040 && cp <= 0x31FF) return true;

    // Enclosed CJK Letters and Months, CJK Compatibility.
    if (cp >= 0x3200 && cp <= 0x33FF) return true;

    // CJK Unified Ideographs Extension A + CJK Unified Ideographs.
    if (cp >= 0x3400 && cp <= 0x9FFF) return true;

    // Yi Syllables, Yi Radicals, Lisu, Vai, Cyrillic Ext-B, Bamum,
    // Latin Ext-D, Syloti Nagri, Phags-pa, Saurashtra, Devanagari Ext,
    // Kayah Li, Rejang, Hangul Jamo Ext-A, Javanese, Myanmar Ext-B, Cham,
    // Myanmar Ext-A, Tai Viet, Meetei Mayek Ext, Ethiopic Ext-A, Meetei Mayek.
    if (cp >= 0xA000 && cp <= 0xABFF) return true;

    // Hangul Syllables + Hangul Jamo Extended-B.
    if (cp >= 0xAC00 && cp <= 0xD7FF) return true;

    // CJK Compatibility Ideographs.
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;

    // Alphabetic Presentation Forms, Arabic Presentation Forms-A.
    if (cp >= 0xFB00 && cp <= 0xFDFF) return true;

    // Arabic Presentation Forms-B.
    if (cp >= 0xFE70 && cp <= 0xFEFF) return true;

    // Fullwidth digits.
    if (cp >= 0xFF10 && cp <= 0xFF19) return true;
    // Fullwidth Latin letters.
    if ((cp >= 0xFF21 && cp <= 0xFF3A) ||
        (cp >= 0xFF41 && cp <= 0xFF5A)) return true;
    // Halfwidth Katakana.
    if (cp >= 0xFF66 && cp <= 0xFF9F) return true;
    // Halfwidth Hangul.
    if (cp >= 0xFFA0 && cp <= 0xFFDC) return true;

    // Supplementary Multilingual Plane letter/script blocks
    // (Linear B through Old Persian, plus misc.).
    if (cp >= 0x10000 && cp <= 0x103FF) return true;
    // Coptic, Gothic, Old Permic, Ugaritic, Old Persian, Deseret, Shavian,
    // Osmanya, Osage, Elbasan, Caucasian Albanian, Vithkuqi, Linear A,
    // Cypriot Syllabary, Imperial Aramaic, Palmyrene, Nabataean, Hatran,
    // Phoenician, Lydian, Meroitic, Kharoshthi, Old South Arabian, ...
    if (cp >= 0x10400 && cp <= 0x10FFF) return true;

    // CJK Unified Ideographs Extensions B..H (astral plane).
    if (cp >= 0x20000 && cp <= 0x323AF) return true;

    return false;
}

bool IsHan(uint32_t cp) {
    // CJK Unified Ideographs Extension A.
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;
    // CJK Unified Ideographs.
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    // CJK Compatibility Ideographs.
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    // CJK Unified Ideographs Extensions B..H (supplementary plane).
    if (cp >= 0x20000 && cp <= 0x323AF) return true;
    return false;
}

bool IsPunctuationToken(std::string_view text) {
    if (text.empty()) return false;
    size_t mblen;
    const uint32_t cp = DecodeUTF8(text, &mblen);
    if (mblen == 0 || mblen != text.size()) return false;

    // Anything that isn't a word character, whitespace, or a C0/C1 control
    // character is treated as punctuation/symbol (matches rustbpe's
    // [^\s\p{L}\p{N}] punctuation class).
    if (IsWordChar(cp)) return false;
    if (IsWhitespaceCodepoint(cp)) return false;
    if (cp < 0x20 || cp == 0x7F) return false;      // C0 controls
    if (cp >= 0x80 && cp <= 0x9F) return false;     // C1 controls
    return true;
}

std::vector<std::string_view> SplitText(std::string_view text,
                                        std::string_view space,
                                        int cut) {
    struct CachedTokenizer {
        std::string space;
        int cut;
        std::unique_ptr<regex::TokenizerRegex> tokenizer;
    };
    thread_local std::vector<CachedTokenizer> tokenizers;
    for (const auto& cached : tokenizers) {
        if (cached.cut == cut && cached.space == space) {
            return cached.tokenizer->Split(text);
        }
    }
    tokenizers.push_back({
        std::string(space), cut,
        std::make_unique<regex::TokenizerRegex>(space, cut),
    });
    return tokenizers.back().tokenizer->Split(text);
}

std::vector<std::string> SplitTextCn(std::string_view text,
                                     std::string_view space,
                                     const CnCutFn& cn_cut,
                                     int cut,
                                     bool split_digits) {
    std::vector<std::string> result;
    const auto pieces = SplitText(text, space, cut);

    // SplitText already splits at Han / non-Han boundaries and peels
    // space prefixes from Han runs. Each piece is either entirely Han
    // or contains no Han at all. Pass Han pieces through cn_cut. When
    // split_digits is true, also split digit runs into single codepoints.
    for (const auto piece : pieces) {
        if (piece.empty()) continue;

        size_t mb = 0;
        const uint32_t cp = DecodeUTF8(piece.data(),
                                       piece.data() + piece.size(), &mb);
        if (IsHan(cp) && cn_cut) {
            for (auto& w : cn_cut(piece)) result.emplace_back(std::move(w));
        } else if (split_digits && IsDigitCodepoint(cp)) {
            const char* p = piece.data();
            const char* end = p + piece.size();
            while (p < end) {
                const int n = std::min<int>(OneUTF8Size(p), end - p);
                result.emplace_back(p, n);
                p += n;
            }
        } else {
            result.emplace_back(piece);
        }
    }

    return result;
}

} // namespace
