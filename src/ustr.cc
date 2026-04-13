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

// Whitespace codepoints per Unicode (subset sufficient for pre-normalized text).
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

// Pre-tokenizes text into runs, with behavior modeled after rustbpe's
// GPT-4 regex (`[^\r\n\p{L}\p{N}]?+\p{L}+ | \p{N}{1,3} | ?[^\s\p{L}\p{N}]++...`).
// Rules:
//   1. Word runs are maximal sequences of word characters (letters, digits,
//      CJK, ...). A single space may attach as a prefix to the run.
//   2. Punct runs are maximal sequences of non-word non-space characters.
//      A single space may attach as a prefix. Additionally, exactly 1 punct
//      char immediately before a word run is absorbed into that word run
//      as a prefix (only when no pending space is waiting).
//   3. Consecutive spaces: N-1 spaces become standalone tokens; the last
//      space attaches to the following run.
// Assumes the normalizer has already stripped leading/trailing whitespace
// and replaced ' ' with the `space` sentinel (which may be multi-byte).
std::vector<std::string_view> SplitText(std::string_view text,
                                        std::string_view space,
                                        int cut) {
    std::vector<std::string_view> result;
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    if (begin >= end) return result;

    // cut=1: spaces and punctuation each become independent tokens.
    if (cut == 1) {
        const char* p = begin;
        while (p < end) {
            const int clen = std::min<int>(OneUTF8Size(p), end - p);
            std::string_view ch(p, clen);

            if (ch == space) {
                // Space → standalone token.
                result.emplace_back(p, clen);
                p += clen;
            } else if (IsPunctuationToken(ch)) {
                // Punctuation → standalone token.
                result.emplace_back(p, clen);
                p += clen;
            } else {
                // Word char → consume maximal run.
                const char* run_start = p;
                p += clen;
                while (p < end) {
                    const int wlen = std::min<int>(OneUTF8Size(p), end - p);
                    std::string_view wch(p, wlen);
                    if (wch == space || IsPunctuationToken(wch)) break;
                    p += wlen;
                }
                result.emplace_back(run_start, p - run_start);
            }
        }
        return result;
    }

    auto char_len = [&](const char* p) -> int {
        return std::min<int>(ustr::OneUTF8Size(p), end - p);
    };

    enum Kind { kSpace, kWord, kPunct };
    auto classify = [&](const char* p, int len) -> Kind {
        std::string_view c(p, len);
        if (c == space) return kSpace;
        size_t mblen = 0;
        const uint32_t cp = DecodeUTF8(p, end, &mblen);
        if (IsWordChar(cp)) return kWord;
        return kPunct;
    };

    // pending_space, when non-null, points at the most recent space char
    // that has not yet been committed to a run. Invariant: it is always
    // immediately before `begin`, so [pending_space, run_end) is a
    // contiguous slice we can emit as a single string_view.
    const char* pending_space = nullptr;
    int pending_space_len = 0;

    while (begin < end) {
        const int clen = char_len(begin);
        const Kind kind = classify(begin, clen);

        if (kind == kSpace) {
            if (pending_space != nullptr) {
                // Two spaces in a row: flush the previous one as a
                // standalone token.
                result.emplace_back(pending_space, pending_space_len);
            }
            pending_space = begin;
            pending_space_len = clen;
            begin += clen;
            continue;
        }

        if (kind == kWord) {
            // Determine Han-ness of first word char for boundary splitting.
            size_t mb = 0;
            const uint32_t first_cp = DecodeUTF8(begin, end, &mb);
            const bool first_han = IsHan(first_cp);

            const char* run_start;
            if (pending_space != nullptr) {
                if (first_han) {
                    // Han words never carry a space prefix — emit standalone.
                    result.emplace_back(pending_space, pending_space_len);
                    run_start = begin;
                } else {
                    run_start = pending_space;
                }
                pending_space = nullptr;
            } else {
                run_start = begin;
            }

            // Consume word chars with the same Han-ness.
            const char* run_end = begin;
            while (run_end < end) {
                const int wlen = char_len(run_end);
                if (classify(run_end, wlen) != kWord) break;
                size_t mb2 = 0;
                const uint32_t cp = DecodeUTF8(run_end, end, &mb2);
                if (IsHan(cp) != first_han) break;
                run_end += wlen;
            }
            result.emplace_back(run_start, run_end - run_start);
            begin = run_end;
            continue;
        }

        // kind == kPunct
        // Special case: when there's no pending space and the very next
        // character is a non-Han word char, the punct is absorbed as a
        // 1-char prefix of the upcoming word run (rustbpe alt 2 behavior).
        // Han word runs never absorb a preceding punct.
        if (pending_space == nullptr && begin + clen < end) {
            const int nlen = char_len(begin + clen);
            if (classify(begin + clen, nlen) == kWord) {
                size_t mb = 0;
                const uint32_t wcp = DecodeUTF8(begin + clen, end, &mb);
                if (!IsHan(wcp)) {
                    const char* run_start = begin;
                    const char* run_end = begin + clen;
                    while (run_end < end) {
                        const int wlen = char_len(run_end);
                        if (classify(run_end, wlen) != kWord) break;
                        size_t mb2 = 0;
                        const uint32_t cp = DecodeUTF8(run_end, end, &mb2);
                        if (IsHan(cp)) break;
                        run_end += wlen;
                    }
                    result.emplace_back(run_start, run_end - run_start);
                    begin = run_end;
                    continue;
                }
            }
        }

        // Regular punct run, optionally prefixed with pending space.
        const char* run_start =
            pending_space != nullptr ? pending_space : begin;
        pending_space = nullptr;
        const char* run_end = begin;
        while (run_end < end) {
            const int plen = char_len(run_end);
            if (classify(run_end, plen) != kPunct) break;
            run_end += plen;
        }
        result.emplace_back(run_start, run_end - run_start);
        begin = run_end;
    }

    // Trailing space (normalizer usually strips this, but be safe).
    if (pending_space != nullptr) {
        result.emplace_back(pending_space, pending_space_len);
    }

    return result;
}

std::vector<std::string> SplitTextCn(std::string_view text,
                                     std::string_view space,
                                     const CnCutFn& cn_cut,
                                     int cut) {
    std::vector<std::string> result;
    const auto pieces = SplitText(text, space, cut);

    // SplitText already splits at Han / non-Han boundaries and peels
    // space prefixes from Han runs. Each piece is either entirely Han
    // or contains no Han at all. Just pass Han pieces through cn_cut.
    for (const auto piece : pieces) {
        if (piece.empty()) continue;

        size_t mb = 0;
        const uint32_t cp = DecodeUTF8(piece.data(),
                                       piece.data() + piece.size(), &mb);
        if (IsHan(cp)) {
            for (auto& w : cn_cut(piece)) result.emplace_back(std::move(w));
        } else {
            result.emplace_back(piece);
        }
    }

    return result;
}

} // namespace
