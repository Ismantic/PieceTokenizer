#include "ustr.h"

#include "common.h"
#include "test.h"

#include <iostream>

namespace ustr {

TEST(UstrTest, EncodePODTest) {
    std::string tmp;
    {
        float v = 0.0;
        tmp = EncodePOD<float>(10.0);
        EXPECT_TRUE(DecodePOD<float>(tmp, &v));
        EXPECT_EQ(10.0, v);
    }

    {
        piece::float_t v = 0.0;
        tmp = EncodePOD<piece::float_t>(10.0);
        EXPECT_TRUE(DecodePOD<piece::float_t>(tmp, &v));
        EXPECT_EQ(10.0, v);
    }

    {
        int32_t v = 0;
        tmp = ustr::EncodePOD<int32_t>(10);
        EXPECT_TRUE(DecodePOD<int32_t>(tmp, &v));
        EXPECT_EQ(10, v);
    }

    {
        int16_t v = 0;
        tmp = EncodePOD<int16_t>(10);
        EXPECT_TRUE(DecodePOD<int16_t>(tmp, &v));
        EXPECT_EQ(10, v);
    }

    {
        int64_t v = 0;
        tmp = ustr::EncodePOD<int64_t>(10);
        EXPECT_TRUE(DecodePOD<int64_t>(tmp, &v));
        EXPECT_EQ(10, v);
    }

    // Invalid data
    {
        int32_t v = 0;
        tmp = EncodePOD<int64_t>(10);
        EXPECT_FALSE(DecodePOD<int32_t>(tmp, &v));
    }
}

// U+2581 ▁ (SpaceSymbol used by SentencePiece-style normalizers).
static const char* kSp = "\xE2\x96\x81";

TEST(UstrTest, SplitTextLeadingSpaceAttachesToWord) {
    std::string input = std::string("hello") + kSp + "world";
    auto r = SplitText(input, kSp);
    ASSERT_EQ(static_cast<size_t>(2), r.size());
    EXPECT_EQ(std::string("hello"), std::string(r[0]));
    EXPECT_EQ(std::string(kSp) + "world", std::string(r[1]));
}

TEST(UstrTest, SplitTextConsecutiveSpacesStandaloneThenAttached) {
    // "a▁▁b" -> ["a", "▁", "▁b"]  (1 extra space as standalone, last attaches)
    std::string input = std::string("a") + kSp + kSp + "b";
    auto r = SplitText(input, kSp);
    ASSERT_EQ(static_cast<size_t>(3), r.size());
    EXPECT_EQ(std::string("a"), std::string(r[0]));
    EXPECT_EQ(std::string(kSp), std::string(r[1]));
    EXPECT_EQ(std::string(kSp) + "b", std::string(r[2]));
}

TEST(UstrTest, SplitTextThreeSpaces) {
    // "a▁▁▁b" -> ["a", "▁", "▁", "▁b"]  (each extra space standalone, last attaches)
    std::string input = std::string("a") + kSp + kSp + kSp + "b";
    auto r = SplitText(input, kSp);
    ASSERT_EQ(static_cast<size_t>(4), r.size());
    EXPECT_EQ(std::string("a"), std::string(r[0]));
    EXPECT_EQ(std::string(kSp), std::string(r[1]));
    EXPECT_EQ(std::string(kSp), std::string(r[2]));
    EXPECT_EQ(std::string(kSp) + "b", std::string(r[3]));
}

TEST(UstrTest, SplitTextSpaceBeforePunctAttaches) {
    // "hello▁," -> ["hello", "▁,"]
    std::string input = std::string("hello") + kSp + ",";
    auto r = SplitText(input, kSp);
    ASSERT_EQ(static_cast<size_t>(2), r.size());
    EXPECT_EQ(std::string("hello"), std::string(r[0]));
    EXPECT_EQ(std::string(kSp) + ",", std::string(r[1]));
}

TEST(UstrTest, SplitTextPunctAsWordPrefix) {
    // "hello,world" -> ["hello", ",world"]
    auto r = SplitText("hello,world", kSp);
    ASSERT_EQ(static_cast<size_t>(2), r.size());
    EXPECT_EQ(std::string("hello"), std::string(r[0]));
    EXPECT_EQ(std::string(",world"), std::string(r[1]));
}

TEST(UstrTest, SplitTextConsecutivePunctBecomesRun) {
    // "hello,,world" -> ["hello", ",,", "world"]
    auto r = SplitText("hello,,world", kSp);
    ASSERT_EQ(static_cast<size_t>(3), r.size());
    EXPECT_EQ(std::string("hello"), std::string(r[0]));
    EXPECT_EQ(std::string(",,"), std::string(r[1]));
    EXPECT_EQ(std::string("world"), std::string(r[2]));
}

TEST(UstrTest, SplitTextPendingSpaceGoesToPunctNotWordPrefix) {
    // "▁,a" -> ["▁,", "a"] (pending space forces alt-4 punct-run branch)
    std::string input = std::string(kSp) + ",a";
    auto r = SplitText(input, kSp);
    ASSERT_EQ(static_cast<size_t>(2), r.size());
    EXPECT_EQ(std::string(kSp) + ",", std::string(r[0]));
    EXPECT_EQ(std::string("a"), std::string(r[1]));
}

TEST(UstrTest, SplitTextDigitsSeparateFromLetters) {
    // "hello123world" -> ["hello", "123", "world"]
    auto r = SplitText("hello123world", kSp);
    ASSERT_EQ(static_cast<size_t>(3), r.size());
    EXPECT_EQ(std::string("hello"), std::string(r[0]));
    EXPECT_EQ(std::string("123"), std::string(r[1]));
    EXPECT_EQ(std::string("world"), std::string(r[2]));
}

TEST(UstrTest, SplitTextDigitRunSpaceStandalone) {
    // "year▁2024" -> ["year", "▁", "2024"] (space doesn't attach to digits)
    std::string input = std::string("year") + kSp + "2024";
    auto r = SplitText(input, kSp);
    ASSERT_EQ(static_cast<size_t>(3), r.size());
    EXPECT_EQ(std::string("year"), std::string(r[0]));
    EXPECT_EQ(std::string(kSp), std::string(r[1]));
    EXPECT_EQ(std::string("2024"), std::string(r[2]));
}

TEST(UstrTest, SplitTextCJKWordRun) {
    // "你好世界" -> single run.
    auto r = SplitText("你好世界", kSp);
    ASSERT_EQ(static_cast<size_t>(1), r.size());
    EXPECT_EQ(std::string("你好世界"), std::string(r[0]));
}

TEST(UstrTest, SplitTextCJKPunctBreaksRun) {
    // "你好，世界" -> ["你好", "，", "世界"]
    // Punct is NOT absorbed into Han word runs.
    auto r = SplitText("你好，世界", kSp);
    ASSERT_EQ(static_cast<size_t>(3), r.size());
    EXPECT_EQ(std::string("你好"), std::string(r[0]));
    EXPECT_EQ(std::string("，"), std::string(r[1]));
    EXPECT_EQ(std::string("世界"), std::string(r[2]));
}

TEST(UstrTest, SplitTextHanSpacePeeled) {
    // "▁你好" -> ["▁", "你好"] (space peeled before Han)
    std::string input = std::string(kSp) + "你好";
    auto r = SplitText(input, kSp);
    ASSERT_EQ(static_cast<size_t>(2), r.size());
    EXPECT_EQ(std::string(kSp), std::string(r[0]));
    EXPECT_EQ(std::string("你好"), std::string(r[1]));
}

TEST(UstrTest, SplitTextHanNonHanBoundary) {
    // "hello你好world" -> ["hello", "你好", "world"]
    auto r = SplitText("hello你好world", kSp);
    ASSERT_EQ(static_cast<size_t>(3), r.size());
    EXPECT_EQ(std::string("hello"), std::string(r[0]));
    EXPECT_EQ(std::string("你好"), std::string(r[1]));
    EXPECT_EQ(std::string("world"), std::string(r[2]));
}

TEST(UstrTest, SplitTextEmojiIsPunct) {
    // "hi🙂bye" -> ["hi", "🙂bye"] (emoji is not word, followed by word).
    auto r = SplitText("hi\xF0\x9F\x99\x82""bye", kSp);
    ASSERT_EQ(static_cast<size_t>(2), r.size());
    EXPECT_EQ(std::string("hi"), std::string(r[0]));
    EXPECT_EQ(std::string("\xF0\x9F\x99\x82""bye"), std::string(r[1]));
}

TEST(UstrTest, IsHanBasics) {
    EXPECT_TRUE(IsHan(0x4E2D));   // 中
    EXPECT_TRUE(IsHan(0x9FFF));   // last in CJK Unified
    EXPECT_TRUE(IsHan(0x3400));   // first in Ext A
    EXPECT_TRUE(IsHan(0x20000));  // Ext B
    EXPECT_TRUE(IsHan(0xF900));   // CJK Compat
    EXPECT_FALSE(IsHan('a'));
    EXPECT_FALSE(IsHan('5'));
    EXPECT_FALSE(IsHan(0x3042));  // あ
    EXPECT_FALSE(IsHan(0xAC00));  // 가
    EXPECT_FALSE(IsHan(0xFF0C));  // ，
}

// Test cutter: cuts each Han codepoint into its own token, mimicking
// a per-character cutter so we can verify SplitTextCn's structure.
static std::vector<std::string> CharCutter(std::string_view s) {
    std::vector<std::string> out;
    const char* p = s.data();
    const char* end = p + s.size();
    while (p < end) {
        const int n = std::min<int>(OneUTF8Size(p), end - p);
        out.emplace_back(p, n);
        p += n;
    }
    return out;
}

TEST(UstrTest, SplitTextCnBasicMix) {
    // "Tom▁他是英国人Bat" — Han run cut per-char, space peeled off
    // because the next thing after it is Han.
    std::string input = std::string("Tom") + kSp + "他是英国人Bat";
    auto r = SplitTextCn(input, kSp, CharCutter);
    ASSERT_EQ(static_cast<size_t>(8), r.size());
    EXPECT_EQ(std::string("Tom"), r[0]);
    EXPECT_EQ(std::string(kSp), r[1]);
    EXPECT_EQ(std::string("他"), r[2]);
    EXPECT_EQ(std::string("是"), r[3]);
    EXPECT_EQ(std::string("英"), r[4]);
    EXPECT_EQ(std::string("国"), r[5]);
    EXPECT_EQ(std::string("人"), r[6]);
    EXPECT_EQ(std::string("Bat"), r[7]);
}

TEST(UstrTest, SplitTextCnSpaceAttachesToNonHan) {
    // "你好▁world" — SplitText emits ["你好", "▁world"]; non-Han keeps space.
    std::string input = std::string("你好") + kSp + "world";
    auto r = SplitTextCn(input, kSp, CharCutter);
    ASSERT_EQ(static_cast<size_t>(3), r.size());
    EXPECT_EQ(std::string("你"), r[0]);
    EXPECT_EQ(std::string("好"), r[1]);
    EXPECT_EQ(std::string(kSp) + "world", r[2]);
}

TEST(UstrTest, SplitTextCnStandaloneSpacePassthrough) {
    // "a▁▁世界" — each extra space standalone, Han separate.
    std::string input = std::string("a") + kSp + kSp + "世界";
    auto r = SplitTextCn(input, kSp, CharCutter);
    ASSERT_EQ(static_cast<size_t>(5), r.size());
    EXPECT_EQ(std::string("a"), r[0]);
    EXPECT_EQ(std::string(kSp), r[1]);
    EXPECT_EQ(std::string(kSp), r[2]);
    EXPECT_EQ(std::string("世"), r[3]);
    EXPECT_EQ(std::string("界"), r[4]);
}

TEST(UstrTest, SplitTextCnPureNonHanUnchanged) {
    // No Han chars: should match SplitText output (modulo string vs view).
    std::string input = std::string("hello") + kSp + "world";
    auto r = SplitTextCn(input, kSp, CharCutter);
    ASSERT_EQ(static_cast<size_t>(2), r.size());
    EXPECT_EQ(std::string("hello"), r[0]);
    EXPECT_EQ(std::string(kSp) + "world", r[1]);
}

TEST(UstrTest, IsWordCharBasics) {
    EXPECT_TRUE(IsWordChar('a'));
    EXPECT_TRUE(IsWordChar('Z'));
    EXPECT_TRUE(IsWordChar('5'));
    EXPECT_TRUE(IsWordChar(0x4E2D));  // 中
    EXPECT_TRUE(IsWordChar(0x3042));  // あ
    EXPECT_TRUE(IsWordChar(0xAC00));  // 가
    EXPECT_TRUE(IsWordChar(0x00E9));  // é
    EXPECT_TRUE(IsWordChar(0x03B1));  // α
    EXPECT_TRUE(IsWordChar(0x0410));  // А (Cyrillic)
    EXPECT_FALSE(IsWordChar(' '));
    EXPECT_FALSE(IsWordChar(','));
    EXPECT_FALSE(IsWordChar(0xFF0C));  // ，
    EXPECT_FALSE(IsWordChar(0x3002));  // 。
    EXPECT_FALSE(IsWordChar(0x00D7));  // ×
    EXPECT_FALSE(IsWordChar(0x1F642)); // 🙂
}

// ----------- cut=1 tests -----------

TEST(UstrTest, SplitTextCut1English) {
    // "▁Hello,▁World!" -> "▁" "Hello" "," "▁" "World" "!"
    std::string input = std::string(kSp) + "Hello," + std::string(kSp) + "World!";
    auto r = SplitText(input, kSp, 1);
    ASSERT_EQ(static_cast<size_t>(6), r.size());
    EXPECT_EQ(std::string(kSp), std::string(r[0]));
    EXPECT_EQ(std::string("Hello"), std::string(r[1]));
    EXPECT_EQ(std::string(","), std::string(r[2]));
    EXPECT_EQ(std::string(kSp), std::string(r[3]));
    EXPECT_EQ(std::string("World"), std::string(r[4]));
    EXPECT_EQ(std::string("!"), std::string(r[5]));
}

TEST(UstrTest, SplitTextCut1Contraction) {
    // "▁don't" -> "don" "'" "t"
    std::string input = std::string(kSp) + "don't";
    auto r = SplitText(input, kSp, 1);
    ASSERT_EQ(static_cast<size_t>(4), r.size());
    EXPECT_EQ(std::string(kSp), std::string(r[0]));
    EXPECT_EQ(std::string("don"), std::string(r[1]));
    EXPECT_EQ(std::string("'"), std::string(r[2]));
    EXPECT_EQ(std::string("t"), std::string(r[3]));
}

TEST(UstrTest, SplitTextCut1Chinese) {
    // "▁你好，世界。" -> "▁" "你好" "，" "世界" "。"
    std::string input = std::string(kSp)
        + "\xe4\xbd\xa0\xe5\xa5\xbd"      // 你好
        + "\xef\xbc\x8c"                    // ，
        + "\xe4\xb8\x96\xe7\x95\x8c"      // 世界
        + "\xe3\x80\x82";                   // 。
    auto r = SplitText(input, kSp, 1);
    ASSERT_EQ(static_cast<size_t>(5), r.size());
    EXPECT_EQ(std::string(kSp), std::string(r[0]));
    EXPECT_EQ(std::string("\xe4\xbd\xa0\xe5\xa5\xbd"), std::string(r[1]));
    EXPECT_EQ(std::string("\xef\xbc\x8c"), std::string(r[2]));
    EXPECT_EQ(std::string("\xe4\xb8\x96\xe7\x95\x8c"), std::string(r[3]));
    EXPECT_EQ(std::string("\xe3\x80\x82"), std::string(r[4]));
}

TEST(UstrTest, SplitTextCut1MixedTypes) {
    // "hello123你好" -> "hello" "123" "你好"
    std::string input = std::string("hello123")
        + "\xe4\xbd\xa0\xe5\xa5\xbd";  // 你好
    auto r = SplitText(input, kSp, 1);
    ASSERT_EQ(static_cast<size_t>(3), r.size());
    EXPECT_EQ(std::string("hello"), std::string(r[0]));
    EXPECT_EQ(std::string("123"), std::string(r[1]));
    EXPECT_EQ(std::string("\xe4\xbd\xa0\xe5\xa5\xbd"), std::string(r[2]));
}

// ----------- GPT-4 alignment tests -----------

TEST(UstrTest, SplitTextPunctPrefixOnlyForLetters) {
    // "$100" -> "$" "100" (punct does NOT prefix digit runs)
    auto r = SplitText("$100", kSp);
    ASSERT_EQ(static_cast<size_t>(2), r.size());
    EXPECT_EQ(std::string("$"), std::string(r[0]));
    EXPECT_EQ(std::string("100"), std::string(r[1]));
}

TEST(UstrTest, SplitTextPunctPrefixForLetters) {
    // "$hello" -> "$hello" (punct prefixes letter run)
    auto r = SplitText("$hello", kSp);
    ASSERT_EQ(static_cast<size_t>(1), r.size());
    EXPECT_EQ(std::string("$hello"), std::string(r[0]));
}

TEST(UstrTest, SplitTextSpacePunctDigit) {
    // "▁$100" -> "▁$" "100" (space attaches to punct, digit separate)
    std::string input = std::string(kSp) + "$100";
    auto r = SplitText(input, kSp);
    ASSERT_EQ(static_cast<size_t>(2), r.size());
    EXPECT_EQ(std::string(kSp) + "$", std::string(r[0]));
    EXPECT_EQ(std::string("100"), std::string(r[1]));
}

TEST(UstrTest, SplitTextDigitPercent) {
    // "100%" -> "100" "%"
    auto r = SplitText("100%", kSp);
    ASSERT_EQ(static_cast<size_t>(2), r.size());
    EXPECT_EQ(std::string("100"), std::string(r[0]));
    EXPECT_EQ(std::string("%"), std::string(r[1]));
}

TEST(UstrTest, SplitTextSpace24h) {
    // "▁24h" -> "▁" "24" "h"
    std::string input = std::string(kSp) + "24h";
    auto r = SplitText(input, kSp);
    ASSERT_EQ(static_cast<size_t>(3), r.size());
    EXPECT_EQ(std::string(kSp), std::string(r[0]));
    EXPECT_EQ(std::string("24"), std::string(r[1]));
    EXPECT_EQ(std::string("h"), std::string(r[2]));
}

TEST(UstrTest, SplitTextContractionLL) {
    // "they'll" -> "they" "'ll"
    auto r = SplitText("they'll", kSp);
    ASSERT_EQ(static_cast<size_t>(2), r.size());
    EXPECT_EQ(std::string("they"), std::string(r[0]));
    EXPECT_EQ(std::string("'ll"), std::string(r[1]));
}

TEST(UstrTest, SplitTextHelloCommaWorld) {
    // "hello,world" -> "hello" ",world" (punct prefixes letter run)
    auto r = SplitText("hello,world", kSp);
    ASSERT_EQ(static_cast<size_t>(2), r.size());
    EXPECT_EQ(std::string("hello"), std::string(r[0]));
    EXPECT_EQ(std::string(",world"), std::string(r[1]));
}

TEST(UstrTest, SplitTextHelloSpaceCommaWorld) {
    // "hello▁,world" -> "hello" "▁," "world"
    // (space attaches to punct, but punct+space can't prefix)
    std::string input = std::string("hello") + kSp + ",world";
    auto r = SplitText(input, kSp);
    ASSERT_EQ(static_cast<size_t>(3), r.size());
    EXPECT_EQ(std::string("hello"), std::string(r[0]));
    EXPECT_EQ(std::string(kSp) + ",", std::string(r[1]));
    EXPECT_EQ(std::string("world"), std::string(r[2]));
}

} // namespace ustr
