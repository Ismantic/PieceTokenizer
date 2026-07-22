#include <string>
#include <unordered_map>
#include <vector>

#include "bytepiece_tokenizer.h"
#include "normalizer.h"
#include "test.h"

namespace {

std::unordered_map<std::string, piece::float_t> MakeEnglishDict() {
    return {
        {"hello", 100.0},
        {"world", 80.0},
        {"he", 10.0},
        {"llo", 10.0},
        {"wor", 10.0},
        {"ld", 10.0},
        {"h", 1.0},
        {"e", 1.0},
        {"l", 1.0},
        {"o", 1.0},
        {"w", 1.0},
        {"r", 1.0},
        {"d", 1.0},
        {" ", 1.0},
    };
}

std::unordered_map<std::string, piece::float_t> MakeChineseDict() {
    return {
        {"我们", 8.0},
        {"在", 6.0},
        {"学习", 7.0},
        {"人工智能", 10.0},
        {"人工", 3.0},
        {"智能", 3.0},
        {"我", 1.0},
        {"们", 1.0},
        {"学", 1.0},
        {"习", 1.0},
        {"人", 1.0},
        {"工", 1.0},
        {"智", 1.0},
        {"能", 1.0},
    };
}

}  // namespace

TEST(BytePieceTokenizerTest, PrefersHigherScoreLongPieces) {
    piece::BytePieceTokenizer tokenizer(MakeEnglishDict());

    const auto tokens = tokenizer.Tokenize("hello world");
    ASSERT_EQ(3u, tokens.size());
    EXPECT_EQ("hello", tokens[0]);
    EXPECT_EQ(" ", tokens[1]);
    EXPECT_EQ("world", tokens[2]);
}

TEST(BytePieceTokenizerTest, FallsBackToCharacterBoundaries) {
    piece::BytePieceTokenizer tokenizer(MakeChineseDict());

    const auto tokens = tokenizer.Tokenize("我们在学习未知");
    ASSERT_EQ(5u, tokens.size());
    EXPECT_EQ("我们", tokens[0]);
    EXPECT_EQ("在", tokens[1]);
    EXPECT_EQ("学习", tokens[2]);
    EXPECT_EQ("未", tokens[3]);
    EXPECT_EQ("知", tokens[4]);
}

TEST(NormalizerTest, NmtNfkcNormalizesCompatibilityChars) {
    piece::PreTokenizerSpec spec;
    spec.SetName("NMT_NFKC");

    piece::Normalizer normalizer(spec);
    EXPECT_EQ("123", normalizer.Normalize("①②③"));
}

int main() {
    return test::RunTests();
}
