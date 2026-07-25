#include <string>
#include <unordered_map>
#include <vector>

#include "bytepiece_tokenizer.h"
#include "cut.h"
#include "normalizer.h"
#include "piece_spec.h"
#include "piece_tokenizer.h"
#include "sentencepiece_tokenizer.h"
#include "sentencepiece_counter.h"
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

TEST(SentencePieceTokenizerTest, HonorsIsolatedPretokenBoundaries) {
    piece::Model model;
    auto* spec = model.GetMutablePreTokenizerSpec();
    spec->SetName("no");
    spec->SetSpace(" ");
    spec->SetCut(1);

    for (const auto& [text, score] :
         std::vector<std::pair<std::string, float>>{
             {"a", -3.0f}, {",", -4.0f}, {"b", -5.0f}, {"a,", 0.0f}}) {
        auto* token = model.InsertPieces();
        token->SetPiece(text);
        token->SetScore(score);
    }

    piece::SentencePieceTokenizer tokenizer(model);
    const auto tokens = tokenizer.Tokenize("a,b");
    ASSERT_EQ(3u, tokens.size());
    EXPECT_EQ("a", tokens[0]);
    EXPECT_EQ(",", tokens[1]);
    EXPECT_EQ("b", tokens[2]);
}

TEST(SentencePieceTokenizerTest, HonorsSplitDigitPretokenBoundaries) {
    piece::Model model;
    auto* spec = model.GetMutablePreTokenizerSpec();
    spec->SetName("no");
    spec->SetSpace(" ");
    spec->SetSplitDigits(true);

    for (const auto& [text, score] :
         std::vector<std::pair<std::string, float>>{
             {"1", -1.0f}, {"2", -2.0f}, {"12", 0.0f}}) {
        auto* token = model.InsertPieces();
        token->SetPiece(text);
        token->SetScore(score);
    }

    piece::SentencePieceTokenizer tokenizer(model);
    const auto tokens = tokenizer.Tokenize("12");
    ASSERT_EQ(2u, tokens.size());
    EXPECT_EQ("1", tokens[0]);
    EXPECT_EQ("2", tokens[1]);
}

TEST(SentencePieceCounterTest, RejectsInvalidDictionary) {
    piece::CounterSpec counter_spec;
    counter_spec.set_method("sentencepiece");
    counter_spec.set_dict("/definitely/missing/piece-tokenizer-dict.txt");
    piece::PreTokenizerSpec pretokenizer_spec;

    piece::SentencePieceCounter counter(counter_spec, pretokenizer_spec);
    EXPECT_FALSE(counter.Count());
}

TEST(TokenizerTest, RejectsInvalidDictionaryAtInference) {
    piece::Model model;
    const std::string missing = "/definitely/missing/piece-tokenizer-dict.txt";
    piece::PieceTokenizer piece_tokenizer(model, missing);
    piece::SentencePieceTokenizer sentencepiece_tokenizer(model, missing);
    EXPECT_FALSE(piece_tokenizer.valid());
    EXPECT_FALSE(sentencepiece_tokenizer.valid());
}

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

TEST(CnCutterTest, SegmentsWithoutBytePieceTokenizer) {
    piece::CnCutter cutter(MakeChineseDict());
    const auto tokens = cutter.Cut("我们在学习人工智能");
    ASSERT_EQ(4u, tokens.size());
    EXPECT_EQ("我们", tokens[0]);
    EXPECT_EQ("在", tokens[1]);
    EXPECT_EQ("学习", tokens[2]);
    EXPECT_EQ("人工智能", tokens[3]);
}

TEST(CnCutterTest, HandlesEmptyAndUnknownHanAtCodepointBoundaries) {
    piece::CnCutter cutter(MakeChineseDict());
    EXPECT_TRUE(cutter.Cut("").empty());

    const auto tokens = cutter.Cut("未知𠀀");
    ASSERT_EQ(3u, tokens.size());
    EXPECT_EQ("未", tokens[0]);
    EXPECT_EQ("知", tokens[1]);
    EXPECT_EQ("𠀀", tokens[2]);
}

TEST(CnCutterTest, KeepsOptimalCandidateBeyondSixteenPrefixes) {
    std::unordered_map<std::string, piece::float_t> dict;
    std::string word = "中";
    for (int i = 0; i < 20; ++i) {
        dict[word] = 1.0;
        word += "人";
    }
    dict[word] = 1e12;

    piece::CnCutter cutter(dict);
    const auto tokens = cutter.Cut(word);
    ASSERT_EQ(1u, tokens.size());
    EXPECT_EQ(word, tokens[0]);
}

TEST(TokenizerTest, PieceAndBytePieceHonorPretokenBoundaries) {
    piece::Model piece_model;
    piece_model.GetMutablePreTokenizerSpec()->SetName("no");
    piece_model.GetMutablePreTokenizerSpec()->SetSpace(" ");
    piece_model.GetMutablePreTokenizerSpec()->SetCut(1);
    for (const auto& fields :
         std::vector<std::vector<std::string>>{
             {"a", "", ""}, {",", "", ""}, {"b", "", ""},
             {"a,", "a", ","}}) {
        piece_model.InsertPieces()->SetPiece(fields[0], fields[1], fields[2]);
    }
    piece::PieceTokenizer piece_tokenizer(piece_model);
    const auto piece_tokens = piece_tokenizer.Tokenize("a,b");
    ASSERT_EQ(3u, piece_tokens.size());
    EXPECT_EQ("a", piece_tokens[0]);
    EXPECT_EQ(",", piece_tokens[1]);
    EXPECT_EQ("b", piece_tokens[2]);

    piece::Model byte_model;
    byte_model.GetMutablePreTokenizerSpec()->SetName("no");
    byte_model.GetMutablePreTokenizerSpec()->SetSpace(" ");
    byte_model.GetMutablePreTokenizerSpec()->SetCut(1);
    for (const auto& [text, score] :
         std::vector<std::pair<std::string, float>>{
             {"a", 1.0f}, {",", 1.0f}, {"b", 1.0f}, {"a,", 100.0f}}) {
        auto* token = byte_model.InsertPieces();
        token->SetPiece(text);
        token->SetScore(score);
    }
    piece::BytePieceTokenizer byte_tokenizer(byte_model);
    const auto byte_tokens = byte_tokenizer.Tokenize("a,b");
    ASSERT_EQ(3u, byte_tokens.size());
    EXPECT_EQ("a", byte_tokens[0]);
    EXPECT_EQ(",", byte_tokens[1]);
    EXPECT_EQ("b", byte_tokens[2]);
}

TEST(NormalizerTest, NmtNfkcNormalizesCompatibilityChars) {
    piece::PreTokenizerSpec spec;
    spec.SetName("NMT_NFKC");

    piece::Normalizer normalizer(spec);
    EXPECT_EQ("123", normalizer.Normalize("①②③"));
}

TEST(NormalizerTest, CompiledMapRoundTripsThroughNewTrie) {
    piece::MapBuilder::UstrMap expected = {
        {{0x41}, {0x61}},
        {{0xFF11}, {0x31}},
        {{0x4F60, 0x597D}, {0x4E16, 0x754C}},
    };
    std::string blob;
    ASSERT_TRUE(piece::MapBuilder::CompileUstrMap(expected, &blob));

    piece::MapBuilder::UstrMap actual;
    ASSERT_TRUE(piece::MapBuilder::DecompileUstrMap(blob, &actual));
    EXPECT_EQ(expected, actual);
}

TEST(TrieTest, EmptyAndMalformedArraysReturnMisses) {
    trie::DoubleArray<int> trie;
    EXPECT_EQ(-1, trie.exactMatchSearch<int>("a"));

    trie::DoubleArray<int>::ResultPair results[1];
    EXPECT_EQ(0u, trie.commonPrefixSearch("a", results, 1));

    std::size_t node_pos = 0;
    std::size_t key_pos = 0;
    EXPECT_EQ(-2, trie.traverse("a", node_pos, key_pos));

    const uint32_t malformed[] = {0xFFFFFFFFu};
    trie.set_array(malformed, 1);
    EXPECT_EQ(-1, trie.exactMatchSearch<int>("a"));
    EXPECT_EQ(0u, trie.commonPrefixSearch("a", results, 1));
    EXPECT_EQ(-2, trie.traverse("a", node_pos, key_pos));
}

TEST(ModelTest, RejectsMissingAndEmptyModels) {
    piece::Model model;
    EXPECT_FALSE(model.Load("/definitely/missing/piece-tokenizer.model"));
    EXPECT_FALSE(model.FromStr(""));
}

TEST(ModelTest, RejectsNonSequentialPieceIds) {
    const std::string data =
        "[CounterSpec]\nmethod=bytepiece\n\n"
        "[PreTokenizerSpec]\nname=no\nspace=\\x20\n\n"
        "[Pieces]\nsize=1\n1\ta\t0\t1\t\t\n";
    piece::Model model;
    EXPECT_FALSE(model.FromStr(data));
}

TEST(ModelTest, ReportsOutputOpenFailure) {
    piece::Model model;
    EXPECT_FALSE(model.Save("/definitely/missing/piece-tokenizer.model"));
}

int main() {
    return test::RunTests();
}
