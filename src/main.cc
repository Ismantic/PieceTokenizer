#include <iostream>
#include <unordered_map>

#include "test.h"

#include <string>
#include <vector>
#include <memory>

#include "piece_spec.h"

#include "sentencepiece_counter.h"
#include "sentencepiece_tokenizer.h"
#include "new_normalizer.h"

#include "naive_counter.h"
#include "naive_tokenizer.h"

#include "piece_counter.h"
#include "piece_tokenizer.h"

namespace piece {

/*
void run_test_naive() {
  std::string text = "We are open-weighting two MoE models: Qwen3-235B-A22B, a large model with 235 billion total parameters and 22 billion activated parameters, and Qwen3-30B-A3B, a smaller MoE model with 30 billion total parameters and 3 billion activated parameters. Additionally, six dense models are also open-weighted, including Qwen3-32B, Qwen3-14B, Qwen3-8B, Qwen3-4B, Qwen3-1.7B, and Qwen3-0.6B, under Apache 2.0 license.";
  NaiveCounter counter;
  NaiveModel model = counter.Count(text, 300);

  NaiveTokenizer tokenizer(model);

  std::vector<int> ids;
  tokenizer.Encode("Hello world", ids);

  for (auto i : ids) {
    std::cout << i << " ";
  }
  std::cout << "\n";
  
  std::string decoded_text;
  tokenizer.Decode(ids, decoded_text);

  std::cout  << decoded_text << "\n";
}
*/

/*
void run_test_simple() {

    SimpleCounter counter(300);

    std::vector<std::string> texts = {
        "Hello, world!",
        "Another example text",
        "More training data"
    };

    SimpleModel model = counter.Count(texts);

    SimpleTokenizer tokenizer(model);

    std::string test_text = "Hello, world!";
    auto tokens = tokenizer.Tokenize(test_text);
    for (auto t : tokens) {
        std::cout << t << " ";
    }
    std::cout << std::endl;
    auto decoded = tokenizer.Detokenize(tokens);
    
    std::cout << "Decoded text: " << decoded << std::endl;
}
*/



void run_sentencepiece_counter_test_naive() {
  const std::string input_file = "sentencepiece_counter.cc";
  const std::string model_prefix = "test";
  int size = 200+256+3;

  std::cout << "In " << input_file << std::endl;
  std::cout << "Mp " << model_prefix << std::endl;

  CounterSpec counter_spec;
  counter_spec.add_input(input_file);
  counter_spec.set_vocab_size(size); // remove <unk>, <s>, </s>
  counter_spec.set_model_prefix(model_prefix);

  NormalizerSpec normalizer_spec;
  normalizer_spec.SetName("identity");

  std::cout << "SentencePieceCounter" << std::endl;

  NaiveCounter counter(counter_spec, normalizer_spec);

  counter.Count();
  counter.Save();

  Model proto;

  counter.Serialize(&proto);

  NaiveTokenizer m(proto);

  std::string text = "hello world 中文";

  //NormalizerSpec spec;
  //normalizer_spec.set_name("none");
  //spec.set_name("nmt_nfkc");
  //auto s = MapBuilder::GetPrecompiledUstrMap(spec.name(), spec.mutable_precompiled_charsmap());
  //Normalizer n(spec);

  //auto str = n.Normalize(text);
  //std::cout << str << std::endl;
  auto rs = m.Encode(text);
  for (auto& r : rs) {
    std::cout << r.first << " " << r.second << std::endl;
  }

}

void run_sentencepiece_counter_test_simple() {
  const std::string input_file = "text.txt";
  const std::string model_prefix = "test";
  int size = 1500+3;

  std::cout << "In " << input_file << std::endl;
  std::cout << "Mp " << model_prefix << std::endl;

  CounterSpec counter_spec;
  counter_spec.add_input(input_file);
  counter_spec.set_vocab_size(size); // remove <unk>, <s>, </s>
  counter_spec.set_model_prefix(model_prefix);

  NormalizerSpec normalizer_spec;
  normalizer_spec.SetName("identity");

  std::cout << "SentencePieceCounter" << std::endl;

  SimpleCounter counter(counter_spec, normalizer_spec);

  counter.Count();
  counter.Save();

  Model proto;
  //counter.Serialize(&proto);
  proto.Load("test.model");

  SimpleTokenizer m(proto);

  std::string text = "hello world 中文";

  NormalizerSpec spec;
  //normalizer_spec.set_name("none");
  //spec.set_name("nmt_nfkc");
  //auto s = MapBuilder::GetPrecompiledUstrMap(spec.name(), spec.mutable_precompiled_charsmap());

  Normalizer n(spec);

  auto str = n.Normalize(text);
  std::cout << str << std::endl;
  auto rs = m.Encode(str);
  for (auto& r : rs) {
    std::cout << r.first << " " << r.second << std::endl;
  }

}



void run_sentencepiece_counter_test_piece() {
  const std::string input_file = "text.txt";
  const std::string model_prefix = "test";
  int size = 1500+256+3;

  std::cout << "In " << input_file << std::endl;
  std::cout << "Mp " << model_prefix << std::endl;

  CounterSpec counter_spec;
  counter_spec.add_input(input_file);
  counter_spec.set_vocab_size(size); // remove <unk>, <s>, </s>
  counter_spec.set_model_prefix(model_prefix);

  NormalizerSpec normalizer_spec;

  std::cout << "SentencePieceCounter" << std::endl;

  BytePieceCounter counter(counter_spec, normalizer_spec);

  counter.Count();
  counter.Save();

  Model proto;
  proto.Load("test.model");

  //counter.Serialize(&proto);
  //proto.Save("test_x.model");

  BytePieceTokenizer m(proto);

  std::string text = "hello world 中文";
  Normalizer n(normalizer_spec);
  //std::string str = n.Normalize(text);
  std::string str = text;

  std::cout << str << std::endl;
  auto rs = m.Encode(str);
  for (auto& r : rs) {
    std::cout << r.first << " " << r.second << std::endl;
  }

  std::string u = m.Decode(rs);
  std::cout << u << std::endl;

  auto xs = m.Tokenize(n.Normalize(str));
  for (auto& r : xs) {
    std::cout << r << " ";
  }
  std::cout << std::endl;

}

void run_normalizer_test() {
  NormalizerSpec spec;
  spec.SetName("NMT_NFKC");

  Normalizer n(spec);

  //std::string str = " hello       world  ";
  std::string str = "①②③";
  std::string output;
  std::vector<size_t> n2o;
  n.Normalize(str, &output, &n2o);
  std::cout << str << " " << str.length() << " -> ";
  std::cout << output.length() << " <" << output << ">" << std::endl;
  for (size_t i = 0; i < n2o.size(); ++i) {
    std::cout << i << " : " << n2o[i] << " ";
    int n = ustr::OneUTF8Size(&str[n2o[i]]);
    std::string s = str.substr(n2o[i],n);
    std::cout << output[i] << " : " << s << std::endl;
  }

  std::cout << n.Normalize("①②③") << std::endl;
 
}

void run_test() {
  std::string_view text = "this▁is▁a▁pen.中国，美国，日本。南方-北方";
  const std::string_view space = "▁";
  auto rs = ustr::SplitText(text, space);
  for (auto s : rs) {
    std::cout << s << " ";
  }
  std::cout << std::endl;
}

void run_sentencepiece_counter_test() {
  std::cout << "run_sentencepiece_counter_test" << std::endl;
  const std::string input_file = "text.txt";
  const std::string model_prefix = "test";
  int size = 8000+256+3;

  std::cout << "In " << input_file << std::endl;
  std::cout << "Mp " << model_prefix << std::endl;

  CounterSpec counter_spec;
  counter_spec.add_input(input_file);
  counter_spec.set_vocab_size(size); // remove <unk>, <s>, </s>
  counter_spec.set_model_prefix(model_prefix);

  NormalizerSpec normalizer_spec;

  std::cout << "SentencePieceCounter" << std::endl;

  SentencePieceCounter counter(counter_spec, normalizer_spec);

  counter.Count();
  counter.Save();

  Model proto;
  proto.Load("test.model");
  //counter.Serialize(&proto);

  SentencePieceTokenizer m(proto);

  std::string text = "hello world 中文";

  std::cout << text << "\n";
  auto rs = m.Encode(text);
  for (auto& r : rs) {
    std::cout << r.first << " " << r.second << std::endl;
  }

  auto u = m.Decode(rs);
  std::cout << u << std::endl;

}

void run_test_common() {
  LOG(INFO) << "这是一条信息日志";
  LOG(WARNING) << "这是一条警告日志";
  LOG(ERROR) << "这是一条错误日志";
  LOG(FATAL) << "这是一条致命错误日志，程序将终止";
  
  // 这行不会执行，因为上面的FATAL日志已经终止了程序
  LOG(INFO) << "这条日志不会被打印";
}


}

inline void test_byte_piece_tokenizer() {
  // 创建一个简单的词典
  std::unordered_map<std::string, piece::float_t> dict = {
      {"我们", 2.0},
      {"在", 1.0},
      {"学习", 1.7},
      {"Python", 1.5},
      {"编程", 1.6},
      {"机器学习", 2.2},
      {"人工智能", 2.1},
      {"人工", 3.1},
      {"智能", 4.1},
      {"x", 2.1}
  };
  
  // 创建分词器实例
  piece::BytePieceTokenizer tokenizer(dict);
  
  // 准备测试句子
  std::vector<std::string> test_cases = {
      "我们在学习Python",
      "我们正在学习机器学习",
      "人工智能编程ABC",
      "未知词分词测试"
  };
  
  // 对每个测试句子进行分词并打印结果
  for (const auto& sentence : test_cases) {
      std::cout << "原句: " << sentence << std::endl;
      auto tokens = tokenizer.Tokenize(sentence);
      std::cout << "分词结果: ";
      for (size_t i = 0; i < tokens.size(); ++i) {
          std::cout << tokens[i];
          if (i < tokens.size() - 1) std::cout << " / ";
      }
      std::cout << std::endl << std::endl;
  }
}

inline void test_byte_piece_tokenizer_advanced() {
  // Create a more extensive dictionary with overlapping substrings
  std::unordered_map<std::string, piece::float_t> dict = {
      {"the", 5.0},
      {"quick", 4.0},
      {"brown", 3.5},
      {"fox", 3.0},
      {"jumps", 2.5},
      {"over", 2.0},
      {"lazy", 1.5},
      {"dog", 1.0},
      {"qu", 0.9},     // Substring of "quick"
      {"ick", 0.8},    // Substring of "quick"
      {"own", 0.7},    // Substring of "brown"
      {"jump", 0.6},   // Substring of "jumps"
      {"la", 0.5},     // Substring of "lazy"
      {"zy", 0.4},     // Substring of "lazy"
      {"do", 0.3},     // Substring of "dog"
      {"he", 0.8},     // Substring of "the"
      {"ro", 0.75},    // Substring of "brown"
      {"ox", 0.5},     // Substring of "fox"
      {"ver", 0.4}     // Substring of "over"
  };
  
  // Add all single characters for completeness
  for (char c = 'a'; c <= 'z'; c++) {
      dict[std::string(1, c)] = 0.1;
  }
  
  // Create tokenizer instance
  piece::BytePieceTokenizer tokenizer(dict);
  
  // Test sentences to examine how the tokenizer handles different scenarios
  std::vector<std::string> test_cases = {
      "the quick brown fox",                          // Basic case with full tokens
      "thequickbrownfox",                             // No spaces
      "the quickbrown fox",                           // Mixed spacing
      "the quick brown fox jumps over the lazy dog",  // Complete sentence
      "unknown words should be tokenized by character", // Words not in dictionary
      "fox jumping over dogs",                        // Variations of dictionary words
      "THE QUICK BROWN FOX",                          // Uppercase (to test case sensitivity)
      "123 quick brown 456"                           // Mixed with numbers
  };
  
  // Tokenize and print results
  for (const auto& sentence : test_cases) {
      std::cout << "Original: " << sentence << std::endl;
      auto tokens = tokenizer.Tokenize(sentence);
      std::cout << "Tokenized (" << tokens.size() << " tokens): ";
      for (size_t i = 0; i < tokens.size(); ++i) {
          std::cout << tokens[i];
          if (i < tokens.size() - 1) std::cout << " / ";
      }
      std::cout << std::endl;
      
      // Add score information if relevant
      std::cout << "Token scores: ";
      for (size_t i = 0; i < tokens.size(); ++i) {
          auto it = dict.find(tokens[i]);
          float score = (it != dict.end()) ? it->second : 0.0;
          std::cout << tokens[i] << "=" << score;
          if (i < tokens.size() - 1) std::cout << ", ";
      }
      std::cout << std::endl << std::endl;
  }
}

inline void test_new_byte_piece_tokenizer() {
  // 创建一个明显偏向长词的词典
  std::unordered_map<std::string, piece::float_t> dict = {
      {"hello", 100.0},       // 高频
      {"world", 80.0},        // 高频
      {"learning", 70.0},     // 高频
      {"Python", 60.0},       // 高频
      {"programming", 50.0},  // 高频
      {"machine", 40.0},      // 高频
      {"learning", 35.0},     // 高频
      {"artificial", 30.0},   // 高频
      {"intelligence", 25.0}, // 高频
      {"the", 20.0},          // 高频，常见词
      {"quick", 15.0},        // 高频，常见词
      {"brown", 10.0},        // 中频，常见词
      {"fox", 5.0},           // 中频，常见词
      // 添加单字符，但频率较低
      {"h", 1.0},
      {"e", 1.0},
      {"l", 1.0},
      {"o", 1.0},
      {"w", 1.0},
      {"r", 1.0},
      {"d", 1.0},
      {"P", 1.0},
      {"y", 1.0},
      {"t", 1.0},
      {"h", 1.0},
      {"o", 1.0},
      {"n", 1.0}
  };
  
  // 创建分词器实例
  piece::BytePieceTokenizer tokenizer(dict);
  
  // 准备测试句子
  std::vector<std::string> test_cases = {
      "hello world Python",
      "helloworld",
      "Python programming",
      "machine learning artificial intelligence"
  };
  
  // 对每个测试句子进行分词并打印结果
  for (const auto& sentence : test_cases) {
      std::cout << "原句: " << sentence << std::endl;
      auto tokens = tokenizer.Tokenize(sentence);
      std::cout << "分词结果: ";
      for (size_t i = 0; i < tokens.size(); ++i) {
          std::cout << tokens[i];
          if (i < tokens.size() - 1) std::cout << " / ";
      }
      std::cout << std::endl << std::endl;
  }
}

inline void test_byte_piece_tokenizer_x() {
  // 创建一个简单的词典，包括空格
  std::unordered_map<std::string, piece::float_t> dict = {
      {"he", 100.0},
      {"llo", 100.0},
      {"wor", 80.0},
      {"ld", 80.0},
      {" ", 50.0},     // 显式添加空格
      {"Python", 60.0},
      {"programming", 40.0}
  };
  
  // 创建分词器实例
  piece::BytePieceTokenizer tokenizer(dict);
  
  // 准备测试句子
  std::vector<std::string> test_cases = {
      "hello",
      "world",
      "hello world",
      "Python programming",
      "helloPython"
  };
  
  // 对每个测试句子进行分词并打印结果
  for (const auto& sentence : test_cases) {
      std::cout << "原句: " << sentence << std::endl;
      auto tokens = tokenizer.Tokenize(sentence);
      std::cout << "分词结果: ";
      if (tokens.empty()) {
          std::cout << "<空>";
      } else {
          for (size_t i = 0; i < tokens.size(); ++i) {
              std::cout << tokens[i];
              if (i < tokens.size() - 1) std::cout << " / ";
          }
      }
      std::cout << std::endl << std::endl;
  }
}

int main(int argc, char *argv[]) {

    std::cout << "test" << std::endl;

    //piece::run_test_naive();
    //piece::run_test_simple();
    //piece::run_normalizer_test();
    //piece::run_sentencepiece_counter_test();
    //piece::run_sentencepiece_counter_test_naive();
    //piece::run_sentencepiece_counter_test_simple();
    piece::run_sentencepiece_counter_test_piece();
    //piece::run_test();
    //piece::run_test_common();

    //test_byte_piece_tokenizer();
    //test_byte_piece_tokenizer_advanced();

    //test_new_byte_piece_tokenizer();
    //test_byte_piece_tokenizer_x();

    return 0;
}
