#pragma once

#include <math.h>

#include <iostream>
#include <sstream>
#include <string>

namespace test {

int RunTests();

class Tester {
 public:
  Tester(const char *filename, int line)
   : ok_(true), filename_(filename), line_(line) {} 
  ~Tester() {
    if (!ok_) {
        std::cerr << "[     NG ] " << filename_ << ":" << line_ << ":" 
                  << ss_.str() << std::endl;
        exit(-1);
    }
  }

  Tester  &Is(bool b, const char *msg) {
    if (!b) {
        ss_ << " failed: " << msg;
        ok_ = false;
    }
    return *this;
  }
  Tester &IsNear(float val1, float val2, float abs_error, 
                const char *msg1, const char *msg2) {
    const double diff = std::fabs(val1 - val2);
    if (diff > abs_error) {
      ss_ << "The difference between (" << msg1 << ") and (" << msg2 << ") is "
          << diff << ", which exceeds " << abs_error << ", where\n"
          << msg1 << " evaluates to " << val1 << ",\n"
          << msg2 << " evaluates to " << val2;
      ok_ = false;
    }
    return *this;
   }

#define BINARY_OP(name, op) \
  template <class X, class Y>   \
  Tester &name(const X &x, const Y &y, const char *msg1, const char *msg2) { \
    if (! (x op y)) { \
      ss_ << " failed: " << msg1 << (" " #op " ") << msg2; \
      ok_ = false; \
    } \
    return *this; \
  } \

  BINARY_OP(IsEq, ==)
  BINARY_OP(IsNe, !=)
  BINARY_OP(IsGe, >=)
  BINARY_OP(IsGt, >)
  BINARY_OP(IsLe, <=)
  BINARY_OP(IsLt, <)
#undef BINARY_OP

    // Attach the specified value to the error message if an error has occurred 
    template <class V>
    Tester &operator<<(const V &value) {
        if (!ok_) {
            ss_ << " " << value;
        }
        return *this;
    }
 private:
  bool ok_;
  const char *filename_;
  int line_;
  std::stringstream ss_;
};

#define EXPECT_TRUE(c) \
  test::Tester(__FILE__, __LINE__).Is((c), #c)
#define EXPECT_FALSE(c) \
  test::Tester(__FILE__, __LINE__).Is((!(c)), #c)
#define EXPECT_STREQ(a, b) \
  test::Tester(__FILE__, __LINE__) \
    .IsEq(std::string(a), std::string(b), #a, #b)
#define EXPECT_EQ(a, b) \
  test::Tester(__FILE__, __LINE__).IsEq((a), (b), #a, #b)
#define EXPECT_NE(a, b) \
  test::Tester(__FILE__, __LINE__).IsNe((a), (b), #a, #b)
#define EXPECT_GE(a, b) \
  test::Tester(__FILE__, __LINE__).IsGe((a), (b), #a, #b)
#define EXPECT_GT(a, b) \
  test::Tester(__FILE__, __LINE__).IsGt((a), (b), #a, #b)
#define EXPECT_LE(a, b) \
  test::Tester(__FILE__, __LINE__).IsLe((a), (b), #a, #b)
#define EXPECT_LT(a, b) \
  test::Tester(__FILE__, __LINE__).IsLt((a), (b), #a, #b)
#define EXPECT_NEAR(a, b, c) \
  test::Tester(__FILE__, __LINE__).IsNear((a), (b), (c), #a, #b)

#define ASSERT_TRUE EXPECT_TRUE
#define ASSERT_FALSE EXPECT_FALSE
#define ASSERT_STREQ EXPECT_STREQ
#define ASSERT_EQ EXPECT_EQ
#define ASSERT_NE EXPECT_NE
#define ASSERT_GE EXPECT_GE
#define ASSERT_GT EXPECT_GT
#define ASSERT_LE EXPECT_LE
#define ASSERT_LT EXPECT_LT
#define ASSERT_NEAR EXPECT_NEAR

// Register the specified test. Typically not used directly, but 
// invoked via the macro expansion of TEST.
extern bool RegisterTest(const char *base, const char *name, void (*func)());

#define TCONCAT(a, b, c) TCONCAT1(a, b, c)
#define TCONCAT1(a, b, c) a##b##c

#define TEST(base, name)                                                       \
  class TCONCAT(base, _Test_, name) {                                          \
   public:                                                                     \
    void _Run();                                                               \
    static void _RunIt() {                                                     \
      TCONCAT(base, _Test_, name) t;                                           \
      t._Run();                                                                \
    }                                                                          \
  };                                                                           \
  bool TCONCAT(base, _Test_ignored_, name) =                                   \
      test::RegisterTest(#base, #name,                          \
                         &TCONCAT(base, _Test_, name)::_RunIt); \
  void TCONCAT(base, _Test_, name)::_Run()

} // namespace test
