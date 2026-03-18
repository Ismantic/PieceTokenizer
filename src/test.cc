
#include "test.h"

#include <vector>
#include <iostream>

namespace test {

namespace {
struct Test {
  const char *base;
  const char *name;
  void (*func)();
};
std::vector<Test> *tests;
} // namespace

bool RegisterTest(const char *base, const char *name, void (*func)()) {
    if (tests == nullptr) {
        tests = new std::vector<Test>;
    }
    Test t;
    t.base = base;
    t.name = name;
    t.func = func;
    tests->emplace_back(t);
    return true;
}

int RunTests() {
  if (tests == nullptr) {
    std::cerr << "No tests are found" << std::endl;
    return 0;
  }

  int num = 0;
  for (const Test &t : *(tests)) {
    std::cerr << "[ RUN      ] " << t.base << "." << t.name << std::endl;
    (*t.func)();
    std::cerr << "[       OK ] " << t.base << "." << t.name << std::endl;
    ++num;
  }
  std::cerr << "==== PASSED " << num << " tests" << std::endl;    

  return 0;
}

} // namespace