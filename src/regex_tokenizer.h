#pragma once

#include <memory>
#include <string_view>
#include <vector>

namespace regex {

class TokenizerRegex {
public:
    explicit TokenizerRegex(std::string_view space, int cut = 0);
    ~TokenizerRegex();

    TokenizerRegex(TokenizerRegex&&) noexcept;
    TokenizerRegex& operator=(TokenizerRegex&&) noexcept;

    TokenizerRegex(const TokenizerRegex&) = delete;
    TokenizerRegex& operator=(const TokenizerRegex&) = delete;

    std::vector<std::string_view> Split(std::string_view text) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace regex
