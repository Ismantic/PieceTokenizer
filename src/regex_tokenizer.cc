// Parser -> Thompson NFA -> frozen DFA for tokenizer pre-segmentation.

#include "regex_tokenizer.h"
#include "ustr.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <functional>
#include <set>
#include <map>
#include <queue>
#include <stack>
#include <stdexcept>
#include <unordered_map>

namespace regex {

// UTF-8

size_t BytesOneUTF8(const char* src) {
    return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
}

uint32_t DecodeUTF8At(std::string_view str, size_t pos, size_t* bytes) {
    if (pos >= str.size()) { *bytes = 0; return 0; }
    const uint8_t* data = reinterpret_cast<const uint8_t*>(str.data() + pos);
    *bytes = BytesOneUTF8(str.data() + pos);
    if (pos + *bytes > str.size()) { *bytes = 0; return 0; }
    switch (*bytes) {
        case 1: return data[0];
        case 2: return ((data[0]&0x1F)<<6)|(data[1]&0x3F);
        case 3: return ((data[0]&0x0F)<<12)|((data[1]&0x3F)<<6)|(data[2]&0x3F);
        case 4: return ((data[0]&0x07)<<18)|((data[1]&0x3F)<<12)|((data[2]&0x3F)<<6)|(data[3]&0x3F);
        default: *bytes = 0; return 0;
    }
}

// Unicode predicates

static bool IsHan(uint32_t c) {
    return ustr::IsHan(c);
}
static bool IsDigit(uint32_t c) {
    return ustr::IsDigitCodepoint(c);
}
static bool IsWordChar(uint32_t c) {
    return ustr::IsWordChar(c);
}
static bool IsAlpha(uint32_t c) { return IsWordChar(c)&&!IsDigit(c)&&!IsHan(c); }

using CharPred = std::function<bool(uint32_t)>;
using ClassKey = std::function<uint32_t(uint32_t)>;

// AST

class EmptyAst;
class LiteralAst;
class CharClassAst;
class SequenceAst;
class AlternativeAst;
class StarAst;
class PlusAst;
class OptionalAst;

class AstVisitor {
public:
    virtual ~AstVisitor() = default;
    virtual void Visit(const EmptyAst*) = 0;
    virtual void Visit(const LiteralAst*) = 0;
    virtual void Visit(const CharClassAst*) = 0;
    virtual void Visit(const SequenceAst*) = 0;
    virtual void Visit(const AlternativeAst*) = 0;
    virtual void Visit(const StarAst*) = 0;
    virtual void Visit(const PlusAst*) = 0;
    virtual void Visit(const OptionalAst*) = 0;
};

class Ast {
public:
    virtual ~Ast() = default;
    virtual void Accept(AstVisitor* v) const = 0;
};

class EmptyAst : public Ast {
public:
    void Accept(AstVisitor* v) const override { v->Visit(this); }
};

class LiteralAst : public Ast {
    uint32_t point_;
public:
    explicit LiteralAst(uint32_t p) : point_(p) {}
    uint32_t GetPoint() const { return point_; }
    void Accept(AstVisitor* v) const override { v->Visit(this); }
};

class CharClassAst : public Ast {
    CharPred pred_;
public:
    explicit CharClassAst(CharPred p) : pred_(std::move(p)) {}
    const CharPred& GetPred() const { return pred_; }
    void Accept(AstVisitor* v) const override { v->Visit(this); }
};

class SequenceAst : public Ast {
    std::vector<std::unique_ptr<Ast>> elements_;
public:
    void Add(std::unique_ptr<Ast> e) { elements_.push_back(std::move(e)); }
    const std::vector<std::unique_ptr<Ast>>& GetElements() const { return elements_; }
    void Accept(AstVisitor* v) const override { v->Visit(this); }
};

class AlternativeAst : public Ast {
    std::vector<std::unique_ptr<Ast>> branches_;
public:
    void Add(std::unique_ptr<Ast> b) { branches_.push_back(std::move(b)); }
    const std::vector<std::unique_ptr<Ast>>& GetBranches() const { return branches_; }
    void Accept(AstVisitor* v) const override { v->Visit(this); }
};

class StarAst : public Ast {
    std::unique_ptr<Ast> element_;
public:
    explicit StarAst(std::unique_ptr<Ast> e) : element_(std::move(e)) {}
    const Ast* GetElement() const { return element_.get(); }
    void Accept(AstVisitor* v) const override { v->Visit(this); }
};

class PlusAst : public Ast {
    std::unique_ptr<Ast> element_;
public:
    explicit PlusAst(std::unique_ptr<Ast> e) : element_(std::move(e)) {}
    const Ast* GetElement() const { return element_.get(); }
    void Accept(AstVisitor* v) const override { v->Visit(this); }
};

class OptionalAst : public Ast {
    std::unique_ptr<Ast> element_;
public:
    explicit OptionalAst(std::unique_ptr<Ast> e) : element_(std::move(e)) {}
    const Ast* GetElement() const { return element_.get(); }
    void Accept(AstVisitor* v) const override { v->Visit(this); }
};

// Recursive-descent parser

class RegexParser {
    std::string pattern_;
    size_t pos_ = 0;

    bool AtEnd() { return pos_ >= pattern_.size(); }
    bool Match(char c) { if (!AtEnd() && pattern_[pos_] == c) { ++pos_; return true; } return false; }

    uint32_t NextChar() {
        size_t bytes;
        uint32_t c = DecodeUTF8At(pattern_, pos_, &bytes);
        if (bytes == 0) throw std::runtime_error("invalid UTF-8");
        pos_ += bytes;
        return c;
    }

    // Parse \p{Name} for the tokenizer pattern's A, H, and N properties.
    CharPred ParseUnicodeProperty() {
        if (!Match('{')) throw std::runtime_error("expected {");
        std::string name;
        while (!AtEnd() && pattern_[pos_] != '}') name += pattern_[pos_++];
        if (!Match('}')) throw std::runtime_error("expected }");

        CharPred pred;
        if (name == "A") pred = IsAlpha;
        else if (name == "H") pred = IsHan;
        else if (name == "N") pred = IsDigit;
        else throw std::runtime_error("unknown property: " + name);
        return pred;
    }

    CharPred ParseEscape() {
        if (AtEnd()) throw std::runtime_error("unexpected end after \\");
        char c = pattern_[pos_++];
        switch (c) {
            case 'p': return ParseUnicodeProperty();
            default: {
                uint32_t cc = c;
                return [cc](uint32_t x) { return x == cc; };
            }
        }
    }

    std::unique_ptr<Ast> ParseCharClass() {
        bool negated = Match('^');
        CharPred pred = [](uint32_t) { return false; };

        while (!AtEnd() && pattern_[pos_] != ']') {
            CharPred cp;
            if (pattern_[pos_] == '\\') {
                ++pos_;
                cp = ParseEscape();
            } else {
                uint32_t c = NextChar();
                cp = [c](uint32_t x) { return x==c; };
            }
            pred = [a=std::move(pred),b=std::move(cp)](uint32_t x) { return a(x)||b(x); };
        }
        if (!Match(']')) throw std::runtime_error("expected ]");
        if (negated) pred = [p=std::move(pred)](uint32_t x) { return !p(x); };
        return std::make_unique<CharClassAst>(std::move(pred));
    }

    // Parse atom: literal | . | (...) | [...] | \escape
    std::unique_ptr<Ast> ParseAtom() {
        if (AtEnd()) throw std::runtime_error("unexpected end");

        // Group: (...)
        if (Match('(')) {
            auto ast = ParseAlternation();
            if (!Match(')')) throw std::runtime_error("expected )");
            return ast;
        }

        if (Match('[')) return ParseCharClass();
        if (Match('\\')) {
            return std::make_unique<CharClassAst>(ParseEscape());
        }

        return std::make_unique<LiteralAst>(NextChar());
    }

    // Parse the *, +, and ? quantifiers used by the tokenizer pattern.
    std::unique_ptr<Ast> ParseQuantified() {
        auto atom = ParseAtom();
        if (AtEnd()) return atom;

        if (Match('*')) return std::make_unique<StarAst>(std::move(atom));
        if (Match('+')) return std::make_unique<PlusAst>(std::move(atom));
        if (Match('?')) return std::make_unique<OptionalAst>(std::move(atom));

        return atom;
    }

    std::unique_ptr<Ast> ParseSequence() {
        auto seq = std::make_unique<SequenceAst>();
        while (!AtEnd() && pattern_[pos_] != '|' && pattern_[pos_] != ')') {
            seq->Add(ParseQuantified());
        }
        if (seq->GetElements().empty()) return std::make_unique<EmptyAst>();
        return seq;
    }

    std::unique_ptr<Ast> ParseAlternation() {
        auto first = ParseSequence();
        if (AtEnd() || pattern_[pos_] != '|') return first;
        auto alt = std::make_unique<AlternativeAst>();
        alt->Add(std::move(first));
        while (Match('|')) alt->Add(ParseSequence());
        return alt;
    }

public:
    explicit RegexParser(std::string p) : pattern_(std::move(p)) {}

    std::unique_ptr<Ast> Parse() {
        pos_ = 0;
        auto ast = ParseAlternation();
        if (pos_ < pattern_.size())
            throw std::runtime_error("unexpected at pos " + std::to_string(pos_));
        return ast;
    }
};

// Thompson NFA

struct NFAState {
    int id;
    bool accept = false;

    struct Edge {
        CharPred pred;
        NFAState* to;
    };
    std::vector<Edge> edges;
    std::vector<NFAState*> epsilons;

    static int next_id;
    NFAState() : id(next_id++) {}
};
int NFAState::next_id = 0;

struct NFAFrag {
    NFAState* start;
    NFAState* end;
};

class NFABuilder : public AstVisitor {
    std::vector<std::unique_ptr<NFAState>> alloc_;
    std::stack<NFAFrag> stack_;

    NFAState* NewState() {
        auto s = std::make_unique<NFAState>();
        auto* p = s.get();
        alloc_.push_back(std::move(s));
        return p;
    }

    void PushPred(CharPred pred) {
        auto *s = NewState(), *e = NewState();
        e->accept = true;
        s->edges.push_back({std::move(pred), e});
        stack_.push({s, e});
    }

public:
    void Visit(const EmptyAst*) override {
        auto *s = NewState(), *e = NewState();
        e->accept = true;
        s->epsilons.push_back(e);
        stack_.push({s, e});
    }

    void Visit(const LiteralAst* n) override {
        uint32_t p = n->GetPoint();
        PushPred([p](uint32_t c) { return c == p; });
    }

    void Visit(const CharClassAst* n) override {
        PushPred(n->GetPred());
    }

    void Visit(const SequenceAst* n) override {
        auto& elems = n->GetElements();
        if (elems.empty()) { Visit((const EmptyAst*)nullptr); return; }
        elems[0]->Accept(this);
        for (size_t i = 1; i < elems.size(); i++) {
            auto u = stack_.top(); stack_.pop();
            elems[i]->Accept(this);
            auto v = stack_.top(); stack_.pop();
            u.end->accept = false;
            u.end->epsilons.push_back(v.start);
            stack_.push({u.start, v.end});
        }
    }

    void Visit(const AlternativeAst* n) override {
        auto& branches = n->GetBranches();
        if (branches.empty()) { Visit((const EmptyAst*)nullptr); return; }
        branches[0]->Accept(this);
        for (size_t i = 1; i < branches.size(); i++) {
            auto u = stack_.top(); stack_.pop();
            branches[i]->Accept(this);
            auto v = stack_.top(); stack_.pop();
            auto *s = NewState(), *e = NewState();
            e->accept = true;
            s->epsilons.push_back(u.start);
            s->epsilons.push_back(v.start);
            u.end->accept = v.end->accept = false;
            u.end->epsilons.push_back(e);
            v.end->epsilons.push_back(e);
            stack_.push({s, e});
        }
    }

    void Visit(const StarAst* n) override {
        n->GetElement()->Accept(this);
        auto inner = stack_.top(); stack_.pop();
        auto *s = NewState(), *e = NewState();
        e->accept = true; inner.end->accept = false;
        s->epsilons.push_back(inner.start);
        s->epsilons.push_back(e);
        inner.end->epsilons.push_back(inner.start);
        inner.end->epsilons.push_back(e);
        stack_.push({s, e});
    }

    void Visit(const PlusAst* n) override {
        n->GetElement()->Accept(this);
        auto inner = stack_.top(); stack_.pop();
        auto *s = NewState(), *e = NewState();
        e->accept = true; inner.end->accept = false;
        s->epsilons.push_back(inner.start);
        inner.end->epsilons.push_back(inner.start);
        inner.end->epsilons.push_back(e);
        stack_.push({s, e});
    }

    void Visit(const OptionalAst* n) override {
        n->GetElement()->Accept(this);
        auto inner = stack_.top(); stack_.pop();
        auto *s = NewState(), *e = NewState();
        e->accept = true; inner.end->accept = false;
        s->epsilons.push_back(inner.start);
        s->epsilons.push_back(e);
        inner.end->epsilons.push_back(e);
        stack_.push({s, e});
    }

    NFAFrag GetResult() { return stack_.top(); }
};

// Subset-construction DFA with predicate equivalence classes

class LazyDFA {
    NFAFrag nfa_;
    using StateSet = std::set<NFAState*>;

    struct DState {
        bool accept;
        std::vector<int> trans;
    };
    std::vector<DState> states_;
    std::vector<StateSet> state_sets_;
    std::map<StateSet, int> set_to_id_;

    // Equivalence classes: group codepoints by their predicate signature
    std::vector<CharPred*> all_preds_;
    std::map<std::vector<bool>, int> sig_map_;
    std::unordered_map<uint32_t, int> cp_cache_;
    std::array<int, 128> ascii_cache_;
    ClassKey class_key_;
    int next_cls_ = 0;
    bool frozen_ = false;

    void EpsClosure(StateSet& s) {
        std::vector<NFAState*> stk(s.begin(), s.end());
        while (!stk.empty()) {
            auto* st = stk.back(); stk.pop_back();
            for (auto* e : st->epsilons)
                if (s.insert(e).second) stk.push_back(e);
        }
    }

    int GetOrCreate(StateSet& ss) {
        EpsClosure(ss);
        auto it = set_to_id_.find(ss);
        if (it != set_to_id_.end()) return it->second;
        int id = states_.size();
        bool acc = false;
        for (auto* s : ss) if (s->accept) { acc = true; break; }
        states_.push_back({acc, {}});
        state_sets_.push_back(ss);
        set_to_id_[ss] = id;
        return id;
    }

    int ClassifyCP(uint32_t cp) {
        const uint32_t cache_key = class_key_ ? class_key_(cp) : cp;
        if (cache_key < ascii_cache_.size() && ascii_cache_[cache_key] >= 0) {
            return ascii_cache_[cache_key];
        }
        auto it = cp_cache_.find(cache_key);
        if (it != cp_cache_.end()) return it->second;
        std::vector<bool> sig;
        sig.reserve(all_preds_.size());
        for (auto* p : all_preds_) sig.push_back((*p)(cp));
        auto sit = sig_map_.find(sig);
        int cls;
        if (sit != sig_map_.end()) cls = sit->second;
        else { cls = next_cls_++; sig_map_[sig] = cls; }
        if (cache_key < ascii_cache_.size()) ascii_cache_[cache_key] = cls;
        else cp_cache_[cache_key] = cls;
        return cls;
    }

    int Step(int dfa_st, uint32_t cp) {
        int cls = ClassifyCP(cp);
        auto& trans = states_[dfa_st].trans;
        if (trans.size() <= static_cast<size_t>(cls)) trans.resize(cls + 1, -2);
        if (trans[cls] != -2) return trans[cls];

        StateSet next;
        for (auto* s : state_sets_[dfa_st])
            for (auto& e : s->edges)
                if (e.pred(cp)) next.insert(e.to);

        if (next.empty()) {
            trans[cls] = -1;
            return -1;
        }
        int next_id = GetOrCreate(next);
        states_[dfa_st].trans[cls] = next_id;
        return next_id;
    }

    int ClassifyFrozen(uint32_t cp) const {
        const uint32_t cache_key = class_key_ ? class_key_(cp) : cp;
        if (cache_key < ascii_cache_.size()) return ascii_cache_[cache_key];
        const auto it = cp_cache_.find(cache_key);
        return it == cp_cache_.end() ? -1 : it->second;
    }

    int StepFrozen(int dfa_st, uint32_t cp) const {
        const int cls = ClassifyFrozen(cp);
        if (cls < 0) return -3;
        const auto& trans = states_[dfa_st].trans;
        return static_cast<size_t>(cls) < trans.size() ? trans[cls] : -2;
    }

public:
    void Build(NFAFrag nfa, ClassKey class_key = {}) {
        nfa_ = nfa;
        class_key_ = std::move(class_key);
        ascii_cache_.fill(-1);
        // Collect all predicates from reachable NFA states
        std::set<NFAState*> visited;
        std::queue<NFAState*> queue;
        queue.push(nfa.start);
        visited.insert(nfa.start);
        while (!queue.empty()) {
            auto* st = queue.front(); queue.pop();
            for (auto& e : st->edges) {
                all_preds_.push_back(&e.pred);
                if (visited.insert(e.to).second) queue.push(e.to);
            }
            for (auto* e : st->epsilons)
                if (visited.insert(e).second) queue.push(e);
        }

        StateSet start = {nfa.start};
        GetOrCreate(start);
    }

    void Freeze(const std::vector<uint32_t>& representatives) {
        for (uint32_t cp : representatives) ClassifyCP(cp);
        for (size_t state = 0; state < states_.size(); ++state) {
            for (uint32_t cp : representatives) Step(state, cp);
        }
        frozen_ = true;
    }

    int MatchAtFrozen(const char* data, size_t len) const {
        int cur = 0;
        int last = states_[0].accept ? 0 : -1;
        size_t pos = 0;
        while (pos < len) {
            size_t bytes = 0;
            const uint32_t cp =
                DecodeUTF8At(std::string_view(data, len), pos, &bytes);
            if (bytes == 0) break;
            const int next = StepFrozen(cur, cp);
            if (next == -3) {
                throw std::logic_error("frozen DFA received an unknown class");
            }
            if (next == -2) {
                throw std::logic_error("frozen DFA is missing a transition");
            }
            if (next < 0) break;
            cur = next;
            pos += bytes;
            if (states_[cur].accept) last = pos;
        }
        return last;
    }

    std::vector<std::string_view> SegmentFrozen(std::string_view text) const {
        if (!frozen_) throw std::logic_error("DFA is not frozen");
        std::vector<std::string_view> result;
        const char* data = text.data();
        size_t len = text.size(), pos = 0;
        while (pos < len) {
            const int n = MatchAtFrozen(data + pos, len - pos);
            if (n > 0) {
                result.emplace_back(data + pos, n);
                pos += n;
            } else {
                pos += BytesOneUTF8(data + pos);
            }
        }
        return result;
    }

};

class Regex {
    std::unique_ptr<Ast> ast_;
    NFABuilder nfa_builder_;
    LazyDFA dfa_;

public:
    explicit Regex(const std::string& pattern, ClassKey class_key = {}) {
        RegexParser parser(pattern);
        ast_ = parser.Parse();
        ast_->Accept(&nfa_builder_);
        auto nfa = nfa_builder_.GetResult();
        dfa_.Build(nfa, std::move(class_key));
    }

    void Freeze(const std::vector<uint32_t>& representatives) {
        dfa_.Freeze(representatives);
    }
    std::vector<std::string_view> SegmentFrozen(std::string_view text) const {
        return dfa_.SegmentFrozen(text);
    }
};

std::string EscapeLiteral(std::string_view value) {
    if (value.empty()) throw std::invalid_argument("space must not be empty");
    size_t bytes = 0;
    DecodeUTF8At(value, 0, &bytes);
    if (bytes != value.size()) {
        throw std::invalid_argument("space must contain one UTF-8 codepoint");
    }
    if (value.size() == 1 &&
        std::string_view("\\()[]|*+?.").find(value[0]) != std::string_view::npos) {
        return "\\" + std::string(value);
    }
    return std::string(value);
}

uint32_t DecodeSingle(std::string_view value) {
    size_t bytes = 0;
    const uint32_t cp = DecodeUTF8At(value, 0, &bytes);
    if (bytes == 0 || bytes != value.size()) {
        throw std::invalid_argument("space must contain one UTF-8 codepoint");
    }
    return cp;
}

std::string EscapeCharClass(std::string_view value) {
    EscapeLiteral(value);
    if (value.size() == 1 &&
        std::string_view("\\]^").find(value[0]) != std::string_view::npos) {
        return "\\" + std::string(value);
    }
    return std::string(value);
}

std::string TokenizerPattern(std::string_view space, int cut) {
    if (cut == 1) {
        return "\\p{A}+('\\p{A}+)*|\\p{H}+|\\p{N}+|"
               "[^\\p{A}\\p{H}\\p{N}]";
    }
    const std::string literal = EscapeLiteral(space);
    const std::string punct =
        "[^\\p{A}\\p{H}\\p{N}" + EscapeCharClass(space) + "]";
    return "[^\\p{A}\\p{H}\\p{N}]?\\p{A}+"
           "|\\p{H}+"
           "|\\p{N}+"
           "|" + literal + "?" + punct + "+"
           "|" + literal;
}

class TokenizerRegexImpl {
    Regex regex_;

public:
    explicit TokenizerRegexImpl(std::string_view space, int cut = 0)
        : regex_(TokenizerPattern(space, cut),
                 [space_cp = DecodeSingle(space)](uint32_t cp) {
                     if (cp == space_cp) return 0U;
                     if (IsHan(cp)) return 1U;
                     if (IsDigit(cp)) return 2U;
                     if (IsAlpha(cp)) return 3U;
                     if (cp == '\'') return 4U;
                     return 5U;
                 }) {
        const uint32_t space_cp = DecodeSingle(space);
        regex_.Freeze({
            space_cp, U'中', U'文', '0', '1', 'a', 'b', '\'', ',', '!',
        });
    }

    std::vector<std::string_view> Split(std::string_view text) const {
        return regex_.SegmentFrozen(text);
    }
};

class TokenizerRegex::Impl : public TokenizerRegexImpl {
public:
    using TokenizerRegexImpl::TokenizerRegexImpl;
};

TokenizerRegex::TokenizerRegex(std::string_view space, int cut)
    : impl_(std::make_unique<Impl>(space, cut)) {}

TokenizerRegex::~TokenizerRegex() = default;
TokenizerRegex::TokenizerRegex(TokenizerRegex&&) noexcept = default;
TokenizerRegex& TokenizerRegex::operator=(TokenizerRegex&&) noexcept = default;

std::vector<std::string_view> TokenizerRegex::Split(
    std::string_view text) const {
    return impl_->Split(text);
}

} // namespace regex
