#ifndef NEW_DARTS_H_
#define NEW_DARTS_H_

#include <array>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <functional>
#include <algorithm>
#include <bit>

// 版本信息
constexpr auto NEW_DARTS_VERSION = "1.0.0";

namespace new_darts {

// 自定义异常类
class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& message)
        : std::runtime_error(std::string(__FILE__) + ":" + 
                          std::to_string(__LINE__) + ": exception: " + message) {}
};

namespace detail {
    // 核心类型定义
    using char_type = char;
    using uchar_type = uint8_t;
    using value_type = int32_t;
    using id_type = uint32_t;
    
    // 定义进度回调函数类型
    using progress_func_type = std::function<int(std::size_t, std::size_t)>;

    // DoubleArrayUnit类，封装了双数组节点的结构
    class DoubleArrayUnit {
    public:
        DoubleArrayUnit() : unit_(0) {}

        bool has_leaf() const noexcept {
            return ((unit_ >> 8) & 1) == 1;
        }

        value_type value() const noexcept {
            return static_cast<value_type>(unit_ & ((1U << 31) - 1));
        }

        id_type label() const noexcept {
            return unit_ & ((1U << 31) | 0xFF);
        }

        id_type offset() const noexcept {
            return (unit_ >> 10) << ((unit_ & (1U << 9)) >> 6);
        }

    private:
        id_type unit_;

        friend class DoubleArrayBuilderUnit;
    };
} // namespace detail

// 面向使用者的DoubleArray类模板，T为值类型
template <typename T = int32_t>
class DoubleArray {
public:
    using value_type = T;
    using key_type = detail::char_type;
    using result_type = value_type;

    // 结果对类型，存储值和长度信息
    struct ResultPair {
        value_type value;
        std::size_t length;
    };

    DoubleArray() : size_(0), array_(nullptr), buf_(nullptr) {}
    ~DoubleArray() {
        clear();
    }

    // 移动构造和赋值
    DoubleArray(DoubleArray&& other) noexcept 
        : size_(other.size_), array_(other.array_), buf_(other.buf_) {
        other.size_ = 0;
        other.array_ = nullptr;
        other.buf_ = nullptr;
    }

    DoubleArray& operator=(DoubleArray&& other) noexcept {
        if (this != &other) {
            clear();
            size_ = other.size_;
            array_ = other.array_;
            buf_ = other.buf_;
            other.size_ = 0;
            other.array_ = nullptr;
            other.buf_ = nullptr;
        }
        return *this;
    }

    // 禁止复制
    DoubleArray(const DoubleArray&) = delete;
    DoubleArray& operator=(const DoubleArray&) = delete;

    // 设置单个结果
    void set_result(value_type* result, value_type value, std::size_t) const {
        *result = value;
    }

    // 设置结果对
    void set_result(ResultPair* result, value_type value, std::size_t length) const {
        result->value = value;
        result->length = length;
    }

    // 设置数组
    void set_array(const void* ptr, std::size_t size = 0) {
        clear();
        array_ = static_cast<const detail::DoubleArrayUnit*>(ptr);
        size_ = size;
    }

    // 获取数组指针
    const void* array() const noexcept {
        return array_;
    }

    // 清理资源
    void clear() {
        size_ = 0;
        array_ = nullptr;
        delete[] buf_;
        buf_ = nullptr;
    }

    // 返回单元大小
    std::size_t unit_size() const noexcept {
        return sizeof(detail::DoubleArrayUnit);
    }

    // 返回单元数量
    std::size_t size() const noexcept {
        return size_;
    }

    // 返回总字节大小
    std::size_t total_size() const noexcept {
        return unit_size() * size();
    }

    // 返回非零单元数量（兼容性函数）
    std::size_t nonzero_size() const noexcept {
        return size();
    }

    // 构建双数组
    int build(std::size_t num_keys, 
              const key_type* const* keys,
              const std::size_t* lengths = nullptr, 
              const value_type* values = nullptr,
              detail::progress_func_type progress_func = nullptr);

    // 从文件加载双数组
    int open(const std::string& filename, 
             const std::string& mode = "rb",
             std::size_t offset = 0, 
             std::size_t size = 0);
    
    // 保存双数组到文件
    int save(const std::string& filename, 
             const std::string& mode = "wb",
             std::size_t offset = 0) const;

    // 精确匹配搜索，更新result参数
    template <typename U>
    void exactMatchSearch(const key_type* key, U& result,
                         std::size_t length = 0, 
                         std::size_t node_pos = 0) const {
        result = exactMatchSearch<U>(key, length, node_pos);
    }

    // 精确匹配搜索，返回结果
    template <typename U>
    U exactMatchSearch(const key_type* key, 
                     std::size_t length = 0,
                     std::size_t node_pos = 0) const;

    // 公共前缀搜索
    template <typename U>
    std::size_t commonPrefixSearch(const key_type* key, 
                                 U* results,
                                 std::size_t max_num_results, 
                                 std::size_t length = 0,
                                 std::size_t node_pos = 0) const;

    // 遍历查找
    value_type traverse(const key_type* key, 
                      std::size_t& node_pos,
                      std::size_t& key_pos, 
                      std::size_t length = 0) const;

private:
    using uchar_type = detail::uchar_type;
    using id_type = detail::id_type;
    using unit_type = detail::DoubleArrayUnit;

    std::size_t size_;
    const unit_type* array_;
    unit_type* buf_;
};

// 精确匹配搜索实现
template <typename T>
template <typename U>
U DoubleArray<T>::exactMatchSearch(const key_type* key,
                                std::size_t length, 
                                std::size_t node_pos) const {
    U result;
    set_result(&result, static_cast<value_type>(-1), 0);

    unit_type unit = array_[node_pos];
    if (length != 0) {
        for (std::size_t i = 0; i < length; ++i) {
            node_pos ^= unit.offset() ^ static_cast<uchar_type>(key[i]);
            unit = array_[node_pos];
            if (unit.label() != static_cast<uchar_type>(key[i])) {
                return result;
            }
        }
    } else {
        for (; key[length] != '\0'; ++length) {
            node_pos ^= unit.offset() ^ static_cast<uchar_type>(key[length]);
            unit = array_[node_pos];
            if (unit.label() != static_cast<uchar_type>(key[length])) {
                return result;
            }
        }
    }

    if (!unit.has_leaf()) {
        return result;
    }
    unit = array_[node_pos ^ unit.offset()];
    set_result(&result, static_cast<value_type>(unit.value()), length);
    return result;
}

// 公共前缀搜索实现
template <typename T>
template <typename U>
std::size_t DoubleArray<T>::commonPrefixSearch(const key_type* key, 
                                            U* results,
                                            std::size_t max_num_results,
                                            std::size_t length, 
                                            std::size_t node_pos) const {
    std::size_t num_results = 0;

    unit_type unit = array_[node_pos];
    node_pos ^= unit.offset();
    if (length != 0) {
        for (std::size_t i = 0; i < length; ++i) {
            node_pos ^= static_cast<uchar_type>(key[i]);
            unit = array_[node_pos];
            if (unit.label() != static_cast<uchar_type>(key[i])) {
                return num_results;
            }

            node_pos ^= unit.offset();
            if (unit.has_leaf()) {
                if (num_results < max_num_results) {
                    set_result(&results[num_results], static_cast<value_type>(
                        array_[node_pos].value()), i + 1);
                }
                ++num_results;
            }
        }
    } else {
        for (; key[length] != '\0'; ++length) {
            node_pos ^= static_cast<uchar_type>(key[length]);
            unit = array_[node_pos];
            if (unit.label() != static_cast<uchar_type>(key[length])) {
                return num_results;
            }

            node_pos ^= unit.offset();
            if (unit.has_leaf()) {
                if (num_results < max_num_results) {
                    set_result(&results[num_results], static_cast<value_type>(
                        array_[node_pos].value()), length + 1);
                }
                ++num_results;
            }
        }
    }

    return num_results;
}

// 遍历查找实现
template <typename T>
typename DoubleArray<T>::value_type DoubleArray<T>::traverse(
    const key_type* key,
    std::size_t& node_pos, 
    std::size_t& key_pos, 
    std::size_t length) const {
    id_type id = static_cast<id_type>(node_pos);
    unit_type unit = array_[id];

    if (length != 0) {
        for (; key_pos < length; ++key_pos) {
            id ^= unit.offset() ^ static_cast<uchar_type>(key[key_pos]);
            unit = array_[id];
            if (unit.label() != static_cast<uchar_type>(key[key_pos])) {
                return static_cast<value_type>(-2);
            }
            node_pos = id;
        }
    } else {
        for (; key[key_pos] != '\0'; ++key_pos) {
            id ^= unit.offset() ^ static_cast<uchar_type>(key[key_pos]);
            unit = array_[id];
            if (unit.label() != static_cast<uchar_type>(key[key_pos])) {
                return static_cast<value_type>(-2);
            }
            node_pos = id;
        }
    }

    if (!unit.has_leaf()) {
        return static_cast<value_type>(-1);
    }
    unit = array_[id ^ unit.offset()];
    return static_cast<value_type>(unit.value());
}

// 从文件加载双数组
template <typename T>
int DoubleArray<T>::open(const std::string& filename, 
                        const std::string& mode,
                        std::size_t offset, 
                        std::size_t size) {
    std::FILE* file = std::fopen(filename.c_str(), mode.c_str());
    if (file == nullptr) {
        return -1;
    }

    if (size == 0) {
        if (std::fseek(file, 0, SEEK_END) != 0) {
            std::fclose(file);
            return -1;
        }
        size = std::ftell(file) - offset;
    }

    size /= unit_size();
    if (size < 256 || (size & 0xFF) != 0) {
        std::fclose(file);
        return -1;
    }

    if (std::fseek(file, offset, SEEK_SET) != 0) {
        std::fclose(file);
        return -1;
    }

    std::array<unit_type, 256> units;
    if (std::fread(units.data(), unit_size(), 256, file) != 256) {
        std::fclose(file);
        return -1;
    }

    if (units[0].label() != '\0' || units[0].has_leaf() ||
        units[0].offset() == 0 || units[0].offset() >= 512) {
        std::fclose(file);
        return -1;
    }
    
    for (id_type i = 1; i < 256; ++i) {
        if (units[i].label() <= 0xFF && units[i].offset() >= size) {
            std::fclose(file);
            return -1;
        }
    }

    unit_type* buf = nullptr;
    try {
        buf = new unit_type[size];
        for (id_type i = 0; i < 256; ++i) {
            buf[i] = units[i];
        }
    } catch (const std::bad_alloc&) {
        std::fclose(file);
        throw Exception("failed to open double-array: std::bad_alloc");
    }

    if (size > 256) {
        if (std::fread(buf + 256, unit_size(), size - 256, file) != size - 256) {
            std::fclose(file);
            delete[] buf;
            return -1;
        }
    }
    std::fclose(file);

    clear();

    size_ = size;
    array_ = buf;
    buf_ = buf;
    return 0;
}

// 保存双数组到文件
template <typename T>
int DoubleArray<T>::save(const std::string& filename, 
                        const std::string& mode,
                        std::size_t offset) const {
    if (size() == 0) {
        return -1;
    }

    std::FILE* file = std::fopen(filename.c_str(), mode.c_str());
    if (file == nullptr) {
        return -1;
    }

    if (std::fwrite(array_, unit_size(), size(), file) != size()) {
        std::fclose(file);
        return -1;
    }
    std::fclose(file);
    return 0;
}

namespace detail {
    // BitVector类，位向量实现，用于构建过程中
    class BitVector {
    public:
        BitVector() : num_ones_(0), size_(0) {}

        bool operator[](std::size_t id) const {
            return (units_[id / UNIT_SIZE] >> (id % UNIT_SIZE) & 1) == 1;
        }

        id_type rank(std::size_t id) const {
            std::size_t unit_id = id / UNIT_SIZE;
            return ranks_[unit_id] + pop_count(units_[unit_id]
                & (~0U >> (UNIT_SIZE - (id % UNIT_SIZE) - 1)));
        }
        
        // 自定义pop_count函数替代std::popcount
        static id_type pop_count(id_type unit) {
            unit = ((unit & 0xAAAAAAAA) >> 1) + (unit & 0x55555555);
            unit = ((unit & 0xCCCCCCCC) >> 2) + (unit & 0x33333333);
            unit = ((unit >> 4) + unit) & 0x0F0F0F0F;
            unit += unit >> 8;
            unit += unit >> 16;
            return unit & 0xFF;
        }

        void set(std::size_t id, bool bit) {
            if (bit) {
                units_[id / UNIT_SIZE] |= 1U << (id % UNIT_SIZE);
            } else {
                units_[id / UNIT_SIZE] &= ~(1U << (id % UNIT_SIZE));
            }
        }

        bool empty() const {
            return units_.empty();
        }
        
        std::size_t num_ones() const {
            return num_ones_;
        }
        
        std::size_t size() const {
            return size_;
        }

        void append() {
            if ((size_ % UNIT_SIZE) == 0) {
                units_.push_back(0);
            }
            ++size_;
        }
        
        void build() {
            ranks_ = std::make_unique<id_type[]>(units_.size());
            
            num_ones_ = 0;
            for (std::size_t i = 0; i < units_.size(); ++i) {
                ranks_[i] = num_ones_;
                num_ones_ += pop_count(units_[i]);
            }
        }

        void clear() {
            units_.clear();
            ranks_.reset();
            num_ones_ = 0;
            size_ = 0;
        }

    private:
        static constexpr std::size_t UNIT_SIZE = sizeof(id_type) * 8;

        std::vector<id_type> units_;
        std::unique_ptr<id_type[]> ranks_;
        std::size_t num_ones_;
        std::size_t size_;
    };

    // Keyset类，用于构建过程中处理键集合
    template <typename T>
    class Keyset {
    public:
        Keyset(std::size_t num_keys, 
              const char_type* const* keys,
              const std::size_t* lengths, 
              const T* values)
            : num_keys_(num_keys), keys_(keys), lengths_(lengths), values_(values) {}

        std::size_t num_keys() const {
            return num_keys_;
        }
        
        const char_type* keys(std::size_t id) const {
            return keys_[id];
        }
        
        uchar_type keys(std::size_t key_id, std::size_t char_id) const {
            if (has_lengths() && char_id >= lengths_[key_id])
                return '\0';
            return keys_[key_id][char_id];
        }

        bool has_lengths() const {
            return lengths_ != nullptr;
        }
        
        std::size_t lengths(std::size_t id) const {
            if (has_lengths()) {
                return lengths_[id];
            }
            std::size_t length = 0;
            while (keys_[id][length] != '\0') {
                ++length;
            }
            return length;
        }

        bool has_values() const {
            return values_ != nullptr;
        }
        
        const value_type values(std::size_t id) const {
            if (has_values()) {
                return static_cast<value_type>(values_[id]);
            }
            return static_cast<value_type>(id);
        }

    private:
        std::size_t num_keys_;
        const char_type* const* keys_;
        const std::size_t* lengths_;
        const T* values_;
    };

    // DawgNode类，DAWG节点实现
    class DawgNode {
    public:
        DawgNode() : child_(0), sibling_(0), label_('\0'),
                    is_state_(false), has_sibling_(false) {}

        void set_child(id_type child) {
            child_ = child;
        }
        
        void set_sibling(id_type sibling) {
            sibling_ = sibling;
        }
        
        void set_value(value_type value) {
            child_ = value;
        }
        
        void set_label(uchar_type label) {
            label_ = label;
        }
        
        void set_is_state(bool is_state) {
            is_state_ = is_state;
        }
        
        void set_has_sibling(bool has_sibling) {
            has_sibling_ = has_sibling;
        }

        id_type child() const {
            return child_;
        }
        
        id_type sibling() const {
            return sibling_;
        }
        
        value_type value() const {
            return static_cast<value_type>(child_);
        }
        
        uchar_type label() const {
            return label_;
        }
        
        bool is_state() const {
            return is_state_;
        }
        
        bool has_sibling() const {
            return has_sibling_;
        }

        id_type unit() const {
            if (label_ == '\0') {
                return (child_ << 1) | (has_sibling_ ? 1 : 0);
            }
            return (child_ << 2) | (is_state_ ? 2 : 0) | (has_sibling_ ? 1 : 0);
        }

    private:
        id_type child_;
        id_type sibling_;
        uchar_type label_;
        bool is_state_;
        bool has_sibling_;
    };

    // DawgUnit类，DAWG单元实现
    class DawgUnit {
    public:
        explicit DawgUnit(id_type unit = 0) : unit_(unit) {}

        DawgUnit& operator=(id_type unit) {
            unit_ = unit;
            return *this;
        }

        id_type unit() const {
            return unit_;
        }

        id_type child() const {
            return unit_ >> 2;
        }
        
        bool has_sibling() const {
            return (unit_ & 1) == 1;
        }
        
        value_type value() const {
            return static_cast<value_type>(unit_ >> 1);
        }
        
        bool is_state() const {
            return (unit_ & 2) == 2;
        }

    private:
        id_type unit_;
    };

    // DawgBuilder类，DAWG构建器
    class DawgBuilder {
    public:
        DawgBuilder() : num_states_(0) {}

        id_type root() const {
            return 0;
        }

        id_type child(id_type id) const {
            return units_[id].child();
        }
        
        id_type sibling(id_type id) const {
            return units_[id].has_sibling() ? (id + 1) : 0;
        }
        
        int value(id_type id) const {
            return units_[id].value();
        }

        bool is_leaf(id_type id) const {
            return label(id) == '\0';
        }
        
        uchar_type label(id_type id) const {
            return labels_[id];
        }

        bool is_intersection(id_type id) const {
            return is_intersections_[id];
        }
        
        id_type intersection_id(id_type id) const {
            return is_intersections_.rank(id) - 1;
        }

        std::size_t num_intersections() const {
            return is_intersections_.num_ones();
        }

        std::size_t size() const {
            return units_.size();
        }

        void init() {
            table_.resize(INITIAL_TABLE_SIZE, 0);

            nodes_.push_back(DawgNode());
            units_.push_back(DawgUnit());
            labels_.push_back('\0');

            num_states_ = 1;

            nodes_[0].set_label(0xFF);
            node_stack_.push_back(0);
        }

        void finish() {
            flush(0);

            units_[0] = nodes_[0].unit();
            labels_[0] = nodes_[0].label();

            nodes_.clear();
            table_.clear();
            node_stack_.clear();
            recycle_bin_.clear();

            is_intersections_.build();
        }

        void insert(const char* key, std::size_t length, value_type value) {
            if (value < 0) {
                throw Exception("failed to insert key: negative value");
            } else if (length == 0) {
                throw Exception("failed to insert key: zero-length key");
            }

            id_type id = 0;
            std::size_t key_pos = 0;

            for (; key_pos <= length; ++key_pos) {
                id_type child_id = nodes_[id].child();
                if (child_id == 0) {
                    break;
                }

                uchar_type key_label = static_cast<uchar_type>(key[key_pos]);
                if (key_pos < length && key_label == '\0') {
                    throw Exception("failed to insert key: invalid null character");
                }

                uchar_type unit_label = nodes_[child_id].label();
                if (key_label < unit_label) {
                    throw Exception("failed to insert key: wrong key order");
                } else if (key_label > unit_label) {
                    nodes_[child_id].set_has_sibling(true);
                    flush(child_id);
                    break;
                }
                id = child_id;
            }

            if (key_pos > length) {
                return;
            }

            for (; key_pos <= length; ++key_pos) {
                uchar_type key_label = static_cast<uchar_type>(
                    (key_pos < length) ? key[key_pos] : '\0');
                id_type child_id = append_node();

                if (nodes_[id].child() == 0) {
                    nodes_[child_id].set_is_state(true);
                }
                nodes_[child_id].set_sibling(nodes_[id].child());
                nodes_[child_id].set_label(key_label);
                nodes_[id].set_child(child_id);
                node_stack_.push_back(child_id);

                id = child_id;
            }
            nodes_[id].set_value(value);
        }

        void clear() {
            nodes_.clear();
            units_.clear();
            labels_.clear();
            is_intersections_.clear();
            table_.clear();
            node_stack_.clear();
            recycle_bin_.clear();
            num_states_ = 0;
        }

    private:
        static constexpr std::size_t INITIAL_TABLE_SIZE = 1 << 10;

        std::vector<DawgNode> nodes_;
        std::vector<DawgUnit> units_;
        std::vector<uchar_type> labels_;
        BitVector is_intersections_;
        std::vector<id_type> table_;
        std::vector<id_type> node_stack_;
        std::vector<id_type> recycle_bin_;
        std::size_t num_states_;

        void flush(id_type id) {
            while (!node_stack_.empty() && node_stack_.back() != id) {
                id_type node_id = node_stack_.back();
                node_stack_.pop_back();

                if (num_states_ >= table_.size() - (table_.size() >> 2)) {
                    expand_table();
                }

                id_type num_siblings = 0;
                for (id_type i = node_id; i != 0; i = nodes_[i].sibling()) {
                    ++num_siblings;
                }

                id_type hash_id;
                id_type match_id = find_node(node_id, &hash_id);
                if (match_id != 0) {
                    is_intersections_.set(match_id, true);
                } else {
                    id_type unit_id = units_.size();
                    for (id_type i = 0; i < num_siblings; ++i) {
                        units_.push_back(DawgUnit());
                        labels_.push_back('\0');
                        is_intersections_.append();
                    }
                    unit_id = units_.size() - 1;
                    
                    for (id_type i = node_id; i != 0; i = nodes_[i].sibling()) {
                        units_[unit_id] = nodes_[i].unit();
                        labels_[unit_id] = nodes_[i].label();
                        --unit_id;
                    }
                    match_id = unit_id + 1;
                    table_[hash_id] = match_id;
                    ++num_states_;
                }

                for (id_type i = node_id, next; i != 0; i = next) {
                    next = nodes_[i].sibling();
                    free_node(i);
                }

                nodes_[node_stack_.back()].set_child(match_id);
            }
            
            if (!node_stack_.empty()) {
                node_stack_.pop_back();
            }
        }

        void expand_table() {
            std::size_t table_size = table_.size() << 1;
            table_.clear();
            table_.resize(table_size, 0);

            for (std::size_t i = 1; i < units_.size(); ++i) {
                id_type id = static_cast<id_type>(i);
                if (labels_[id] == '\0' || units_[id].is_state()) {
                    id_type hash_id;
                    find_unit(id, &hash_id);
                    table_[hash_id] = id;
                }
            }
        }

        id_type find_unit(id_type id, id_type* hash_id) const {
            *hash_id = hash_unit(id) % table_.size();
            for (;;) {
                id_type unit_id = table_[*hash_id];
                if (unit_id == 0) {
                    break;
                }
                *hash_id = (*hash_id + 1) % table_.size();
            }
            return 0;
        }

        id_type find_node(id_type node_id, id_type* hash_id) const {
            *hash_id = hash_node(node_id) % table_.size();
            for (;;) {
                id_type unit_id = table_[*hash_id];
                if (unit_id == 0) {
                    break;
                }

                if (are_equal(node_id, unit_id)) {
                    return unit_id;
                }
                *hash_id = (*hash_id + 1) % table_.size();
            }
            return 0;
        }

        bool are_equal(id_type node_id, id_type unit_id) const {
            for (id_type i = nodes_[node_id].sibling(); i != 0;
                i = nodes_[i].sibling()) {
                if (!units_[unit_id].has_sibling()) {
                    return false;
                }
                ++unit_id;
            }
            if (units_[unit_id].has_sibling()) {
                return false;
            }

            for (id_type i = node_id; i != 0; i = nodes_[i].sibling(), --unit_id) {
                if (nodes_[i].unit() != units_[unit_id].unit() ||
                    nodes_[i].label() != labels_[unit_id]) {
                    return false;
                }
            }
            return true;
        }

        id_type hash_unit(id_type id) const {
            id_type hash_value = 0;
            for (; id != 0; ++id) {
                id_type unit = units_[id].unit();
                uchar_type label = labels_[id];
                hash_value ^= hash((label << 24) ^ unit);

                if (!units_[id].has_sibling()) {
                    break;
                }
            }
            return hash_value;
        }

        id_type hash_node(id_type id) const {
            id_type hash_value = 0;
            for (; id != 0; id = nodes_[id].sibling()) {
                id_type unit = nodes_[id].unit();
                uchar_type label = nodes_[id].label();
                hash_value ^= hash((label << 24) ^ unit);
            }
            return hash_value;
        }

        id_type append_node() {
            id_type id;
            if (recycle_bin_.empty()) {
                id = static_cast<id_type>(nodes_.size());
                nodes_.push_back(DawgNode());
            } else {
                id = recycle_bin_.back();
                nodes_[id] = DawgNode();
                recycle_bin_.pop_back();
            }
            return id;
        }

        void free_node(id_type id) {
            recycle_bin_.push_back(id);
        }

        static id_type hash(id_type key) {
            key = ~key + (key << 15);  // key = (key << 15) - key - 1;
            key = key ^ (key >> 12);
            key = key + (key << 2);
            key = key ^ (key >> 4);
            key = key * 2057;  // key = (key + (key << 3)) + (key << 11);
            key = key ^ (key >> 16);
            return key;
        }
    };

    // DoubleArrayBuilderUnit类，双数组构建单元
    class DoubleArrayBuilderUnit {
    public:
        DoubleArrayBuilderUnit() : unit_(0) {}

        void set_has_leaf(bool has_leaf) {
            if (has_leaf) {
                unit_ |= 1U << 8;
            } else {
                unit_ &= ~(1U << 8);
            }
        }
        
        void set_value(value_type value) {
            unit_ = value | (1U << 31);
        }
        
        void set_label(uchar_type label) {
            unit_ = (unit_ & ~0xFFU) | label;
        }
        
        void set_offset(id_type offset) {
            if (offset >= 1U << 29) {
                throw Exception("failed to modify unit: too large offset");
            }
            unit_ &= (1U << 31) | (1U << 8) | 0xFF;
            if (offset < 1U << 21) {
                unit_ |= (offset << 10);
            } else {
                unit_ |= (offset << 2) | (1U << 9);
            }
        }

        id_type unit() const {
            return unit_;
        }

    private:
        id_type unit_;

        friend class DoubleArrayUnit;
    };

    // DoubleArrayBuilderExtraUnit类，双数组构建额外单元
    class DoubleArrayBuilderExtraUnit {
    public:
        DoubleArrayBuilderExtraUnit() : prev_(0), next_(0),
            is_fixed_(false), is_used_(false) {}

        void set_prev(id_type prev) {
            prev_ = prev;
        }
        
        void set_next(id_type next) {
            next_ = next;
        }
        
        void set_is_fixed(bool is_fixed) {
            is_fixed_ = is_fixed;
        }
        
        void set_is_used(bool is_used) {
            is_used_ = is_used;
        }

        id_type prev() const {
            return prev_;
        }
        
        id_type next() const {
            return next_;
        }
        
        bool is_fixed() const {
            return is_fixed_;
        }
        
        bool is_used() const {
            return is_used_;
        }

    private:
        id_type prev_;
        id_type next_;
        bool is_fixed_;
        bool is_used_;
    };

    // DoubleArrayBuilder类，双数组构建器
    class DoubleArrayBuilder {
    public:
        explicit DoubleArrayBuilder(progress_func_type progress_func)
            : progress_func_(progress_func), extras_head_(0) {}

        template <typename T>
        void build(const Keyset<T>& keyset);
        
        void copy(std::size_t* size_ptr, DoubleArrayUnit** buf_ptr) const;
        void clear();

    private:
        static constexpr std::size_t BLOCK_SIZE = 256;
        static constexpr std::size_t NUM_EXTRA_BLOCKS = 16;
        static constexpr std::size_t NUM_EXTRAS = BLOCK_SIZE * NUM_EXTRA_BLOCKS;

        static constexpr std::size_t UPPER_MASK = 0xFF << 21;
        static constexpr std::size_t LOWER_MASK = 0xFF;

        using unit_type = DoubleArrayBuilderUnit;
        using extra_type = DoubleArrayBuilderExtraUnit;

        progress_func_type progress_func_;
        std::vector<unit_type> units_;
        std::unique_ptr<extra_type[]> extras_;
        std::vector<uchar_type> labels_;
        std::unique_ptr<id_type[]> table_;
        id_type extras_head_;

        std::size_t num_blocks() const {
            return units_.size() / BLOCK_SIZE;
        }

        const extra_type& extras(id_type id) const {
            return extras_[id % NUM_EXTRAS];
        }
        
        extra_type& extras(id_type id) {
            return extras_[id % NUM_EXTRAS];
        }

        template <typename T>
        void build_dawg(const Keyset<T>& keyset, DawgBuilder* dawg_builder);
        
        void build_from_dawg(const DawgBuilder& dawg);
        
        void build_from_dawg(const DawgBuilder& dawg,
            id_type dawg_id, id_type dic_id);
            
        id_type arrange_from_dawg(const DawgBuilder& dawg,
            id_type dawg_id, id_type dic_id);

        template <typename T>
        void build_from_keyset(const Keyset<T>& keyset);
        
        template <typename T>
        void build_from_keyset(const Keyset<T>& keyset, std::size_t begin,
            std::size_t end, std::size_t depth, id_type dic_id);
            
        template <typename T>
        id_type arrange_from_keyset(const Keyset<T>& keyset, std::size_t begin,
            std::size_t end, std::size_t depth, id_type dic_id);

        id_type find_valid_offset(id_type id) const;
        bool is_valid_offset(id_type id, id_type offset) const;

        void reserve_id(id_type id);
        void expand_units();

        void fix_all_blocks();
        void fix_block(id_type block_id);
    };

    template <typename T>
    void DoubleArrayBuilder::build(const Keyset<T>& keyset) {
        if (keyset.has_values()) {
            DawgBuilder dawg_builder;
            build_dawg(keyset, &dawg_builder);
            build_from_dawg(dawg_builder);
            dawg_builder.clear();
        } else {
            build_from_keyset(keyset);
        }
    }

    inline void DoubleArrayBuilder::copy(std::size_t* size_ptr,
        DoubleArrayUnit** buf_ptr) const {
        if (size_ptr != nullptr) {
            *size_ptr = units_.size();
        }
        if (buf_ptr != nullptr) {
            *buf_ptr = new DoubleArrayUnit[units_.size()];
            unit_type* units = reinterpret_cast<unit_type*>(*buf_ptr);
            for (std::size_t i = 0; i < units_.size(); ++i) {
                units[i] = units_[i];
            }
        }
    }

    inline void DoubleArrayBuilder::clear() {
        units_.clear();
        extras_.reset();
        labels_.clear();
        table_.reset();
        extras_head_ = 0;
    }

    template <typename T>
    void DoubleArrayBuilder::build_dawg(const Keyset<T>& keyset,
        DawgBuilder* dawg_builder) {
        dawg_builder->init();
        for (std::size_t i = 0; i < keyset.num_keys(); ++i) {
            dawg_builder->insert(keyset.keys(i), keyset.lengths(i), keyset.values(i));
            if (progress_func_ != nullptr) {
                progress_func_(i + 1, keyset.num_keys() + 1);
            }
        }
        dawg_builder->finish();
    }

    inline void DoubleArrayBuilder::build_from_dawg(const DawgBuilder& dawg) {
        std::size_t num_units = 1;
        while (num_units < dawg.size()) {
            num_units <<= 1;
        }
        units_.reserve(num_units);

        table_ = std::make_unique<id_type[]>(dawg.num_intersections());
        for (std::size_t i = 0; i < dawg.num_intersections(); ++i) {
            table_[i] = 0;
        }

        extras_ = std::make_unique<extra_type[]>(NUM_EXTRAS);

        reserve_id(0);
        extras(0).set_is_used(true);
        units_[0].set_offset(1);
        units_[0].set_label('\0');

        if (dawg.child(dawg.root()) != 0) {
            build_from_dawg(dawg, dawg.root(), 0);
        }

        fix_all_blocks();

        extras_.reset();
        labels_.clear();
        table_.reset();
    }

    inline void DoubleArrayBuilder::build_from_dawg(const DawgBuilder& dawg,
        id_type dawg_id, id_type dic_id) {
        id_type dawg_child_id = dawg.child(dawg_id);
        if (dawg.is_intersection(dawg_child_id)) {
            id_type intersection_id = dawg.intersection_id(dawg_child_id);
            id_type offset = table_[intersection_id];
            if (offset != 0) {
                offset ^= dic_id;
                if (!(offset & UPPER_MASK) || !(offset & LOWER_MASK)) {
                    if (dawg.is_leaf(dawg_child_id)) {
                        units_[dic_id].set_has_leaf(true);
                    }
                    units_[dic_id].set_offset(offset);
                    return;
                }
            }
        }

        id_type offset = arrange_from_dawg(dawg, dawg_id, dic_id);
        if (dawg.is_intersection(dawg_child_id)) {
            table_[dawg.intersection_id(dawg_child_id)] = offset;
        }

        do {
            uchar_type child_label = dawg.label(dawg_child_id);
            id_type dic_child_id = offset ^ child_label;
            if (child_label != '\0') {
                build_from_dawg(dawg, dawg_child_id, dic_child_id);
            }
            dawg_child_id = dawg.sibling(dawg_child_id);
        } while (dawg_child_id != 0);
    }

    inline id_type DoubleArrayBuilder::arrange_from_dawg(const DawgBuilder& dawg,
        id_type dawg_id, id_type dic_id) {
        labels_.clear();

        id_type dawg_child_id = dawg.child(dawg_id);
        while (dawg_child_id != 0) {
            labels_.push_back(dawg.label(dawg_child_id));
            dawg_child_id = dawg.sibling(dawg_child_id);
        }

        id_type offset = find_valid_offset(dic_id);
        units_[dic_id].set_offset(dic_id ^ offset);

        dawg_child_id = dawg.child(dawg_id);
        for (std::size_t i = 0; i < labels_.size(); ++i) {
            id_type dic_child_id = offset ^ labels_[i];
            reserve_id(dic_child_id);

            if (dawg.is_leaf(dawg_child_id)) {
                units_[dic_id].set_has_leaf(true);
                units_[dic_child_id].set_value(dawg.value(dawg_child_id));
            } else {
                units_[dic_child_id].set_label(labels_[i]);
            }

            dawg_child_id = dawg.sibling(dawg_child_id);
        }
        extras(offset).set_is_used(true);

        return offset;
    }

    template <typename T>
    void DoubleArrayBuilder::build_from_keyset(const Keyset<T>& keyset) {
        std::size_t num_units = 1;
        while (num_units < keyset.num_keys()) {
            num_units <<= 1;
        }
        units_.reserve(num_units);

        extras_ = std::make_unique<extra_type[]>(NUM_EXTRAS);

        reserve_id(0);
        extras(0).set_is_used(true);
        units_[0].set_offset(1);
        units_[0].set_label('\0');

        if (keyset.num_keys() > 0) {
            build_from_keyset(keyset, 0, keyset.num_keys(), 0, 0);
        }

        fix_all_blocks();

        extras_.reset();
        labels_.clear();
    }

    template <typename T>
    void DoubleArrayBuilder::build_from_keyset(const Keyset<T>& keyset,
        std::size_t begin, std::size_t end, std::size_t depth, id_type dic_id) {
        id_type offset = arrange_from_keyset(keyset, begin, end, depth, dic_id);

        while (begin < end) {
            if (keyset.keys(begin, depth) != '\0') {
                break;
            }
            ++begin;
        }
        if (begin == end) {
            return;
        }

        std::size_t last_begin = begin;
        uchar_type last_label = keyset.keys(begin, depth);
        while (++begin < end) {
            uchar_type label = keyset.keys(begin, depth);
            if (label != last_label) {
                build_from_keyset(keyset, last_begin, begin,
                    depth + 1, offset ^ last_label);
                last_begin = begin;
                last_label = keyset.keys(begin, depth);
            }
        }
        build_from_keyset(keyset, last_begin, end, depth + 1, offset ^ last_label);
    }

    template <typename T>
    id_type DoubleArrayBuilder::arrange_from_keyset(const Keyset<T>& keyset,
        std::size_t begin, std::size_t end, std::size_t depth, id_type dic_id) {
        labels_.clear();

        value_type value = -1;
        for (std::size_t i = begin; i < end; ++i) {
            uchar_type label = keyset.keys(i, depth);
            if (label == '\0') {
                if (keyset.has_lengths() && depth < keyset.lengths(i)) {
                    throw Exception("failed to build double-array: "
                        "invalid null character");
                } else if (keyset.values(i) < 0) {
                    throw Exception("failed to build double-array: negative value");
                }

                if (value == -1) {
                    value = keyset.values(i);
                }
                if (progress_func_ != nullptr) {
                    progress_func_(i + 1, keyset.num_keys() + 1);
                }
            }

            if (labels_.empty()) {
                labels_.push_back(label);
            } else if (label != labels_[labels_.size() - 1]) {
                if (label < labels_[labels_.size() - 1]) {
                    throw Exception("failed to build double-array: wrong key order");
                }
                labels_.push_back(label);
            }
        }

        id_type offset = find_valid_offset(dic_id);
        units_[dic_id].set_offset(dic_id ^ offset);

        for (std::size_t i = 0; i < labels_.size(); ++i) {
            id_type dic_child_id = offset ^ labels_[i];
            reserve_id(dic_child_id);
            if (labels_[i] == '\0') {
                units_[dic_id].set_has_leaf(true);
                units_[dic_child_id].set_value(value);
            } else {
                units_[dic_child_id].set_label(labels_[i]);
            }
        }
        extras(offset).set_is_used(true);

        return offset;
    }

    inline id_type DoubleArrayBuilder::find_valid_offset(id_type id) const {
        if (extras_head_ >= units_.size()) {
            return units_.size() | (id & LOWER_MASK);
        }

        id_type unfixed_id = extras_head_;
        do {
            id_type offset = unfixed_id ^ labels_[0];
            if (is_valid_offset(id, offset)) {
                return offset;
            }
            unfixed_id = extras(unfixed_id).next();
        } while (unfixed_id != extras_head_);

        return units_.size() | (id & LOWER_MASK);
    }

    inline bool DoubleArrayBuilder::is_valid_offset(id_type id,
        id_type offset) const {
        if (extras(offset).is_used()) {
            return false;
        }

        id_type rel_offset = id ^ offset;
        if ((rel_offset & LOWER_MASK) && (rel_offset & UPPER_MASK)) {
            return false;
        }

        for (std::size_t i = 1; i < labels_.size(); ++i) {
            if (extras(offset ^ labels_[i]).is_fixed()) {
                return false;
            }
        }

        return true;
    }

    inline void DoubleArrayBuilder::reserve_id(id_type id) {
        if (id >= units_.size()) {
            expand_units();
        }

        if (id == extras_head_) {
            extras_head_ = extras(id).next();
            if (extras_head_ == id) {
                extras_head_ = units_.size();
            }
        }
        extras(extras(id).prev()).set_next(extras(id).next());
        extras(extras(id).next()).set_prev(extras(id).prev());
        extras(id).set_is_fixed(true);
    }

    inline void DoubleArrayBuilder::expand_units() {
        id_type src_num_units = units_.size();
        id_type src_num_blocks = num_blocks();

        id_type dest_num_units = src_num_units + BLOCK_SIZE;
        id_type dest_num_blocks = src_num_blocks + 1;

        if (dest_num_blocks > NUM_EXTRA_BLOCKS) {
            fix_block(src_num_blocks - NUM_EXTRA_BLOCKS);
        }

        units_.resize(dest_num_units);

        if (dest_num_blocks > NUM_EXTRA_BLOCKS) {
            for (std::size_t id = src_num_units; id < dest_num_units; ++id) {
                extras(id).set_is_used(false);
                extras(id).set_is_fixed(false);
            }
        }

        for (id_type i = src_num_units + 1; i < dest_num_units; ++i) {
            extras(i - 1).set_next(i);
            extras(i).set_prev(i - 1);
        }

        extras(src_num_units).set_prev(dest_num_units - 1);
        extras(dest_num_units - 1).set_next(src_num_units);

        extras(src_num_units).set_prev(extras(extras_head_).prev());
        extras(dest_num_units - 1).set_next(extras_head_);

        extras(extras(extras_head_).prev()).set_next(src_num_units);
        extras(extras_head_).set_prev(dest_num_units - 1);
    }

    inline void DoubleArrayBuilder::fix_all_blocks() {
        id_type begin = 0;
        if (num_blocks() > NUM_EXTRA_BLOCKS) {
            begin = num_blocks() - NUM_EXTRA_BLOCKS;
        }
        id_type end = num_blocks();

        for (id_type block_id = begin; block_id != end; ++block_id) {
            fix_block(block_id);
        }
    }

    inline void DoubleArrayBuilder::fix_block(id_type block_id) {
        id_type begin = block_id * BLOCK_SIZE;
        id_type end = begin + BLOCK_SIZE;

        id_type unused_offset = 0;
        for (id_type offset = begin; offset != end; ++offset) {
            if (!extras(offset).is_used()) {
                unused_offset = offset;
                break;
            }
        }

        for (id_type id = begin; id != end; ++id) {
            if (!extras(id).is_fixed()) {
                reserve_id(id);
                units_[id].set_label(static_cast<uchar_type>(id ^ unused_offset));
            }
        }
    }
}  // namespace detail

// build方法实现
template <typename T>
int DoubleArray<T>::build(std::size_t num_keys,
    const key_type* const* keys, const std::size_t* lengths,
    const value_type* values, detail::progress_func_type progress_func) {
    detail::Keyset<value_type> keyset(num_keys, keys, lengths, values);

    detail::DoubleArrayBuilder builder(progress_func);
    builder.build(keyset);

    std::size_t size = 0;
    unit_type* buf = nullptr;
    builder.copy(&size, &buf);

    clear();

    size_ = size;
    array_ = buf;
    buf_ = buf;

    if (progress_func != nullptr) {
        progress_func(num_keys + 1, num_keys + 1);
    }

    return 0;
}

}  // namespace new_darts

#endif  // DARTS_H_
