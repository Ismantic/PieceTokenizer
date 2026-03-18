#include "new_normalizer.h"
#include "normalization_data.h"

#include <utility>
#include <vector>
#include <iostream>

#include "ustr.h"
#include "common.h"
#include "misc.h"
#include "sentence.h"

namespace piece {

// <trie size><double array trie><normalized string>
std::string MapBuilder::EncodePrecompiledMap(
    std::string_view trie_blob, std::string_view normalized) {
    std::string blob;
    blob.append(ustr::EncodePOD<uint32_t>(trie_blob.size()));
    blob.append(trie_blob.data(), trie_blob.size());
    blob.append(normalized.data(), normalized.size());
    return blob;
}

bool MapBuilder::DecodePrecompiledMap(
    std::string_view blob, std::string_view* trie_blob,
                           std::string_view* normalized) {
    uint32_t trie_blob_size = 0;
    if (blob.size() <= sizeof(trie_blob_size) || 
        !ustr::DecodePOD<uint32_t>(
            std::string_view(blob.data(), sizeof(trie_blob_size)),
            &trie_blob_size)) {
                return false;
    }
    if (trie_blob_size >= blob.size()) {
        return false;
    }
    blob.remove_prefix(sizeof(trie_blob_size));
    *trie_blob = std::string_view(blob.data(), trie_blob_size);
    blob.remove_prefix(trie_blob_size);
    *normalized = std::string_view(blob.data(), blob.size());
    return true;
}

bool MapBuilder::CompileUstrMap(const UstrMap& ustr_map, 
                                std::string* output) {
    if (ustr_map.empty()) return false;
    if (output->size() != 0) return false;

    LOG(INFO) << "Loading UstrMap of size=" << ustr_map.size();    

    // Aggregates the same target strings to save footprint
    std::map<Ustr, int> normalized2pos;
    for (const auto &p : ustr_map) {
        normalized2pos[p.second] = 0;
    }

    std::string normalized;
    for (auto &p : normalized2pos) {
        p.second = normalized.size(); // stores the pointer (position)
        const std::string utf8_out = ustr::UnicodeTextToUTF8(p.first);
        if (!ustr::IsStructurallyValid(utf8_out)) return false;
        normalized += utf8_out;
        normalized += '\0';
    }

    std::vector<std::pair<std::string,int>> kv; // key-value of Trie
    for (const auto& p : ustr_map) {
        const std::string utf8_in = ustr::UnicodeTextToUTF8(p.first);
        if (utf8_in.empty()) return false;
        if (!ustr::IsStructurallyValid(utf8_in)) return false;
        kv.emplace_back(utf8_in, misc::FindOrDie(normalized2pos, p.second));
    }

    std::sort(kv.begin(), kv.end());
    std::vector<const char*> key(kv.size());
    std::vector<int> value(kv.size());
    for (size_t i = 0; i < kv.size(); ++i) {
        key[i] = kv[i].first.c_str();
        value[i] = kv[i].second;
    }

    new_darts::DoubleArray<int> trie;
    trie.build(key.size(), const_cast<char **>(&key[0]), nullptr, &value[0]);

    int max_nodes_size = 0;
    std::vector<new_darts::DoubleArray<int>::ResultPair> results(
        2 * Normalizer::kMaxTrieResultSize);
    for (const char *str : key) {
        const int num_nodes = trie.commonPrefixSearch<new_darts::DoubleArray<int>::ResultPair>(str, 
                                                      results.data(),
                                                      results.size(), 
                                                      strlen(str));
        max_nodes_size = std::max(num_nodes, max_nodes_size);
    }
    if (max_nodes_size >= Normalizer::kMaxTrieResultSize) {
        LOG(INFO) << "This charmaps contains many shared prefix. "
                  << "The number of shared prefix must be less then "
                  << Normalizer::kMaxTrieResultSize;
        return false;
    }

    std::string_view trie_blob(static_cast<const char *>(trie.array()),
                               trie.size()*trie.unit_size());
    *output = EncodePrecompiledMap(trie_blob, normalized);

    LOG(INFO) << "Generated normalizer blob. size=" << output->size();

    return true;
}

bool MapBuilder::DecompileUstrMap(std::string_view blob, 
                                  UstrMap *ustr_map) {
    if (ustr_map == nullptr) return false;
    ustr_map->clear();

    std::string_view trie_blob, normalized;
    DecodePrecompiledMap(blob, &trie_blob, &normalized);

    Darts::DoubleArray trie;
    trie.set_array(const_cast<char*>(trie_blob.data()),
                   trie_blob.size()/trie.unit_size());
    
    std::string key;
    std::function<void(size_t, size_t)> traverse;

    // Given a Trie node at node_pos and the key position at key_position,
    // Expands children nodes from node_pos.
    // When leaf nodes are found, stores them into ustr_map
    traverse = [&traverse, &key, &trie, &normalized, &ustr_map] (
                size_t node_pos, size_t key_pos) -> void {
        for (int c = 0; c <= 255; ++c) {
            key.push_back(static_cast<char>(c));
            size_t node_pos_ = node_pos;
            size_t key_pos_ = key_pos_;
            const new_darts::DoubleArray<int>::value_type result = 
                trie.traverse(key.data(), node_pos_, key_pos_, key.size());
            if (result >= -1) {
                if (result >= 0) {
                    const std::string_view value = normalized.data()+result;
                    Ustr key_ustr, value_ustr;
                    for (const auto c : ustr::UTF8ToUnicodeText(key)) {
                        key_ustr.push_back(c);
                    }
                    for (const auto c : ustr::UTF8ToUnicodeText(value)) {
                        value_ustr.push_back(c);
                    }
                    (*ustr_map)[key_ustr] = value_ustr;
                }
                traverse(node_pos_, key_pos_);
            }
            key.pop_back();
        }
    };

    traverse(0, 0);
    return true;
}

bool MapBuilder::GetPrecompiledUstrMap(std::string_view name, 
                                       std::string* output) {
    if (output == nullptr) return false;

    if (name == "none") {
        output->clear();
        return true;
    }

    for (size_t i = 0; i < kNormalizationRules_size; ++i) {
        const auto *blob = &kNormalizationRules_blob[i];
        if (blob->name == name) {
            output->assign(blob->data, blob->size);
            LOG(INFO) << "Assign NormlizationRule " << name;
            return true;
        }
    }

    LOG(INFO) << "No precompiled ustr map is found: " << name;
    return false;
}

Normalizer::Normalizer(const NormalizerSpec& spec)
    : spec_(&spec) {
        Init();
}
Normalizer::~Normalizer() {}

void Normalizer::Init() {
    if (spec_->GetName() == "NMT_NFKC" ||
        spec_->GetName() == "NFKC" ||
        spec_->GetName() == "NFKC_CF" || 
        spec_->GetName() == "NMT_NFKC_CF")  {
        MapBuilder::GetPrecompiledUstrMap(spec_->GetName(), &map_data_);
    }
    std::string_view data = map_data_;
    if (!data.empty()) {
        std::string_view trie_blob, normalized;
        MapBuilder::DecodePrecompiledMap(data, &trie_blob, &normalized);

        trie_ = std::make_unique<new_darts::DoubleArray<int>>();
        trie_->set_array(const_cast<char *>(trie_blob.data()),
                         trie_blob.size() / trie_->unit_size());
        normalized_ = normalized.data(); 
    }
}

std::pair<std::string_view, int> Normalizer::ProcessTrie(std::string_view input) const {
    std::pair<std::string_view, int> p;

    if (trie_ != nullptr) {
        new_darts::DoubleArray<int>::ResultPair rs[kMaxTrieResultSize];
        const size_t num = trie_->commonPrefixSearch<new_darts::DoubleArray<int>::ResultPair>(
            input.data(), rs, kMaxTrieResultSize, input.size());
        
        if (num > 0) {
            size_t max = 0;
            int max_value = 0;
            for (size_t k = 0; k < num; ++k) {
                if (max == 0 || rs[k].length > max) {
                    max = rs[k].length;
                    max_value = rs[k].value;
                }
            }
            p.first = std::string_view(&normalized_[max_value]);
            p.second = max;
            return p;
        }
    }

    size_t n = 0;
    if (ustr::IsValidDecodeUTF8(input, &n)) {
        p.second = n;
        p.first = input.substr(0, n);
    } else {
        p.second = 1;
        static const char kReplacementChar[] = "\xEF\xBF\xBD"; // �
        p.first = kReplacementChar;
    }
    return p;
}

bool Normalizer::Normalize(std::string_view input,
                           std::string* output,
                           std::vector<size_t>* n2o) const {
    n2o->clear();
    output->clear();

    if (input.empty())
        return true;
    
    int consume = 0;
    std::pair<std::string_view,int> p;
    while (!input.empty()) {
        p = ProcessTrie(input);
        if (p.first == " ") {
            input.remove_prefix(p.second);
            consume += p.second;
        } else {
            break;
        }
    }
    if (input.empty()) return true;

    const size_t reserve_size = input.size() * 3;
    output->reserve(reserve_size);
    n2o->reserve(reserve_size);

    const std::string_view space = spec_->GetSpace();
    bool is_prev_space = true;
    while (!input.empty()) {
        p = ProcessTrie(input);
        std::string_view sp = p.first;

        while (is_prev_space && sp.size() > 0 && sp[0] == ' ') {
            sp.remove_prefix(1);
        }
          
        if (!sp.empty()) {
            for (size_t i = 0; i < sp.size(); ++i) {
                if (sp[i] == ' ') {
                    // replace SpaceSymbol
                    output->append(space.data(), space.size());
                    for (size_t j = 0; j < space.size(); ++j) {
                        n2o->push_back(consume);
                    }
                } else {
                    (*output) += sp[i];
                    n2o->push_back(consume);
                }
            }
            
            is_prev_space = !sp.empty() && sp.back() == ' ';
        }
        
        consume += p.second;
        input.remove_prefix(p.second);
    }

    while (output->size() >= space.size() && 
           output->substr(output->size() - space.size()) == space) {
      const int length = output->size() - space.size();
      consume = (*n2o)[length];
      output->resize(length);
      n2o->resize(length);
    }

    n2o->push_back(consume);

    return true;
}

std::string Normalizer::Normalize(std::string_view input) const {
    std::vector<size_t> n2o;
    std::string output;
    Normalize(input, &output, &n2o);
    return output;
}

std::string Normalizer::ReplaceSpace(std::string_view input) const {
    if (input.empty()) {
        return "";
    }

    const std::string_view space = spec_->GetSpace();

    if (space.empty() || input.find(space) == std::string_view::npos) {
        return std::string(input);
    }

    std::string output;
    
    size_t pos = 0;
    while (pos < input.size()) {
        size_t space_pos = input.find(space, pos);

        if (space_pos == std::string_view::npos) {
            output.append(input.substr(pos));
            break;
        }

        output.append(input.substr(pos, space_pos - pos));

        output.append(" ");

        pos = space_pos + space.size();
    }

    return output;
}


} // namespace piece