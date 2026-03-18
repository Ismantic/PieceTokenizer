#pragma once

#include <random>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include <stdint.h>
#include <assert.h>

namespace misc {

template <class Collection, class Key>
bool ContainsKey(const Collection &collection, const Key &key) {
  return collection.find(key) != collection.end();
}

template <class Collection>
const typename Collection::value_type::second_type &FindOrDie(
    const Collection &collection,
    const typename Collection::value_type::first_type &key) {
  typename Collection::const_iterator it = collection.find(key);
  assert(it != collection.end());
  return it->second;
}

template <class Collection>
const typename Collection::value_type::second_type &FindWithDefault(
    const Collection &collection,
    const typename Collection::value_type::first_type &key,
    const typename Collection::value_type::second_type &value) {
  typename Collection::const_iterator it = collection.find(key);
  if (it == collection.end()) {
    return value;
  }
  return it->second;
}

template <class Collection>
bool InsertIfNotPresent(Collection *const collection,
                        const typename Collection::value_type &vt) {
  return collection->insert(vt).second;
}

template <class Collection>
bool InsertIfNotPresent(
    Collection *const collection,
    const typename Collection::value_type::first_type &key,
    const typename Collection::value_type::second_type &value) {
  return InsertIfNotPresent(collection,
                            typename Collection::value_type(key, value));
}

template <class Collection>
void InsertOrDie(Collection *const collection,
                 const typename Collection::value_type::first_type &key,
                 const typename Collection::value_type::second_type &data) {
  assert(InsertIfNotPresent(collection, key, data));
}

template <typename T>
void STLDeleteElements(std::vector<T *> *vec) {
  for (auto item : *vec) {
    delete item;
  }
  vec->clear();
}

template <typename K, typename V>
std::vector<std::pair<K, V>> Sorted(const std::vector<std::pair<K, V>> &m) {
  std::vector<std::pair<K, V>> v = m;
  std::sort(v.begin(), v.end(),
            [](const std::pair<K, V> &p1, const std::pair<K, V> &p2) {
              return (p1.second > p2.second ||
                      (p1.second == p2.second && p1.first < p2.first));
            });
  return v;
}

template <typename K, typename V>
std::vector<std::pair<K, V>> Sorted(const std::unordered_map<K, V> &m) {
  std::vector<std::pair<K, V>> v(m.begin(), m.end());
  return Sorted(v);
}

inline void mix(uint64_t &a, uint64_t &b, uint64_t &c) {  // 64bit version
  a -= b;
  a -= c;
  a ^= (c >> 43);
  b -= c;
  b -= a;
  b ^= (a << 9);
  c -= a;
  c -= b;
  c ^= (b >> 8);
  a -= b;
  a -= c;
  a ^= (c >> 38);
  b -= c;
  b -= a;
  b ^= (a << 23);
  c -= a;
  c -= b;
  c ^= (b >> 5);
  a -= b;
  a -= c;
  a ^= (c >> 35);
  b -= c;
  b -= a;
  b ^= (a << 49);
  c -= a;
  c -= b;
  c ^= (b >> 11);
  a -= b;
  a -= c;
  a ^= (c >> 12);
  b -= c;
  b -= a;
  b ^= (a << 18);
  c -= a;
  c -= b;
  c ^= (b >> 22);
}

inline uint64_t FingerprintCat(uint64_t x, uint64_t y) {
  uint64_t b = 0xe08c1d668b756f82;  // more of the golden ratio
  mix(x, b, y);
  return y;
}

} // namespace misc
