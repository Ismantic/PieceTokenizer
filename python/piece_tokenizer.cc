#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <variant>

#include "piece_spec.h"
#include "normalizer.h"
#include "pretokenizer.h"
#include "naive_tokenizer.h"
#include "piece_tokenizer.h"
#include "sentencepiece_tokenizer.h"
#include "bytepiece_tokenizer.h"

namespace py = pybind11;
using namespace piece;

class PyTokenizer {
public:
    PyTokenizer() = default;

    bool Load(const std::string& model_file, const std::string& dict = "") {
        Model model;
        if (!model.Load(model_file)) {
            return false;
        }
        const std::string method = model.GetCounterSpec().method();
        if (method != "naive" && method != "piece" &&
            method != "sentencepiece" && method != "bytepiece") {
            return false;
        }
        tokenizer_ = std::monostate{};
        normalizer_.reset();
        model_ = std::move(model);

        const auto& pretokenizer_spec = model_.GetPreTokenizerSpec();
        normalizer_ = std::make_unique<Normalizer>(pretokenizer_spec);

        if (method == "naive") {
            tokenizer_ = std::make_unique<NaiveTokenizer>(model_);
        } else if (method == "piece") {
            auto tokenizer = std::make_unique<PieceTokenizer>(model_, dict);
            if (!tokenizer->valid()) return false;
            tokenizer_ = std::move(tokenizer);
        } else if (method == "sentencepiece") {
            auto tokenizer =
                std::make_unique<SentencePieceTokenizer>(model_, dict);
            if (!tokenizer->valid()) return false;
            tokenizer_ = std::move(tokenizer);
        } else {
            tokenizer_ = std::make_unique<BytePieceTokenizer>(model_);
        }
        return true;
    }

    std::vector<std::pair<std::string, int>> Encode(const std::string& text) const {
        EnsureLoaded();
        return std::visit([&](const auto& tokenizer)
                -> std::vector<std::pair<std::string, int>> {
            using Alternative = std::decay_t<decltype(tokenizer)>;
            if constexpr (std::is_same_v<Alternative, std::monostate>) {
                return {};
            } else if constexpr (std::is_same_v<
                                     typename Alternative::element_type,
                                     NaiveTokenizer>) {
                // NaiveTokenizer does not normalize internally.
                return tokenizer->Encode(normalizer_->Normalize(text));
            } else {
                return tokenizer->Encode(text);
            }
        }, tokenizer_);
    }

    std::vector<int> EncodeAsIds(const std::string& text) const {
        EnsureLoaded();
        return std::visit([&](const auto& tokenizer) -> std::vector<int> {
            using Alternative = std::decay_t<decltype(tokenizer)>;
            if constexpr (std::is_same_v<Alternative, std::monostate>) {
                return {};
            } else if constexpr (std::is_same_v<
                                     typename Alternative::element_type,
                                     NaiveTokenizer>) {
                return tokenizer->EncodeAsIds(normalizer_->Normalize(text));
            } else {
                return tokenizer->EncodeAsIds(text);
            }
        }, tokenizer_);
    }

    std::vector<std::string> EncodeAsPieces(const std::string& text) const {
        auto result = Encode(text);
        std::vector<std::string> pieces;
        pieces.reserve(result.size());
        for (const auto& [piece, id] : result) {
            pieces.push_back(piece);
        }
        return pieces;
    }

    std::vector<std::pair<py::bytes, int>> EncodeBytes(
            const std::string& text) const {
        const auto result = Encode(text);
        std::vector<std::pair<py::bytes, int>> encoded;
        encoded.reserve(result.size());
        for (const auto& [piece, id] : result) {
            encoded.emplace_back(py::bytes(piece.data(), piece.size()), id);
        }
        return encoded;
    }

    std::vector<py::bytes> EncodeAsPieceBytes(const std::string& text) const {
        const auto result = Encode(text);
        std::vector<py::bytes> pieces;
        pieces.reserve(result.size());
        for (const auto& [piece, id] : result) {
            pieces.emplace_back(piece.data(), piece.size());
        }
        return pieces;
    }

    std::string Decode(const std::vector<int>& ids) const {
        EnsureLoaded();
        return std::visit([&](const auto& tokenizer) -> std::string {
            using Alternative = std::decay_t<decltype(tokenizer)>;
            if constexpr (std::is_same_v<Alternative, std::monostate>) {
                return "";
            } else {
                return tokenizer->Decode(ids);
            }
        }, tokenizer_);
    }

    int PieceToId(const std::string& piece) const {
        EnsureLoaded();
        return std::visit([&](const auto& tokenizer) {
            using Alternative = std::decay_t<decltype(tokenizer)>;
            if constexpr (std::is_same_v<Alternative, std::monostate>) {
                return -1;
            } else {
                return tokenizer->PieceID(piece);
            }
        }, tokenizer_);
    }

    std::string IdToPiece(int id) const {
        EnsureLoaded();
        const auto& pieces = model_.GetPieces();
        if (id >= 0 && id < static_cast<int>(pieces.size())) {
            return pieces[id].GetPiece();
        }
        return "";
    }

    // Bytes-typed accessor: BPE fragment pieces may not form valid UTF-8 by
    // themselves, so this returns the raw bytes without trying to decode.
    py::bytes IdToPieceBytes(int id) const {
        EnsureLoaded();
        const auto& pieces = model_.GetPieces();
        if (id >= 0 && id < static_cast<int>(pieces.size())) {
            const std::string& s = pieces[id].GetPiece();
            return py::bytes(s.data(), s.size());
        }
        return py::bytes();
    }

    int VocabSize() const {
        EnsureLoaded();
        return static_cast<int>(model_.PiecesSize());
    }

    const std::string& Method() const {
        EnsureLoaded();
        return model_.GetCounterSpec().method();
    }

private:
    using Tokenizer = std::variant<
        std::monostate,
        std::unique_ptr<NaiveTokenizer>,
        std::unique_ptr<PieceTokenizer>,
        std::unique_ptr<SentencePieceTokenizer>,
        std::unique_ptr<BytePieceTokenizer>>;

    void EnsureLoaded() const {
        if (std::holds_alternative<std::monostate>(tokenizer_)) {
            throw std::runtime_error("tokenizer is not loaded");
        }
    }

    Model model_;
    std::unique_ptr<Normalizer> normalizer_;
    Tokenizer tokenizer_;
};

// Model-free Normalize + Split, exposing the PreTokenizer's three axes:
//   split ∈ {word, isolate}   num ∈ {keep, split}   cn ∈ {"", "no", path}
class PyPreTokenizer {
public:
    PyPreTokenizer(const std::string& normalize = "no",
                   const std::string& split = "word",
                   const std::string& num = "keep",
                   const std::string& cn = "",
                   bool reconstruct = false) {
        spec_.SetName(normalize);
        spec_.SetCut(split == "isolate" ? 1 : 0);
        spec_.SetSplitDigits(num == "split");
        spec_.SetReconstruct(reconstruct);
        tokenizer_ = std::make_unique<piece::PreTokenizer>(spec_, cn);
    }

    std::vector<std::string> tokenize(const std::string& text) const {
        return tokenizer_->PreTokenize(text);
    }

private:
    PreTokenizerSpec spec_;
    std::unique_ptr<piece::PreTokenizer> tokenizer_;
};

PYBIND11_MODULE(_core, m) {
    m.doc() = "PieceTokenizer Python bindings";

    py::class_<PyPreTokenizer>(m, "PreTokenizer")
        .def(py::init<const std::string&, const std::string&, const std::string&,
                      const std::string&, bool>(),
             py::arg("normalize") = "no", py::arg("split") = "word",
             py::arg("num") = "keep", py::arg("cn") = "",
             py::arg("reconstruct") = false,
             "Create a pre-tokenizer (normalize + split). Axes: "
             "split=word|isolate, num=keep|split, cn=''|no|<dict path>")
        .def(py::init<const std::string&, const std::string&, const std::string&,
                      const std::string&, bool>(),
             py::arg("normalize") = "no", py::arg("split") = "word",
             py::arg("digit") = "keep", py::arg("cn") = "",
             py::arg("reconstruct") = false,
             "Backward-compatible constructor; use num instead of digit")
        .def("tokenize", &PyPreTokenizer::tokenize, py::arg("text"),
             "Pre-tokenize text into tokens");

    py::class_<PyTokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def("load", &PyTokenizer::Load, py::arg("model_file"), py::arg("dict") = "",
             "Load a trained model file, optionally with a dictionary for CN mode")
        .def("encode", &PyTokenizer::Encode, py::arg("text"),
             "Encode text into (piece, id) pairs")
        .def("encode_as_ids", &PyTokenizer::EncodeAsIds, py::arg("text"),
             "Encode text into token ids")
        .def("encode_as_pieces", &PyTokenizer::EncodeAsPieces, py::arg("text"),
             "Encode text into UTF-8 piece strings; byte fragments may raise UnicodeDecodeError")
        .def("encode_bytes", &PyTokenizer::EncodeBytes, py::arg("text"),
             "Encode text into (raw piece bytes, id) pairs")
        .def("encode_as_piece_bytes", &PyTokenizer::EncodeAsPieceBytes,
             py::arg("text"), "Encode text into raw piece byte strings")
        .def("decode", &PyTokenizer::Decode, py::arg("ids"),
             "Decode token ids back to text")
        .def("piece_to_id", &PyTokenizer::PieceToId, py::arg("piece"),
             "Convert a piece string to its id")
        .def("id_to_piece", &PyTokenizer::IdToPiece, py::arg("id"),
             "Convert an id to its piece string")
        .def("id_to_piece_bytes", &PyTokenizer::IdToPieceBytes, py::arg("id"),
             "Convert an id to its raw piece bytes (safe for non-UTF-8 fragments)")
        .def("vocab_size", &PyTokenizer::VocabSize,
             "Get vocabulary size")
        .def_property_readonly("method", &PyTokenizer::Method,
             "Get the tokenization method");
}
