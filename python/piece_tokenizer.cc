#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <memory>

#include "piece_spec.h"
#include "normalizer.h"
#include "tokenizer.h"
#include "naive_counter.h"
#include "naive_tokenizer.h"
#include "piece_counter.h"
#include "piece_tokenizer.h"
#include "sentencepiece_tokenizer.h"
#include "bytepiece_counter.h"

namespace py = pybind11;
using namespace piece;

class PyTokenizer {
public:
    PyTokenizer() = default;

    bool Load(const std::string& model_file, const std::string& cn_dict = "") {
        if (!model_.Load(model_file)) {
            return false;
        }
        method_ = model_.GetCounterSpec().method();

        const auto& normalizer_spec = model_.GetNormalizerSpec();
        normalizer_ = std::make_unique<Normalizer>(normalizer_spec);

        if (method_ == "naive") {
            naive_tok_ = std::make_unique<NaiveTokenizer>(model_);
        } else if (method_ == "piece") {
            piece_tok_ = std::make_unique<PieceTokenizer>(model_, cn_dict);
        } else if (method_ == "sentencepiece") {
            sp_tok_ = std::make_unique<SentencePieceTokenizer>(model_);
        } else if (method_ == "bytepiece") {
            bp_tok_ = std::make_unique<BytePieceTokenizer>(model_);
        } else {
            return false;
        }
        return true;
    }

    std::vector<std::pair<std::string, int>> Encode(const std::string& text) const {
        if (method_ == "naive") {
            return naive_tok_->Encode(normalizer_->Normalize(text));
        } else if (method_ == "piece") {
            return piece_tok_->Encode(normalizer_->Normalize(text));
        } else if (method_ == "sentencepiece") {
            return sp_tok_->Encode(text);
        } else if (method_ == "bytepiece") {
            return bp_tok_->Encode(text);
        }
        return {};
    }

    std::vector<int> EncodeAsIds(const std::string& text) const {
        auto result = Encode(text);
        std::vector<int> ids;
        ids.reserve(result.size());
        for (const auto& [piece, id] : result) {
            ids.push_back(id);
        }
        return ids;
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

    std::string Decode(const std::vector<int>& ids) const {
        if (method_ == "naive") {
            return naive_tok_->Decode(ids);
        } else if (method_ == "piece") {
            return piece_tok_->Decode(ids);
        } else if (method_ == "sentencepiece") {
            return sp_tok_->Decode(ids);
        } else if (method_ == "bytepiece") {
            return bp_tok_->Decode(ids);
        }
        return "";
    }

    int PieceToId(const std::string& piece) const {
        if (method_ == "sentencepiece" && sp_tok_) {
            return sp_tok_->PieceID(piece);
        } else if (method_ == "naive" && naive_tok_) {
            return naive_tok_->PieceID(piece);
        } else if (method_ == "piece" && piece_tok_) {
            return piece_tok_->PieceID(piece);
        }
        // Fallback: linear search
        const auto& pieces = model_.GetPieces();
        for (size_t i = 0; i < pieces.size(); i++) {
            if (pieces[i].GetPiece() == piece) return static_cast<int>(i);
        }
        return -1;
    }

    std::string IdToPiece(int id) const {
        const auto& pieces = model_.GetPieces();
        if (id >= 0 && id < static_cast<int>(pieces.size())) {
            return pieces[id].GetPiece();
        }
        return "";
    }

    // Bytes-typed accessor: BPE fragment pieces may not form valid UTF-8 by
    // themselves, so this returns the raw bytes without trying to decode.
    py::bytes IdToPieceBytes(int id) const {
        const auto& pieces = model_.GetPieces();
        if (id >= 0 && id < static_cast<int>(pieces.size())) {
            const std::string& s = pieces[id].GetPiece();
            return py::bytes(s.data(), s.size());
        }
        return py::bytes();
    }

    int VocabSize() const {
        return static_cast<int>(model_.PiecesSize());
    }

    const std::string& Method() const { return method_; }

private:
    Model model_;
    std::string method_;
    std::unique_ptr<Normalizer> normalizer_;
    std::unique_ptr<NaiveTokenizer> naive_tok_;
    std::unique_ptr<PieceTokenizer> piece_tok_;
    std::unique_ptr<SentencePieceTokenizer> sp_tok_;
    std::unique_ptr<BytePieceTokenizer> bp_tok_;
};

class PreTokenizer {
public:
    PreTokenizer(const std::string& normalize = "no", int cut = 0, bool reconstruct = false) {
        spec_.SetName(normalize);
        spec_.SetCut(cut);
        spec_.SetReconstruct(reconstruct);
        tokenizer_ = std::make_unique<piece::Tokenizer>(spec_);
    }

    std::vector<std::string> tokenize(const std::string& text) const {
        return tokenizer_->Tokenize(text);
    }

private:
    NormalizerSpec spec_;
    std::unique_ptr<piece::Tokenizer> tokenizer_;
};

PYBIND11_MODULE(piece_tokenizer, m) {
    m.doc() = "PieceTokenizer Python bindings";

    py::class_<PreTokenizer>(m, "PreTokenizer")
        .def(py::init<const std::string&, int, bool>(),
             py::arg("normalize") = "no", py::arg("cut") = 0, py::arg("reconstruct") = false,
             "Create a pre-tokenizer (normalize + split)")
        .def("tokenize", &PreTokenizer::tokenize, py::arg("text"),
             "Pre-tokenize text into tokens");

    py::class_<PyTokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def("load", &PyTokenizer::Load, py::arg("model_file"), py::arg("cn_dict") = "",
             "Load a trained model file, optionally with cn-dict for piece method")
        .def("encode", &PyTokenizer::Encode, py::arg("text"),
             "Encode text into (piece, id) pairs")
        .def("encode_as_ids", &PyTokenizer::EncodeAsIds, py::arg("text"),
             "Encode text into token ids")
        .def("encode_as_pieces", &PyTokenizer::EncodeAsPieces, py::arg("text"),
             "Encode text into piece strings")
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
