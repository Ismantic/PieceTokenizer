#include "sentence.h"
#include "common.h"

#include <fstream>

namespace piece {


class PosixReadableFile : public ReadableFile {
  public:
    PosixReadableFile(std::string_view filename, bool is_binary)
        : is_(new std::ifstream(filename.data(),
                  is_binary ? std::ios::binary | std::ios::in : std::ios::in)) {
    }
  
    ~PosixReadableFile() {
      delete is_;
    }
  
    bool ReadLine(std::string *line) {
      return static_cast<bool>(std::getline(*is_, *line));
    }
  
    bool ReadAll(std::string *line) {
      line->assign(std::istreambuf_iterator<char>(*is_),
                  std::istreambuf_iterator<char>());
      return true;
    }
  
  private:
    std::istream *is_;
  };
  
  class PosixWritableFile : public WritableFile {
  public:
    PosixWritableFile(std::string_view filename, bool is_binary)
        : os_(new std::ofstream(filename.data(),
                                is_binary ? std::ios::binary | std::ios::out : std::ios::out)) {
    }
  
    ~PosixWritableFile() {
      delete os_;
    }
  
    bool Write(std::string_view text) {
      os_->write(text.data(), text.size());
      return os_->good();
    }
  
    bool WriteLine(std::string_view text) { return Write(text) && Write("\n"); }
  
  private:
    std::ostream *os_;
  };
  
  using DefaultReadableFile = PosixReadableFile;
  using DefaultWritableFile = PosixWritableFile;
  
  std::unique_ptr<ReadableFile> NewReadableFile(std::string_view filename, bool is_binary) {
    return std::make_unique<DefaultReadableFile>(filename, is_binary);
  }
  
  std::unique_ptr<WritableFile> NewWritableFile(std::string_view filename, bool is_binary) {
    return std::make_unique<DefaultWritableFile>(filename, is_binary);
  }

MultiFileSentenceIterator::MultiFileSentenceIterator(
    const std::vector<std::string> &files)
    : files_(files) {
  Next();
}

bool MultiFileSentenceIterator::done() const {
  return (!read_done_ && file_index_ == files_.size());
}

void MultiFileSentenceIterator::Next() {
  TryRead();

  if (!read_done_ && file_index_ < files_.size()) {
    const auto &filename = files_[file_index_++];
    fp_ = NewReadableFile(filename);
    LOG(INFO) << "Loading corpus: " << filename;

    TryRead();
  }
}

void MultiFileSentenceIterator::TryRead() {
  read_done_ = fp_ && fp_->ReadLine(&value_);
}


} // namespace