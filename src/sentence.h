#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <memory>


namespace piece {

class ReadableFile {
    public:
        ReadableFile() {}
        explicit ReadableFile(std::string_view filename, bool is_binary = false) {}
        virtual ~ReadableFile() {}

        virtual bool ReadLine(std::string *line) = 0;
        virtual bool ReadAll(std::string *line) = 0;
};

class WritableFile {
    public:
        WritableFile() {}
        explicit WritableFile(std::string_view filename, bool is_binary = false) {}
        virtual ~WritableFile() {}

        virtual bool Write(std::string_view text) = 0;
        virtual bool WriteLine(std::string_view text) = 0;
};

std::unique_ptr<ReadableFile> NewReadableFile(std::string_view filename, bool is_binary = false);
std::unique_ptr<WritableFile> NewWritableFile(std::string_view filename, bool is_binary = false);

class SentenceIterator {
    public:
        virtual ~SentenceIterator() {}

        virtual bool done() const = 0;
        virtual void Next() = 0;
        virtual const std::string &value() const = 0;
};

class MultiFileSentenceIterator : public SentenceIterator {
    public:
        explicit MultiFileSentenceIterator(const std::vector<std::string> &files);
        ~MultiFileSentenceIterator() {}

        bool done() const override;
        void Next() override;
        const std::string &value() const override { return value_; }

    private:
        void TryRead();

        bool read_done_ = false;
        size_t file_index_ = 0;
        std::vector<std::string> files_;
        std::string value_;
        std::unique_ptr<ReadableFile> fp_;
};

} // namespace