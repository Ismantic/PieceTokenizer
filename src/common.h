#pragma once

#include <string.h> 
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <memory>

#include <time.h>

namespace piece {

enum LogLevel {
    INFO = 0,
    WARNING = 1,
    ERROR = 2,
    FATAL = 3
};

static int MinLogLevel = INFO;

inline int GetMinLogLevel() { return MinLogLevel; }
inline void SetMinLogLevel(int level) { MinLogLevel = level; }

inline const char* GetBaseName(const char* filename) {
  const char* base = strrchr(filename, '/');
  return base ? base + 1 : filename;
}

inline std::string GetTimeStr() {
  auto now = std::chrono::system_clock::now();  
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
  return ss.str();
}

inline const char* LogLevelToStr(LogLevel level) {
  static const char* level_strings[] = {
    "INFO", "WARNING", "ERROR", "FATAL"
  };
  return level_strings[level];
}

class Logger {
public:
    Logger(const char* file, int line, LogLevel level) 
        : file_(GetBaseName(file)), line_(line), level_(level) {
        stream_ << GetTimeStr() << " " 
                << LogLevelToStr(level) << " "
                << file_ << ":" << line_ << " ";
    }

    ~Logger() {
        stream_ << std::endl;
        std::cerr << stream_.str();
        
        if (level_ == FATAL) {
            std::cerr << "Program terminated due to fatal error." << std::endl;
            exit(1);
        }
    }

    std::ostream& stream() { return stream_; }

private:
    std::ostringstream stream_;
    const char* file_;
    int line_;
    LogLevel level_;
};

class NoLogger {
public:
    NoLogger() {}
    ~NoLogger() {}
    
    std::ostream& stream() {
        static std::ostream* disabled_stream = nullptr;
        if (!disabled_stream) {
            disabled_stream = new std::ostream(nullptr);
            disabled_stream->setstate(std::ios::badbit); 
        }
        return *disabled_stream;
    }
};

static thread_local std::unique_ptr<Logger> g_logger;
inline std::ostream& Log(const char* file, int line, LogLevel severity) {
  if (GetMinLogLevel() > severity) {
      static NoLogger no_logger;
      return no_logger.stream();
  } else {
      g_logger.reset(new Logger(file, line, severity));
      return g_logger->stream();
  }
}

class LogMessage {
    public:
        LogMessage(const char* file, int line, LogLevel level) 
            : logger_(new Logger(file, line, level)) {}
        
        ~LogMessage() { 
            // 析构时自动完成日志输出
        }
        
        std::ostream& stream() { return logger_->stream(); }
    
    private:
        std::unique_ptr<Logger> logger_;
};
} // namespace piece

//#define LOG(severity) piece::Log(__FILE__, __LINE__, piece::severity)

#define LOG(severity) \
    piece::LogMessage(__FILE__, __LINE__, piece::severity).stream()
