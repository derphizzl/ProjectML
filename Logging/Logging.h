#pragma once
#include <iostream>
#include <string>
#include <map>
#include <memory>
#include <mutex>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <algorithm>

#define LOG_TRACE(mod, msg)   Logging::Instance().log(mod, Logging::Level::TRACE, msg)
#define LOG_DEBUG(mod, msg)   Logging::Instance().log(mod, Logging::Level::DEBUG, msg)
#define LOG_INFO(mod, msg)    Logging::Instance().log(mod, Logging::Level::INFO, msg)
#define LOG_WARN(mod, msg)    Logging::Instance().log(mod, Logging::Level::WARNING, msg)
#define LOG_ERROR(mod, msg)   Logging::Instance().log(mod, Logging::Level::ERROR, msg)

#define TOS(msg)              std::to_string(msg)

class Logging {
public:
    enum class Level {
        NONE = 0,
        ERROR,
        WARNING,
        INFO,
        DEBUG,
        TRACE
    };

    static Logging& Instance() {
        static Logging instance;
        return instance;
    }

    void log(const std::string& module, Level level, const std::string& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        getLogger(module)->log(level, msg);
    }

private:
    class LoggerInstance {
    public:
        LoggerInstance(const std::string& module) : moduleName_(module) {
            initLogLevel();
        }

        void log(Level level, const std::string& msg) {
            if (level > level_) return;

            std::cout << timestamp() << " [" << levelToString(level) << "] "
                      << moduleName_ << ": " << msg << std::endl;
        }

    private:
        std::string moduleName_;
        Level level_ = Level::NONE;

        void initLogLevel() {
            std::string envName = "LOG_" + toUpper(moduleName_);
            const char* levelStr = std::getenv(envName.c_str());

            if (!levelStr) levelStr = std::getenv("LOG_LEVEL");
            if (!levelStr) {
                level_ = Level::NONE;
                return;
            }

            std::string val(levelStr);
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            if      (val == "trace")   level_ = Level::TRACE;
            else if (val == "debug")   level_ = Level::DEBUG;
            else if (val == "info")    level_ = Level::INFO;
            else if (val == "warn")    level_ = Level::WARNING;
            else if (val == "error")   level_ = Level::ERROR;
            else                       level_ = Level::NONE;
        }

        static std::string levelToString(Level level) {
            switch (level) {
                case Level::ERROR:   return "ERROR";
                case Level::WARNING: return "WARN";
                case Level::INFO:    return "INFO";
                case Level::DEBUG:   return "DEBUG";
                case Level::TRACE:   return "TRACE";
                default:             return "NONE";
            }
        }

        static std::string timestamp() {
            using namespace std::chrono;
            auto now = system_clock::now();
            auto in_time = system_clock::to_time_t(now);
            auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

            std::stringstream ss;
            ss << std::put_time(std::localtime(&in_time), "%F %T")
               << '.' << std::setw(3) << std::setfill('0') << ms.count();
            return ss.str();
        }

        static std::string toUpper(const std::string& str) {
            std::string res = str;
            std::transform(res.begin(), res.end(), res.begin(), ::toupper);
            return res;
        }
    };

    std::map<std::string, std::unique_ptr<LoggerInstance>> loggers_;
    std::mutex mutex_;

    LoggerInstance* getLogger(const std::string& module) {
        if (loggers_.find(module) == loggers_.end()) {
            loggers_[module] = std::make_unique<LoggerInstance>(module);
        }
        return loggers_[module].get();
    }

    // Prevent direct construction
    Logging() = default;
    ~Logging() = default;
    Logging(const Logging&) = delete;
    Logging& operator=(const Logging&) = delete;
};

