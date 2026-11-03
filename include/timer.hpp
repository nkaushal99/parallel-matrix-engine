#pragma once

#include <chrono>

class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}

    double elapsed_ms() const {
        const auto elapsed = std::chrono::steady_clock::now() - start_;
        return std::chrono::duration<double, std::milli>(elapsed).count();
    }

private:
    std::chrono::steady_clock::time_point start_;
};
