/*
MIT License

Copyright (c) 2026 Magnus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace magnus {
struct Position;
namespace memory {
struct Memory;
}
}

namespace magnus::mnue::x1 {

class AccumulatorStack {
public:
    struct Stats {
        std::size_t feature_builds = 0;
        std::size_t rebuilds = 0;
        std::size_t diff_updates = 0;
        std::size_t added_rows = 0;
        std::size_t removed_rows = 0;
    };

    AccumulatorStack() noexcept;
    ~AccumulatorStack();

    AccumulatorStack(AccumulatorStack&&) noexcept;
    AccumulatorStack& operator=(AccumulatorStack&&) noexcept;

    AccumulatorStack(const AccumulatorStack&) = delete;
    AccumulatorStack& operator=(const AccumulatorStack&) = delete;

    void reset() noexcept;
    void push() noexcept;
    void pop() noexcept;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] Stats stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend int evaluate_incremental(
        const Position&,
        const memory::Memory&,
        AccumulatorStack&
    ) noexcept;
};

bool load(const std::string& path);
void unload() noexcept;

[[nodiscard]] bool loaded() noexcept;
[[nodiscard]] const std::string& path() noexcept;
[[nodiscard]] const std::string& last_error() noexcept;

// Scalar, full-rebuild implementation used to validate trained networks and
// the file format before an incremental SIMD search integration is attempted.
[[nodiscard]] int evaluate_reference(
    const Position& pos,
    const memory::Memory& mem
) noexcept;

// Full-rebuild inference with the production integer representation:
// i16 SIMD feature accumulation, u8 pairwise activation, and sparse SIMD L1.
[[nodiscard]] int evaluate_fast(
    const Position& pos,
    const memory::Memory& mem
) noexcept;

[[nodiscard]] int evaluate_incremental(
    const Position& pos,
    const memory::Memory& mem,
    AccumulatorStack& stack
) noexcept;

} // namespace magnus::mnue::x1
