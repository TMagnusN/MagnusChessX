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
#include <iosfwd>
#include <string>

namespace magnus {
struct Position;
namespace memory {
struct Memory;
}
namespace nnue {
struct WdlTriplet;
}
}

namespace magnus::mnue::x2 {

struct Layout {
    static constexpr int InputBuckets = 10;
    static constexpr int OutputBuckets = 8;
    static constexpr int RelativeColors = 2;
    static constexpr int PieceTypes = 6;
    static constexpr int VictimClasses = RelativeColors * PieceTypes;
    static constexpr int Squares = 64;

    static constexpr int PieceInputSize =
        InputBuckets * RelativeColors * PieceTypes * Squares;
    static constexpr int AttackInputSize = 90048;
    static constexpr int InputSize = PieceInputSize + AttackInputSize;

    static constexpr int HiddenSize = 768;
    static constexpr int L1Size = 16;
    static constexpr int L2Size = 32;
    static constexpr int ArchId = 6;
    static constexpr int FeatureVersion = 1;
    static constexpr std::size_t MaxActive = 320;
};

static_assert(Layout::PieceInputSize == 7680);
static_assert(Layout::AttackInputSize == 90048);
static_assert(Layout::InputSize == 97728);

bool load(const std::string& path);
void unload() noexcept;

[[nodiscard]] bool loaded() noexcept;
[[nodiscard]] const std::string& path() noexcept;
[[nodiscard]] const std::string& last_error() noexcept;
[[nodiscard]] std::size_t network_bytes() noexcept;

[[nodiscard]] int evaluate_reference(
    const Position& pos,
    const memory::Memory& mem
) noexcept;

[[nodiscard]] int search_score(int v, const Position& pos) noexcept;
[[nodiscard]] int to_cp(int v, const Position& pos) noexcept;
[[nodiscard]] int search_score_to_cp(int score, const Position& pos) noexcept;
[[nodiscard]] nnue::WdlTriplet search_score_to_wdl(
    int score,
    const Position& pos
) noexcept;

void debug_dump_network(std::ostream& output);

} // namespace magnus::mnue::x2
