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
#include <cstdint>
#include <iosfwd>
#include <string>

namespace magnus {
struct Position;
namespace memory {
struct Memory;
}
}

namespace magnus::mnue::x2k16 {

struct Layout {
    static constexpr int InputBuckets = 16;
    static constexpr int OutputBuckets = 8;
    static constexpr int RelativeColors = 2;
    static constexpr int PieceTypes = 6;
    static constexpr int VictimClasses = RelativeColors * PieceTypes;
    static constexpr int Squares = 64;
    static constexpr int PawnSquares = 48;
    static constexpr int PawnTokens = RelativeColors * PawnSquares;

    static constexpr int PieceInputSize =
        InputBuckets * RelativeColors * PieceTypes * Squares;
    static constexpr int AttackInputSize = 90048;
    static constexpr int PawnPairInputSize =
        PawnTokens * (PawnTokens - 1) / 2;

    static constexpr int PieceHiddenSize = 768;
    static constexpr int AttackHiddenSize = 384;
    static constexpr int PawnPairHiddenSize = 768;
    static constexpr int MergedHiddenSize = 768;
    static constexpr int PairwiseSize = MergedHiddenSize / 2;
    static constexpr int HeadInputSize = PairwiseSize * 2;
    static constexpr int L1Size = 16;
    static constexpr int L2Size = 32;

    static constexpr int ArchId = 8;
    static constexpr int FeatureVersion = 2;
};

static_assert(Layout::PieceInputSize == 12288);
static_assert(Layout::AttackInputSize == 90048);
static_assert(Layout::PawnPairInputSize == 4560);
static_assert(Layout::HeadInputSize == 768);

bool load(const std::string& path);
void unload() noexcept;

[[nodiscard]] bool loaded() noexcept;
[[nodiscard]] const std::string& path() noexcept;
[[nodiscard]] const std::string& last_error() noexcept;
[[nodiscard]] std::size_t network_bytes() noexcept;
[[nodiscard]] std::uintmax_t expected_payload_bytes() noexcept;
[[nodiscard]] std::uintmax_t expected_file_bytes() noexcept;

[[nodiscard]] int evaluate_reference(
    const Position& pos,
    const memory::Memory& mem
) noexcept;

void debug_dump_network(std::ostream& output);
void debug_dump_features(const Position& pos, std::ostream& output);
void debug_dump_evaluation(
    const Position& pos,
    const memory::Memory& mem,
    std::ostream& output
);

} // namespace magnus::mnue::x2k16
