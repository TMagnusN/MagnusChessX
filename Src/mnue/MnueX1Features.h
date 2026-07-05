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

#include <array>
#include <cstddef>
#include <iosfwd>

#include "Types.h"

namespace magnus {
struct Position;
namespace memory {
struct Memory;
}
}

namespace magnus::mnue::x1 {

struct Layout {
    static constexpr int InputBuckets = 16;
    static constexpr int OutputBuckets = 8;
    static constexpr int RelativeColors = 2;
    static constexpr int NonKingPieceTypes = 5;
    static constexpr int PieceTypes = 6;
    static constexpr int Squares = 64;
    static constexpr int TacticalStates = 64;

    static constexpr int PieceInputSize =
        InputBuckets * RelativeColors * NonKingPieceTypes * Squares;
    static constexpr int TacticalInputSize =
        RelativeColors * PieceTypes * Squares * TacticalStates;
    static constexpr int InputSize = PieceInputSize + TacticalInputSize;

    static constexpr int HiddenSize = 768;
    static constexpr int L1Size = 16;
    static constexpr int L2Size = 32;
    static constexpr int ArchId = 5;
    static constexpr int FeatureVersion = 1;
    static constexpr std::size_t MaxActive = 62;
};

static_assert(Layout::PieceInputSize == 10240);
static_assert(Layout::TacticalInputSize == 49152);
static_assert(Layout::InputSize == 59392);
static_assert(Layout::InputSize <= 65536);

enum TacticalStatus : u8 {
    EnemyPawn       = 1u << 0,
    EnemyKnight     = 1u << 1,
    EnemyDiagonal   = 1u << 2,
    EnemyOrthogonal = 1u << 3,
    EnemyKing       = 1u << 4,
    Defended        = 1u << 5
};

using FeatureList = std::array<u16, Layout::MaxActive>;
using PerspectiveFeatureLists = std::array<FeatureList, COLOR_NB>;
using PerspectiveFeatureCounts = std::array<std::size_t, COLOR_NB>;

[[nodiscard]] int input_bucket(
    const Position& pos,
    Color perspective
) noexcept;

[[nodiscard]] int output_bucket(const Position& pos) noexcept;

[[nodiscard]] u8 tactical_status(
    const Position& pos,
    const memory::Memory& mem,
    Square square
) noexcept;

[[nodiscard]] int piece_feature_index(
    const Position& pos,
    Color perspective,
    Piece piece,
    Square square
) noexcept;

[[nodiscard]] int tactical_feature_index(
    Color perspective,
    Piece piece,
    Square square,
    u8 status
) noexcept;

[[nodiscard]] std::size_t collect_features(
    const Position& pos,
    const memory::Memory& mem,
    Color perspective,
    FeatureList& output
) noexcept;

void collect_feature_pairs(
    const Position& pos,
    const memory::Memory& mem,
    PerspectiveFeatureLists& output,
    PerspectiveFeatureCounts& counts
) noexcept;

void debug_dump_features(
    const Position& pos,
    const memory::Memory& mem,
    std::ostream& output
);

} // namespace magnus::mnue::x1
