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
#include <limits>
#include <string>

#include "Types.h"

namespace magnus {
struct Position;
namespace memory {
struct Memory;
}
}

namespace magnus::mnue::v2 {

struct Layout {
    static constexpr u32 FormatVersion = 4;
    static constexpr u32 ArchitectureId = 7;
    static constexpr u32 BucketMappingVersion = 1;

    static constexpr int PositionFeatureCount = 12'288;
    static constexpr int AttackFeatureCount = 153'216;
    static constexpr int StructureFeatureCount = 34'685;
    static constexpr int AttackStatusEnd = 49'152;
    static constexpr int AttackEdgeEnd = 139'200;
    static constexpr int AttackRelationEnd = 151'488;

    // Raw accumulator widths. Pairwise CReLU halves each perspective, then
    // STM and NTM are concatenated to restore these final branch widths.
    static constexpr int PositionAccumulatorWidth = 320;
    static constexpr int AttackAccumulatorWidth = 288;
    static constexpr int StructureAccumulatorWidth = 160;
    static constexpr int PositionPerspectiveWidth = 160;
    static constexpr int AttackPerspectiveWidth = 144;
    static constexpr int StructurePerspectiveWidth = 80;
    static constexpr int ConcatWidth = 768;

    static constexpr int HeadHidden1 = 32;
    static constexpr int HeadHidden2 = 32;
    static constexpr int OutputBuckets = 12;

    static constexpr std::size_t PositionMaxActive = 32;
    static constexpr std::size_t AttackMaxActive = 384;
    static constexpr std::size_t StructureMaxActive = 64;

    static constexpr u32 NoFeature =
        std::numeric_limits<u32>::max();

    static constexpr std::size_t PositionSlotCount = 64;
    static constexpr std::size_t AttackStatusSlotCount = 64;
    static constexpr std::size_t AttackRelationSlotCount = 64;
    static constexpr std::size_t AttackEdgeSlotCount = 64 * 64;
    static constexpr std::size_t AttackPressureSlotCount = 2 * 6;
    static constexpr std::size_t AttackSlotCount =
        AttackStatusSlotCount
        + AttackRelationSlotCount
        + AttackEdgeSlotCount
        + AttackPressureSlotCount;

    static constexpr std::size_t StructurePawnSlotCount = 2 * 64;
    static constexpr std::size_t StructureFileSlotCount = 8;
    static constexpr std::size_t StructureIslandSlotCount = 2;
    static constexpr std::size_t StructureCenterSlotCount = 1;
    static constexpr std::size_t StructureShelterSlotCount = 2;
    static constexpr std::size_t StructureOutpostSlotCount = 64;
    static constexpr std::size_t StructureComplexSlotCount = 2;
    static constexpr std::size_t StructureBlockerSlotCount = 2 * 64;
    static constexpr std::size_t StructureSlotCount =
        StructurePawnSlotCount
        + StructureFileSlotCount
        + StructureIslandSlotCount
        + StructureCenterSlotCount
        + StructureShelterSlotCount
        + StructureOutpostSlotCount
        + StructureComplexSlotCount
        + StructureBlockerSlotCount;
};

static_assert(
    Layout::PositionAccumulatorWidth
        + Layout::AttackAccumulatorWidth
        + Layout::StructureAccumulatorWidth
    == Layout::ConcatWidth
);
static_assert(
    Layout::PositionPerspectiveWidth * 2
    == Layout::PositionAccumulatorWidth
);
static_assert(
    Layout::AttackPerspectiveWidth * 2
    == Layout::AttackAccumulatorWidth
);
static_assert(
    Layout::StructurePerspectiveWidth * 2
    == Layout::StructureAccumulatorWidth
);
static_assert(Layout::ConcatWidth == 768);

template<std::size_t Capacity>
struct FeatureList {
    std::array<u32, Capacity> indices{};
    std::size_t count = 0;
};

using PositionFeatureList = FeatureList<Layout::PositionMaxActive>;
using AttackFeatureList = FeatureList<Layout::AttackMaxActive>;
using StructureFeatureList = FeatureList<Layout::StructureMaxActive>;

struct EncodedFeatures {
    std::array<PositionFeatureList, COLOR_NB> position{};
    std::array<AttackFeatureList, COLOR_NB> attack{};
    std::array<StructureFeatureList, COLOR_NB> structure{};
};

struct AttackGraph {
    std::array<Bitboard, SQ_NB> attacks_from{};
    std::array<Bitboard, SQ_NB> attackers_to{};
};

[[nodiscard]] Bitboard compute_piece_attacks(
    const Position& pos,
    const memory::Memory& mem,
    Square source
) noexcept;

[[nodiscard]] u32 compute_position_slot(
    const Position& pos,
    Color perspective,
    Square square
) noexcept;

[[nodiscard]] bool build_attack_graph(
    const Position& pos,
    const memory::Memory& mem,
    AttackGraph& graph
) noexcept;

[[nodiscard]] u32 compute_attack_slot(
    const Position& pos,
    const memory::Memory& mem,
    const AttackGraph& graph,
    Color perspective,
    std::size_t slot
) noexcept;

void compute_attack_slot_pair(
    const Position& pos,
    const memory::Memory& mem,
    const AttackGraph& graph,
    std::size_t slot,
    std::array<u32, COLOR_NB>& output
) noexcept;

void compute_attack_summary_pairs(
    const Position& pos,
    const AttackGraph& graph,
    Square square,
    std::array<u32, COLOR_NB>& status_output,
    std::array<u32, COLOR_NB>& relation_output
) noexcept;

[[nodiscard]] u32 compute_structure_slot(
    const Position& pos,
    Color perspective,
    std::size_t slot
) noexcept;

void compute_structure_slot_pair(
    const Position& pos,
    std::size_t slot,
    std::array<u32, COLOR_NB>& output
) noexcept;

[[nodiscard]] bool debug_attack_graph_invariant(
    const Position& pos,
    const memory::Memory& mem,
    const AttackGraph& graph,
    std::ostream& output
) noexcept;

[[nodiscard]] bool encode_position_features(
    const Position& pos,
    Color perspective,
    PositionFeatureList& output
) noexcept;

[[nodiscard]] bool encode_attack_features(
    const Position& pos,
    const memory::Memory& mem,
    Color perspective,
    AttackFeatureList& output
) noexcept;

[[nodiscard]] bool encode_attack_features_pair(
    const Position& pos,
    const memory::Memory& mem,
    std::array<AttackFeatureList, COLOR_NB>& output
) noexcept;

[[nodiscard]] bool encode_structure_features(
    const Position& pos,
    Color perspective,
    StructureFeatureList& output
) noexcept;

[[nodiscard]] bool encode_all_features(
    const Position& pos,
    const memory::Memory& mem,
    EncodedFeatures& output
) noexcept;

[[nodiscard]] int material_units(const Position& pos) noexcept;
[[nodiscard]] int material_bucket_from_units(int units) noexcept;
[[nodiscard]] int material_bucket(const Position& pos) noexcept;
[[nodiscard]] bool debug_bucket_selftest(std::ostream& output);

[[nodiscard]] std::string decode_position_feature(u32 index);
[[nodiscard]] std::string decode_attack_feature(u32 index);
[[nodiscard]] std::string decode_structure_feature(u32 index);

void debug_dump_features(
    const Position& pos,
    const memory::Memory& mem,
    std::ostream& output
);

} // namespace magnus::mnue::v2
