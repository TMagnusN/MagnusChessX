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
#include <ostream>
#include <string>

#include "mnue/MnueV2Features.h"

#ifndef MAGNUS_MNUEV2_TELEMETRY
#define MAGNUS_MNUEV2_TELEMETRY 0
#endif

namespace magnus {
struct Position;
namespace memory {
struct Memory;
}
}

namespace magnus::mnue::v2 {

enum class AttackWeightType : u8 {
    Int8 = 1,
    Int16 = 2
};

enum class MoveClass : u8 {
    Quiet = 0,
    Capture,
    PawnMove,
    PawnCapture,
    KingMove,
    Castling,
    Promotion,
    EnPassant,
    Count
};

constexpr std::size_t MoveClassCount =
    static_cast<std::size_t>(MoveClass::Count);
constexpr std::size_t DeltaHistogramBins = 769;

struct BranchTelemetry {
    u64 rows_added = 0;
    u64 rows_removed = 0;
    u64 unique_rows_changed = 0;
    u64 duplicate_deltas = 0;
    u64 cancelled_deltas = 0;
    u64 full_refreshes = 0;
    u64 cache_hits = 0;
    u64 cache_misses = 0;
    u64 accumulator_rebuilds = 0;
    u64 moves = 0;
    u64 max_rows_changed = 0;
#if MAGNUS_MNUEV2_TELEMETRY
    std::array<u64, DeltaHistogramBins> rows_changed_histogram{};
    std::array<u64, MoveClassCount> updates_by_move_class{};
    std::array<u64, MoveClassCount> rows_by_move_class{};
#endif
};

struct Telemetry {
    BranchTelemetry position{};
    BranchTelemetry attack{};
    BranchTelemetry structure{};
    i32 attack_accumulator_min = 0;
    i32 attack_accumulator_max = 0;
    u64 attack_accumulator_samples = 0;
    u64 attack_activation_clips = 0;
    u64 affected_pieces = 0;
    u64 changed_slider_rays = 0;
    u64 tactical_summary_transitions = 0;
    u64 structure_rebuilds_on_non_pawn_moves = 0;
#if MAGNUS_MNUEV2_TELEMETRY
    std::array<u64, 4097> attack_abs_histogram{};
#endif
};

struct GoldenSnapshot {
    EncodedFeatures features{};
    int material = 0;
    int bucket = 0;
    std::array<u64, COLOR_NB> position_hash{};
    std::array<u64, COLOR_NB> attack_hash{};
    std::array<u64, COLOR_NB> structure_hash{};
    double output = 0.0;
    int score = 0;
};

class AccumulatorStack {
public:
    AccumulatorStack() noexcept;
    ~AccumulatorStack();

    AccumulatorStack(AccumulatorStack&&) noexcept;
    AccumulatorStack& operator=(AccumulatorStack&&) noexcept;

    AccumulatorStack(const AccumulatorStack&) = delete;
    AccumulatorStack& operator=(const AccumulatorStack&) = delete;

    void reset() noexcept;
    void push(
        const Position& pos,
        const memory::Memory& mem,
        Move move
    ) noexcept;
    void after_make(
        const Position& pos,
        const memory::Memory& mem
    ) noexcept;
    void pop(
        const Position& pos,
        const memory::Memory& mem
    ) noexcept;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] Telemetry telemetry() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend int evaluate_incremental(
        const Position&,
        const memory::Memory&,
        AccumulatorStack&
    ) noexcept;
    friend bool debug_check_incremental(
        const Position&,
        const memory::Memory&,
        AccumulatorStack&,
        std::ostream&
    ) noexcept;
    friend std::size_t accumulator_stack_bytes() noexcept;
};

[[nodiscard]] bool load(const std::string& path);
void unload() noexcept;

[[nodiscard]] bool loaded() noexcept;
[[nodiscard]] const std::string& path() noexcept;
[[nodiscard]] const std::string& last_error() noexcept;
[[nodiscard]] AttackWeightType attack_weight_type() noexcept;
[[nodiscard]] const char* backend_name() noexcept;
void set_force_scalar(bool force) noexcept;
[[nodiscard]] bool force_scalar() noexcept;
[[nodiscard]] bool telemetry_enabled() noexcept;
[[nodiscard]] std::size_t network_bytes() noexcept;
[[nodiscard]] std::size_t accumulator_stack_bytes() noexcept;

// Scalar full-refresh oracle. All sparse accumulation is i32. Dense inference
// follows the exported dequantised reference and returns score_scale * output,
// rounded to nearest integer.
[[nodiscard]] int evaluate_reference(
    const Position& pos,
    const memory::Memory& mem
) noexcept;

[[nodiscard]] int evaluate_incremental(
    const Position& pos,
    const memory::Memory& mem,
    AccumulatorStack& stack
) noexcept;

[[nodiscard]] double evaluate_reference_output(
    const Position& pos,
    const memory::Memory& mem
) noexcept;

[[nodiscard]] bool debug_check_incremental(
    const Position& pos,
    const memory::Memory& mem,
    AccumulatorStack& stack,
    std::ostream& output
) noexcept;

void debug_dump_network(std::ostream& output);
void debug_dump_position(
    const Position& pos,
    const memory::Memory& mem,
    std::ostream& output
);
void print_telemetry(const Telemetry& telemetry, std::ostream& output);
[[nodiscard]] bool debug_attack_kernel_selftest(std::ostream& output) noexcept;
[[nodiscard]] bool debug_loader_selftest(std::ostream& output);
[[nodiscard]] bool make_golden_snapshot(
    const Position& pos,
    const memory::Memory& mem,
    GoldenSnapshot& snapshot
) noexcept;

} // namespace magnus::mnue::v2
