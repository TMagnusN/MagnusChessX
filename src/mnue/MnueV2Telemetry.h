#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>

#include "Types.h"

#ifndef MAGNUS_MNUEV2_TELEMETRY
#define MAGNUS_MNUEV2_TELEMETRY 0
#endif

namespace magnus::mnue::v2 {

enum class CycleKind : u8 {
    FeatureReconstruction,
    PositionGeneration,
    AttackGeneration,
    StructureGeneration,
    Deduplication,
    SetDiff,
    RowApplication,
    SelectedHead,
    SemanticMake,
    SemanticUnmake,
    PositionSemantic,
    AttackSemantic,
    StructureSemantic,
    Materialisation,
    Count
};

struct SampledCycles {
    u64 events = 0;
    u64 samples = 0;
    u64 cycles = 0;
};

struct CycleTelemetry {
    u64 nodes = 0;
    u64 static_eval_calls = 0;
    u64 full_head_calls = 0;
    u64 tt_eval_reuse = 0;
    std::array<
        SampledCycles,
        static_cast<std::size_t>(CycleKind::Count)
    > sampled{};
};

#if MAGNUS_MNUEV2_TELEMETRY
extern thread_local CycleTelemetry g_cycle_telemetry;

[[nodiscard]] u64 read_cycle_clock() noexcept;

class CycleScope {
public:
    explicit CycleScope(CycleKind kind) noexcept;
    ~CycleScope();

    CycleScope(const CycleScope&) = delete;
    CycleScope& operator=(const CycleScope&) = delete;

private:
    SampledCycles* counter_ = nullptr;
    u64 start_ = 0;
};

inline void telemetry_note_node() noexcept {
    ++g_cycle_telemetry.nodes;
}

inline void telemetry_note_static_eval() noexcept {
    ++g_cycle_telemetry.static_eval_calls;
}

inline void telemetry_note_full_head() noexcept {
    ++g_cycle_telemetry.full_head_calls;
}

inline void telemetry_note_tt_eval_reuse() noexcept {
    ++g_cycle_telemetry.tt_eval_reuse;
}
#else
class CycleScope {
public:
    explicit CycleScope(CycleKind) noexcept {}
};

inline void telemetry_note_node() noexcept {}
inline void telemetry_note_static_eval() noexcept {}
inline void telemetry_note_full_head() noexcept {}
inline void telemetry_note_tt_eval_reuse() noexcept {}
#endif

void reset_cycle_telemetry() noexcept;
[[nodiscard]] CycleTelemetry cycle_telemetry() noexcept;
void print_cycle_telemetry(std::ostream& output);

} // namespace magnus::mnue::v2
