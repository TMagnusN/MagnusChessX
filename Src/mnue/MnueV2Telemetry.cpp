#include "mnue/MnueV2Telemetry.h"

#include <array>
#include <immintrin.h>
#include <ostream>

namespace magnus::mnue::v2 {

#if MAGNUS_MNUEV2_TELEMETRY
thread_local CycleTelemetry g_cycle_telemetry{};

u64 read_cycle_clock() noexcept {
    _mm_lfence();
    const u64 value = __rdtsc();
    _mm_lfence();
    return value;
}

CycleScope::CycleScope(CycleKind kind) noexcept {
    SampledCycles& counter = g_cycle_telemetry.sampled[
        static_cast<std::size_t>(kind)
    ];
    ++counter.events;
    constexpr u64 SampleMask = 63;
    if ((counter.events & SampleMask) == 0) {
        counter_ = &counter;
        ++counter.samples;
        start_ = read_cycle_clock();
    }
}

CycleScope::~CycleScope() {
    if (counter_ != nullptr)
        counter_->cycles += read_cycle_clock() - start_;
}
#endif

void reset_cycle_telemetry() noexcept {
#if MAGNUS_MNUEV2_TELEMETRY
    g_cycle_telemetry = {};
#endif
}

CycleTelemetry cycle_telemetry() noexcept {
#if MAGNUS_MNUEV2_TELEMETRY
    return g_cycle_telemetry;
#else
    return {};
#endif
}

void print_cycle_telemetry(std::ostream& output) {
    output << "mnue v2 cycle telemetry enabled "
        << (MAGNUS_MNUEV2_TELEMETRY ? 1 : 0) << '\n';
#if MAGNUS_MNUEV2_TELEMETRY
    constexpr std::array<const char*, 14> Names{{
        "feature_reconstruction",
        "position_generation",
        "attack_generation",
        "structure_generation",
        "deduplication",
        "set_diff",
        "row_application",
        "selected_head",
        "semantic_make",
        "semantic_unmake",
        "position_semantic",
        "attack_semantic",
        "structure_semantic",
        "materialisation"
    }};
    output << "mnue v2 cycle totals nodes " << g_cycle_telemetry.nodes
        << " static_eval_calls " << g_cycle_telemetry.static_eval_calls
        << " full_head_calls " << g_cycle_telemetry.full_head_calls
        << " tt_eval_reuse " << g_cycle_telemetry.tt_eval_reuse
        << " evals_per_node "
        << (
            g_cycle_telemetry.nodes == 0
                ? 0.0
                : static_cast<double>(
                    g_cycle_telemetry.static_eval_calls
                ) / static_cast<double>(g_cycle_telemetry.nodes)
        )
        << '\n';
    for (std::size_t index = 0; index < Names.size(); ++index) {
        const SampledCycles& counter =
            g_cycle_telemetry.sampled[index];
        const double cycles_per_sample = counter.samples == 0
            ? 0.0
            : static_cast<double>(counter.cycles)
                / static_cast<double>(counter.samples);
        output << "mnue v2 cycles " << Names[index]
            << " events " << counter.events
            << " samples " << counter.samples
            << " sampled_cycles " << counter.cycles
            << " cycles_per_event " << cycles_per_sample
            << '\n';
    }
#else
    output << "mnue v2 cycle telemetry rebuild with "
        << "MNUEV2_TELEMETRY=1\n";
#endif
}

} // namespace magnus::mnue::v2
