/*
MIT License

Copyright (c) 2026 TMagnusN

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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string_view>

namespace magnus::search {

/*
 * Runtime SPSA parameters.
 *
 * Keep this to scalar search heuristics: margins, thresholds, divisors and
 * small depth gates. Structural constants stay constexpr below because tuning
 * them would alter array sizes, TT format, telemetry shape, or object layout.
 */
inline int ASPIRATION_DELTA = 24;
inline int REPETITION_AVOID_BASE = 16;
inline int IIR_MIN_DEPTH = 6;
inline int IMPROVING_MARGIN = 16;

inline int RFP_BASE_MARGIN = 40;
inline int RFP_DEPTH_MARGIN = 64;
inline int RFP_IMPROVING_MARGIN = 40;
inline int RFP_OPPONENT_WORSENING_MARGIN = 24;
inline int RFP_CORRECTION_THRESHOLD = 128;
inline int RFP_CORRECTION_MARGIN_BONUS = 8;
inline int RFP_TT_CAPTURE_MARGIN_REDUCTION = 16;
inline int RFP_TT_QUIET_FAIL_HIGH_BONUS = 24;

inline int RAZOR_MARGIN_1 = 280;
inline int RAZOR_MARGIN_2 = 420;

inline int NMP_STATIC_BASE = 128;
inline int NMP_STATIC_DEPTH_SLOPE = 6;
inline int NMP_IMPROVING_MARGIN = 48;
inline int NMP_EVAL_BUCKET = 96;
inline int NMP_MIN_REDUCTION = 2;
inline int NMP_VERIFICATION_MIN_DEPTH = 16;
inline int NMP_VERIFICATION_MIN_SPAN = 2;

inline int FUTILITY_BASE_MARGIN = 56;
inline int FUTILITY_DEPTH_MARGIN = 56;
inline int FUTILITY_IMPROVING_MARGIN = 24;
inline int FUTILITY_HISTORY_DIVISOR = 128;

inline int SEE_PRUNE_DEPTH_LIMIT = 8;
inline int SEE_LATE_BAD_CAPTURE_GATE_THRESHOLD = -60;
inline int SEE_LATE_BAD_CAPTURE_GATE_MIN_DEPTH = 3;
inline int SEE_LATE_BAD_CAPTURE_GATE_MAX_DEPTH = 10;
inline int SEE_LATE_BAD_CAPTURE_GATE_MIN_CAPTURE_INDEX = 4;

inline int DELTA_MARGIN = 176;
inline int QS_ADJ_SHUFFLE_CAP = 80;

inline int PROBCUT_MIN_DEPTH = 5;
inline int PROBCUT_MARGIN = 224;
inline int PROBCUT_REDUCTION = 5;
inline int PROBCUT_TT_DEPTH_MARGIN = 3;
inline int SMALL_PROBCUT_MARGIN = 416;
inline int SMALL_PROBCUT_TT_DEPTH_MARGIN = 4;

inline int CAPTURE_HISTORY_HIGH_THRESHOLD = 128;

inline int SINGULAR_MIN_DEPTH = 8;
inline int SINGULAR_TT_DEPTH_MARGIN = 3;
inline int SINGULAR_MARGIN_BASE = 24;
inline int SINGULAR_MARGIN_PER_DEPTH = 4;
inline int SINGULAR_DOUBLE_MARGIN_BASE = 48;
inline int SINGULAR_DOUBLE_MARGIN_PER_DEPTH = 8;
inline int SINGULAR_DOUBLE_MIN_DEPTH = 12;
inline int SINGULAR_TRIPLE_MARGIN_BASE = 72;
inline int SINGULAR_TRIPLE_MARGIN_PER_DEPTH = 12;
inline int SINGULAR_TRIPLE_MIN_DEPTH = 18;
inline int SINGULAR_SCORE_NEAR_BETA = 32;
inline int SINGULAR_SCORE_STRONG = 64;
inline int SINGULAR_GOOD_HISTORY = 1024;
inline int SINGULAR_RECENT_EXTENSION_PLIES = 8;
inline int SINGULAR_RECENT_EXTENSION_LIMIT = 3;
inline int SINGULAR_COST_RATIO_PERCENT = 5;
inline int SINGULAR_COST_GATE_MIN_NODES = 4096;
inline int SINGULAR_TRUST_THRESHOLD_PV = 6;
inline int SINGULAR_TRUST_THRESHOLD_CUTLIKE = 6;
inline int SINGULAR_TRUST_THRESHOLD_ALLLIKE = 9;

inline int CORRECTION_HISTORY_GRAIN = 16;
inline int CORRECTION_HISTORY_CLAMP = 256;
inline int CORRECTION_HISTORY_WEIGHT_MAX = 96;
inline int CORRECTION_POSITION_WEIGHT = 2;
inline int CORRECTION_PAWN_WEIGHT = 2;
inline int CORRECTION_MATERIAL_WEIGHT = 1;

inline int LMR_TABLE_LOG_SCALE_X128 = 2747;
inline int QUIET_HISTORY_FP_DIVISOR = 12;
inline int CAPTURE_HISTORY_FP_DIVISOR = 16;
inline int LMR_DEEPER_RESEARCH_MARGIN = 96;
inline int LMR_SHALLOWER_RESEARCH_MARGIN = 8;

inline constexpr int CORRECTION_HISTORY_SIZE = 16384;
inline constexpr int FP_ONE_PLY = 1024;
inline constexpr int LMR_TABLE_MAX_INDEX = 64;
inline constexpr int CAPTURE_TOPK = 3;

inline constexpr std::size_t SINGULAR_NODE_KINDS = 3;
inline constexpr std::size_t SINGULAR_DEPTH_BANDS = 3;
inline constexpr std::size_t SINGULAR_SCORE_BANDS = 3;
inline constexpr std::size_t SINGULAR_TELEMETRY_BUCKETS =
    SINGULAR_NODE_KINDS * SINGULAR_DEPTH_BANDS * SINGULAR_SCORE_BANDS;
inline constexpr std::size_t SINGULAR_TELEMETRY_EXTENSION_LEVELS = 3;

[[nodiscard]] inline int razor_margin(int depth) noexcept {
    return depth <= 1 ? RAZOR_MARGIN_1
         : depth == 2 ? RAZOR_MARGIN_2
                      : 0;
}

[[nodiscard]] inline int correction_weight_sum() noexcept {
    return std::max(
        1,
        CORRECTION_POSITION_WEIGHT
            + CORRECTION_PAWN_WEIGHT
            + CORRECTION_MATERIAL_WEIGHT
    );
}

[[nodiscard]] inline int singular_trust_threshold(std::size_t node_kind) noexcept {
    if (node_kind == 0)
        return SINGULAR_TRUST_THRESHOLD_PV;
    if (node_kind == 1)
        return SINGULAR_TRUST_THRESHOLD_CUTLIKE;
    return SINGULAR_TRUST_THRESHOLD_ALLLIKE;
}

namespace tuning {

struct IntParam {
    std::string_view name;
    int default_value;
    int min_value;
    int max_value;
    int step;
    int spsa_delta;
    int* value;
};

[[nodiscard]] inline std::uint64_t& generation_ref() noexcept {
    static std::uint64_t generation = 0;
    return generation;
}

[[nodiscard]] inline std::uint64_t generation() noexcept {
    return generation_ref();
}

[[nodiscard]] inline const auto& int_params() noexcept {
    static const auto params = std::to_array<IntParam>({
        {"AspirationDelta", 24, 4, 160, 4, 8, &ASPIRATION_DELTA},
        {"RepetitionAvoidBase", 16, -128, 128, 4, 8, &REPETITION_AVOID_BASE},
        {"IirMinDepth", 6, 3, 12, 1, 1, &IIR_MIN_DEPTH},
        {"ImprovingMargin", 16, -64, 128, 4, 8, &IMPROVING_MARGIN},
        {"RfpBaseMargin", 40, 0, 240, 4, 8, &RFP_BASE_MARGIN},
        {"RfpDepthMargin", 64, 0, 240, 4, 8, &RFP_DEPTH_MARGIN},
        {"RfpImprovingMargin", 40, -64, 192, 4, 8, &RFP_IMPROVING_MARGIN},
        {"RfpOpponentWorseningMargin", 24, -64, 160, 4, 8, &RFP_OPPONENT_WORSENING_MARGIN},
        {"RfpCorrectionThreshold", 128, -256, 512, 8, 16, &RFP_CORRECTION_THRESHOLD},
        {"RfpCorrectionMarginBonus", 8, -64, 160, 4, 8, &RFP_CORRECTION_MARGIN_BONUS},
        {"RfpTtCaptureMarginReduction", 16, -64, 160, 4, 8, &RFP_TT_CAPTURE_MARGIN_REDUCTION},
        {"RfpTtQuietFailHighBonus", 24, -64, 192, 4, 8, &RFP_TT_QUIET_FAIL_HIGH_BONUS},
        {"RazorMargin1", 280, 0, 900, 8, 24, &RAZOR_MARGIN_1},
        {"RazorMargin2", 420, 0, 1300, 8, 32, &RAZOR_MARGIN_2},
        {"NmpStaticBase", 128, -128, 512, 8, 16, &NMP_STATIC_BASE},
        {"NmpStaticDepthSlope", 6, -32, 64, 1, 2, &NMP_STATIC_DEPTH_SLOPE},
        {"NmpImprovingMargin", 48, -128, 256, 8, 16, &NMP_IMPROVING_MARGIN},
        {"NmpEvalBucket", 96, 16, 384, 4, 16, &NMP_EVAL_BUCKET},
        {"NmpMinReduction", 2, 1, 5, 1, 1, &NMP_MIN_REDUCTION},
        {"NmpVerificationMinDepth", 16, 8, 32, 1, 2, &NMP_VERIFICATION_MIN_DEPTH},
        {"NmpVerificationMinSpan", 2, 1, 8, 1, 1, &NMP_VERIFICATION_MIN_SPAN},
        {"FutilityBaseMargin", 56, -128, 320, 4, 12, &FUTILITY_BASE_MARGIN},
        {"FutilityDepthMargin", 56, 0, 240, 4, 12, &FUTILITY_DEPTH_MARGIN},
        {"FutilityImprovingMargin", 24, -128, 192, 4, 12, &FUTILITY_IMPROVING_MARGIN},
        {"FutilityHistoryDivisor", 128, 16, 512, 4, 16, &FUTILITY_HISTORY_DIVISOR},
        {"SeePruneDepthLimit", 8, 2, 18, 1, 1, &SEE_PRUNE_DEPTH_LIMIT},
        {"SeeLateBadCaptureGateThreshold", -60, -400, 120, 4, 16, &SEE_LATE_BAD_CAPTURE_GATE_THRESHOLD},
        {"SeeLateBadCaptureGateMinDepth", 3, 1, 10, 1, 1, &SEE_LATE_BAD_CAPTURE_GATE_MIN_DEPTH},
        {"SeeLateBadCaptureGateMaxDepth", 10, 4, 24, 1, 2, &SEE_LATE_BAD_CAPTURE_GATE_MAX_DEPTH},
        {"SeeLateBadCaptureGateMinCaptureIndex", 4, 1, 16, 1, 1, &SEE_LATE_BAD_CAPTURE_GATE_MIN_CAPTURE_INDEX},
        {"DeltaMargin", 176, 0, 700, 8, 24, &DELTA_MARGIN},
        {"QsAdjShuffleCap", 80, 0, 320, 4, 12, &QS_ADJ_SHUFFLE_CAP},
        {"ProbcutMinDepth", 5, 3, 12, 1, 1, &PROBCUT_MIN_DEPTH},
        {"ProbcutMargin", 224, 0, 800, 8, 32, &PROBCUT_MARGIN},
        {"ProbcutReduction", 5, 2, 12, 1, 1, &PROBCUT_REDUCTION},
        {"ProbcutTtDepthMargin", 3, 1, 10, 1, 1, &PROBCUT_TT_DEPTH_MARGIN},
        {"SmallProbcutMargin", 416, 0, 1200, 8, 40, &SMALL_PROBCUT_MARGIN},
        {"SmallProbcutTtDepthMargin", 4, 1, 10, 1, 1, &SMALL_PROBCUT_TT_DEPTH_MARGIN},
        {"CaptureHistoryHighThreshold", 128, -512, 2048, 16, 64, &CAPTURE_HISTORY_HIGH_THRESHOLD},
        {"SingularMinDepth", 8, 5, 18, 1, 1, &SINGULAR_MIN_DEPTH},
        {"SingularTtDepthMargin", 3, 1, 10, 1, 1, &SINGULAR_TT_DEPTH_MARGIN},
        {"SingularMarginBase", 24, 0, 160, 4, 8, &SINGULAR_MARGIN_BASE},
        {"SingularMarginPerDepth", 4, 0, 24, 1, 2, &SINGULAR_MARGIN_PER_DEPTH},
        {"SingularDoubleMarginBase", 48, 0, 240, 4, 12, &SINGULAR_DOUBLE_MARGIN_BASE},
        {"SingularDoubleMarginPerDepth", 8, 0, 36, 1, 2, &SINGULAR_DOUBLE_MARGIN_PER_DEPTH},
        {"SingularDoubleMinDepth", 12, 8, 28, 1, 2, &SINGULAR_DOUBLE_MIN_DEPTH},
        {"SingularTripleMarginBase", 72, 0, 320, 4, 16, &SINGULAR_TRIPLE_MARGIN_BASE},
        {"SingularTripleMarginPerDepth", 12, 0, 48, 1, 3, &SINGULAR_TRIPLE_MARGIN_PER_DEPTH},
        {"SingularTripleMinDepth", 18, 12, 36, 1, 2, &SINGULAR_TRIPLE_MIN_DEPTH},
        {"SingularScoreNearBeta", 32, 0, 200, 4, 8, &SINGULAR_SCORE_NEAR_BETA},
        {"SingularScoreStrong", 64, 0, 320, 4, 12, &SINGULAR_SCORE_STRONG},
        {"SingularGoodHistory", 1024, -4096, 8192, 64, 256, &SINGULAR_GOOD_HISTORY},
        {"SingularRecentExtensionPlies", 8, 2, 24, 1, 2, &SINGULAR_RECENT_EXTENSION_PLIES},
        {"SingularRecentExtensionLimit", 3, 1, 10, 1, 1, &SINGULAR_RECENT_EXTENSION_LIMIT},
        {"SingularCostRatioPercent", 5, 1, 50, 1, 2, &SINGULAR_COST_RATIO_PERCENT},
        {"SingularCostGateMinNodes", 4096, 0, 65536, 256, 2048, &SINGULAR_COST_GATE_MIN_NODES},
        {"SingularTrustThresholdPv", 6, 1, 16, 1, 1, &SINGULAR_TRUST_THRESHOLD_PV},
        {"SingularTrustThresholdCutLike", 6, 1, 16, 1, 1, &SINGULAR_TRUST_THRESHOLD_CUTLIKE},
        {"SingularTrustThresholdAllLike", 9, 1, 20, 1, 1, &SINGULAR_TRUST_THRESHOLD_ALLLIKE},
        {"CorrectionHistoryGrain", 16, 1, 64, 1, 2, &CORRECTION_HISTORY_GRAIN},
        {"CorrectionHistoryClamp", 256, 32, 1024, 8, 32, &CORRECTION_HISTORY_CLAMP},
        {"CorrectionHistoryWeightMax", 96, 1, 256, 4, 12, &CORRECTION_HISTORY_WEIGHT_MAX},
        {"CorrectionPositionWeight", 2, 0, 8, 1, 1, &CORRECTION_POSITION_WEIGHT},
        {"CorrectionPawnWeight", 2, 0, 8, 1, 1, &CORRECTION_PAWN_WEIGHT},
        {"CorrectionMaterialWeight", 1, 0, 8, 1, 1, &CORRECTION_MATERIAL_WEIGHT},
        {"LmrTableLogScaleX128", 2747, 512, 8192, 16, 64, &LMR_TABLE_LOG_SCALE_X128},
        {"QuietHistoryFpDivisor", 12, 1, 64, 1, 2, &QUIET_HISTORY_FP_DIVISOR},
        {"CaptureHistoryFpDivisor", 16, 1, 64, 1, 2, &CAPTURE_HISTORY_FP_DIVISOR},
        {"LmrDeeperResearchMargin", 96, 0, 512, 4, 16, &LMR_DEEPER_RESEARCH_MARGIN},
        {"LmrShallowerResearchMargin", 8, -128, 128, 4, 8, &LMR_SHALLOWER_RESEARCH_MARGIN}
    });
    return params;
}

[[nodiscard]] inline const IntParam* find_int_param(std::string_view name) noexcept {
    for (const IntParam& param : int_params())
        if (param.name == name)
            return &param;
    return nullptr;
}

inline bool set_int_param(
    std::string_view name,
    int requested_value,
    int* applied_value = nullptr
) noexcept {
    const IntParam* param = find_int_param(name);
    if (!param)
        return false;

    const int clamped = std::clamp(
        requested_value,
        param->min_value,
        param->max_value
    );
    if (*param->value != clamped) {
        *param->value = clamped;
        ++generation_ref();
    }
    if (applied_value)
        *applied_value = clamped;
    return true;
}

inline void reset() noexcept {
    bool changed = false;
    for (const IntParam& param : int_params()) {
        if (*param.value != param.default_value) {
            *param.value = param.default_value;
            changed = true;
        }
    }
    if (changed)
        ++generation_ref();
}

inline void emit_uci_options(std::ostream& out) {
    for (const IntParam& param : int_params()) {
        out << "option name " << param.name
            << " type spin default " << param.default_value
            << " min " << param.min_value
            << " max " << param.max_value
            << '\n';
    }
}

inline void emit_values(std::ostream& out) {
    for (const IntParam& param : int_params()) {
        out << "info string tuning " << param.name
            << " current " << *param.value
            << " default " << param.default_value
            << " min " << param.min_value
            << " max " << param.max_value
            << '\n';
    }
}

inline void emit_spsa_csv(std::ostream& out) {
    for (const IntParam& param : int_params()) {
        out << param.name
            << ", int, "
            << param.default_value
            << ", "
            << param.min_value
            << ", "
            << param.max_value
            << ", "
            << param.step
            << ", "
            << param.spsa_delta
            << '\n';
    }
}

inline void emit_spsa_json(std::ostream& out) {
    out << "[\n";
    for (std::size_t i = 0; i < int_params().size(); ++i) {
        const IntParam& param = int_params()[i];
        out << "  {"
            << "\"name\":\"" << param.name << "\","
            << "\"type\":\"int\","
            << "\"default\":" << param.default_value << ','
            << "\"min\":" << param.min_value << ','
            << "\"max\":" << param.max_value << ','
            << "\"step\":" << param.step << ','
            << "\"spsa_delta\":" << param.spsa_delta << ','
            << "\"current\":" << *param.value
            << '}';
        if (i + 1 != int_params().size())
            out << ',';
        out << '\n';
    }
    out << "]\n";
}

inline void emit_spsa(std::ostream& out) {
    emit_spsa_json(out);
}

} // namespace tuning

} // namespace magnus::search
