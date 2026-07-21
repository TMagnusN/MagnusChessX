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

#include <atomic>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "HistoryContext.h"
#include "Memory.h"
#include "Tuning.h"
#include "Types.h"

namespace magnus {
struct Position;
}

/*
 * Search public interface — the core search engine of MagnusChessX Thinking.
 *
 * This header exposes two entry points:
 *   1. iterative_deepening() — launches an iterative-deepening PVS search from the
 *      root position, used by bench and the UCI loop
 *   2. move_to_uci() — converts an internal 16-bit move to a UCI coordinate string
 *
 * The internal implementation uses the classic iterative deepening + principal
 * variation search (PVS) framework, with transposition-table-driven ordering,
 * null-move pruning, singular extensions, probabilistic cut-offs,
 * and a suite of low-overhead heuristic pruning techniques.
 *
 * All search state is encapsulated in an anonymous namespace inside Search.cpp
 * and is completely opaque to external callers.
 */
namespace magnus::search {

constexpr std::size_t TT_MOVE_TRUST_SIZE = 8192;
constexpr int TT_MOVE_TRUST_LIMIT = 8192;
constexpr std::size_t TT_MOVE_TRUST_BUCKETS = 8;

struct TtTrustBucketCounters {
    u64 searched = 0;
    u64 tt_best = 0;
    u64 other_best = 0;
    u64 fail_low = 0;
    u64 single_extension = 0;
    u64 double_extension = 0;
    u64 triple_extension = 0;
    u64 multi_cut = 0;
    u64 depth_sum = 0;
    u64 updates = 0;
    u64 saturated = 0;
};

struct TtTrustTelemetry {
    std::array<TtTrustBucketCounters, TT_MOVE_TRUST_BUCKETS> bucket{};
};

struct WorkerPersistentState {
    std::array<std::array<i16, COLOR_NB>, TT_MOVE_TRUST_SIZE> tt_move_trust{};
    TtTrustTelemetry telemetry{};
    u64 game_epoch = 0;
    u64 search_sequence = 0;

    [[nodiscard]] int trust(const Position& pos) const noexcept;
    void update_trust(
        const Position& pos,
        int bonus,
        std::size_t telemetry_bucket = TT_MOVE_TRUST_BUCKETS
    ) noexcept;
    void clear() noexcept;
};

static_assert(sizeof(WorkerPersistentState::tt_move_trust) == 32 * 1024);

class SearchSessionState {
public:
    SearchSessionState();
    void ensure_workers(std::size_t count);
    [[nodiscard]] WorkerPersistentState& worker(std::size_t id) noexcept;
    void clear() noexcept;
    void new_game() noexcept;
    [[nodiscard]] u64 begin_search(std::size_t active);

private:
    std::vector<WorkerPersistentState> workers_{};
    u64 game_epoch_ = 0;
    u64 search_sequence_ = 0;
};

[[nodiscard]] std::size_t tt_move_trust_bucket(int trust) noexcept;
[[nodiscard]] constexpr int tt_trust_required_gap(
    int base_gap,
    int pre_update_trust
) noexcept {
    const int raw_adjustment = pre_update_trust / 512;
    const int adjustment = raw_adjustment < -8
        ? -8
        : raw_adjustment > 8 ? 8 : raw_adjustment;
    const int required_gap = base_gap - adjustment;
    return required_gap < 1 ? 1 : required_gap;
}
[[nodiscard]] bool capture_stage(Move move) noexcept;
[[nodiscard]] bool is_shuffling(
    Move move,
    const Position& pos,
    Move move_at_ply_minus_2,
    Move move_at_ply_minus_4,
    int ply
) noexcept;
[[nodiscard]] constexpr int singular_child_depth(
    int move_base_depth,
    int extension
) noexcept {
    return move_base_depth + extension;
}
[[nodiscard]] constexpr int depth_after_alpha_improvement(
    int node_depth,
    bool decisive
) noexcept {
    return node_depth > 2 && node_depth < 14 && !decisive
        ? node_depth - 2
        : node_depth;
}
[[nodiscard]] constexpr int fail_high_softbound(
    int best_value,
    int beta,
    int node_depth
) noexcept {
    return (best_value * node_depth + beta) / (node_depth + 1);
}

/*
 * Search-layer global constants:
 *   MAX_PLY            — search stack ply budget
 *   MAX_SEARCH_DEPTH   — maximum completed root depth reported by iterative deepening
 *   MAX_GAME_HISTORY   — maximum number of game history entries, used for repetition detection
 */
constexpr int MAX_PLY = 246;
constexpr int MAX_SEARCH_DEPTH = MAX_PLY - 1;
constexpr int MAX_GAME_HISTORY = 128;

// ============================================================
// Search Tuning Parameters — every numeric knob in one place
// ============================================================
// inline constexpr → zero runtime overhead, each .cpp that
// includes this header gets its own copy; the optimizer folds.
//
// Sections mirror the search execution order inside pvs().
// Conservative adjustment: ±10 % then test. ±50 % will break.
// ============================================================

// --- Value / Score Bounds ------------------------------------
inline constexpr int VALUE_INF                  = 32000;
inline constexpr int VALUE_MATE                 = 31000;
inline constexpr int VALUE_TB                   = 30000;
inline constexpr int VALUE_NONE                 = 32002;
inline constexpr int VALUE_MATE_IN_MAX_PLY      = VALUE_MATE - MAX_PLY;
inline constexpr int VALUE_MATED_IN_MAX_PLY     = -VALUE_MATE_IN_MAX_PLY;
inline constexpr int VALUE_TB_WIN_IN_MAX_PLY    = VALUE_TB - MAX_PLY;
inline constexpr int VALUE_TB_LOSS_IN_MAX_PLY   = -VALUE_TB_WIN_IN_MAX_PLY;

struct RootMsvEntry {
    int credit = 0;
    int exactHits = 0;
    int boundHits = 0;
    int maxDepth = 0;
    int maxSelDepth = 0;
    int stableCount = 0;
    std::uint64_t workerMask = 0;
    int workerHits = 0;
    int activeWorkers = 0;
    int lastScore = 0;
    memory::Bound lastBound = memory::BOUND_NONE;
    bool pvValid = false;
    int pvLength = 0;
    bool stopped = false;
};

struct RootMsvShared;
struct SearchTimeSignals;

struct TimeManagementState {
    Score previous_score = VALUE_NONE;
    Score previous_average_score = VALUE_NONE;
    double previous_time_reduction = 0.85;
};

enum class SearchEvalKind : std::uint8_t {
    None,
    P2,
    X2K16
};

enum class TtTrustStage : std::uint8_t {
    A = 1,
    B = 2,
    C = 3,
    D = 4
};

// The normal engine runs one fixed experiment stage. Other stages are selected
// only by the standalone bench command, never through the UCI protocol.
inline constexpr TtTrustStage ACTIVE_TT_TRUST_STAGE = TtTrustStage::C;

#ifndef MAGNUS_SEARCH_ENABLE_ASPIRATION
#define MAGNUS_SEARCH_ENABLE_ASPIRATION 1
#endif
#ifndef MAGNUS_SEARCH_ENABLE_IIR
#define MAGNUS_SEARCH_ENABLE_IIR 1
#endif
#ifndef MAGNUS_SEARCH_ENABLE_RAZORING
#define MAGNUS_SEARCH_ENABLE_RAZORING 1
#endif
#ifndef MAGNUS_SEARCH_ENABLE_RFP
#define MAGNUS_SEARCH_ENABLE_RFP 1
#endif
#ifndef MAGNUS_SEARCH_ENABLE_NMP
#define MAGNUS_SEARCH_ENABLE_NMP 1
#endif
#ifndef MAGNUS_SEARCH_ENABLE_PROBCUT
#define MAGNUS_SEARCH_ENABLE_PROBCUT 1
#endif
#ifndef MAGNUS_SEARCH_ENABLE_SMALL_PROBCUT
#define MAGNUS_SEARCH_ENABLE_SMALL_PROBCUT 1
#endif
#ifndef MAGNUS_SEARCH_ENABLE_SEE_BAD_CAPTURE_GATE
#define MAGNUS_SEARCH_ENABLE_SEE_BAD_CAPTURE_GATE 1
#endif
#ifndef MAGNUS_SEARCH_ENABLE_CAPTURE_FUTILITY
#define MAGNUS_SEARCH_ENABLE_CAPTURE_FUTILITY 1
#endif
#ifndef MAGNUS_SEARCH_ENABLE_SEE_PRUNING
#define MAGNUS_SEARCH_ENABLE_SEE_PRUNING 1
#endif
#ifndef MAGNUS_SEARCH_ENABLE_QUIET_FUTILITY
#define MAGNUS_SEARCH_ENABLE_QUIET_FUTILITY 1
#endif
#ifndef MAGNUS_SEARCH_ENABLE_LMP
#define MAGNUS_SEARCH_ENABLE_LMP 1
#endif
#ifndef MAGNUS_SEARCH_ENABLE_HISTORY_PRUNING
#define MAGNUS_SEARCH_ENABLE_HISTORY_PRUNING 1
#endif
#ifndef MAGNUS_SEARCH_ENABLE_SINGULAR
#define MAGNUS_SEARCH_ENABLE_SINGULAR 1
#endif
#ifndef MAGNUS_SEARCH_ENABLE_LMR
#define MAGNUS_SEARCH_ENABLE_LMR 1
#endif
#ifndef MAGNUS_SEARCH_ENABLE_CORRECTION_HISTORY
#define MAGNUS_SEARCH_ENABLE_CORRECTION_HISTORY 1
#endif

struct SearchComponents {
    static constexpr bool aspiration = MAGNUS_SEARCH_ENABLE_ASPIRATION != 0;

    /*
    --------------------------------------------------
    Results of HaveAspiration vs NoAspiration (10+0.1, 1t, 128MB, UHO_4060_v4.epd):
    Elo: 61.29 +/- 31.79, nElo: 119.78 +/- 60.66
    LOS: 99.99 %, DrawRatio: 55.56 %, PairsRatio: 4.60
    Games: 126, Wins: 33, Losses: 11, Draws: 82, Points: 74.0 (58.73 %)
    Ptnml(0-2): [0, 5, 35, 19, 4], WL/DD Ratio: 0.21
    --------------------------------------------------
    */

    static constexpr bool iir = MAGNUS_SEARCH_ENABLE_IIR != 0;
    /*
    --------------------------------------------------
    Results of HaveIIR vs NoIIR (10+0.1, 1t, 128MB, UHO_4060_v4.epd):
    Elo: 28.08 +/- 22.27, nElo: 54.82 +/- 43.24
    LOS: 99.35 %, DrawRatio: 52.42 %, PairsRatio: 1.95
    Games: 248, Wins: 51, Losses: 31, Draws: 166, Points: 134.0 (54.03 %)
    Ptnml(0-2): [1, 19, 65, 37, 2], WL/DD Ratio: 0.18
    --------------------------------------------------
    */

    static constexpr bool razoring = MAGNUS_SEARCH_ENABLE_RAZORING != 0;
    /*
    --------------------------------------------------
    Results of HaveRazoring vs NoRazoring (10+0.1, 1t, 128MB, UHO_4060_v4.epd):
    Elo: 13.09 +/- 11.93, nElo: 25.30 +/- 23.01
    LOS: 98.44 %, DrawRatio: 56.16 %, PairsRatio: 1.43
    Games: 876, Wins: 157, Losses: 124, Draws: 595, Points: 454.5 (51.88 %)
    Ptnml(0-2): [8, 71, 246, 106, 7], WL/DD Ratio: 0.18
    --------------------------------------------------
    */

    static constexpr bool reverse_futility = MAGNUS_SEARCH_ENABLE_RFP != 0;
    /*
    --------------------------------------------------
    Results of HaveReverse_futility vs NoReverse_futility (10+0.1, 1t, 128MB, UHO_4060_v4.epd):
    Elo: 48.42 +/- 30.77, nElo: 95.46 +/- 59.72
    LOS: 99.91 %, DrawRatio: 60.00 %, PairsRatio: 3.33
    Games: 130, Wins: 29, Losses: 11, Draws: 90, Points: 74.0 (56.92 %)
    Ptnml(0-2): [0, 6, 39, 16, 4], WL/DD Ratio: 0.15
    --------------------------------------------------
    */

    static constexpr bool null_move = MAGNUS_SEARCH_ENABLE_NMP != 0;

    static constexpr bool probcut = MAGNUS_SEARCH_ENABLE_PROBCUT != 0;
    static constexpr bool small_probcut = MAGNUS_SEARCH_ENABLE_SMALL_PROBCUT != 0;

    /*
    --------------------------------------------------
    Results of HaveProbCut vs NoProbCut (10+0.1, 1t, 128MB, UHO_4060_v4.epd):
    Elo: 18.32 +/- 12.16, nElo: 33.74 +/- 22.33
    LOS: 99.85 %, DrawRatio: 52.90 %, PairsRatio: 1.58
    Games: 930, Wins: 169, Losses: 120, Draws: 641, Points: 489.5 (52.63 %)
    Ptnml(0-2): [10, 75, 246, 124, 10], WL/DD Ratio: 0.11
    --------------------------------------------------
    */

    static constexpr bool see_bad_capture_gate = MAGNUS_SEARCH_ENABLE_SEE_BAD_CAPTURE_GATE != 0;
    static constexpr bool capture_futility = MAGNUS_SEARCH_ENABLE_CAPTURE_FUTILITY != 0;
    static constexpr bool see_pruning = MAGNUS_SEARCH_ENABLE_SEE_PRUNING != 0;
    static constexpr bool quiet_futility = MAGNUS_SEARCH_ENABLE_QUIET_FUTILITY != 0;
    static constexpr bool late_move_pruning = MAGNUS_SEARCH_ENABLE_LMP != 0;
    static constexpr bool history_pruning = MAGNUS_SEARCH_ENABLE_HISTORY_PRUNING != 0;
    static constexpr bool singular_extension = MAGNUS_SEARCH_ENABLE_SINGULAR != 0;
    static constexpr bool lmr = MAGNUS_SEARCH_ENABLE_LMR != 0;
    static constexpr bool correction_history = MAGNUS_SEARCH_ENABLE_CORRECTION_HISTORY != 0;
};

/*
 * SearchLimits — collection of search-limit parameters
 *
 * Constructed by the bench command or the UCI "go" command, controlling the
 * termination conditions of the search.
 * Supports multiple limit types: depth, node count, soft/hard time limits,
 * infinite search, and ponder mode.
 *
 * Multi-threading support:
 *   shared_nodes  — node counter shared across threads (for Lazy SMP)
 *   stop / external_stop — cooperative stop signals (atomic booleans)
 *   pondering — ponder mode state; time spent on a successful ponder is kept
 *               on the same search clock
 *   thread_id / thread_count — thread identifier and total count
 */
struct SearchLimits {
    // --- Search depth & node limits ---
    int depth = MAX_SEARCH_DEPTH;       // maximum search depth (ply), default is unlimited
    u64 node_limit = 0;                 // maximum node count limit, 0 = no limit

    // --- Time control ---
    int soft_time_ms = 0;               // soft time limit (ms), used by time management
    int hard_time_ms = 0;               // hard time limit (ms), forces stop
    bool ponder = false;                // whether in ponder mode (search on opponent's time)
    bool infinite = false;              // whether in infinite search mode
    bool use_time_management = false;   // whether time management is enabled
    bool recover_ponder_pv = false;     // when pondering, fill second PV move via full-window search if needed
    int syzygy_probe_depth = 1;          // minimum depth at which to start probing given the max piece count
    int syzygy_probe_limit = 0;          // maximum Syzygy piece count, 0 = disabled
    bool syzygy_50_move_rule = true;     // whether to distinguish cursed win / blessed loss
    bool root_in_tb = false;             // root moves already ranked by Syzygy
    int root_tb_wdl = 0;                 // root WDL, range -2..2
    u64 root_tb_hits = 0;                // root probe hit count

    // --- Engine options ---
    int contempt = 0;                   // contempt value: positive = avoid draws, negative = accept draws
    bool full_pv = false;               // whether to fill short exact PV from TT chain in UCI info
    bool singular_telemetry = false;    // whether to collect singular extension contextual telemetry
    TtTrustStage tt_trust_stage = ACTIVE_TT_TRUST_STAGE;
    bool use_msv_smp = false;           // Search-local MSV-SMP root scheduling credit
    bool msv_info = false;              // Emit MSV-SMP debug info strings
    int multipv = 1;                    // Number of root principal variations to report
    SearchEvalKind eval_kind = SearchEvalKind::P2; // Active MNUE evaluator selected by UCI
    SearchComponents components{};      // Runtime search-component switches; defaults preserve normal search

    // --- Game history (for repetition detection) ---
    Key game_history_keys[MAX_GAME_HISTORY]{}; // Zobrist keys of historical positions
    int game_history_count = 0;                // number of recorded historical positions

    // --- Root move restriction (UCI searchmoves) ---
    Move root_moves[256]{};             // list of root moves to restrict search to
    int root_move_count = 0;            // number of restricted moves, 0 = all legal moves

    // --- Multi-thread synchronization ---
    std::atomic<bool>* stop = nullptr;                      // stop signal for this thread
    const std::atomic<bool>* external_stop = nullptr;       // external stop signal (cross-thread)
    const std::atomic<bool>* pondering = nullptr;           // ponder mode active flag
    std::atomic<u64>* shared_nodes = nullptr;               // shared node counter
    std::atomic<u64>* shared_tb_hits = nullptr;             // shared Syzygy hit counter
    RootMsvShared* root_msv = nullptr;                      // MSV-SMP root-local credit table
    SearchTimeSignals* time_signals = nullptr;              // Lazy-SMP dynamic time signals
    TimeManagementState* time_state = nullptr;              // cross-move time management history (main thread only)
    std::chrono::steady_clock::time_point start_time{};     // search start time after UCI receives "go"

    // --- Thread info ---
    int thread_id = 0;                  // this thread's ID (0 = main thread)
    int thread_count = 1;               // total number of threads
    bool report_info = true;            // whether to emit UCI info output (false for helper threads)
};

/*
 * SearchStackEntry — per-ply search state
 *
 * During recursive search, each ply must retain some state for use by later plies.
 * This state is passed between plies via the search_stack[] array.
 */
struct SearchStackEntry {
    Move current_move = 0;
    ContinuationHistoryContext continuation{};
    int static_eval = 0;
    int stat_score = 0;
    int reduction_fp = 0;
    int extension = 0;
    int move_count = 0;
    int cutoff_count = 0;
    bool in_check = false;
    bool tt_hit = false;
    bool tt_pv = false;
};

/*
 * SearchResult — search result
 *
 * Structure returned after iterative deepening completes, containing the best move,
 * score, the completed principal variation, and overall search statistics
 * (node count, depth, selective depth).
 */
struct SearchResult {
    Move best_move = 0;                 // best move found by the search (0 = no legal moves)
    Move pv[MAX_PLY]{};                 // principal variation at the completed depth; pv[0] should equal best_move
    Score score = 0;                    // score of the best move (cp units, from the root's perspective)
    u64 nodes = 0;                      // total number of nodes explored by the search
    u64 tb_hits = 0;                    // number of successful Syzygy probes
    int depth = 0;                      // completed search depth (ply)
    int seldepth = 0;                   // selective depth: deepest branch actually searched
    int pv_length = 0;                  // principal variation length
};

/*
 * move_to_uci — converts the internal 16-bit move format to UCI coordinate notation
 *
 * Parameters:
 *   m — internal Move format (16-bit, containing origin/destination/promotion info)
 *
 * Returns:
 *   A UCI string in the form "e2e4" or "e7e8q" (promotion)
 *   If m is a null move (0), returns "0000"
 */
[[nodiscard]] std::string move_to_uci(Move m);

/*
 * iterative_deepening — search entry point
 *
 * Launches a complete iterative-deepening search from the root position.
 * Automatically chooses between single-threaded or Lazy SMP parallel paths
 * based on thread_count.
 *
 * Parameters:
 *   root   — root position (const reference; a copy is made internally)
 *   mem    — shared memory (transposition table, pawn table, material table, attack table)
 *   limits — search limit parameters
 *   out    — optional output stream for UCI info output (nullptr = no output)
 *
 * Returns:
 *   SearchResult — best move, score, search statistics
 */
[[nodiscard]] SearchResult iterative_deepening(
    const Position& root,
    memory::Memory& mem,
    SearchSessionState& session_state,
    const SearchLimits& limits,
    std::ostream* out = nullptr
);

} // namespace magnus::search
