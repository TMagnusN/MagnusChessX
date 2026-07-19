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
 * 搜尋層公開介面 — MagnusChessX Thinking 的核心搜尋引擎。
 *
 * 本標頭對外暴露兩個入口點：
 *   1. iterative_deepening() — 從根節點啟動反覆加深 PVS 搜尋，供 bench 與 UCI 迴圈調用
 *   2. move_to_uci() — 將內部 16 位元著法轉換為 UCI 座標字串
 *
 * 內部實作採用經典的迭代加深 + 主變例搜尋 (PVS) 架構，
 * 搭配置換表導向排序、空步剪枝、奇異延伸、機率截斷、
 * 以及一組低開銷的啟發式剪枝技術。
 *
 * 所有搜尋狀態封裝在 Search.cpp 的匿名命名空間中，
 * 對外部完全透明。
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
 * 搜尋層全局常數：
 *   MAX_PLY            — search stack ply budget
 *   MAX_SEARCH_DEPTH   — maximum completed root depth reported by iterative deepening
 *   MAX_GAME_HISTORY   — 最大對局歷史記錄數，用於重複局面檢測
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
 * SearchLimits — 搜尋限制參數集合
 *
 * 由 bench 命令或 UCI "go" 命令建構，控制搜尋的終止條件。
 * 支援多種限制類型：深度、節點數、軟/硬時間限制、無限搜尋、沉思模式。
 *
 * 多線程支援：
 *   shared_nodes  — 跨線程共享的節點計數器（Lazy SMP 用）
 *   stop / external_stop — 合作式停止信號（原子布林）
 *   pondering — 沉思模式狀態；成功 ponder 的耗時保留在同一搜尋時鐘中
 *   thread_id / thread_count — 線程標識與總數
 */
struct SearchLimits {
    // --- 搜尋深度與節點限制 ---
    int depth = MAX_SEARCH_DEPTH;       // 最大搜索深度（ply），預設為無限
    u64 node_limit = 0;                 // 最大節點數限制，0 表示無限制

    // --- 時間控制 ---
    int soft_time_ms = 0;               // 軟時間限制（毫秒），用於時間管理
    int hard_time_ms = 0;               // 硬時間限制（毫秒），強制停止
    bool ponder = false;                // 是否為沉思模式（對手時間內搜尋）
    bool infinite = false;              // 是否為無限搜尋模式
    bool use_time_management = false;   // 是否啟用時間管理
    bool recover_ponder_pv = false;     // Ponder 開啟時，必要時 full-window 補主變例第二手
    int syzygy_probe_depth = 1;          // 同等最大子力數時開始探測的最小深度
    int syzygy_probe_limit = 0;          // 最大 Syzygy 子力數，0 = 停用
    bool syzygy_50_move_rule = true;     // 是否區分 cursed win / blessed loss
    bool root_in_tb = false;             // root 著法已由 Syzygy 排名
    int root_tb_wdl = 0;                 // root WDL，範圍 -2..2
    u64 root_tb_hits = 0;                // root 探測成功次數

    // --- 引擎選項 ---
    int contempt = 0;                   // 輕視值：正值傾向避免和棋，負值傾向接受和棋
    bool full_pv = false;               // UCI info 的短 exact PV 是否從 TT chain 補全
    bool singular_telemetry = false;    // 是否收集 singular extension contextual telemetry
    TtTrustStage tt_trust_stage = ACTIVE_TT_TRUST_STAGE;
    bool use_msv_smp = false;           // Search-local MSV-SMP root scheduling credit
    bool msv_info = false;              // Emit MSV-SMP debug info strings
    int multipv = 1;                    // Number of root principal variations to report
    SearchEvalKind eval_kind = SearchEvalKind::P2; // Active MNUE evaluator selected by UCI
    SearchComponents components{};      // Runtime search-component switches; defaults preserve normal search

    // --- 對局歷史（供重複局面檢測）---
    Key game_history_keys[MAX_GAME_HISTORY]{}; // 歷史局面的 Zobrist 鍵值
    int game_history_count = 0;                // 已記錄的歷史局面數量

    // --- 根節點著法限制（UCI searchmoves）---
    Move root_moves[256]{};             // 限制搜尋的根節點著法列表
    int root_move_count = 0;            // 限制的著法數量，0 = 全部合法著法

    // --- 多線程同步 ---
    std::atomic<bool>* stop = nullptr;                      // 本線程的停止信號
    const std::atomic<bool>* external_stop = nullptr;       // 外部停止信號（跨線程）
    const std::atomic<bool>* pondering = nullptr;           // 沉思模式啟用標誌
    std::atomic<u64>* shared_nodes = nullptr;               // 共享節點計數器
    std::atomic<u64>* shared_tb_hits = nullptr;             // 共享 Syzygy 命中計數器
    RootMsvShared* root_msv = nullptr;                      // MSV-SMP root-local credit table
    SearchTimeSignals* time_signals = nullptr;              // Lazy-SMP 動態時間訊號
    TimeManagementState* time_state = nullptr;              // 跨著時間管理歷史（僅主控提交）
    std::chrono::steady_clock::time_point start_time{};     // UCI 收到 go 後的搜尋起點

    // --- 線程資訊 ---
    int thread_id = 0;                  // 本線程的 ID（0 = 主線程）
    int thread_count = 1;               // 總線程數
    bool report_info = true;            // 是否輸出 UCI info 資訊（輔助線程設為 false）
};

/*
 * SearchStackEntry — 每個 ply 的搜尋狀態
 *
 * 在搜尋遞歸過程中，每個 ply 都需要保留一些狀態供後續 ply 使用。
 * 這些狀態透過 search_stack[] 陣列在 ply 之間傳遞。
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
 * SearchResult — 搜尋結果
 *
 * 反覆加深完成後回傳的結構，包含最佳著法、評分、
 * 完成的主變例，以及整體搜尋統計（節點數、深度、選擇性深度）。
 */
struct SearchResult {
    Move best_move = 0;                 // 搜尋找到的最佳著法（0 = 無合法著法）
    Move pv[MAX_PLY]{};                 // 完成深度的主變例；pv[0] 應等於 best_move
    Score score = 0;                    // 最佳著法的評分（cp 單位，從根節點視角）
    u64 nodes = 0;                      // 搜尋探索的總節點數
    u64 tb_hits = 0;                    // 成功的 Syzygy 探測次數
    int depth = 0;                      // 完成的搜索深度（ply）
    int seldepth = 0;                   // 選擇性深度：最深的分支實際搜索深度
    int pv_length = 0;                  // 主變例長度
};

/*
 * move_to_uci — 將內部 16 位元著法格式轉換為 UCI 座標表示法
 *
 * 參數：
 *   m — 內部 Move 格式（16 位元，包含起點/終點/升變資訊）
 *
 * 回傳：
 *   格式為 "e2e4" 或 "e7e8q"（升變）的 UCI 字串
 *   若 m 為空著法（0），回傳 "0000"
 */
[[nodiscard]] std::string move_to_uci(Move m);

/*
 * iterative_deepening — 搜尋入口點
 *
 * 從根節點位置啟動完整的反覆加深搜尋。
 * 根據 thread_count 自動選擇單線程或 Lazy SMP 並行路徑。
 *
 * 參數：
 *   root   — 根節點局面（const 參考，內部會複製）
 *   mem    — 共享記憶體（置換表、兵表、材料表、攻擊表）
 *   limits — 搜尋限制參數
 *   out    — 可選的輸出串流，用於 UCI info 輸出（nullptr = 不輸出）
 *
 * 回傳：
 *   SearchResult — 最佳著法、評分、搜尋統計
 */
[[nodiscard]] SearchResult iterative_deepening(
    const Position& root,
    memory::Memory& mem,
    SearchSessionState& session_state,
    const SearchLimits& limits,
    std::ostream* out = nullptr
);

} // namespace magnus::search
