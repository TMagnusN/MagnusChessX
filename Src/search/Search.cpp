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

#include "Search.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "History.h"
#include "Lmr.h"
#include "MovePicker.h"
#include "board/MoveGen.h"


#include "board/Position.h"
#include "Nmp.h"
#include "mnue/Mnue.h"
#include "mnue/MnueX2K16PawnNetwork.h"
#include "See.h"
#include "syzygy/Syzygy.h"

/*
 * MagnusChessX Thinking 搜尋引擎核心 — Search Engine Core
 *
 * 採用經典的迭代加深 + 主變例搜尋 (PVS) 架構，包含以下關鍵組件：
 *
 * 1. 反覆加深 (Iterative Deepening)
 *    - 從深度 1 逐步增加到最大深度
 *    - 每層使用 aspiration window 加速收斂
 *    - Stockfish 風格的時間管理決定何時停止
 *
 * 2. 主變例搜尋 (PVS — Principal Variation Search)
 *    - 第一個著法用完整窗口搜索
 *    - 後續著法用零窗口 (null window) 搜索
 *    - 零窗口 fail-high 時觸發完整窗口 re-search
 *
 * 3. 靜態搜索 (Quiescence Search)
 *    - 只搜索戰術著法（捕獲、升變）
 *    - stand-pat 原則：靜態評估已達 beta 即可截斷
 *    - delta pruning 跳過無法提升 alpha 的捕獲
 *
 * 4. 剪枝技術
 *    - 空步剪枝 (NMP) — 給對方免費回合測試截斷
 *    - 反向虛無剪枝 (RFP) — 靜態評估遠高於 beta 時提前截斷
 *    - 剃刀剪枝 (Razoring) — 淺層評估極差時跳至 qsearch
 *    - 機率截斷 (ProbCut) — 用淺層搜索預測深層截斷
 *    - 奇異延伸 (Singular Extension) — TT 著法為唯一好著時延伸
 *    - 延遲著法減免 (LMR) — 排序靠後的著法用縮減深度搜索
 *    - 歷史啟發式剪枝 — 歷史分數極差的安靜著法直接跳過
 *    - 捕獲 futility 剪枝 — 淺層捕獲不可能達到 alpha 時跳過
 *
 * 5. 並行搜索 (Lazy SMP)
 *    - 多線程各自獨立搜索，共享置換表
 *    - 加權投票選擇最佳著法
 *
 * 核心結構體 Searcher 封裝了單次迭代加深會話的所有可變狀態。
 */

namespace magnus::search {

namespace {

[[nodiscard]] std::size_t tt_move_trust_index(const Position& pos) noexcept {
    const Key pawn_key = pieces(pos, WHITE, PAWN)
        ^ std::rotl(pieces(pos, BLACK, PAWN), 7);
    return static_cast<std::size_t>(pawn_key) & (TT_MOVE_TRUST_SIZE - 1);
}

} // namespace

std::size_t tt_move_trust_bucket(int trust) noexcept {
    constexpr int upper[] = {-4096, -2048, -1024, 0, 1024, 2048, 4096};
    for (std::size_t i = 0; i < std::size(upper); ++i)
        if (trust < upper[i])
            return i;
    return TT_MOVE_TRUST_BUCKETS - 1;
}

bool stockfish_capture_stage(Move move) noexcept {
    return move_is_capture(move)
        || (move_is_promotion(move) && promo_piece(move) == QUEEN);
}

bool stockfish_is_shuffling(
    Move move,
    const Position& pos,
    Move move2,
    Move move4,
    int ply
) noexcept {
    // Frozen reference: Stockfish 18 commit
    // cb3d4ee9b47d0c5aae855b12379378ea1439675c.
    if (stockfish_capture_stage(move) || pos.halfmove_clock < 10)
        return false;
    if (pos.plies_from_null <= 6 || ply < 20)
        return false;
    return from_sq(move) == to_sq(move2)
        && from_sq(move2) == to_sq(move4);
}

int WorkerPersistentState::trust(const Position& pos) const noexcept {
    return tt_move_trust[tt_move_trust_index(pos)][pos.side_to_move];
}

void WorkerPersistentState::update_trust(
    const Position& pos,
    int bonus,
    std::size_t telemetry_bucket
) noexcept {
    i16& slot = tt_move_trust[tt_move_trust_index(pos)][pos.side_to_move];
    const int old = slot;
    const int bounded = std::clamp(bonus, -TT_MOVE_TRUST_LIMIT, TT_MOVE_TRUST_LIMIT);
    const int updated = std::clamp(
        old + bounded - old * std::abs(bounded) / TT_MOVE_TRUST_LIMIT,
        -TT_MOVE_TRUST_LIMIT,
        TT_MOVE_TRUST_LIMIT
    );
    slot = static_cast<i16>(updated);
    if (telemetry_bucket >= TT_MOVE_TRUST_BUCKETS)
        telemetry_bucket = tt_move_trust_bucket(old);
    TtTrustBucketCounters& c = telemetry.bucket[telemetry_bucket];
    ++c.updates;
    c.saturated += std::abs(updated) == TT_MOVE_TRUST_LIMIT;
}

void WorkerPersistentState::clear() noexcept {
    tt_move_trust = {};
    telemetry = {};
    game_epoch = 0;
    search_sequence = 0;
}

SearchSessionState::SearchSessionState() {
    ensure_workers(1);
}

void SearchSessionState::ensure_workers(std::size_t count) {
    count = std::max<std::size_t>(1, count);
    const std::size_t old_size = workers_.size();
    if (old_size < count) {
        workers_.resize(count);
        for (std::size_t i = old_size; i < count; ++i)
            workers_[i].game_epoch = game_epoch_;
    }
}

WorkerPersistentState& SearchSessionState::worker(std::size_t id) noexcept {
    return workers_[id];
}

void SearchSessionState::clear() noexcept {
    for (WorkerPersistentState& worker_state : workers_)
        worker_state.clear();
    game_epoch_ = 0;
    search_sequence_ = 0;
}

void SearchSessionState::new_game() noexcept {
    const u64 next_epoch = game_epoch_ + 1;
    clear();
    game_epoch_ = next_epoch;
    for (WorkerPersistentState& worker_state : workers_)
        worker_state.game_epoch = game_epoch_;
}

u64 SearchSessionState::begin_search(std::size_t active) {
    ensure_workers(active);
    ++search_sequence_;
    for (std::size_t i = 0; i < active; ++i) {
        workers_[i].game_epoch = game_epoch_;
        workers_[i].search_sequence = search_sequence_;
    }
    return search_sequence_;
}

struct RootMsvShared {
    struct Record {
        Move move = 0;
        RootMsvEntry entry{};
        int lastPriority = 0;
    };

    std::mutex mutex;
    std::vector<Record> records;
};

struct SearchTimeSignals {
    std::atomic<int> best_move_changes{0};
    std::atomic<bool> increase_depth{true};
};

namespace {

[[nodiscard]] constexpr bool is_valid_score(int score) noexcept {
    return score != VALUE_NONE;
}

[[nodiscard]] constexpr bool is_win(int score) noexcept {
    return is_valid_score(score) && score >= VALUE_TB_WIN_IN_MAX_PLY;
}

[[nodiscard]] constexpr bool is_loss(int score) noexcept {
    return is_valid_score(score) && score <= VALUE_TB_LOSS_IN_MAX_PLY;
}

[[nodiscard]] constexpr bool is_decisive(int score) noexcept {
    return is_win(score) || is_loss(score);
}

[[nodiscard]] constexpr bool is_mate_score(int score) noexcept {
    return is_valid_score(score)
        && (score >= VALUE_MATE_IN_MAX_PLY || score <= VALUE_MATED_IN_MAX_PLY);
}

[[nodiscard]] constexpr int score_to_tt(int score, int ply) noexcept {
    if (score >= VALUE_MATE_IN_MAX_PLY)
        return score + ply;
    if (score <= VALUE_MATED_IN_MAX_PLY)
        return score - ply;
    return score;
}

[[nodiscard]] constexpr int score_from_tt(
    int score,
    int ply,
    int halfmove_clock
) noexcept {
    if (!is_valid_score(score))
        return VALUE_NONE;

    const int remaining = std::max(0, 100 - halfmove_clock);
    if (score >= VALUE_MATE_IN_MAX_PLY) {
        if (VALUE_MATE - score > remaining)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;
        return score - ply;
    }

    if (score <= VALUE_MATED_IN_MAX_PLY) {
        if (VALUE_MATE + score > remaining)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;
        return score + ply;
    }

    return score;
}

static_assert(score_to_tt(123, 17) == 123);
static_assert(
    score_from_tt(score_to_tt(VALUE_MATE - 23, 7), 19, 0)
    == VALUE_MATE - 35
);
static_assert(score_to_tt(VALUE_TB - 11, 7) == VALUE_TB - 11);
static_assert(score_to_tt(-VALUE_TB + 11, 7) == -VALUE_TB + 11);
static_assert(!is_decisive(score_from_tt(VALUE_MATE - 5, 0, 96)));

#ifndef MAGNUS_SEARCH_OBS
#define MAGNUS_SEARCH_OBS 0
#endif

#ifndef MAGNUS_SEARCHSTATS_OBS
#define MAGNUS_SEARCHSTATS_OBS 0
#endif

#ifndef MAGNUS_LMR_OBS
#define MAGNUS_LMR_OBS 0
#endif

#ifndef MAGNUS_ENABLE_PROBCUT
#define MAGNUS_ENABLE_PROBCUT 1
#endif

#ifndef MAGNUS_ENABLE_SMALL_PROBCUT
#define MAGNUS_ENABLE_SMALL_PROBCUT 1
#endif

#ifndef MAGNUS_ENABLE_SINGULAR_EXTENSION
#define MAGNUS_ENABLE_SINGULAR_EXTENSION 1
#endif

#ifndef MAGNUS_SEE_TERM_PRESET
#define MAGNUS_SEE_TERM_PRESET 1
#endif

#if MAGNUS_SEE_TERM_PRESET == 0
constexpr SeeScalePreset SEE_TERM_PRESET = SeeScalePreset::Weak;
#elif MAGNUS_SEE_TERM_PRESET == 1
constexpr SeeScalePreset SEE_TERM_PRESET = SeeScalePreset::Medium;
#elif MAGNUS_SEE_TERM_PRESET == 2
constexpr SeeScalePreset SEE_TERM_PRESET = SeeScalePreset::Strong;
#else
#error "MAGNUS_SEE_TERM_PRESET must be 0 (Weak), 1 (Medium), or 2 (Strong)"
#endif

#ifndef MAGNUS_CAPTURE_OBS
#define MAGNUS_CAPTURE_OBS 0
#endif

#ifndef MAGNUS_SEE_LATE_BAD_CAPTURE_GATE
#define MAGNUS_SEE_LATE_BAD_CAPTURE_GATE 1
#endif

#ifndef MAGNUS_MOVEPICKER_OBS
#define MAGNUS_MOVEPICKER_OBS 0
#endif

struct RootLine {
    Move move = 0;
    int score = -VALUE_INF;
    int selection_score = -VALUE_INF;
    int previous_score = -VALUE_INF;
    memory::Bound bound = memory::BOUND_NONE;
    int depth = 0;
    int seldepth = 0;
    Move pv[MAX_PLY]{};
    int pv_length = 0;
    bool searched = false;
};

[[nodiscard]] inline bool has_root_line_score(int score) noexcept {
    return score != -VALUE_INF && score != VALUE_NONE;
}

[[nodiscard]] int root_line_index(
    const std::vector<RootLine>& lines,
    int first,
    Move move
) noexcept {
    if (move_is_none(move))
        return -1;

    const int begin = std::max(0, first);
    for (int i = begin; i < static_cast<int>(lines.size()); ++i)
        if (lines[static_cast<std::size_t>(i)].move == move)
            return i;

    return -1;
}

[[nodiscard]] inline bool root_line_less(
    const RootLine& lhs,
    const RootLine& rhs
) noexcept {
    if (lhs.searched != rhs.searched)
        return lhs.searched;
    const int lhs_order = has_root_line_score(lhs.selection_score)
        ? lhs.selection_score
        : lhs.score;
    const int rhs_order = has_root_line_score(rhs.selection_score)
        ? rhs.selection_score
        : rhs.score;
    if (lhs_order != rhs_order)
        return lhs_order > rhs_order;
    if (lhs.previous_score != rhs.previous_score)
        return lhs.previous_score > rhs.previous_score;
    return false;
}

void stable_sort_root_lines(
    std::vector<RootLine>& lines,
    int first,
    int last
) noexcept {
    const int begin = std::clamp(first, 0, static_cast<int>(lines.size()));
    const int end = std::clamp(last, begin, static_cast<int>(lines.size()));
    for (int i = begin + 1; i < end; ++i) {
        RootLine value = lines[static_cast<std::size_t>(i)];
        int j = i;
        while (j > begin &&
               root_line_less(value, lines[static_cast<std::size_t>(j - 1)])) {
            lines[static_cast<std::size_t>(j)] =
                lines[static_cast<std::size_t>(j - 1)];
            --j;
        }
        lines[static_cast<std::size_t>(j)] = value;
    }
}

void stable_sort_root_lines(
    std::vector<RootLine>& lines,
    int first
) noexcept {
    stable_sort_root_lines(lines, first, static_cast<int>(lines.size()));
}

[[nodiscard]] static inline int mvv_lva_capture_term(
    const Position& pos,
    Move move
) noexcept {
    const PieceType attacker = piece_type_on(pos, from_sq(move));
    const PieceType victim = move_is_ep(move) ? PAWN : piece_type_on(pos, to_sq(move));
    const int attacker_value = is_ok(attacker) ? piece_order_value[attacker] : 0;
    const int victim_value = is_ok(victim) ? piece_order_value[victim] : 0;
    return victim_value * 32 - attacker_value;
}

[[nodiscard]] inline memory::Bound score_bound_from_window(
    int score,
    int alpha,
    int beta
) noexcept {
    if (score <= alpha)
        return memory::BOUND_UPPER;
    if (score >= beta)
        return memory::BOUND_LOWER;
    return memory::BOUND_EXACT;
}

constexpr Move STARTPOS_E2E4_MOVE = make_move(12, 28, MOVE_DOUBLE_PUSH);
constexpr int STARTPOS_E2E4_ROOT_ORDER_SCORE = 31'000'000;
constexpr int STARTPOS_E2E4_ROOT_BONUS = 64;

[[nodiscard]] bool is_classical_startpos(const Position& pos) noexcept {
    return pos.side_to_move == WHITE &&
           pos.ep_sq == NO_SQ &&
           pos.castling_rights == ANY_CASTLING &&
           pos.halfmove_clock == 0 &&
           pos.fullmove_number == 1 &&
           pos.occupied == 0xFFFF00000000FFFFULL &&
           pieces(pos, WHITE, PAWN) == 0x000000000000FF00ULL &&
           pieces(pos, WHITE, KNIGHT) == 0x0000000000000042ULL &&
           pieces(pos, WHITE, BISHOP) == 0x0000000000000024ULL &&
           pieces(pos, WHITE, ROOK) == 0x0000000000000081ULL &&
           pieces(pos, WHITE, QUEEN) == 0x0000000000000008ULL &&
           pieces(pos, WHITE, KING) == 0x0000000000000010ULL &&
           pieces(pos, BLACK, PAWN) == 0x00FF000000000000ULL &&
           pieces(pos, BLACK, KNIGHT) == 0x4200000000000000ULL &&
           pieces(pos, BLACK, BISHOP) == 0x2400000000000000ULL &&
           pieces(pos, BLACK, ROOK) == 0x8100000000000000ULL &&
           pieces(pos, BLACK, QUEEN) == 0x0800000000000000ULL &&
           pieces(pos, BLACK, KING) == 0x1000000000000000ULL;
}

[[nodiscard]] int root_opening_preference_bonus(
    const SearchLimits& limits,
    const Position& root,
    Move move
) noexcept {
    if (limits.root_move_count != 0 ||
        move != STARTPOS_E2E4_MOVE ||
        !is_classical_startpos(root)) {
        return 0;
    }

    return STARTPOS_E2E4_ROOT_BONUS;
}

void promote_startpos_e4_root_line(
    const SearchLimits& limits,
    const Position& root,
    std::vector<RootLine>& lines,
    int first,
    int last
) noexcept {
    if (root_opening_preference_bonus(limits, root, STARTPOS_E2E4_MOVE) == 0)
        return;

    const int begin = std::clamp(first, 0, static_cast<int>(lines.size()));
    const int end = std::clamp(last, begin, static_cast<int>(lines.size()));
    const int preferred_index = root_line_index(
        lines,
        begin,
        STARTPOS_E2E4_MOVE
    );
    if (preferred_index > begin && preferred_index < end) {
        std::swap(
            lines[static_cast<std::size_t>(begin)],
            lines[static_cast<std::size_t>(preferred_index)]
        );
    }
}

inline void append_uci_score_bound(
    std::ostream& out,
    memory::Bound bound
) {
    if (bound == memory::BOUND_UPPER)
        out << " upperbound";
    else if (bound == memory::BOUND_LOWER)
        out << " lowerbound";
}

inline void append_uci_score(
    std::ostream& out,
    int score,
    const Position& root,
    memory::Bound bound
) {
    if (score >= VALUE_MATE - MAX_PLY) {
        const int plies_to_mate = VALUE_MATE - score;
        out << "score mate " << ((plies_to_mate + 1) / 2);
        append_uci_score_bound(out, bound);
        return;
    }

    if (score <= -VALUE_MATE + MAX_PLY) {
        const int plies_to_mate = VALUE_MATE + score;
        out << "score mate -" << ((plies_to_mate + 1) / 2);
        append_uci_score_bound(out, bound);
        return;
    }

    if (std::abs(score) >= VALUE_TB - MAX_PLY &&
        std::abs(score) <= VALUE_TB) {
        constexpr int TB_CP = 20000;
        const int distance = VALUE_TB - std::abs(score);
        const int display_score =
            score > 0 ? TB_CP - distance : -TB_CP + distance;
        out << "score cp " << display_score;
        append_uci_score_bound(out, bound);
        out << " wdl "
            << (score > 0 ? "1000 0 0" : "0 0 1000");
        return;
    }

    const int display_score = mnue::search_score_to_cp(score, root);
    out << "score cp " << display_score;
    append_uci_score_bound(out, bound);

    const mnue::WdlTriplet wdl = mnue::search_score_to_wdl(score, root);
    out << " wdl " << wdl.win << ' ' << wdl.draw << ' ' << wdl.loss;
}

[[nodiscard]] inline bool root_msv_enabled(
    const SearchLimits& limits
) noexcept {
    return limits.use_msv_smp &&
           limits.thread_count > 1 &&
           limits.multipv <= 1 &&
           limits.root_msv != nullptr;
}

[[nodiscard]] RootMsvShared::Record* root_msv_find_locked(
    RootMsvShared& shared,
    Move move
) noexcept {
    for (RootMsvShared::Record& record : shared.records)
        if (record.move == move)
            return &record;
    return nullptr;
}

void root_msv_seed_moves(
    const Position& root,
    const memory::Memory& mem,
    const SearchLimits& limits,
    RootMsvShared& shared
) {
    MoveList list{};
    generate_legal(root, mem, list);

    if (limits.root_move_count > 0) {
        MoveList filtered{};
        for (int i = 0; i < list.size; ++i) {
            const Move move = list.moves[i];
            bool allowed = false;
            for (int j = 0; j < limits.root_move_count; ++j) {
                if (limits.root_moves[j] == move) {
                    allowed = true;
                    break;
                }
            }
            if (allowed)
                filtered.moves[filtered.size++] = move;
        }
        list = filtered;
    }

    std::lock_guard<std::mutex> lock(shared.mutex);
    shared.records.clear();
    shared.records.reserve(static_cast<std::size_t>(list.size));
    for (int i = 0; i < list.size; ++i) {
        const Move move = list.moves[i];
        const auto duplicate = std::find_if(
            shared.records.begin(),
            shared.records.end(),
            [move](const RootMsvShared::Record& record) noexcept {
                return record.move == move;
            }
        );
        if (duplicate == shared.records.end())
            shared.records.push_back({move, RootMsvEntry{}, 0});
    }
}

[[nodiscard]] RootMsvEntry root_msv_snapshot(
    const SearchLimits& limits,
    Move move
) {
    if (!root_msv_enabled(limits))
        return RootMsvEntry{};

    RootMsvShared& shared = *limits.root_msv;
    std::lock_guard<std::mutex> lock(shared.mutex);
    const RootMsvShared::Record* record = root_msv_find_locked(shared, move);
    return record != nullptr ? record->entry : RootMsvEntry{};
}

void root_msv_set_last_priority(
    const SearchLimits& limits,
    Move move,
    int priority
) {
    if (!root_msv_enabled(limits))
        return;

    RootMsvShared& shared = *limits.root_msv;
    std::lock_guard<std::mutex> lock(shared.mutex);
    RootMsvShared::Record* record = root_msv_find_locked(shared, move);
    if (record != nullptr)
        record->lastPriority = priority;
}

[[nodiscard]] bool root_msv_has_signal(
    const SearchLimits& limits
) {
    if (!root_msv_enabled(limits))
        return false;

    RootMsvShared& shared = *limits.root_msv;
    std::lock_guard<std::mutex> lock(shared.mutex);
    for (const RootMsvShared::Record& record : shared.records) {
        const RootMsvEntry& entry = record.entry;
        if (entry.credit > 0)
            return true;
    }
    return false;
}

void root_msv_adjust_active(
    const SearchLimits& limits,
    Move move,
    int delta
) {
    if (!root_msv_enabled(limits) || move_is_none(move))
        return;

    RootMsvShared& shared = *limits.root_msv;
    std::lock_guard<std::mutex> lock(shared.mutex);
    RootMsvShared::Record* record = root_msv_find_locked(shared, move);
    if (record == nullptr)
        return;

    RootMsvEntry& entry = record->entry;
    entry.activeWorkers = std::max(0, entry.activeWorkers + delta);
}

struct RootMsvActiveGuard {
    const SearchLimits& limits;
    Move move = 0;
    bool active = false;

    RootMsvActiveGuard(const SearchLimits& l, Move m)
        : limits(l), move(m), active(root_msv_enabled(l) && !move_is_none(m)) {
        if (active)
            root_msv_adjust_active(limits, move, 1);
    }

    ~RootMsvActiveGuard() {
        if (active)
            root_msv_adjust_active(limits, move, -1);
    }

    RootMsvActiveGuard(const RootMsvActiveGuard&) = delete;
    RootMsvActiveGuard& operator=(const RootMsvActiveGuard&) = delete;
};

[[nodiscard]] inline int root_msv_bad_bound_penalty(
    memory::Bound bound
) noexcept {
    if (bound == memory::BOUND_UPPER)
        return 96;
    if (bound == memory::BOUND_LOWER)
        return 48;
    return 0;
}

[[nodiscard]] inline int root_msv_priority_from_entry(
    int score_part,
    int history_prior,
    int depth,
    const RootMsvEntry& entry
) noexcept {
    const int hit_count = entry.exactHits + entry.boundHits;
    const int credit_part = entry.credit * 2;
    const int stable_bonus = std::min(entry.stableCount, 16) * 12;
    const int uncertainty_bonus = hit_count > 0
        ? std::clamp(depth - entry.maxDepth, 0, 8) * 6
        : 0;
    const int active_worker_penalty = entry.activeWorkers * 96;
    const int bad_bound_penalty = root_msv_bad_bound_penalty(entry.lastBound);

    return score_part
         + history_prior
         + credit_part
         + stable_bonus
         + uncertainty_bonus
         - active_worker_penalty
         - bad_bound_penalty;
}

void root_msv_record_completed(
    const SearchLimits& limits,
    Move move,
    int depth,
    int seldepth,
    int score,
    memory::Bound bound,
    bool stopped,
    bool pv_valid,
    int pv_length
) {
    if (!root_msv_enabled(limits) || move_is_none(move))
        return;

    if (stopped || !pv_valid || pv_length <= 2)
        return;

    RootMsvShared& shared = *limits.root_msv;
    std::lock_guard<std::mutex> lock(shared.mutex);

    for (RootMsvShared::Record& record : shared.records)
        record.entry.credit = (record.entry.credit * 7) / 8;

    RootMsvShared::Record* record = root_msv_find_locked(shared, move);
    if (record == nullptr)
        return;

    RootMsvEntry& entry = record->entry;
    const int previous_max_depth = entry.maxDepth;
    const int previous_score = entry.lastScore;
    const bool had_valid_pv = entry.pvValid;

    entry.stopped = false;
    entry.pvValid = pv_valid;
    entry.pvLength = pv_length;
    entry.lastScore = score;
    entry.lastBound = bound;
    entry.maxDepth = std::max(entry.maxDepth, depth);
    entry.maxSelDepth = std::max(entry.maxSelDepth, seldepth);

    if (bound == memory::BOUND_EXACT)
        ++entry.exactHits;
    else if (bound == memory::BOUND_LOWER || bound == memory::BOUND_UPPER)
        ++entry.boundHits;

    bool new_worker = false;
    if (limits.thread_id >= 0 && limits.thread_id < 64) {
        const std::uint64_t worker_bit = 1ULL << limits.thread_id;
        new_worker = (entry.workerMask & worker_bit) == 0;
        entry.workerMask |= worker_bit;
        if (new_worker)
            ++entry.workerHits;
    }

    const bool score_stable =
        had_valid_pv && std::abs(score - previous_score) <= 32;
    const bool depth_stable =
        previous_max_depth > 0 && std::abs(depth - previous_max_depth) <= 2;
    if (score_stable || depth_stable)
        entry.stableCount = std::min(entry.stableCount + 1, 64);
    else
        entry.stableCount = std::max(1, (entry.stableCount * 3) / 4);

    int bonus = 8; // completed search
    if (bound == memory::BOUND_EXACT)
        bonus += 48;
    else if (bound == memory::BOUND_LOWER || bound == memory::BOUND_UPPER)
        bonus += 12;
    else
        bonus = 0;

    if (pv_length >= 4)
        bonus += 16;

    if (previous_max_depth > 0) {
        const int depth_gap = std::abs(depth - previous_max_depth);
        if (depth_gap <= 4)
            bonus += 24;
        else if (depth_gap <= 8)
            bonus += 12;
        else if (depth_gap <= 16)
            bonus += 6;
    }

    if (new_worker)
        bonus += 16;

    if (bound != memory::BOUND_EXACT)
        bonus /= 2;

    if (bonus > 0)
        entry.credit = std::clamp(entry.credit + bonus, 0, 4096);
}

/*
Searcher owns the mutable state for one iterative-deepening session: node
counting, killer/history tables, PV storage, and stop-condition bookkeeping.
*/
struct Searcher {
    struct CorrectionKeys {
        std::size_t position = 0;
        std::size_t pawn_king = 0;
        std::size_t material = 0;
        std::size_t nonpawn[COLOR_NB]{};
        std::size_t major = 0;
        std::size_t minor = 0;
        std::size_t counter = 0;
        std::size_t followup = 0;
        bool has_counter = false;
        bool has_followup = false;
    };

    static constexpr int CORRECTION_MOVE_HISTORY_SIZE =
        PIECE_TYPE_NB * SQ_NB * SQ_NB;

    struct CorrectionHistoryTables {
        i16 position[COLOR_NB][CORRECTION_HISTORY_SIZE]{};
        i16 pawn[COLOR_NB][CORRECTION_HISTORY_SIZE]{};
        i16 material[COLOR_NB][CORRECTION_HISTORY_SIZE]{};
        i16 nonpawn[COLOR_NB][COLOR_NB][CORRECTION_HISTORY_SIZE]{};
        i16 major[COLOR_NB][CORRECTION_HISTORY_SIZE]{};
        i16 minor[COLOR_NB][CORRECTION_HISTORY_SIZE]{};
        i16 counter[COLOR_NB][CORRECTION_MOVE_HISTORY_SIZE]{};
        i16 followup[COLOR_NB][CORRECTION_MOVE_HISTORY_SIZE]{};
    };

    struct StaticEvalInfo {
        CorrectionKeys keys{};
        int raw = VALUE_NONE;
        int base = 0;
        int search = 0;
        int stand_pat = 0;
    };

#if MAGNUS_SEARCHSTATS_OBS
    struct SearchStats {
        u64 root_moves_searched = 0;
        u64 root_pvs_researches = 0;
        u64 pvs_full_researches = 0;
        u64 pvs_full_pv_node = 0;
        u64 pvs_full_non_pv_node = 0;
        u64 lmr_researches = 0;
        u64 lmr_research_quiet = 0;
        u64 lmr_research_capture = 0;
        u64 lmr_research_check = 0;
        u64 lmr_research_r1 = 0;
        u64 lmr_research_r2 = 0;
        u64 lmr_research_r3_plus = 0;
        std::array<u64, MAX_PLY> pvs_full_by_ply{};
        std::array<u64, MAX_PLY> lmr_by_ply{};
    };
#endif

    using clock = std::chrono::steady_clock;

    memory::Memory& mem;
    WorkerPersistentState& persistent;
    const SearchLimits& limits;

    u64 nodes = 0;
    u64 base_nodes = 0;
    u64 published_nodes = 0;
    u64 tb_hits = 0;
    u64 published_tb_hits = 0;
    int seldepth = 0;
    int completed_depth = 0;
    int root_depth = 0;
    int root_side_to_move = WHITE;
    HistoryTables history_tables{};
    mutable mnue::P2AccumulatorStack p2_accumulator_stack{};
    Move pv_table[MAX_PLY][MAX_PLY]{};
    int pv_length[MAX_PLY + 1]{};
    Key rep_keys[MAX_PLY + 1]{};
    Move move_stack[MAX_PLY]{};
    SearchStackEntry search_stack[MAX_PLY + 2]{};
    int static_eval_stack[MAX_PLY + 1]{};
    bool static_eval_valid[MAX_PLY + 1]{};
    CorrectionHistoryTables correction_history{};
    clock::time_point start_time{};
    u64 limit_poll_mask = 1023ULL;
    bool stopped = false;
    bool hard_stop = false;
    bool stop_on_ponderhit = false;  // time expired during ponder, stop on next ponderhit
    int nmp_min_ply = 0;
    u64 singular_verification_nodes = 0;

    struct RootEffortEntry {
        Move move = 0;
        u64 effort = 0;
        int average_score = VALUE_NONE;
    };

    std::array<RootEffortEntry, 256> root_effort_entries{};
    int root_effort_count = 0;

    struct SingularExtensionOutcome {
        u64 searched = 0;
        u64 alpha_raise = 0;
        u64 cutoff = 0;
        u64 fail_low = 0;
        u64 nodes = 0;
    };

    struct SingularTelemetryBucket {
        u64 candidates = 0;
        u64 tested = 0;
        u64 skipped_trust = 0;
        u64 skipped_cost = 0;
        u64 skipped_path = 0;
        u64 skipped_proximity = 0;
        u64 lower_bound = 0;
        u64 exact_bound = 0;
        u64 tested_lower = 0;
        u64 tested_exact = 0;
        u64 extended_lower = 0;
        u64 extended_exact = 0;
        u64 multi_cut_lower = 0;
        u64 multi_cut_exact = 0;
        u64 tactical_move = 0;
        u64 good_history = 0;
        u64 exclusion_fail_low = 0;
        u64 exclusion_fail_high = 0;
        u64 exclusion_nodes = 0;
        u64 extended = 0;
        u64 double_extended = 0;
        u64 triple_extended = 0;
        u64 multi_cut = 0;
        u64 negative_extended = 0;
        std::array<SingularExtensionOutcome, SINGULAR_TELEMETRY_EXTENSION_LEVELS>
            outcomes{};
    };

    std::array<SingularTelemetryBucket, SINGULAR_TELEMETRY_BUCKETS>
        singular_telemetry_buckets{};
#if MAGNUS_SEARCHSTATS_OBS
    SearchStats stats{};
#endif

#if MAGNUS_LMR_OBS
    struct LmrObservation {
        u64 considered = 0;
        u64 eligible = 0;
        u64 reduced = 0;
        u64 fail_high = 0;
        u64 researches = 0;
        u64 full_confirms = 0;
        u64 quiet = 0;
        u64 capture = 0;
        u64 pv = 0;
        u64 non_pv = 0;
        u64 r1 = 0;
        u64 r2 = 0;
        u64 r3_plus = 0;
    } lmr_obs{};
#endif

#if MAGNUS_CAPTURE_OBS
    struct CaptureObservation {
        u64 main_capture_searches = 0;
        u64 main_capture_cutoffs = 0;
        u64 q_capture_searches = 0;
        u64 q_capture_cutoffs = 0;

        u64 topk_tried[CAPTURE_TOPK]{};
        u64 topk_cutoff[CAPTURE_TOPK]{};

        u64 main_see_tried[3]{};
        u64 main_see_cutoff[3]{};
        u64 q_see_tried[3]{};
        u64 q_see_cutoff[3]{};

        u64 main_high_hist_tried = 0;
        u64 main_high_hist_cutoff = 0;
        u64 q_high_hist_tried = 0;
        u64 q_high_hist_cutoff = 0;

        i64 topk_rank_shift_sum[CAPTURE_TOPK]{};
        u64 topk_rank_shift_count[CAPTURE_TOPK]{};

        u64 gate_checks = 0;
        u64 gate_bad_see = 0;
        u64 gate_pruned = 0;
        u64 gate_exempt_check = 0;
        u64 gate_exempt_recapture = 0;

        u64 cap_lmr_late_simple_total = 0;
        u64 cap_lmr_eligible = 0;
        u64 cap_lmr_considered = 0;
        u64 cap_lmr_reduced = 0;
        u64 cap_lmr_reduced_r1 = 0;
        u64 cap_lmr_reduced_r2 = 0;
        u64 cap_lmr_research = 0;
        u64 cap_lmr_research_r1 = 0;
        u64 cap_lmr_research_r2 = 0;
        u64 cap_lmr_considered_see[3]{};
        u64 cap_lmr_reduced_see[3]{};
        u64 cap_lmr_research_see[3]{};
    } cap_obs{};
#endif

#if MAGNUS_MOVEPICKER_OBS
    enum class MoveStageBucket : std::uint8_t {
        TT = 0,
        GoodCapture,
        Killer,
        Quiet,
        BadCapture,
        Count
    };

    struct MovePickerObservation {
        u64 nodes = 0;
        u64 nodes_with_tt_probe = 0;
        u64 tt_first_try = 0;
        u64 tt_first_cutoff = 0;

        u64 cutoffs_total = 0;
        u64 cutoff_by_stage[static_cast<int>(MoveStageBucket::Count)]{};

        u64 first_good_capture_try = 0;
        u64 first_good_capture_cutoff = 0;
        u64 first_killer_try = 0;
        u64 first_killer_cutoff = 0;
        u64 first_quiet_try = 0;
        u64 first_quiet_cutoff = 0;

        u64 quiet_generated = 0;
        u64 quiet_scored = 0;
        u64 quiet_skipped_by_mp = 0;
        u64 quiet_searched = 0;
        u64 late_quiet_fail_high = 0;
        u64 quiet_fail_high_after_skip_band = 0;
    } mp_obs{};
#endif

#if MAGNUS_SEARCH_OBS
    struct SearchObservation {
        u64 nmp_candidates = 0;
        u64 nmp_tried = 0;
        u64 nmp_fail_high = 0;
        u64 nmp_verification_tried = 0;
        u64 nmp_verified_cutoffs = 0;
        u64 nmp_verification_failed = 0;
        u64 probcut_nodes = 0;
        u64 probcut_moves = 0;
        u64 probcut_cutoffs = 0;
        u64 singular_candidates = 0;
        u64 singular_extend1 = 0;
        u64 singular_extend2 = 0;
        u64 singular_extend3 = 0;
    } search_obs{};
#endif

    explicit Searcher(
        memory::Memory& m,
        WorkerPersistentState& persistent_state,
        const SearchLimits& l
    ) noexcept
        : mem(m),
          persistent(persistent_state),
          limits(l),
          limit_poll_mask(
              l.hard_time_ms > 0 && l.hard_time_ms <= 100
                  ? 63ULL
                  : (l.hard_time_ms > 0 && l.hard_time_ms <= 1000
                         ? 255ULL
                         : 1023ULL)
          ) {}

#if MAGNUS_CAPTURE_OBS
    [[nodiscard]] static inline int see_bucket(int see_value) noexcept {
        if (see_value < 0) return 0;
        if (see_value == 0) return 1;
        return 2;
    }

    [[nodiscard]] static inline int ratio_percent(u64 num, u64 den) noexcept {
        if (den == 0) return 0;
        return static_cast<int>((num * 100ULL) / den);
    }

    [[nodiscard]] static inline int lookup_capture_base_rank(
        Move move,
        const Move* cap_moves,
        const int* cap_ranks,
        int cap_count
    ) noexcept {
        for (int i = 0; i < cap_count; ++i) {
            if (cap_moves[i] == move)
                return cap_ranks[i];
        }
        return 0;
    }

    inline void record_main_capture_try(
        int capture_order,
        int see_value,
        int capture_hist,
        int base_rank
    ) noexcept {
        ++cap_obs.main_capture_searches;
        const int bucket = see_bucket(see_value);
        ++cap_obs.main_see_tried[bucket];
        if (capture_hist >= CAPTURE_HISTORY_HIGH_THRESHOLD)
            ++cap_obs.main_high_hist_tried;

        if (capture_order >= 0 && capture_order < CAPTURE_TOPK) {
            ++cap_obs.topk_tried[capture_order];
            if (base_rank > 0) {
                cap_obs.topk_rank_shift_sum[capture_order] += static_cast<i64>(base_rank - (capture_order + 1));
                ++cap_obs.topk_rank_shift_count[capture_order];
            }
        }
    }

    inline void record_main_capture_cutoff(
        int capture_order,
        int see_value,
        int capture_hist
    ) noexcept {
        ++cap_obs.main_capture_cutoffs;
        const int bucket = see_bucket(see_value);
        ++cap_obs.main_see_cutoff[bucket];
        if (capture_hist >= CAPTURE_HISTORY_HIGH_THRESHOLD)
            ++cap_obs.main_high_hist_cutoff;
        if (capture_order >= 0 && capture_order < CAPTURE_TOPK)
            ++cap_obs.topk_cutoff[capture_order];
    }

    inline void record_q_capture_try(
        int see_value,
        int capture_hist
    ) noexcept {
        ++cap_obs.q_capture_searches;
        const int bucket = see_bucket(see_value);
        ++cap_obs.q_see_tried[bucket];
        if (capture_hist >= CAPTURE_HISTORY_HIGH_THRESHOLD)
            ++cap_obs.q_high_hist_tried;
    }

    inline void record_q_capture_cutoff(
        int see_value,
        int capture_hist
    ) noexcept {
        ++cap_obs.q_capture_cutoffs;
        const int bucket = see_bucket(see_value);
        ++cap_obs.q_see_cutoff[bucket];
        if (capture_hist >= CAPTURE_HISTORY_HIGH_THRESHOLD)
            ++cap_obs.q_high_hist_cutoff;
    }

    inline void record_gate_check(
        bool bad_see,
        bool pruned,
        bool exempt_check,
        bool exempt_recapture
    ) noexcept {
        ++cap_obs.gate_checks;
        if (bad_see)
            ++cap_obs.gate_bad_see;
        if (pruned)
            ++cap_obs.gate_pruned;
        if (exempt_check)
            ++cap_obs.gate_exempt_check;
        if (exempt_recapture)
            ++cap_obs.gate_exempt_recapture;
    }

    inline void record_cap_lmr_considered(
        int see_value,
        int reduction
    ) noexcept {
        ++cap_obs.cap_lmr_considered;
        const int bucket = see_bucket(see_value);
        ++cap_obs.cap_lmr_considered_see[bucket];

        if (reduction <= 0)
            return;

        ++cap_obs.cap_lmr_reduced;
        ++cap_obs.cap_lmr_reduced_see[bucket];
        if (reduction == 1)
            ++cap_obs.cap_lmr_reduced_r1;
        else
            ++cap_obs.cap_lmr_reduced_r2;
    }

    inline void record_cap_lmr_research(
        int see_value,
        int reduction
    ) noexcept {
        if (reduction <= 0)
            return;

        ++cap_obs.cap_lmr_research;
        const int bucket = see_bucket(see_value);
        ++cap_obs.cap_lmr_research_see[bucket];
        if (reduction == 1)
            ++cap_obs.cap_lmr_research_r1;
        else
            ++cap_obs.cap_lmr_research_r2;
    }

    inline void emit_capture_observation(std::ostream& out) const {
        const auto sb_raw = [&](DepthClass dc, SeeClass sc) -> int {
            return static_cast<int>(
                history_tables.see_bias.value[static_cast<int>(dc)][static_cast<int>(sc)]
            );
        };
        const auto sb_term = [](int raw) -> int {
            return std::clamp(raw / 4, -96, 96);
        };

        out << "info string capobs main_capture "
            << cap_obs.main_capture_cutoffs << '/' << cap_obs.main_capture_searches
            << " (" << ratio_percent(cap_obs.main_capture_cutoffs, cap_obs.main_capture_searches) << "%)"
            << " q_capture "
            << cap_obs.q_capture_cutoffs << '/' << cap_obs.q_capture_searches
            << " (" << ratio_percent(cap_obs.q_capture_cutoffs, cap_obs.q_capture_searches) << "%)\n";

        out << "info string capobs topk "
            << "k1 " << cap_obs.topk_cutoff[0] << '/' << cap_obs.topk_tried[0]
            << " (" << ratio_percent(cap_obs.topk_cutoff[0], cap_obs.topk_tried[0]) << "%) "
            << "k2 " << cap_obs.topk_cutoff[1] << '/' << cap_obs.topk_tried[1]
            << " (" << ratio_percent(cap_obs.topk_cutoff[1], cap_obs.topk_tried[1]) << "%) "
            << "k3 " << cap_obs.topk_cutoff[2] << '/' << cap_obs.topk_tried[2]
            << " (" << ratio_percent(cap_obs.topk_cutoff[2], cap_obs.topk_tried[2]) << "%)\n";

        out << "info string capobs see_cutoff_rate "
            << "main_bad " << cap_obs.main_see_cutoff[0] << '/' << cap_obs.main_see_tried[0]
            << " (" << ratio_percent(cap_obs.main_see_cutoff[0], cap_obs.main_see_tried[0]) << "%) "
            << "main_eq " << cap_obs.main_see_cutoff[1] << '/' << cap_obs.main_see_tried[1]
            << " (" << ratio_percent(cap_obs.main_see_cutoff[1], cap_obs.main_see_tried[1]) << "%) "
            << "main_good " << cap_obs.main_see_cutoff[2] << '/' << cap_obs.main_see_tried[2]
            << " (" << ratio_percent(cap_obs.main_see_cutoff[2], cap_obs.main_see_tried[2]) << "%) "
            << "q_bad " << cap_obs.q_see_cutoff[0] << '/' << cap_obs.q_see_tried[0]
            << " (" << ratio_percent(cap_obs.q_see_cutoff[0], cap_obs.q_see_tried[0]) << "%) "
            << "q_eq " << cap_obs.q_see_cutoff[1] << '/' << cap_obs.q_see_tried[1]
            << " (" << ratio_percent(cap_obs.q_see_cutoff[1], cap_obs.q_see_tried[1]) << "%) "
            << "q_good " << cap_obs.q_see_cutoff[2] << '/' << cap_obs.q_see_tried[2]
            << " (" << ratio_percent(cap_obs.q_see_cutoff[2], cap_obs.q_see_tried[2]) << "%)\n";

        out << "info string capobs gate "
            << "checked " << cap_obs.gate_checks
            << " bad_see " << cap_obs.gate_bad_see
            << " pruned " << cap_obs.gate_pruned
            << " (" << ratio_percent(cap_obs.gate_pruned, cap_obs.gate_checks) << "%)"
            << " exempt_check " << cap_obs.gate_exempt_check
            << " exempt_recapture " << cap_obs.gate_exempt_recapture << '\n';

        out << "info string capobs caplmr "
            << "late_simple " << cap_obs.cap_lmr_late_simple_total
            << " eligible " << cap_obs.cap_lmr_eligible
            << " considered " << cap_obs.cap_lmr_considered
            << " reduced " << cap_obs.cap_lmr_reduced
            << " (r1 " << cap_obs.cap_lmr_reduced_r1 << " r2 " << cap_obs.cap_lmr_reduced_r2 << ")"
            << " re_search " << cap_obs.cap_lmr_research
            << " (r1 " << cap_obs.cap_lmr_research_r1 << " r2 " << cap_obs.cap_lmr_research_r2 << ")\n";

        out << "info string capobs caplmr_by_see "
            << "considered bad " << cap_obs.cap_lmr_considered_see[0]
            << " eq " << cap_obs.cap_lmr_considered_see[1]
            << " good " << cap_obs.cap_lmr_considered_see[2]
            << " | reduced bad " << cap_obs.cap_lmr_reduced_see[0]
            << " eq " << cap_obs.cap_lmr_reduced_see[1]
            << " good " << cap_obs.cap_lmr_reduced_see[2]
            << " | re_search bad " << cap_obs.cap_lmr_research_see[0]
            << " eq " << cap_obs.cap_lmr_research_see[1]
            << " good " << cap_obs.cap_lmr_research_see[2] << '\n';

        const int sb_s_bad = sb_raw(DepthClass::Shallow, SeeClass::LossSmall);
        const int sb_s_eq = sb_raw(DepthClass::Shallow, SeeClass::Equal);
        const int sb_s_good_s = sb_raw(DepthClass::Shallow, SeeClass::GainSmall);
        const int sb_s_good_b = sb_raw(DepthClass::Shallow, SeeClass::GainBig);
        const int sb_ml_bad = sb_raw(DepthClass::MediumLow, SeeClass::LossSmall);
        const int sb_ml_eq = sb_raw(DepthClass::MediumLow, SeeClass::Equal);
        const int sb_ml_good_s = sb_raw(DepthClass::MediumLow, SeeClass::GainSmall);
        const int sb_ml_good_b = sb_raw(DepthClass::MediumLow, SeeClass::GainBig);
        const int sb_mh_bad = sb_raw(DepthClass::MediumHigh, SeeClass::LossSmall);
        const int sb_mh_eq = sb_raw(DepthClass::MediumHigh, SeeClass::Equal);
        const int sb_mh_good_s = sb_raw(DepthClass::MediumHigh, SeeClass::GainSmall);
        const int sb_mh_good_b = sb_raw(DepthClass::MediumHigh, SeeClass::GainBig);
        const int sb_d_bad = sb_raw(DepthClass::Deep, SeeClass::LossSmall);
        const int sb_d_eq = sb_raw(DepthClass::Deep, SeeClass::Equal);
        const int sb_d_good_s = sb_raw(DepthClass::Deep, SeeClass::GainSmall);
        const int sb_d_good_b = sb_raw(DepthClass::Deep, SeeClass::GainBig);

        out << "info string capobs see_bias_raw "
            << "shallow bad " << sb_s_bad << " eq " << sb_s_eq
            << " good_s " << sb_s_good_s << " good_b " << sb_s_good_b
            << " | med_low bad " << sb_ml_bad << " eq " << sb_ml_eq
            << " good_s " << sb_ml_good_s << " good_b " << sb_ml_good_b
            << " | med_high bad " << sb_mh_bad << " eq " << sb_mh_eq
            << " good_s " << sb_mh_good_s << " good_b " << sb_mh_good_b
            << " | deep bad " << sb_d_bad << " eq " << sb_d_eq
            << " good_s " << sb_d_good_s << " good_b " << sb_d_good_b << '\n';

        out << "info string capobs see_bias_term "
            << "shallow bad " << sb_term(sb_s_bad) << " eq " << sb_term(sb_s_eq)
            << " good_s " << sb_term(sb_s_good_s) << " good_b " << sb_term(sb_s_good_b)
            << " | med_low bad " << sb_term(sb_ml_bad) << " eq " << sb_term(sb_ml_eq)
            << " good_s " << sb_term(sb_ml_good_s) << " good_b " << sb_term(sb_ml_good_b)
            << " | med_high bad " << sb_term(sb_mh_bad) << " eq " << sb_term(sb_mh_eq)
            << " good_s " << sb_term(sb_mh_good_s) << " good_b " << sb_term(sb_mh_good_b)
            << " | deep bad " << sb_term(sb_d_bad) << " eq " << sb_term(sb_d_eq)
            << " good_s " << sb_term(sb_d_good_s) << " good_b " << sb_term(sb_d_good_b)
            << '\n';

        out << "info string capobs capture_hist_high "
            << "main " << cap_obs.main_high_hist_cutoff << '/' << cap_obs.main_high_hist_tried
            << " (" << ratio_percent(cap_obs.main_high_hist_cutoff, cap_obs.main_high_hist_tried) << "%) "
            << "q " << cap_obs.q_high_hist_cutoff << '/' << cap_obs.q_high_hist_tried
            << " (" << ratio_percent(cap_obs.q_high_hist_cutoff, cap_obs.q_high_hist_tried) << "%)";

        for (int i = 0; i < CAPTURE_TOPK; ++i) {
            const i64 sum = cap_obs.topk_rank_shift_sum[i];
            const u64 cnt = cap_obs.topk_rank_shift_count[i];
            const int avg_x100 = cnt == 0 ? 0 : static_cast<int>((sum * 100) / static_cast<i64>(cnt));
            out << " d" << (i + 1) << "_avg_rank_shift_x100 " << avg_x100;
        }
        out << '\n';
    }
#endif

#if MAGNUS_MOVEPICKER_OBS
    [[nodiscard]] static inline int mp_ratio_percent(u64 num, u64 den) noexcept {
        if (den == 0)
            return 0;
        return static_cast<int>((num * 100ULL) / den);
    }

    [[nodiscard]] static inline MoveStageBucket classify_stage_bucket(
        Move move,
        Move tt_move,
        Move killer1,
        Move killer2,
        bool capture_move,
        bool bad_capture
    ) noexcept {
        if (!move_is_none(tt_move) && move == tt_move)
            return MoveStageBucket::TT;
        if (capture_move)
            return bad_capture ? MoveStageBucket::BadCapture : MoveStageBucket::GoodCapture;
        if (move == killer1 || move == killer2)
            return MoveStageBucket::Killer;
        return MoveStageBucket::Quiet;
    }

    inline void emit_movepicker_observation(std::ostream& out) const {
        const u64 tt_probe_rate_num = mp_obs.nodes_with_tt_probe;
        const u64 tt_probe_rate_den = mp_obs.nodes;
        const u64 tt_first_rate_num = mp_obs.tt_first_try;
        const u64 tt_first_rate_den = mp_obs.nodes_with_tt_probe;
        const u64 tt_first_cut_num = mp_obs.tt_first_cutoff;
        const u64 tt_first_cut_den = mp_obs.tt_first_try;

        out << "info string mpobs nodes " << mp_obs.nodes
            << " tt_probe " << tt_probe_rate_num << '/' << tt_probe_rate_den
            << " (" << mp_ratio_percent(tt_probe_rate_num, tt_probe_rate_den) << "%)"
            << " tt_first_try " << tt_first_rate_num << '/' << tt_first_rate_den
            << " (" << mp_ratio_percent(tt_first_rate_num, tt_first_rate_den) << "%)"
            << " tt_first_cut " << tt_first_cut_num << '/' << tt_first_cut_den
            << " (" << mp_ratio_percent(tt_first_cut_num, tt_first_cut_den) << "%)\n";

        out << "info string mpobs cutoff_by_stage "
            << "total " << mp_obs.cutoffs_total
            << " tt " << mp_obs.cutoff_by_stage[static_cast<int>(MoveStageBucket::TT)]
            << " goodcap " << mp_obs.cutoff_by_stage[static_cast<int>(MoveStageBucket::GoodCapture)]
            << " killer " << mp_obs.cutoff_by_stage[static_cast<int>(MoveStageBucket::Killer)]
            << " quiet " << mp_obs.cutoff_by_stage[static_cast<int>(MoveStageBucket::Quiet)]
            << " badcap " << mp_obs.cutoff_by_stage[static_cast<int>(MoveStageBucket::BadCapture)]
            << '\n';

        out << "info string mpobs first_stage_cutoff_rate "
            << "goodcap " << mp_obs.first_good_capture_cutoff << '/' << mp_obs.first_good_capture_try
            << " (" << mp_ratio_percent(mp_obs.first_good_capture_cutoff, mp_obs.first_good_capture_try) << "%) "
            << "killer " << mp_obs.first_killer_cutoff << '/' << mp_obs.first_killer_try
            << " (" << mp_ratio_percent(mp_obs.first_killer_cutoff, mp_obs.first_killer_try) << "%) "
            << "quiet " << mp_obs.first_quiet_cutoff << '/' << mp_obs.first_quiet_try
            << " (" << mp_ratio_percent(mp_obs.first_quiet_cutoff, mp_obs.first_quiet_try) << "%)\n";

        out << "info string mpobs quiet_work "
            << "generated " << mp_obs.quiet_generated
            << " scored " << mp_obs.quiet_scored
            << " quiet_skipped_by_mp " << mp_obs.quiet_skipped_by_mp
            << " searched " << mp_obs.quiet_searched << '\n';

        out << "info string mpobs quiet_failhigh "
            << "late_quiet_fail_high " << mp_obs.late_quiet_fail_high
            << " quiet_fail_high_after_skip_band " << mp_obs.quiet_fail_high_after_skip_band
            << '\n';
    }
#endif

#if MAGNUS_SEARCH_OBS
    [[nodiscard]] static inline int search_ratio_percent(u64 num, u64 den) noexcept {
        if (den == 0)
            return 0;
        return static_cast<int>((num * 100ULL) / den);
    }

    inline void emit_search_observation(std::ostream& out) const {
        out << "info string searchobs nmp candidates "
            << search_obs.nmp_candidates
            << " tried " << search_obs.nmp_tried
            << " failhigh " << search_obs.nmp_fail_high
            << " verify " << search_obs.nmp_verification_tried
            << " verified " << search_obs.nmp_verified_cutoffs
            << " failed " << search_obs.nmp_verification_failed
            << '\n';
        out << "info string searchobs probcut_nodes "
            << search_obs.probcut_nodes
            << " probcut_moves " << search_obs.probcut_moves
            << " probcut_cutoffs " << search_obs.probcut_cutoffs
            << " (" << search_ratio_percent(search_obs.probcut_cutoffs, search_obs.probcut_moves) << "%)\n";
        out << "info string searchobs singular "
            << search_obs.singular_extend1 << '/' << search_obs.singular_candidates
            << " (" << search_ratio_percent(search_obs.singular_extend1, search_obs.singular_candidates) << "%)"
            << " double " << search_obs.singular_extend2
            << " triple " << search_obs.singular_extend3 << '\n';
    }
#endif

    [[nodiscard]] static inline i16 score_to_tt_i16(int score, int ply) noexcept {
        return static_cast<i16>(
            std::clamp(score_to_tt(score, ply), -VALUE_INF, VALUE_INF)
        );
    }

    [[nodiscard]] static inline i16 raw_eval_to_tt_i16(int raw_eval) noexcept {
        if (raw_eval == VALUE_NONE)
            return static_cast<i16>(VALUE_NONE);
        return static_cast<i16>(std::clamp(raw_eval, -VALUE_INF, VALUE_INF));
    }

    [[nodiscard]] static inline memory::Bound bound_from_score(
        int score,
        int alpha,
        int beta
    ) noexcept {
        return score_bound_from_window(score, alpha, beta);
    }

    [[nodiscard]] static inline bool is_mate_window(int score) noexcept {
        return is_mate_score(score);
    }

    [[nodiscard]] static inline int tablebase_score(
        syzygy::Wdl wdl,
        int ply,
        bool use_rule50
    ) noexcept {
        (void)ply;
        const int value = static_cast<int>(wdl);
        const int draw_threshold = use_rule50 ? 1 : 0;
        if (value < -draw_threshold)
            return -VALUE_TB;
        if (value > draw_threshold)
            return VALUE_TB;
        return 2 * value * draw_threshold;
    }

    [[nodiscard]] bool probe_tablebase(
        const Position& pos,
        int depth,
        int alpha,
        int beta,
        int ply,
        bool pv_node,
        bool exclusion_search,
        int& score,
        memory::Bound& bound,
        bool& resolved
    ) noexcept {
        resolved = false;
        bound = memory::BOUND_NONE;
        if (exclusion_search ||
            limits.syzygy_probe_limit <= 0 ||
            pos.halfmove_clock != 0 ||
            pos.castling_rights != NO_CASTLING) {
            return false;
        }

        const int count = std::popcount(pos.occupied);
        if (count > limits.syzygy_probe_limit ||
            (count == limits.syzygy_probe_limit &&
             depth < limits.syzygy_probe_depth)) {
            return false;
        }

        syzygy::Wdl wdl = syzygy::Wdl::Draw;
        if (!syzygy::probe_wdl(pos, limits.syzygy_probe_limit, wdl))
            return false;

        resolved = true;
        ++tb_hits;
        score = tablebase_score(wdl, ply, limits.syzygy_50_move_rule);
        const int value = static_cast<int>(wdl);
        const int draw_threshold = limits.syzygy_50_move_rule ? 1 : 0;
        bound =
            value < -draw_threshold ? memory::BOUND_UPPER
            : value > draw_threshold ? memory::BOUND_LOWER
                                     : memory::BOUND_EXACT;
        const bool cuts =
            bound == memory::BOUND_EXACT ||
            (bound == memory::BOUND_LOWER ? score >= beta : score <= alpha);
        if (!cuts)
            return false;

        memory::tt_save(
            mem.tt,
            memory::tt_key(pos, mem.tables),
            Move(0),
            score_to_tt_i16(score, ply),
            raw_eval_to_tt_i16(VALUE_NONE),
            static_cast<i16>(std::min(MAX_PLY - 1, depth + 6)),
            bound,
            pv_node
        );
        return true;
    }

    [[nodiscard]] static inline memory::Bound tt_bound_from_probe(
        const memory::TTProbe& probe
    ) noexcept {
        return probe.hit
            ? static_cast<memory::Bound>(probe.data.flags & 0x3U)
            : memory::BOUND_NONE;
    }

    // Delta pruning only needs a rough upper bound on how much a capture can gain.
    [[nodiscard]] static inline int capture_gain_estimate(
        const Position& pos,
        Move move
    ) noexcept {
        int gain = move_is_ep(move)
            ? piece_order_value[PAWN]
            : piece_order_value[piece_type_on(pos, to_sq(move))];

        if (move_is_promotion(move))
            gain += piece_order_value[promo_piece(move)] - piece_order_value[PAWN];

        return gain;
    }

    [[nodiscard]] static inline int reverse_futility_margin(
        int depth,
        bool improving,
        bool opponent_worsening,
        int correction,
        Move tt_move,
        memory::Bound tt_bound,
        int tt_score,
        int beta
    ) noexcept {
        int margin = RFP_BASE_MARGIN
            + depth * RFP_DEPTH_MARGIN
            - (improving ? RFP_IMPROVING_MARGIN : 0)
            - (opponent_worsening ? RFP_OPPONENT_WORSENING_MARGIN : 0);

        if (correction > RFP_CORRECTION_THRESHOLD)
            margin += RFP_CORRECTION_MARGIN_BONUS;

        if (!move_is_none(tt_move) && move_is_capture(tt_move)) {
            margin = std::max(0, margin - RFP_TT_CAPTURE_MARGIN_REDUCTION);
        } else if (!move_is_none(tt_move) &&
                   (tt_bound == memory::BOUND_LOWER || tt_bound == memory::BOUND_EXACT) &&
                   tt_score >= beta) {
            margin += RFP_TT_QUIET_FAIL_HIGH_BONUS;
        }

        return margin;
    }

    [[nodiscard]] static inline int futility_margin(
        int depth,
        bool improving,
        int history_score,
        int correction = 0
    ) noexcept {
        return FUTILITY_BASE_MARGIN
            + depth * FUTILITY_DEPTH_MARGIN
            - (improving ? FUTILITY_IMPROVING_MARGIN : 0)
            + std::clamp(
                history_score / FUTILITY_HISTORY_DIVISOR,
                -64,
                64
            )
            + std::clamp(correction / 4, -32, 32); // eval confidence feedback
    }

    [[nodiscard]] static inline int lmp_limit(int depth, bool improving) noexcept {
        const int d = std::clamp(depth, 1, 11);
        if (improving)
            return 4 + (4 * d * d) / 5;
        return 2 + (2 * d * d) / 5;
    }

    [[nodiscard]] static inline int history_prune_threshold(
        int depth,
        bool improving
    ) noexcept {
        const int coeff = improving ? 5 : 3;
        return -depth * depth * coeff;
    }

    [[nodiscard]] static inline QuietControl quiet_control_for_node(
        int depth,
        bool improving,
        int static_eval,
        int alpha,
        Move tt_move,
        int node_history_signal
    ) noexcept {
        QuietControl control{};
        if (depth < 5)
            return control;

        control.skip_quiets = true;
        control.keep_top_history = 4;
        control.quiet_limit = std::max(6, lmp_limit(depth, improving) / 2);

        if (!improving)
            control.quiet_limit = std::max(6, control.quiet_limit - 1);
        if (static_eval <= alpha)
            control.quiet_limit = std::max(6, control.quiet_limit - 1);
        if (node_history_signal < 0)
            control.quiet_limit = std::max(6, control.quiet_limit - 1);

        if (!move_is_none(tt_move) && !move_is_capture(tt_move)) {
            ++control.quiet_limit;
        } else if (!move_is_none(tt_move)) {
            control.quiet_limit = std::max(6, control.quiet_limit - 1);
        }

        control.history_floor = history_prune_threshold(depth, improving) - 64;
        if (static_eval <= alpha)
            control.history_floor += 32;
        if (!improving)
            control.history_floor += 16;
        if (node_history_signal < 0)
            control.history_floor += std::min(32, -node_history_signal);
        if (!move_is_none(tt_move) && !move_is_capture(tt_move))
            control.history_floor -= 32;

        control.history_floor = std::clamp(control.history_floor, -512, -32);
        return control;
    }

#if MAGNUS_LMR_OBS
    inline void record_lmr_considered(
        const LmrDecision& lmr,
        bool quiet,
        bool capture,
        bool pv_node
    ) noexcept {
        ++lmr_obs.considered;
        if (lmr.eligible)
            ++lmr_obs.eligible;
        if (quiet)
            ++lmr_obs.quiet;
        if (capture)
            ++lmr_obs.capture;
        if (pv_node)
            ++lmr_obs.pv;
        else
            ++lmr_obs.non_pv;

        if (!lmr.eligible)
            return;

        ++lmr_obs.reduced;
        if (lmr.final_reduction <= 1)
            ++lmr_obs.r1;
        else if (lmr.final_reduction == 2)
            ++lmr_obs.r2;
        else
            ++lmr_obs.r3_plus;
    }

    inline void record_lmr_fail_high() noexcept {
        ++lmr_obs.fail_high;
    }

    inline void record_lmr_research() noexcept {
        ++lmr_obs.researches;
    }

    inline void record_lmr_full_confirm() noexcept {
        ++lmr_obs.full_confirms;
    }

    inline void emit_lmr_observation(std::ostream& out) const {
        out << "info string lmrobs considered " << lmr_obs.considered
            << " eligible " << lmr_obs.eligible
            << " reduced " << lmr_obs.reduced
            << " fail_high " << lmr_obs.fail_high
            << " researches " << lmr_obs.researches
            << " full_confirms " << lmr_obs.full_confirms
            << " quiet " << lmr_obs.quiet
            << " capture " << lmr_obs.capture
            << " pv " << lmr_obs.pv
            << " nonpv " << lmr_obs.non_pv
            << " r1 " << lmr_obs.r1
            << " r2 " << lmr_obs.r2
            << " r3plus " << lmr_obs.r3_plus
            << '\n';
    }
#endif

    [[nodiscard]] inline u64 global_nodes() const noexcept {
        if (limits.shared_nodes != nullptr) {
            return limits.shared_nodes->load(std::memory_order_relaxed)
                + (nodes - published_nodes);
        }
        return base_nodes + nodes;
    }

    inline void publish_nodes() noexcept {
        if (limits.shared_nodes == nullptr)
            return;

        const u64 delta = nodes - published_nodes;
        if (delta == 0)
            return;

        limits.shared_nodes->fetch_add(delta, std::memory_order_relaxed);
        published_nodes = nodes;
    }

    inline void publish_tb_hits() noexcept {
        if (limits.shared_tb_hits == nullptr)
            return;

        const u64 delta = tb_hits - published_tb_hits;
        if (delta == 0)
            return;

        limits.shared_tb_hits->fetch_add(delta, std::memory_order_relaxed);
        published_tb_hits = tb_hits;
    }

    inline void update_seldepth(int ply) noexcept {
        const int reported_depth = ply + 1;
        if (reported_depth > seldepth)
            seldepth = reported_depth;
    }

    enum class SingularNodeKind : std::size_t {
        Pv = 0,
        CutLike,
        AllLike
    };

    enum class SingularDepthBand : std::size_t {
        ShallowTt = 0,
        Current,
        DeepTt
    };

    enum class SingularScoreBand : std::size_t {
        NearBeta = 0,
        AtBeta,
        Strong
    };

    struct SingularContext {
        SingularNodeKind node_kind = SingularNodeKind::AllLike;
        SingularDepthBand depth_band = SingularDepthBand::ShallowTt;
        SingularScoreBand score_band = SingularScoreBand::NearBeta;
        memory::Bound tt_bound = memory::BOUND_NONE;
        std::size_t bucket_index = 0;
        int trust = 0;
        int normal_threshold = 0;
        int threshold = 0;
        int tt_depth_gap = 0;
        int tt_score_gap = 0;
        bool tactical_move = false;
        bool good_history = false;
        bool cost_pressure = false;
    };

    [[nodiscard]] static inline std::size_t singular_telemetry_bucket_index(
        SingularNodeKind node_kind,
        SingularDepthBand depth_band,
        SingularScoreBand score_band
    ) noexcept {
        return (
            static_cast<std::size_t>(node_kind) * SINGULAR_DEPTH_BANDS
            + static_cast<std::size_t>(depth_band)
        ) * SINGULAR_SCORE_BANDS + static_cast<std::size_t>(score_band);
    }

    [[nodiscard]] inline bool singular_cost_pressure() const noexcept {
        return nodes >= static_cast<u64>(SINGULAR_COST_GATE_MIN_NODES)
            && singular_verification_nodes * 100
                >= nodes * static_cast<u64>(SINGULAR_COST_RATIO_PERCENT);
    }

    [[nodiscard]] inline int recent_singular_extensions(int ply) const noexcept {
        int extensions = 0;
        const int first = std::max(0, ply - SINGULAR_RECENT_EXTENSION_PLIES);
        for (int i = first; i < ply; ++i)
            extensions += std::max(0, search_stack[i].extension);
        return extensions;
    }

    [[nodiscard]] inline bool is_shuffling(
        Move move,
        const Position& pos,
        int ply
    ) const noexcept {
        const Move move2 = search_stack[ply - 2].current_move;
        const Move move4 = search_stack[ply - 4].current_move;
        return stockfish_is_shuffling(move, pos, move2, move4, ply);
    }

    [[nodiscard]] inline SingularContext make_singular_context(
        bool pv_node,
        memory::Bound tt_bound,
        int tt_depth,
        int depth,
        int tt_score,
        int beta,
        bool tactical_move,
        int history_score
    ) const noexcept {
        SingularContext ctx{};
        ctx.tt_bound = tt_bound;
        ctx.tt_depth_gap = tt_depth - depth;
        ctx.tt_score_gap = tt_score - beta;
        ctx.tactical_move = tactical_move;
        ctx.good_history = history_score >= SINGULAR_GOOD_HISTORY;
        ctx.cost_pressure = singular_cost_pressure();

        const bool cut_like =
            !pv_node
            && (tt_bound == memory::BOUND_LOWER || tt_bound == memory::BOUND_EXACT)
            && tt_score >= beta;
        ctx.node_kind = pv_node
            ? SingularNodeKind::Pv
            : cut_like ? SingularNodeKind::CutLike : SingularNodeKind::AllLike;
        ctx.depth_band = ctx.tt_depth_gap >= 3
            ? SingularDepthBand::DeepTt
            : ctx.tt_depth_gap >= 0
                ? SingularDepthBand::Current
                : SingularDepthBand::ShallowTt;
        ctx.score_band = ctx.tt_score_gap >= SINGULAR_SCORE_STRONG
            ? SingularScoreBand::Strong
            : ctx.tt_score_gap >= 0
                ? SingularScoreBand::AtBeta
                : SingularScoreBand::NearBeta;

        ctx.trust += tt_bound == memory::BOUND_LOWER ? 3 : 2;
        if (ctx.tt_depth_gap >= 0)
            ctx.trust += 2;
        if (ctx.tt_depth_gap >= 3)
            ++ctx.trust;
        if (ctx.tt_score_gap >= 0)
            ctx.trust += 2;
        if (ctx.tt_score_gap >= SINGULAR_SCORE_STRONG)
            ++ctx.trust;
        if (cut_like)
            ++ctx.trust;
        if (ctx.tactical_move)
            ++ctx.trust;
        if (ctx.good_history)
            ++ctx.trust;

        const std::size_t node_index = static_cast<std::size_t>(ctx.node_kind);
        ctx.normal_threshold = singular_trust_threshold(node_index);
        ctx.threshold = ctx.normal_threshold + (ctx.cost_pressure ? 2 : 0);
        ctx.bucket_index = singular_telemetry_bucket_index(
            ctx.node_kind,
            ctx.depth_band,
            ctx.score_band
        );
        return ctx;
    }

    [[nodiscard]] static inline int singular_margin(
        const SingularContext& ctx,
        int depth
    ) noexcept {
        // singular_beta = tt_score - margin: stronger TT evidence lowers the
        // margin, raising the exclusion threshold and easing singular detection.
        int margin = SINGULAR_MARGIN_BASE + SINGULAR_MARGIN_PER_DEPTH * depth;
        margin -= ctx.tt_bound == memory::BOUND_LOWER ? 4 : 2;
        if (ctx.tt_depth_gap >= 3)
            margin -= 4;
        else if (ctx.tt_depth_gap >= 0)
            margin -= 2;
        if (ctx.tt_score_gap >= SINGULAR_SCORE_STRONG)
            margin -= 4;
        else if (ctx.tt_score_gap >= 0)
            margin -= 2;
        if (ctx.node_kind == SingularNodeKind::CutLike)
            margin -= 4;
        else if (ctx.node_kind == SingularNodeKind::AllLike)
            margin += 4;
        if (ctx.tactical_move)
            margin -= 2;
        if (ctx.good_history)
            margin -= 2;
        if (ctx.cost_pressure)
            margin += 8;
        return std::max(16, margin);
    }

    inline void record_singular_candidate(const SingularContext& ctx) noexcept {
        if (!limits.singular_telemetry)
            return;
        SingularTelemetryBucket& bucket = singular_telemetry_buckets[ctx.bucket_index];
        ++bucket.candidates;
        bucket.lower_bound += ctx.tt_bound == memory::BOUND_LOWER;
        bucket.exact_bound += ctx.tt_bound == memory::BOUND_EXACT;
        bucket.tactical_move += ctx.tactical_move;
        bucket.good_history += ctx.good_history;
    }

    inline void record_singular_skip(
        const SingularContext& ctx,
        bool path_limit,
        bool proximity_limit = false
    ) noexcept {
        if (!limits.singular_telemetry)
            return;
        SingularTelemetryBucket& bucket = singular_telemetry_buckets[ctx.bucket_index];
        if (path_limit)
            ++bucket.skipped_path;
        if (ctx.cost_pressure)
            ++bucket.skipped_cost;
        if (ctx.trust < ctx.normal_threshold)
            ++bucket.skipped_trust;
        if (proximity_limit)
            ++bucket.skipped_proximity;
    }

    inline void record_singular_test(
        const SingularContext& ctx,
        int singular_score,
        int singular_beta,
        int beta,
        u64 node_delta
    ) noexcept {
        singular_verification_nodes += node_delta;
        if (!limits.singular_telemetry)
            return;
        SingularTelemetryBucket& bucket = singular_telemetry_buckets[ctx.bucket_index];
        ++bucket.tested;
        bucket.tested_lower += ctx.tt_bound == memory::BOUND_LOWER;
        bucket.tested_exact += ctx.tt_bound == memory::BOUND_EXACT;
        bucket.exclusion_fail_low += singular_score < singular_beta;
        bucket.exclusion_fail_high += singular_score >= beta;
        bucket.exclusion_nodes += node_delta;
    }

    inline void record_singular_decision(
        const SingularContext& ctx,
        int extension,
        bool multi_cut
    ) noexcept {
        if (!limits.singular_telemetry)
            return;
        SingularTelemetryBucket& bucket = singular_telemetry_buckets[ctx.bucket_index];
        bucket.multi_cut += multi_cut;
        bucket.multi_cut_lower += multi_cut && ctx.tt_bound == memory::BOUND_LOWER;
        bucket.multi_cut_exact += multi_cut && ctx.tt_bound == memory::BOUND_EXACT;
        bucket.negative_extended += extension < 0;
        bucket.extended += extension > 0;
        bucket.extended_lower += extension > 0 && ctx.tt_bound == memory::BOUND_LOWER;
        bucket.extended_exact += extension > 0 && ctx.tt_bound == memory::BOUND_EXACT;
        bucket.double_extended += extension >= 2;
        bucket.triple_extended += extension >= 3;
    }

    inline void record_singular_outcome(
        std::size_t bucket_index,
        int extension,
        int score,
        int alpha_before,
        int beta,
        u64 node_delta
    ) noexcept {
        if (!limits.singular_telemetry || extension <= 0)
            return;

        SingularTelemetryBucket& bucket = singular_telemetry_buckets[bucket_index];
        const std::size_t level = static_cast<std::size_t>(
            std::min(extension, static_cast<int>(SINGULAR_TELEMETRY_EXTENSION_LEVELS)) - 1
        );
        SingularExtensionOutcome& outcome = bucket.outcomes[level];
        ++outcome.searched;
        outcome.alpha_raise += score > alpha_before;
        outcome.cutoff += score >= beta;
        outcome.fail_low += score <= alpha_before;
        outcome.nodes += node_delta;
    }

    inline void emit_singular_telemetry(std::ostream& out) const {
        static constexpr const char* NODE_LABELS[] = {"pv", "cut_like", "all_like"};
        static constexpr const char* DEPTH_LABELS[] = {"tt_shallow", "tt_current", "tt_deep"};
        static constexpr const char* SCORE_LABELS[] = {"near_beta", "at_beta", "strong"};

        u64 exclusion_nodes = 0;
        for (const SingularTelemetryBucket& bucket : singular_telemetry_buckets)
            exclusion_nodes += bucket.exclusion_nodes;
        const u64 searched_nodes = std::max<u64>(1, base_nodes + nodes);
        out << "info string singular telemetry summary exclusion_nodes "
            << exclusion_nodes
            << " ratio_permille " << (exclusion_nodes * 1000 / searched_nodes)
            << '\n';

        for (std::size_t node = 0; node < SINGULAR_NODE_KINDS; ++node) {
            for (std::size_t depth = 0; depth < SINGULAR_DEPTH_BANDS; ++depth) {
                for (std::size_t score = 0; score < SINGULAR_SCORE_BANDS; ++score) {
                    const std::size_t index =
                        (node * SINGULAR_DEPTH_BANDS + depth) * SINGULAR_SCORE_BANDS + score;
                    const SingularTelemetryBucket& bucket = singular_telemetry_buckets[index];
                    if (bucket.candidates == 0)
                        continue;

                    out << "info string singular telemetry ctx "
                        << NODE_LABELS[node] << ' '
                        << DEPTH_LABELS[depth] << ' '
                        << SCORE_LABELS[score]
                        << " cand " << bucket.candidates
                        << " test " << bucket.tested
                        << " skip_trust " << bucket.skipped_trust
                        << " skip_cost " << bucket.skipped_cost
                        << " skip_path " << bucket.skipped_path
                        << " skip_proximity " << bucket.skipped_proximity
                        << " lower " << bucket.lower_bound
                        << " exact " << bucket.exact_bound
                        << " test_lower " << bucket.tested_lower
                        << " test_exact " << bucket.tested_exact
                        << " ext_lower " << bucket.extended_lower
                        << " ext_exact " << bucket.extended_exact
                        << " mc_lower " << bucket.multi_cut_lower
                        << " mc_exact " << bucket.multi_cut_exact
                        << " tactical " << bucket.tactical_move
                        << " hist_good " << bucket.good_history
                        << " excl_low " << bucket.exclusion_fail_low
                        << " excl_high " << bucket.exclusion_fail_high
                        << " excl_nodes " << bucket.exclusion_nodes
                        << " ext " << bucket.extended
                        << " d2 " << bucket.double_extended
                        << " d3 " << bucket.triple_extended
                        << " neg " << bucket.negative_extended
                        << " mc " << bucket.multi_cut;
                    for (std::size_t level = 0; level < bucket.outcomes.size(); ++level) {
                        const SingularExtensionOutcome& outcome = bucket.outcomes[level];
                        const u64 average_nodes =
                            outcome.searched == 0 ? 0 : outcome.nodes / outcome.searched;
                        out << " e" << (level + 1) << "_n " << outcome.searched
                            << " e" << (level + 1) << "_raise " << outcome.alpha_raise
                            << " e" << (level + 1) << "_cut " << outcome.cutoff
                            << " e" << (level + 1) << "_low " << outcome.fail_low
                            << " e" << (level + 1) << "_avg_nodes " << average_nodes;
                    }
                    out << '\n';
                }
            }
        }
    }

    [[nodiscard]] inline int elapsed_ms() const noexcept {
        return static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
            clock::now() - start_time
        ).count());
    }

    [[nodiscard]] inline bool pondering_active() const noexcept {
        return limits.pondering != nullptr &&
               limits.pondering->load(std::memory_order_acquire);
    }

    [[nodiscard]] inline int timed_elapsed_ms() const noexcept {
        return elapsed_ms();
    }

    [[nodiscard]] RootEffortEntry* find_root_effort(Move move) noexcept {
        for (int i = 0; i < root_effort_count; ++i)
            if (root_effort_entries[static_cast<std::size_t>(i)].move == move)
                return &root_effort_entries[static_cast<std::size_t>(i)];

        if (move_is_none(move) || root_effort_count >= 256)
            return nullptr;

        RootEffortEntry& entry =
            root_effort_entries[static_cast<std::size_t>(root_effort_count++)];
        entry.move = move;
        return &entry;
    }

    void record_root_effort(Move move, u64 effort, int score) noexcept {
        RootEffortEntry* entry = find_root_effort(move);
        if (entry == nullptr)
            return;

        entry->effort += effort;
        entry->average_score = entry->average_score == VALUE_NONE
            ? score
            : (entry->average_score + score) / 2;
    }

    [[nodiscard]] u64 root_effort(Move move) noexcept {
        RootEffortEntry* entry = find_root_effort(move);
        return entry == nullptr ? 0 : entry->effort;
    }

    [[nodiscard]] int root_average_score(Move move) noexcept {
        RootEffortEntry* entry = find_root_effort(move);
        return entry == nullptr ? VALUE_NONE : entry->average_score;
    }

    inline void signal_best_move_change() noexcept {
        if (limits.time_signals != nullptr)
            limits.time_signals->best_move_changes.fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] inline bool hit_hard_limit() noexcept {
        if (stopped)
            return true;

        if (limits.external_stop != nullptr &&
            limits.external_stop->load(std::memory_order_relaxed)) {
            stopped = true;
            hard_stop = true;
            return true;
        }

        if (limits.stop != nullptr &&
            limits.stop->load(std::memory_order_relaxed)) {
            stopped = true;
            hard_stop = true;
            return true;
        }

        if (limits.node_limit > 0 && global_nodes() >= limits.node_limit) {
            stopped = true;
            hard_stop = true;
            return true;
        }

        if (stop_on_ponderhit && !pondering_active()) {
            stopped = true;
            hard_stop = true;
            return true;
        }

        if (!limits.infinite &&
            limits.hard_time_ms > 0 &&
            completed_depth > 0 &&
            timed_elapsed_ms() >= limits.hard_time_ms) {
            if (pondering_active()) {
                return false;
            } else {
                stopped = true;
                hard_stop = true;
            }
            return true;
        }

        return false;
    }

    inline void poll_limits() noexcept {
        if ((nodes & limit_poll_mask) == 0) {
            publish_nodes();
            (void)hit_hard_limit();
        }
    }

    [[nodiscard]] inline bool stop_after_completed_depth() noexcept {
        return hit_hard_limit();
    }

    [[nodiscard]] bool in_check(const Position& pos) const noexcept {
        const Color side = static_cast<Color>(pos.side_to_move);
        return checkers_bb(pos, mem, side) != 0ULL;
    }

    [[nodiscard]] inline bool is_recapture_move(Move move, int ply) const noexcept {
        if (ply <= 0)
            return false;
        const Move prev = move_stack[ply - 1];
        if (move_is_none(prev))
            return false;
        return to_sq(move) == to_sq(prev);
    }

    inline void set_move_history_context(
        const Position& pos,
        Move move,
        int ply
    ) noexcept {
        move_stack[ply] = move;
        SearchStackEntry& entry = search_stack[ply];
        entry.current_move = move;
        if (move_is_none(move)) {
            entry.continuation = {};
            return;
        }

        const PieceType moved_piece = move_is_promotion(move)
            ? promo_piece(move)
            : piece_type_on(pos, from_sq(move));
        entry.continuation = { moved_piece, to_sq(move) };
    }

    [[nodiscard]] inline bool gives_check_after_move(
        const Position& pos,
        Move move
    ) const noexcept {
        return move_gives_check(pos, mem, move);
    }

    [[nodiscard]] inline bool use_p2_mnue() const noexcept {
        return limits.eval_kind == SearchEvalKind::P2 && mnue::p2_loaded();
    }

    [[nodiscard]] inline bool use_x2k16_mnue() const noexcept {
        return limits.eval_kind == SearchEvalKind::X2K16
            && mnue::x2k16::loaded();
    }

    [[nodiscard]] inline bool use_mnue() const noexcept {
        return use_p2_mnue() || use_x2k16_mnue();
    }

    inline void make_search_move(
        Position& pos,
        Move move,
        StateInfo& st
    ) noexcept {
        if (use_p2_mnue())
            p2_accumulator_stack.push(pos, move);
        make_move(pos, move, mem.tables, st);
    }

    inline void unmake_search_move(
        Position& pos,
        Move move,
        const StateInfo& st
    ) noexcept {
        unmake_move(pos, move, mem.tables, st);
        if (use_p2_mnue())
            p2_accumulator_stack.pop();
    }

    [[nodiscard]] inline int evaluate_raw_position(const Position& pos) const noexcept {
        if (use_p2_mnue())
            return mnue::eval_p2(pos, p2_accumulator_stack);
        if (use_x2k16_mnue())
            return mnue::x2k16::evaluate_lut(pos, mem);
        return 0;
    }

    [[nodiscard]] inline int base_search_eval_from_raw(
        int raw_eval,
        const Position& pos
    ) const noexcept {
        return mnue::search_score(raw_eval, pos);
    }

    [[nodiscard]] inline int evaluate_search_position(const Position& pos) const noexcept {
        return base_search_eval_from_raw(evaluate_raw_position(pos), pos);
    }

    [[nodiscard]] static inline bool tt_raw_eval_available(
        const memory::TTProbe& probe
    ) noexcept {
        return probe.hit && probe.data.eval != VALUE_NONE;
    }

    [[nodiscard]] static inline std::size_t correction_hash_index(Key key) noexcept {
        return static_cast<std::size_t>(mix64(key)) & (CORRECTION_HISTORY_SIZE - 1);
    }

    [[nodiscard]] static inline std::size_t correction_move_index(
        Move move,
        PieceType piece
    ) noexcept {
        if (move_is_none(move) ||
            move_is_capture(move) ||
            move_is_promotion(move) ||
            move_is_castle(move) ||
            !is_ok(piece)) {
            return 0;
        }
        return static_cast<std::size_t>(
            (static_cast<unsigned>(piece) << 12)
            | (static_cast<unsigned>(from_sq(move)) << 6)
            | static_cast<unsigned>(to_sq(move))
        );
    }

    [[nodiscard]] static inline Key correction_pawn_king_key(
        const Position& pos
    ) noexcept {
        return pieces(pos, WHITE, PAWN)
            ^ std::rotl(pieces(pos, BLACK, PAWN), 7)
            ^ std::rotl(pieces(pos, WHITE, KING), 19)
            ^ std::rotl(pieces(pos, BLACK, KING), 29);
    }

    [[nodiscard]] static inline Key correction_nonpawn_key(
        const Position& pos,
        Color color
    ) noexcept {
        return pieces(pos, color, KNIGHT)
            ^ std::rotl(pieces(pos, color, BISHOP), 11)
            ^ std::rotl(pieces(pos, color, ROOK), 23)
            ^ std::rotl(pieces(pos, color, QUEEN), 37)
            ^ std::rotl(pieces(pos, color, KING), 47);
    }

    [[nodiscard]] static inline Key correction_major_key(
        const Position& pos
    ) noexcept {
        return pieces(pos, WHITE, ROOK)
            ^ std::rotl(pieces(pos, WHITE, QUEEN), 11)
            ^ std::rotl(pieces(pos, BLACK, ROOK), 23)
            ^ std::rotl(pieces(pos, BLACK, QUEEN), 37);
    }

    [[nodiscard]] static inline Key correction_minor_key(
        const Position& pos
    ) noexcept {
        return pieces(pos, WHITE, KNIGHT)
            ^ std::rotl(pieces(pos, WHITE, BISHOP), 11)
            ^ std::rotl(pieces(pos, BLACK, KNIGHT), 23)
            ^ std::rotl(pieces(pos, BLACK, BISHOP), 37);
    }

    [[nodiscard]] CorrectionKeys correction_keys(
        const Position& pos,
        int ply
    ) const noexcept {
        CorrectionKeys keys{};
        keys.position = correction_hash_index(pos.key);
        keys.pawn_king = correction_hash_index(correction_pawn_king_key(pos));
        keys.material = correction_hash_index(packed_material_signature(pos));
        if (CORRECTION_NONPAWN_WEIGHT > 0) {
            keys.nonpawn[WHITE] = correction_hash_index(
                correction_nonpawn_key(pos, WHITE)
            );
            keys.nonpawn[BLACK] = correction_hash_index(
                correction_nonpawn_key(pos, BLACK)
            );
        }
        if (CORRECTION_MAJOR_WEIGHT > 0)
            keys.major = correction_hash_index(correction_major_key(pos));
        if (CORRECTION_MINOR_WEIGHT > 0)
            keys.minor = correction_hash_index(correction_minor_key(pos));
        if (CORRECTION_COUNTER_WEIGHT > 0 && ply > 0) {
            const Move move = move_stack[ply - 1];
            const PieceType piece = search_stack[ply - 1].continuation.piece;
            if (!move_is_none(move) && is_ok(piece)) {
                const std::size_t index = correction_move_index(move, piece);
                if (index != 0) {
                    keys.counter = index;
                    keys.has_counter = true;
                }
            }
        }
        if (CORRECTION_FOLLOWUP_WEIGHT > 0 && ply > 1) {
            const Move move = move_stack[ply - 2];
            const PieceType piece = search_stack[ply - 2].continuation.piece;
            if (!move_is_none(move) && is_ok(piece)) {
                const std::size_t index = correction_move_index(move, piece);
                if (index != 0) {
                    keys.followup = index;
                    keys.has_followup = true;
                }
            }
        }
        return keys;
    }

    [[nodiscard]] static inline int correction_slot_value(
        const i16 table[CORRECTION_HISTORY_SIZE],
        std::size_t index
    ) noexcept {
        return static_cast<int>(table[index]);
    }

    [[nodiscard]] static inline int correction_move_slot_value(
        const i16 table[CORRECTION_MOVE_HISTORY_SIZE],
        std::size_t index
    ) noexcept {
        return static_cast<int>(table[index]);
    }

    inline void update_correction_slot(
        i16& slot,
        int target,
        int weight
    ) noexcept {
        const int next =
            static_cast<int>(slot)
            + ((target - static_cast<int>(slot)) * weight) / 128;
        slot = static_cast<i16>(std::clamp(
            next,
            -CORRECTION_HISTORY_CLAMP * CORRECTION_HISTORY_GRAIN,
            CORRECTION_HISTORY_CLAMP * CORRECTION_HISTORY_GRAIN
        ));
    }

    inline void update_correction_slot_if_enabled(
        i16& slot,
        int target,
        int update_weight,
        int history_weight
    ) noexcept {
        if (history_weight <= 0)
            return;
        update_correction_slot(slot, target, update_weight);
    }

    [[nodiscard]] inline int correction_value(
        Color side,
        const CorrectionKeys& keys
    ) const noexcept {
        const CorrectionHistoryTables& hist = correction_history;
        int stored =
            CORRECTION_POSITION_WEIGHT
                * correction_slot_value(hist.position[side], keys.position)
            + CORRECTION_PAWN_WEIGHT
                * correction_slot_value(hist.pawn[side], keys.pawn_king)
            + CORRECTION_MATERIAL_WEIGHT
                * correction_slot_value(hist.material[side], keys.material);
        if (CORRECTION_NONPAWN_WEIGHT > 0) {
            stored += CORRECTION_NONPAWN_WEIGHT
                * correction_slot_value(hist.nonpawn[side][WHITE], keys.nonpawn[WHITE]);
            stored += CORRECTION_NONPAWN_WEIGHT
                * correction_slot_value(hist.nonpawn[side][BLACK], keys.nonpawn[BLACK]);
        }
        if (CORRECTION_MAJOR_WEIGHT > 0) {
            stored += CORRECTION_MAJOR_WEIGHT
                * correction_slot_value(hist.major[side], keys.major);
        }
        if (CORRECTION_MINOR_WEIGHT > 0) {
            stored += CORRECTION_MINOR_WEIGHT
                * correction_slot_value(hist.minor[side], keys.minor);
        }
        if (keys.has_counter) {
            stored += CORRECTION_COUNTER_WEIGHT
                * correction_move_slot_value(hist.counter[side], keys.counter);
        }
        if (keys.has_followup) {
            stored += CORRECTION_FOLLOWUP_WEIGHT
                * correction_move_slot_value(hist.followup[side], keys.followup);
        }
        return std::clamp(
            stored / (
                CORRECTION_HISTORY_GRAIN
                * correction_weight_sum(keys.has_counter, keys.has_followup)
            ),
            -CORRECTION_HISTORY_CLAMP,
            CORRECTION_HISTORY_CLAMP
        );
    }

    inline void update_correction_history(
        Color side,
        const CorrectionKeys& keys,
        int base_eval,
        int score,
        int depth
    ) noexcept {
        if (base_eval == VALUE_NONE || is_decisive(score))
            return;

        const int delta = std::clamp(
            score - base_eval,
            -CORRECTION_HISTORY_CLAMP,
            CORRECTION_HISTORY_CLAMP
        );
        const int target = delta * CORRECTION_HISTORY_GRAIN;
        const int weight = std::min(
            CORRECTION_HISTORY_WEIGHT_MAX,
            16 + std::max(1, depth) * 8
        );
        CorrectionHistoryTables& hist = correction_history;
        update_correction_slot_if_enabled(
            hist.position[side][keys.position],
            target,
            weight,
            CORRECTION_POSITION_WEIGHT
        );
        update_correction_slot_if_enabled(
            hist.pawn[side][keys.pawn_king],
            target,
            weight,
            CORRECTION_PAWN_WEIGHT
        );
        update_correction_slot_if_enabled(
            hist.material[side][keys.material],
            target,
            weight,
            CORRECTION_MATERIAL_WEIGHT
        );
        update_correction_slot_if_enabled(
            hist.nonpawn[side][WHITE][keys.nonpawn[WHITE]],
            target,
            weight,
            CORRECTION_NONPAWN_WEIGHT
        );
        update_correction_slot_if_enabled(
            hist.nonpawn[side][BLACK][keys.nonpawn[BLACK]],
            target,
            weight,
            CORRECTION_NONPAWN_WEIGHT
        );
        update_correction_slot_if_enabled(
            hist.major[side][keys.major],
            target,
            weight,
            CORRECTION_MAJOR_WEIGHT
        );
        update_correction_slot_if_enabled(
            hist.minor[side][keys.minor],
            target,
            weight,
            CORRECTION_MINOR_WEIGHT
        );
        if (keys.has_counter) {
            update_correction_slot_if_enabled(
                hist.counter[side][keys.counter],
                target,
                weight,
                CORRECTION_COUNTER_WEIGHT
            );
        }
        if (keys.has_followup) {
            update_correction_slot_if_enabled(
                hist.followup[side][keys.followup],
                target,
                weight,
                CORRECTION_FOLLOWUP_WEIGHT
            );
        }
    }

    inline void store_static_eval(int ply, int static_eval) noexcept {
        static_eval_stack[ply] = static_eval;
        static_eval_valid[ply] = true;
    }

    [[nodiscard]] inline bool improving_position(
        int ply,
        int static_eval
    ) const noexcept {
        if (ply < 2 || !static_eval_valid[ply - 2])
            return false;
        return static_eval >= static_eval_stack[ply - 2] + IMPROVING_MARGIN;
    }

    [[nodiscard]] bool is_repetition_draw(
        const Position& pos,
        int ply
    ) const noexcept {
        const int back = std::min(ply, pos.halfmove_clock);
        const int min_ply = ply - back;
        for (int p = ply - 2; p >= min_ply; p -= 2) {
            if (rep_keys[p] != pos.key)
                continue;
            return true;
        }

        int matches = 1;
        const int history_window = std::max(0, pos.halfmove_clock - back);
        const int history_count = std::min(
            limits.game_history_count,
            history_window
        );
        const int history_start = limits.game_history_count - history_count;
        for (int i = history_start; i < limits.game_history_count; ++i) {
            if (limits.game_history_keys[i] != pos.key)
                continue;
            if (++matches >= 3)
                return true;
        }

        return false;
    }

    [[nodiscard]] bool previous_repetition_key(
        int ply,
        int plies_back,
        Key& key
    ) const noexcept {
        if (plies_back < 0)
            return false;

        if (plies_back <= ply) {
            key = rep_keys[ply - plies_back];
            return true;
        }

        const int history_plies = plies_back - ply;
        if (history_plies <= 0 ||
            history_plies > limits.game_history_count) {
            return false;
        }

        key = limits.game_history_keys[
            limits.game_history_count - history_plies
        ];
        return true;
    }

    [[nodiscard]] bool has_upcoming_repetition(
        const Position& pos,
        int ply
    ) noexcept {
        if (pos.halfmove_clock < 3 || !mem.tables.cuckoo_repetition.valid) {
            return false;
        }

        const int available_plies = std::min(
            pos.halfmove_clock,
            ply + limits.game_history_count
        );
        if (available_plies < 3)
            return false;

        Key prev_key = 0;
        if (!previous_repetition_key(ply, 1, prev_key))
            return false;

        const CuckooRepetitionTables& cuckoo =
            mem.tables.cuckoo_repetition;
        const Key side_key = mem.tables.zobrist.side;
        Key other = pos.key ^ prev_key ^ side_key;

        for (int i = 3; i <= available_plies; i += 2) {
            Key key_i_minus_1 = 0;
            Key key_i = 0;
            if (!previous_repetition_key(ply, i - 1, key_i_minus_1) ||
                !previous_repetition_key(ply, i, key_i)) {
                break;
            }

            other ^= key_i_minus_1 ^ key_i ^ side_key;
            if (other != 0)
                continue;

            const Key diff = pos.key ^ key_i;
            std::size_t slot = cuckoo_repetition_h1(diff);
            if (cuckoo.keys[slot] != diff) {
                slot = cuckoo_repetition_h2(diff);
                if (cuckoo.keys[slot] != diff)
                    continue;
            }

            const Move move = cuckoo.moves[slot];
            const Square from = from_sq(move);
            const Square to = to_sq(move);
            if ((mem.tables.between[from][to] & pos.occupied) != 0ULL)
                continue;

            if (ply > i)
                return true;

            const Square target =
                piece_on(pos, from) != PIECE_NONE ? from : to;
            return color_on(pos, target) == pos.side_to_move;
        }

        return false;
    }

    [[nodiscard]] inline int draw_score(int side_to_move) const noexcept {
        const int random_component = static_cast<int>(nodes & 0x3ULL) - 2;
        const int contempt_component =
            side_to_move == root_side_to_move ? -limits.contempt : limits.contempt;
        return random_component + contempt_component;
    }

    [[nodiscard]] inline int repetition_score(
        int side_to_move,
        int eval_hint
    ) const noexcept {
        const int base_draw = draw_score(side_to_move);
        if (eval_hint == VALUE_NONE || limits.contempt == 0)
            return base_draw;

        const int avoid_swing = std::max(
            REPETITION_AVOID_BASE,
            std::min(std::abs(limits.contempt), 128)
        );

        if (eval_hint > 0)
            return base_draw - avoid_swing;
        if (eval_hint < 0)
            return base_draw + avoid_swing / 2;
        return base_draw;
    }

    [[nodiscard]] bool has_null_move_pruning_material(
        const Position& pos,
        Color side
    ) const noexcept {
        if (pieces(pos, side, QUEEN) != 0ULL || pieces(pos, side, ROOK) != 0ULL)
            return true;

        const Bitboard minors =
            pieces(pos, side, KNIGHT) |
            pieces(pos, side, BISHOP);
        if (minors == 0ULL)
            return false;

        return (minors & (minors - 1)) != 0ULL || pieces(pos, side, PAWN) != 0ULL;
    }

    [[nodiscard]] inline Move tt_move_from_probe(
        const memory::TTProbe& probe
    ) const noexcept {
        return probe.hit ? static_cast<Move>(probe.data.move) : Move(0);
    }

    [[nodiscard]] inline int tt_raw_eval_from_probe(
        const memory::TTProbe& probe
    ) const noexcept {
        return tt_raw_eval_available(probe) ? probe.data.eval : VALUE_NONE;
    }

    [[nodiscard]] inline bool tt_cutoff(
        const memory::TTProbe& probe,
        int depth,
        int alpha,
        int beta,
        int ply,
        int halfmove_clock,
        int& score
    ) const noexcept {
        if (!probe.hit || probe.data.depth < depth)
            return false;

        score = score_from_tt(probe.data.score, ply, halfmove_clock);

        switch (static_cast<memory::Bound>(probe.data.flags & 0x3U)) {
            case memory::BOUND_EXACT:
                return true;
            case memory::BOUND_LOWER:
                return score >= beta;
            case memory::BOUND_UPPER:
                return score <= alpha;
            default:
                return false;
        }
    }

    inline void save_tt(
        const Position& pos,
        int depth,
        int ply,
        int score,
        int raw_eval,
        Move best_move,
        int alpha,
        int beta,
        bool pv_node
    ) noexcept {
        if (stopped)
            return;

        memory::tt_save(
            mem.tt,
            memory::tt_key(pos, mem.tables),
            best_move,
            score_to_tt_i16(score, ply),
            raw_eval_to_tt_i16(raw_eval),
            static_cast<i16>(depth),
            bound_from_score(score, alpha, beta),
            pv_node
        );
    }

    [[nodiscard]] StaticEvalInfo resolve_static_eval(
        const Position& pos,
        const memory::TTProbe& probe,
        int ply,
        bool checked,
        bool qsearch_node
    ) const noexcept {
        StaticEvalInfo info{};
        if (limits.components.correction_history)
            info.keys = correction_keys(pos, ply);
        const Color side = static_cast<Color>(pos.side_to_move);
        if (tt_raw_eval_available(probe)) {
            info.raw = probe.data.eval;
        } else {
            info.raw = evaluate_raw_position(pos);
        }
        info.base = base_search_eval_from_raw(info.raw, pos);
        const int correction = limits.components.correction_history
            ? correction_value(side, info.keys)
            : 0;
        const int mixed_eval = std::clamp(
            info.base + correction,
            -VALUE_INF,
            VALUE_INF
        );
        info.search = std::clamp(mixed_eval, -VALUE_INF, VALUE_INF);
        info.stand_pat = info.search;

        if (!qsearch_node || checked || !probe.hit)
            return info;

        const int tt_score =
            score_from_tt(probe.data.score, ply, pos.halfmove_clock);
        if (is_decisive(tt_score))
            return info;

        switch (tt_bound_from_probe(probe)) {
            case memory::BOUND_LOWER: {
                const int gap = tt_score - info.stand_pat;
                if (gap > 0)
                    info.stand_pat += std::min(QS_ADJ_SHUFFLE_CAP, std::max(1, gap / 2));
                break;
            }
            case memory::BOUND_UPPER: {
                const int gap = info.stand_pat - tt_score;
                if (gap > 0)
                    info.stand_pat -= std::min(QS_ADJ_SHUFFLE_CAP, std::max(1, gap / 2));
                break;
            }
            default:
                break;
        }

        return info;
    }

    // PV lines are copied upward every time a child improves alpha.
    inline void update_pv(int ply, Move move) noexcept {
        pv_table[ply][0] = move;
        const int child_len = pv_length[ply + 1];
        if (child_len > 0) {
            std::memcpy(
                &pv_table[ply][1],
                pv_table[ply + 1],
                static_cast<std::size_t>(child_len) * sizeof(Move)
            );
        }
        pv_length[ply] = child_len + 1;
    }

    [[nodiscard]] inline i32 score_move(
        const Position& pos,
        Move move,
        Move tt_move,
        int ply,
        int depth,
        int* see_out = nullptr
    ) const noexcept {
        // Ordering priority:
        // 1. TT move
        // 2. captures by MVV-LVA
        // 3. promotions
        // 4. killer moves
        // 5. history heuristic
        if (move == tt_move)
            return 30'000'000;

        if (move_is_capture(move)) {
            const int mvv_lva_term = mvv_lva_capture_term(pos, move);
            const int see_value = search::see_value(pos, mem, move);
            if (see_out) *see_out = see_value;
            const int immediate_see_term = see_immediate_term(see_value, SEE_TERM_PRESET);
            const int see_bias_term = history_tables.see_bias_value_fast(depth, see_value);
            return 20'000'000 + mvv_lva_term
                + history_tables.capture_value_fast(pos, move)
                + immediate_see_term
                + see_bias_term;
        }

        if (move_is_promotion(move))
            return 19'000'000 + piece_order_value[promo_piece(move)];

        if (move == history_tables.killer_fast(ply, 0))
            return 18'000'000;
        if (move == history_tables.killer_fast(ply, 1))
            return 17'999'000;

        return history_tables.quiet_value_fast(pos, move);
    }

    inline void score_moves(
        const Position& pos,
        const MoveList& moves,
        ScoredMoveList& scored,
        Move tt_move,
        int ply,
        int depth
    ) const noexcept {
        scored.size = moves.size;
        for (int i = 0; i < moves.size; ++i) {
            int see_value = 0;
            scored.moves[i].move = moves.moves[i];
            scored.moves[i].score = score_move(pos, moves.moves[i], tt_move, ply, depth, &see_value);
            scored.moves[i].see_value = see_value;
        }
    }

    [[nodiscard]] inline bool root_msv_worker_ordering() const noexcept {
        return root_msv_enabled(limits) &&
               limits.thread_id != 0 &&
               root_msv_has_signal(limits);
    }

    [[nodiscard]] inline int root_msv_history_prior(
        const Position& pos,
        Move move,
        int depth,
        int see_value
    ) const noexcept {
        if (move_is_capture(move))
            return history_tables.capture_ordering_score_fast(
                pos,
                move,
                depth,
                see_value
            ) / 8;

        return history_tables.quiet_ordering_score_fast(
            pos,
            move,
            Move(0),
            ContinuationHistoryContext{},
            ContinuationHistoryContext{},
            ContinuationHistoryContext{}
        ) / 8;
    }

    [[nodiscard]] inline int root_msv_priority(
        const Position& pos,
        const ScoredMove& scored,
        int depth
    ) const {
        const RootMsvEntry entry = root_msv_snapshot(limits, scored.move);
        const int score_part = scored.score / 1024;
        const int history_prior = root_msv_history_prior(
            pos,
            scored.move,
            depth,
            scored.see_value
        );
        const int priority = root_msv_priority_from_entry(
            score_part,
            history_prior,
            depth,
            entry
        );
        root_msv_set_last_priority(limits, scored.move, priority);
        return priority;
    }

    inline void apply_msv_root_order(
        const Position& pos,
        ScoredMoveList& scored,
        int depth
    ) const {
        if (!root_msv_worker_ordering())
            return;

        for (int i = 0; i < scored.size; ++i)
            scored.moves[i].score = root_msv_priority(pos, scored.moves[i], depth);
    }

    inline void apply_startpos_e4_root_order(
        const Position& root,
        ScoredMoveList& scored
    ) const noexcept {
        if (root_opening_preference_bonus(limits, root, STARTPOS_E2E4_MOVE) == 0)
            return;

        for (int i = 0; i < scored.size; ++i) {
            if (scored.moves[i].move == STARTPOS_E2E4_MOVE)
                scored.moves[i].score = std::max(
                    scored.moves[i].score,
                    STARTPOS_E2E4_ROOT_ORDER_SCORE
                );
        }
    }

    void emit_msv_info(
        std::ostream& out,
        const Position& root,
        int depth
    ) const {
        if (!root_msv_enabled(limits) ||
            !limits.msv_info ||
            limits.thread_id != 0) {
            return;
        }

        MoveList list{};
        generate_legal(root, mem, list);

        if (limits.root_move_count > 0) {
            MoveList filtered{};
            for (int i = 0; i < list.size; ++i) {
                const Move move = list.moves[i];
                bool allowed = false;
                for (int j = 0; j < limits.root_move_count; ++j) {
                    if (limits.root_moves[j] == move) {
                        allowed = true;
                        break;
                    }
                }
                if (allowed)
                    filtered.moves[filtered.size++] = move;
            }
            list = filtered;
        }

        const memory::TTProbe probe = memory::tt_probe(mem.tt, memory::tt_key(root, mem.tables));
        const Move tt_move = tt_move_from_probe(probe);
        ScoredMoveList scored{};
        score_moves(root, list, scored, tt_move, 0, depth);

        struct DebugEntry {
            Move move = 0;
            RootMsvEntry entry{};
            int priority = 0;
        };

        std::vector<DebugEntry> entries;
        entries.reserve(static_cast<std::size_t>(scored.size));
        for (int i = 0; i < scored.size; ++i) {
            const int priority = root_msv_priority(root, scored.moves[i], depth);
            const RootMsvEntry entry = root_msv_snapshot(limits, scored.moves[i].move);
            if (entry.credit == 0 &&
                entry.exactHits == 0 &&
                entry.boundHits == 0 &&
                entry.activeWorkers == 0) {
                continue;
            }

            entries.push_back({scored.moves[i].move, entry, priority});
        }

        if (entries.empty())
            return;

        std::stable_sort(
            entries.begin(),
            entries.end(),
            [](const DebugEntry& lhs, const DebugEntry& rhs) noexcept {
                if (lhs.priority != rhs.priority)
                    return lhs.priority > rhs.priority;
                if (lhs.entry.credit != rhs.entry.credit)
                    return lhs.entry.credit > rhs.entry.credit;
                return lhs.move < rhs.move;
            }
        );

        out << "info string msv d=" << depth;
        const int count = std::min<int>(6, static_cast<int>(entries.size()));
        for (int i = 0; i < count; ++i) {
            const DebugEntry& e = entries[static_cast<std::size_t>(i)];
            out << ' ' << move_to_uci(e.move)
                << "[c=" << e.entry.credit
                << ",h=" << (e.entry.exactHits + e.entry.boundHits)
                << ",w=" << e.entry.workerHits
                << ",md=" << e.entry.maxDepth
                << ",a=" << e.entry.activeWorkers
                << ",p=" << e.priority
                << ']';
        }
        out << '\n';
    }

    [[nodiscard]] inline Move pick_next(ScoredMoveList& scored, int index) const noexcept {
        int best = index;
        for (int i = index + 1; i < scored.size; ++i)
            if (scored.moves[i].score > scored.moves[best].score)
                best = i;

        if (best != index)
            std::swap(scored.moves[index], scored.moves[best]);

        return scored.moves[index].move;
    }

    [[nodiscard]] int qsearch(Position& pos, int alpha, int beta, int ply) noexcept {
        // Quiescence search extends only tactical continuations (or all legal
        // evasions when in check) so the engine does not stand pat in unstable positions.
        pv_length[ply] = 0;
        rep_keys[ply] = pos.key;
        update_seldepth(ply);
        ++nodes;
        poll_limits();
        if (stopped)
            return beta;

        if (ply >= MAX_PLY - 1)
            return evaluate_search_position(pos);

        if (pos.halfmove_clock >= 100)
            return draw_score(pos.side_to_move);

        alpha = std::max(alpha, -VALUE_MATE + ply);
        beta = std::min(beta, VALUE_MATE - ply - 1);
        if (alpha >= beta)
            return alpha;

        const int alpha0 = alpha;
        const bool pv_node = (beta - alpha) > 1;
        const int draw_floor = draw_score(pos.side_to_move);
        if (alpha < draw_floor && has_upcoming_repetition(pos, ply)) {
            alpha = draw_floor;
            if (alpha >= beta)
                return alpha;
        }

        const memory::TTProbe probe = memory::tt_probe(mem.tt, memory::tt_key(pos, mem.tables));
        if (is_repetition_draw(pos, ply))
            return repetition_score(pos.side_to_move, tt_raw_eval_from_probe(probe));

        int tt_score = 0;
        if (tt_cutoff(probe, 0, alpha, beta, ply, pos.halfmove_clock, tt_score))
            return tt_score;

        const bool checked = in_check(pos);
        const Move tt_move = tt_move_from_probe(probe);
        const StaticEvalInfo eval_info = resolve_static_eval(pos, probe, ply, checked, true);
        const int raw_eval = eval_info.raw;
        const int static_eval = eval_info.search;
        const int stand_pat_eval = eval_info.stand_pat;
        store_static_eval(ply, static_eval);

        if (!checked) {
            // Stand-pat: if the static position already fails high, no capture
            // search can make it worse for the side to move.
            if (stand_pat_eval >= beta) {
                save_tt(pos, 0, ply, stand_pat_eval, raw_eval, tt_move, alpha0, beta, pv_node);
                return stand_pat_eval;
            }
            if (stand_pat_eval > alpha)
                alpha = stand_pat_eval;
        }

        MoveList list;
        GenInfo info;
        init_gen_info(info, pos, mem);
        Move* qend = generate_pseudo_captures(pos, mem, info, list.moves);
        list.size = static_cast<int>(qend - list.moves);

        if (list.size == 0) {
            const int score = checked ? (-VALUE_MATE + ply) : alpha;
            save_tt(pos, 0, ply, score, raw_eval, 0, alpha0, beta, pv_node);
            return score;
        }

        ScoredMoveList scored{};
        for (int i = 0; i < list.size; ++i) {
            const Move move = list.moves[i];
            if (!legal_fast(pos, mem, info, move))
                continue;

            int see_value = 0;
            scored.moves[scored.size].move = move;
            scored.moves[scored.size].score =
                score_move(pos, move, tt_move, ply, 0, &see_value);
            scored.moves[scored.size].see_value = see_value;
            ++scored.size;
        }

        if (scored.size == 0) {
            const int score = checked ? (-VALUE_MATE + ply) : alpha;
            save_tt(pos, 0, ply, score, raw_eval, 0, alpha0, beta, pv_node);
            return score;
        }

        Move best_move = 0;
        for (int i = 0; i < scored.size; ++i) {
            const Move move = pick_next(scored, i);
            const int cached_see = scored.moves[i].see_value;

            if (!checked &&
                !move_is_promotion(move) &&
                cached_see < -DELTA_MARGIN) {
                continue;
            }

            if (!checked && !move_is_promotion(move)) {
                // Delta pruning skips captures that cannot reasonably raise alpha.
                const int max_gain = capture_gain_estimate(pos, move);
                if (static_eval + max_gain + DELTA_MARGIN <= alpha)
                    continue;
            }

#if MAGNUS_CAPTURE_OBS
            const bool capture_move = move_is_capture(move);
            int obs_see_value = 0;
            int obs_capture_hist = 0;
            if (capture_move) {
                obs_see_value = cached_see;
                obs_capture_hist = history_tables.capture_value_fast(pos, move);
                record_q_capture_try(obs_see_value, obs_capture_hist);
            }
#endif

            StateInfo st;
            make_search_move(pos, move, st);
            memory::tt_prefetch(mem.tt, memory::tt_key(pos, mem.tables));

            const int score = -qsearch(pos, -beta, -alpha, ply + 1);
            unmake_search_move(pos, move, st);
            if (stopped)
                return beta;

            if (score > alpha) {
                alpha = score;
                best_move = move;
                update_pv(ply, move);
                if (alpha >= beta) {
#if MAGNUS_CAPTURE_OBS
                    if (move_is_capture(move))
                        record_q_capture_cutoff(obs_see_value, obs_capture_hist);
#endif
                    break;
                }
            }
        }

        save_tt(pos, 0, ply, alpha, raw_eval, best_move, alpha0, beta, pv_node);
        return alpha;
    }

    [[nodiscard]] int pvs(
        Position& pos,
        int depth,
        int alpha,
        int beta,
        int ply,
        bool allow_null,
        Move excluded_move = Move(0)
    ) noexcept {
        // Principal Variation Search:
        // - first move gets a full window
        // - later moves get a null window first
        // - promising moves are re-searched on a wider window
        const bool previous_tt_pv = search_stack[ply].tt_pv;
        pv_length[ply] = 0;
        rep_keys[ply] = pos.key;
        update_seldepth(ply);

        if (stopped)
            return beta;

        if (ply >= MAX_PLY - 1)
            return evaluate_search_position(pos);

        if (pos.halfmove_clock >= 100)
            return draw_score(pos.side_to_move);

        alpha = std::max(alpha, -VALUE_MATE + ply);
        beta = std::min(beta, VALUE_MATE - ply - 1);
        if (alpha >= beta)
            return alpha;

        if (depth <= 0)
            return qsearch(pos, alpha, beta, ply);

        ++nodes;
        poll_limits();
        if (stopped)
            return beta;

        const int alpha0 = alpha;
        const bool pv_node = (beta - alpha) > 1;
        const bool exclusion_search = !move_is_none(excluded_move);
        const memory::TTProbe probe = memory::tt_probe(mem.tt, memory::tt_key(pos, mem.tables));
        if (is_repetition_draw(pos, ply))
            return repetition_score(pos.side_to_move, tt_raw_eval_from_probe(probe));
        const Move probed_tt_move = tt_move_from_probe(probe);
        const Move tt_move = exclusion_search ? Move(0) : probed_tt_move;
        const memory::Bound tt_bound = tt_bound_from_probe(probe);
        const int probed_tt_score = probe.hit
            ? score_from_tt(probe.data.score, ply, pos.halfmove_clock)
            : 0;

        int node_depth = depth;
        if (limits.components.iir &&
            !pv_node &&
            node_depth >= IIR_MIN_DEPTH &&
            move_is_none(tt_move) &&
            !exclusion_search) {
            --node_depth;
        }
        const int legacy_tt_store_depth = node_depth;
        int tt_score = 0;
        if (!exclusion_search &&
            (limits.tt_trust_stage == TtTrustStage::A || !pv_node) &&
            tt_cutoff(
                probe,
                node_depth,
                alpha,
                beta,
                ply,
                pos.halfmove_clock,
                tt_score
            ))
            return tt_score;

        int tb_score = 0;
        memory::Bound tb_bound = memory::BOUND_NONE;
        bool tablebase_resolved = false;
        if (probe_tablebase(
                pos,
                node_depth,
                alpha,
                beta,
                ply,
                pv_node,
                exclusion_search,
                tb_score,
                tb_bound,
                tablebase_resolved
            )) {
            return tb_score;
        }
        int tablebase_max_score = VALUE_INF;
        if (tablebase_resolved && pv_node) {
            if (tb_bound == memory::BOUND_LOWER)
                alpha = std::max(alpha, tb_score);
            else if (tb_bound == memory::BOUND_UPPER)
                tablebase_max_score = tb_score;
        }

        const bool checked = in_check(pos);
        const StaticEvalInfo eval_info = resolve_static_eval(pos, probe, ply, checked, false);
        const int raw_eval = eval_info.raw;
        const int base_eval = eval_info.base;
        const int static_eval = eval_info.search;
        const int correction = static_eval - base_eval;
        store_static_eval(ply, static_eval);
        const bool improving = !checked && improving_position(ply, static_eval);
        const bool opponent_worsening = !checked && ply > 0
            && static_eval_valid[ply - 1]
            && static_eval > -static_eval_stack[ply - 1];
        const bool can_prune = !pv_node && !checked;

        // Hindsight depth adjustment: compensate for prior LMR under/over-reduction.
        const int prior_reduction_plies = ply > 0
            ? search_stack[ply - 1].reduction_fp / 1024 : 0;
        if (limits.components.lmr && prior_reduction_plies >= 3 && !opponent_worsening)
            node_depth++;
        if (limits.components.lmr &&
            prior_reduction_plies >= 2 && node_depth >= 2
            && ply > 0 && static_eval_valid[ply - 1]
            && static_eval + static_eval_stack[ply - 1] > 195)
            node_depth--;

        const Color side = static_cast<Color>(pos.side_to_move);
        SearchStackEntry& ss = search_stack[ply];
        ss.current_move = Move(0);
        ss.continuation = {};
        ss.static_eval = static_eval;
        ss.stat_score = 0;
        ss.reduction_fp = 0;
        ss.extension = 0;
        ss.move_count = 0;
        ss.cutoff_count = 0;
        ss.in_check = checked;
        ss.tt_hit = probe.hit;
        ss.tt_pv = exclusion_search
            ? previous_tt_pv
            : pv_node || (probe.hit && (probe.data.flags & 0x4U) != 0);

        // Conservative TT-estimated razoring:
        // use TT score only when its bound is directionally useful.
        // Keep the old depth cap and qsearch verification.

        int razor_eval = static_eval;
        if (probe.hit && !checked && !exclusion_search &&
            is_valid_score(probed_tt_score)) {
            if ((tt_bound == memory::BOUND_EXACT) ||
                (tt_bound == memory::BOUND_UPPER &&
                 probed_tt_score < static_eval) ||
                (tt_bound == memory::BOUND_LOWER &&
                 probed_tt_score > static_eval)) {
                razor_eval = probed_tt_score;
            }
        }

        // razor_margin() is only used after depth <= 2 is confirmed through
        // short-circuit evaluation.
        if (limits.components.razoring &&
            can_prune &&
            node_depth <= 2 &&
            razor_eval + razor_margin(node_depth) <= alpha) {
            const int score = qsearch(pos, alpha, beta, ply);
            if (score <= alpha)
                return score;
        }

        if (limits.components.reverse_futility &&
            can_prune &&
            node_depth < 8 &&
            !is_mate_window(beta) &&
            static_eval - reverse_futility_margin(
                node_depth,
                improving,
                opponent_worsening,
                correction,
                tt_move,
                tt_bound,
                probed_tt_score,
                beta
            ) >= beta) {
            // Reverse futility: when static eval is already safely above beta,
            // a shallow node often does not need a full search.
            return beta + (static_eval - beta) / 3;
        }

        const bool nmp_disabled_here = nmp_disabled_for_ply(ply, nmp_min_ply);
#if MAGNUS_SEARCH_OBS
        if (limits.components.null_move &&
            allow_null &&
            !pv_node &&
            !checked &&
            node_depth >= 3 &&
            !exclusion_search &&
            !is_mate_window(beta)) {
            ++search_obs.nmp_candidates;
        }
#endif

        NmpNodeContext nmp_node{};
        nmp_node.depth = node_depth;
        nmp_node.ply = ply;
        nmp_node.alpha = alpha;
        nmp_node.beta = beta;
        nmp_node.static_eval = static_eval;
        nmp_node.tt_score = probed_tt_score;
        nmp_node.nmp_min_ply = nmp_min_ply;
        nmp_node.allow_null = allow_null;
        nmp_node.pv_node = pv_node;
        nmp_node.cut_node =
            !pv_node &&
            (tt_bound == memory::BOUND_LOWER || !move_is_none(tt_move));
        nmp_node.checked = checked;
        nmp_node.improving = improving;
        nmp_node.exclusion_search = exclusion_search;
        nmp_node.tt_hit = probe.hit;
        nmp_node.tt_move_present = !move_is_none(tt_move);
        nmp_node.material_ok = has_null_move_pruning_material(pos, side);
        nmp_node.tt_bound = tt_bound;

        const NmpDecision nmp = limits.components.null_move
            ? decide_null_move(nmp_node)
            : NmpDecision{};
        if (nmp.eligible && !is_mate_window(beta) && !nmp_disabled_here) {
            // Null-move pruning tests whether simply passing still keeps the
            // position above beta. If so, the real position is likely also a cutoff.
#if MAGNUS_SEARCH_OBS
            ++search_obs.nmp_tried;
#endif
            StateInfo null_state{};
            make_null_move(pos, mem.tables, null_state);
            set_move_history_context(pos, Move(0), ply);

            const int score = -pvs(
                pos,
                nmp.null_depth,
                -beta,
                -beta + 1,
                ply + 1,
                false
            );
            unmake_null_move(pos, null_state);
            if (stopped)
                return beta;

            if (score >= beta && !is_mate_window(score)) {
#if MAGNUS_SEARCH_OBS
                ++search_obs.nmp_fail_high;
#endif
                if (!nmp.requires_verification) {
                    save_tt(
                        pos,
                        limits.tt_trust_stage == TtTrustStage::A
                            ? legacy_tt_store_depth : node_depth,
                        ply,
                        score,
                        raw_eval,
                        0,
                        alpha0,
                        beta,
                        pv_node
                    );
                    return score;
                }

#if MAGNUS_SEARCH_OBS
                ++search_obs.nmp_verification_tried;
#endif
                const int old_nmp_min_ply = nmp_min_ply;
                nmp_min_ply = nmp.verify_min_ply;
                const int verify_score = pvs(
                    pos,
                    nmp.verify_depth,
                    beta - 1,
                    beta,
                    ply,
                    true
                );
                nmp_min_ply = old_nmp_min_ply;
                if (stopped)
                    return beta;

                if (verify_score >= beta) {
#if MAGNUS_SEARCH_OBS
                    ++search_obs.nmp_verified_cutoffs;
#endif
                    save_tt(
                        pos,
                        limits.tt_trust_stage == TtTrustStage::A
                            ? legacy_tt_store_depth : node_depth,
                        ply,
                        score,
                        raw_eval,
                        0,
                        alpha0,
                        beta,
                        pv_node
                    );
                    return score;
                }

#if MAGNUS_SEARCH_OBS
                ++search_obs.nmp_verification_failed;
#endif
            }
        }

        GenInfo info;
        init_gen_info(info, pos, mem);

        if (!pv_node) {
            const int draw_floor = draw_score(pos.side_to_move);
            if (alpha < draw_floor &&
                has_upcoming_repetition(pos, ply)) {
                alpha = draw_floor;
                if (alpha >= beta)
                    return alpha;
            }
        }

#if MAGNUS_ENABLE_PROBCUT
        if (limits.components.probcut &&
            can_prune &&
            !exclusion_search &&
            node_depth >= PROBCUT_MIN_DEPTH &&
            !is_mate_window(beta)) {
#if MAGNUS_SEARCH_OBS
            ++search_obs.probcut_nodes;
#endif
            const int probcut_beta = beta + PROBCUT_MARGIN;

            if (probe.hit &&
                tt_bound == memory::BOUND_LOWER &&
                probe.data.depth >= node_depth - PROBCUT_TT_DEPTH_MARGIN &&
                probed_tt_score >= probcut_beta &&
                !is_mate_window(probed_tt_score)) {
                return probed_tt_score;
            }

            MoveList probcut_moves{};
            Move* probcut_end = generate_pseudo_captures_only(pos, mem, info, probcut_moves.moves);
            probcut_moves.size = static_cast<int>(probcut_end - probcut_moves.moves);

            if (probcut_moves.size > 0) {
                ScoredMoveList scored_probcut{};
                score_moves(pos, probcut_moves, scored_probcut, tt_move, ply, node_depth);

                for (int i = 0; i < scored_probcut.size; ++i) {
                    const Move move = pick_next(scored_probcut, i);
                    if (!move_is_capture(move))
                        continue;
                    if (!legal_fast(pos, mem, info, move))
                        continue;
                    if (scored_probcut.moves[i].see_value < 0)
                        continue;
#if MAGNUS_SEARCH_OBS
                    ++search_obs.probcut_moves;
#endif

                    StateInfo st;
                    set_move_history_context(pos, move, ply);
                    make_search_move(pos, move, st);
                    memory::tt_prefetch(mem.tt, memory::tt_key(pos, mem.tables));

                    int score = -qsearch(pos, -probcut_beta, -probcut_beta + 1, ply + 1);
                    if (score >= probcut_beta) {
                        const int probcut_depth = std::max(0, node_depth - PROBCUT_REDUCTION);
                        if (probcut_depth > 0) {
                            score = -pvs(
                                pos,
                                probcut_depth,
                                -probcut_beta,
                                -probcut_beta + 1,
                                ply + 1,
                                false
                            );
                        }
                    }

                    unmake_search_move(pos, move, st);
                    if (stopped)
                        return beta;

                    if (score >= probcut_beta) {
#if MAGNUS_SEARCH_OBS
                        ++search_obs.probcut_cutoffs;
#endif
                        save_tt(pos, std::max(0, node_depth - PROBCUT_REDUCTION), ply, score, raw_eval, move, alpha0, beta, pv_node);
                        return score;
                    }
                }
            }
        }
#endif

#if MAGNUS_ENABLE_SMALL_PROBCUT
        {
            const int small_probcut_beta = beta + SMALL_PROBCUT_MARGIN;
            if (limits.components.small_probcut &&
                !pv_node && probe.hit
                && tt_bound == memory::BOUND_LOWER
                && probe.data.depth >= node_depth - SMALL_PROBCUT_TT_DEPTH_MARGIN
                && probed_tt_score >= small_probcut_beta
                && !is_mate_window(beta)
                && !is_mate_window(probed_tt_score)) {
                return small_probcut_beta;
            }
        }
#endif

        const Move prev_move = (ply > 0) ? move_stack[ply - 1] : Move(0);
        const ContinuationHistoryContext prev2 =
            (ply > 1) ? search_stack[ply - 2].continuation : ContinuationHistoryContext{};
        const ContinuationHistoryContext prev4 =
            (ply > 3) ? search_stack[ply - 4].continuation : ContinuationHistoryContext{};
        const ContinuationHistoryContext prev8 =
            (ply > 7) ? search_stack[ply - 8].continuation : ContinuationHistoryContext{};
        QuietControl quiet_control{};
        if (!pv_node && !checked) {
            int node_history_signal = 0;
            if (ply > 0)
                node_history_signal += search_stack[ply - 1].stat_score / 256;
            if (ply > 1)
                node_history_signal += search_stack[ply - 2].stat_score / 512;

            quiet_control = quiet_control_for_node(
                node_depth,
                improving,
                static_eval,
                alpha,
                tt_move,
                node_history_signal
            );
        }
        MovePicker picker(
            pos,
            mem,
            info,
            history_tables,
            tt_move,
            ply,
            prev_move,
            prev2,
            prev4,
            prev8,
            node_depth,
            quiet_control
        );
#if MAGNUS_MOVEPICKER_OBS
        ++mp_obs.nodes;
        if (!move_is_none(tt_move))
            ++mp_obs.nodes_with_tt_probe;
        const Move killer1 = history_tables.killer_fast(ply, 0);
        const Move killer2 = history_tables.killer_fast(ply, 1);
        bool seen_first_good_capture = false;
        bool seen_first_killer = false;
        bool seen_first_quiet = false;
#endif

#if MAGNUS_CAPTURE_OBS
        Move capture_moves[MAX_MOVES];
        int capture_base_scores[MAX_MOVES];
        int capture_base_ranks[MAX_MOVES];
        int capture_count = 0;
        // Reuse MovePicker's already-generated capture lists instead of regenerating.
        {
            const int good_cnt = picker.good_capture_count();
            for (int i = 0; i < good_cnt; ++i) {
                const Move m = picker.good_captures()[i].move;
                capture_moves[capture_count] = m;
                capture_base_scores[capture_count] = mvv_lva_capture_term(pos, m);
                ++capture_count;
            }
            const int bad_cnt = picker.bad_capture_count();
            for (int i = 0; i < bad_cnt; ++i) {
                const Move m = picker.bad_captures()[i].move;
                capture_moves[capture_count] = m;
                capture_base_scores[capture_count] = mvv_lva_capture_term(pos, m);
                ++capture_count;
            }
        }
        for (int i = 0; i < capture_count; ++i) {
            int rank = 1;
            for (int j = 0; j < capture_count; ++j)
                if (capture_base_scores[j] > capture_base_scores[i])
                    ++rank;
            capture_base_ranks[i] = rank;
        }
        int capture_search_order = 0;
#endif

        bool searched_first = false;
        Move best_move = 0;
        int legal_count = 0;
        int quiet_count = 0;
        Move searched_quiets[MAX_MOVES];
        int searched_quiet_count = 0;
        Move searched_captures[MAX_MOVES];
        int searched_capture_see[MAX_MOVES];
        int searched_capture_count = 0;
        int simple_capture_count = 0;
        int capture_index = 0;
        int moves_tried = 0;
        int best_score = -VALUE_INF;
        bool cutoff = false;
        const int tt_trust_before = !exclusion_search && !move_is_none(tt_move)
            ? persistent.trust(pos)
            : 0;
        const std::size_t tt_trust_bucket =
            tt_move_trust_bucket(tt_trust_before);
        bool tt_move_searched = false;
        int tt_move_extension = 0;

        for (;;) {
            if (stopped)
                return beta;

            const Move move = picker.next();
            if (move_is_none(move))
                break;
            if (move == excluded_move)
                continue;

            ++moves_tried;
            const int move_index = moves_tried - 1;
            ++legal_count;
            const bool capture_move = move_is_capture(move);
            const bool bad_capture = picker.last_was_bad_capture();
            const bool quiet_move =
                !capture_move &&
                !move_is_promotion(move) &&
                !move_is_castle(move);
            const bool simple_capture =
                capture_move &&
                !move_is_promotion(move);
            int move_see_value = 0;
            if (capture_move)
                move_see_value = picker.last_see_value();
            const int history_score = quiet_move
                ? history_tables.quiet_value_fast(pos, move)
                : 0;
            const int quiet_ordering_score = quiet_move
                ? picker.last_score()
                : 0;
            const int capture_history_score = simple_capture
                ? history_tables.capture_value_fast(pos, move)
                : 0;
            const bool lmr_quiet_candidate =
                quiet_move &&
                !checked &&
                ((!pv_node && node_depth >= 3 && move_index >= 2) ||
                 (pv_node && node_depth >= 5 && move_index >= 4));
            const bool lmr_capture_candidate =
                simple_capture &&
                !pv_node &&
                !checked &&
                node_depth >= 4 &&
                simple_capture_count >= 2;
            bool gives_check = false;
            bool gives_check_known = false;
            const auto ensure_gives_check = [&]() noexcept {
                if (!gives_check_known) {
                    gives_check = gives_check_after_move(pos, move);
                    gives_check_known = true;
                }
                return gives_check;
            };
#if MAGNUS_MOVEPICKER_OBS
            const bool first_move_this = (moves_tried == 1);
            const MoveStageBucket stage_bucket = classify_stage_bucket(
                move, tt_move, killer1, killer2, capture_move, bad_capture
            );
            const bool first_good_capture_this =
                stage_bucket == MoveStageBucket::GoodCapture && !seen_first_good_capture;
            const bool first_killer_this =
                stage_bucket == MoveStageBucket::Killer && !seen_first_killer;
            const bool first_quiet_this =
                stage_bucket == MoveStageBucket::Quiet && !seen_first_quiet;
            if (first_move_this && stage_bucket == MoveStageBucket::TT)
                ++mp_obs.tt_first_try;
            if (first_good_capture_this) {
                seen_first_good_capture = true;
                ++mp_obs.first_good_capture_try;
            }
            if (first_killer_this) {
                seen_first_killer = true;
                ++mp_obs.first_killer_try;
            }
            if (first_quiet_this) {
                seen_first_quiet = true;
                ++mp_obs.first_quiet_try;
            }
#endif

            if (quiet_move)
                ++quiet_count;
            if (simple_capture) {
                capture_index = simple_capture_count;
                ++simple_capture_count;
            }

#if MAGNUS_SEE_LATE_BAD_CAPTURE_GATE
            if (limits.components.see_bad_capture_gate &&
                !pv_node &&
                !checked &&
                simple_capture &&
                node_depth >= SEE_LATE_BAD_CAPTURE_GATE_MIN_DEPTH &&
                node_depth <= SEE_LATE_BAD_CAPTURE_GATE_MAX_DEPTH &&
                simple_capture_count >= SEE_LATE_BAD_CAPTURE_GATE_MIN_CAPTURE_INDEX) {
                const bool bad_see = move_see_value < SEE_LATE_BAD_CAPTURE_GATE_THRESHOLD;
                [[maybe_unused]] bool pruned = false;
                bool exempt_check = false;
                bool exempt_recapture = false;

                if (bad_see) {
                    exempt_recapture = is_recapture_move(move, ply);
                    if (!exempt_recapture)
                        exempt_check = ensure_gives_check();

                    if (!exempt_recapture && !exempt_check) {
                        pruned = true;
#if MAGNUS_CAPTURE_OBS
                        record_gate_check(bad_see, pruned, exempt_check, exempt_recapture);
#endif
                        continue;
                    }
                }
#if MAGNUS_CAPTURE_OBS
                record_gate_check(bad_see, pruned, exempt_check, exempt_recapture);
#endif
            }
#endif

            // Capture futility pruning: even with the captured piece value,
            // the position cannot reach alpha at shallow LMR depth.
            if (limits.components.capture_futility &&
                !pv_node &&
                !checked &&
                simple_capture &&
                !ensure_gives_check() &&
                node_depth <= 7) {
                const PieceType captured_pt = move_is_ep(move) ? PAWN : piece_type_on(pos, to_sq(move));
                const int captured_value = is_ok(captured_pt) ? piece_order_value[captured_pt] : 0;
                // SPSA-tuned capture futility margin: base intercept + depth slope
                // + material gain + scaled capture-history signal.
                const int cap_futility = static_eval + 192 + 160 * node_depth
                    + captured_value + 131 * capture_history_score / 1024;
                if (cap_futility <= alpha)
                    continue;
            }

            // Bad capture pruning with dynamic SEE threshold:
            // capture history adjusts the threshold — good captures are harder to prune.
            if (limits.components.see_pruning &&
                !pv_node &&
                !checked &&
                simple_capture &&
                move_index > 1) {
                const int see_margin = std::max(167 * node_depth + capture_history_score * 34 / 1024, 0);
                if (!search::see_ge(pos, mem, move, -see_margin))
                    continue;
            }

            // Combined history for quiet pruning decisions:
            // split continuation history (2/4/8 ply) + pawn history.
            const int combined_history = quiet_move
                ? history_tables.continuation_value_fast(
                        pos,
                        move,
                        prev2,
                        CONTINUATION_PLY2_SLOT
                    )
                    + history_tables.continuation_value_fast(
                        pos,
                        move,
                        prev4,
                        CONTINUATION_PLY4_SLOT
                    ) / 2
                    + history_tables.continuation_value_fast(
                        pos,
                        move,
                        prev8,
                        CONTINUATION_PLY8_SLOT
                    ) / 4
                    + history_tables.pawn_history_value_fast(pos, move)
                : 0;

            if (limits.components.quiet_futility &&
                !pv_node &&
                !checked &&
                node_depth <= 4 &&
                quiet_move &&
                !ensure_gives_check() &&
                static_eval + futility_margin(node_depth, improving, history_score, correction) <= alpha) {
                // Shallow futility pruning skips quiet moves that cannot raise alpha.
                continue;
            }

            if (limits.components.late_move_pruning &&
                !pv_node &&
                !checked &&
                node_depth <= 7 &&
                quiet_move &&
                !ensure_gives_check() &&
                quiet_count > lmp_limit(node_depth, improving) &&
                static_eval <= alpha) {
                // Late move pruning drops very late quiets once enough earlier
                // quiets have already been searched with no improvement.
                continue;
            }

            if (limits.components.history_pruning &&
                !pv_node &&
                !checked &&
                node_depth <= 6 &&
                quiet_move &&
                !ensure_gives_check() &&
                quiet_count > lmp_limit(node_depth, improving) &&
                combined_history <= history_prune_threshold(node_depth, improving)) {
                // History pruning removes quiets that are both late and historically bad.
                continue;
            }

#if MAGNUS_CAPTURE_OBS
            int obs_capture_order = -1;
            int obs_see_value = 0;
            int obs_capture_hist = 0;
            if (capture_move) {
                obs_capture_order = capture_search_order++;
                obs_see_value = move_see_value;
                obs_capture_hist = capture_history_score;
                const int base_rank = lookup_capture_base_rank(
                    move, capture_moves, capture_base_ranks, capture_count
                );
                record_main_capture_try(
                    obs_capture_order, obs_see_value, obs_capture_hist, base_rank
                );
            }
#endif

            const int move_base_depth = node_depth - 1;
            int move_extension = 0;
            std::size_t singular_bucket_index = 0;
#if MAGNUS_ENABLE_SINGULAR_EXTENSION
            if (limits.tt_trust_stage == TtTrustStage::A) {
                const bool legacy_basic_ok =
                    limits.components.singular_extension &&
                    move == tt_move &&
                    ply > 0 &&
                    !checked &&
                    !exclusion_search &&
                    !tablebase_resolved &&
                    node_depth >= SINGULAR_MIN_DEPTH &&
                    probe.hit &&
                    (tt_bound == memory::BOUND_LOWER || tt_bound == memory::BOUND_EXACT) &&
                    probe.data.depth >= node_depth - SINGULAR_TT_DEPTH_MARGIN &&
                    probed_tt_score >= beta - SINGULAR_SCORE_NEAR_BETA &&
                    !is_mate_window(probed_tt_score);

                if (legacy_basic_ok) {
                    const bool tactical_move =
                        capture_move || move_is_promotion(move) || ensure_gives_check();
                    const int singular_history =
                        quiet_move ? history_score : capture_history_score;
                    const SingularContext singular_ctx = make_singular_context(
                        pv_node,
                        tt_bound,
                        probe.data.depth,
                        node_depth,
                        probed_tt_score,
                        beta,
                        tactical_move,
                        singular_history
                    );
                    singular_bucket_index = singular_ctx.bucket_index;
                    record_singular_candidate(singular_ctx);
                    const bool path_limited = recent_singular_extensions(ply)
                        >= SINGULAR_RECENT_EXTENSION_LIMIT;
                    if (path_limited || singular_ctx.trust < singular_ctx.threshold) {
                        record_singular_skip(singular_ctx, path_limited);
                    } else {
#if MAGNUS_SEARCH_OBS
                        ++search_obs.singular_candidates;
#endif
                        const int singular_beta = probed_tt_score
                            - singular_margin(singular_ctx, node_depth);
                        const int singular_depth = std::max(1, (node_depth - 1) / 2);
                        const u64 singular_nodes_before = nodes;
                        const int singular_score = pvs(
                            pos,
                            singular_depth,
                            singular_beta - 1,
                            singular_beta,
                            ply,
                            false,
                            move
                        );
                        if (stopped)
                            return beta;
                        record_singular_test(
                            singular_ctx,
                            singular_score,
                            singular_beta,
                            beta,
                            nodes - singular_nodes_before
                        );
                        if (singular_score < singular_beta) {
                            move_extension = 1;
#if MAGNUS_SEARCH_OBS
                            ++search_obs.singular_extend1;
#endif
                            if (node_depth >= SINGULAR_DOUBLE_MIN_DEPTH
                                && singular_score < singular_beta - (
                                    SINGULAR_DOUBLE_MARGIN_BASE
                                    + SINGULAR_DOUBLE_MARGIN_PER_DEPTH * node_depth
                                )) {
                                move_extension = 2;
#if MAGNUS_SEARCH_OBS
                                ++search_obs.singular_extend2;
#endif
                                if (node_depth >= SINGULAR_TRIPLE_MIN_DEPTH
                                    && singular_score < singular_beta - (
                                        SINGULAR_TRIPLE_MARGIN_BASE
                                        + SINGULAR_TRIPLE_MARGIN_PER_DEPTH * node_depth
                                    )) {
                                    move_extension = 3;
#if MAGNUS_SEARCH_OBS
                                    ++search_obs.singular_extend3;
#endif
                                }
                            }
                        } else if (
                            singular_ctx.node_kind == SingularNodeKind::CutLike
                            && singular_score >= beta
                            && !is_mate_window(singular_score)
                        ) {
                            record_singular_decision(singular_ctx, 0, true);
                            return singular_score;
                        } else if (tt_bound == memory::BOUND_LOWER && !pv_node) {
                            move_extension = probed_tt_score >= beta ? -3 : -2;
                        }
                        record_singular_decision(
                            singular_ctx,
                            move_extension,
                            false
                        );
                    }
                }
                if (move_extension < 0) {
                    const int max_negative = -(node_depth / 3);
                    move_extension = std::max(move_extension, max_negative);
                }
            } else {
            const bool singular_basic_ok =
                limits.components.singular_extension &&
                move == tt_move &&
                ply > 0 &&
                !exclusion_search &&
                node_depth >= 6 + static_cast<int>(ss.tt_pv) &&
                probe.hit &&
                is_valid_score(probed_tt_score) &&
                !is_decisive(probed_tt_score) &&
                (tt_bound == memory::BOUND_LOWER || tt_bound == memory::BOUND_EXACT) &&
                probe.data.depth >= node_depth - 3 &&
                !is_shuffling(move, pos, ply);

            if (singular_basic_ok) {
                const bool tactical_move =
                    capture_move || move_is_promotion(move) || ensure_gives_check();
                const int singular_history =
                    quiet_move ? history_score : capture_history_score;
                const SingularContext singular_ctx = make_singular_context(
                    pv_node,
                    tt_bound,
                    probe.data.depth,
                    node_depth,
                    probed_tt_score,
                    beta,
                    tactical_move,
                    singular_history
                );
                singular_bucket_index = singular_ctx.bucket_index;
                record_singular_candidate(singular_ctx);

                const bool path_limited =
                    recent_singular_extensions(ply) >= SINGULAR_RECENT_EXTENSION_LIMIT;
                // Legacy trust/cost/path gates are observation-only in the
                // fixed-reference port; they never suppress verification.
                record_singular_skip(
                    singular_ctx,
                    path_limited,
                    probed_tt_score < beta - SINGULAR_SCORE_NEAR_BETA
                );
#if MAGNUS_SEARCH_OBS
                ++search_obs.singular_candidates;
#endif
                const int base_gap = 53
                    + 75 * static_cast<int>(ss.tt_pv && !pv_node);
                const int trust_adjustment =
                    limits.tt_trust_stage >= TtTrustStage::C
                    ? std::clamp(tt_trust_before / 512, -8, 8)
                    : 0;
                const int required_gap = std::max(1, base_gap - trust_adjustment);
                const int singular_beta = probed_tt_score
                    - required_gap * node_depth / 60;
                const int singular_depth = (node_depth - 1) / 2;
                const u64 singular_nodes_before = nodes;
                const int singular_score = pvs(
                    pos,
                    singular_depth,
                    singular_beta - 1,
                    singular_beta,
                    ply,
                    false,
                    move
                );
                if (stopped)
                    return beta;

                record_singular_test(
                    singular_ctx,
                    singular_score,
                    singular_beta,
                    beta,
                    nodes - singular_nodes_before
                );

                if (singular_score < singular_beta) {
                    const int corr_val_adj =
                        std::abs(correction) * 131072 / 230673;
                    const bool tt_capture = stockfish_capture_stage(move);
                    const int double_margin =
                        -4
                        + 199 * static_cast<int>(pv_node)
                        - 201 * static_cast<int>(!tt_capture)
                        - corr_val_adj
                        - 42 * static_cast<int>(ply > root_depth);
                    const int triple_margin =
                        73
                        + 302 * static_cast<int>(pv_node)
                        - 248 * static_cast<int>(!tt_capture)
                        + 90 * static_cast<int>(ss.tt_pv)
                        - corr_val_adj
                        - 50 * static_cast<int>(ply * 2 > root_depth * 3);
                    move_extension = 1
                        + static_cast<int>(singular_score < singular_beta - double_margin)
                        + static_cast<int>(singular_score < singular_beta - triple_margin);
                    ++node_depth;
#if MAGNUS_SEARCH_OBS
                    ++search_obs.singular_extend1;
                    search_obs.singular_extend2 += move_extension >= 2;
                    search_obs.singular_extend3 += move_extension >= 3;
#endif
                } else if (singular_score >= beta && !is_decisive(singular_score)) {
                    record_singular_decision(
                        singular_ctx,
                        0,
                        true
                    );
                    TtTrustBucketCounters& counters =
                        persistent.telemetry.bucket[tt_trust_bucket];
                    ++counters.multi_cut;
                    if (limits.tt_trust_stage >= TtTrustStage::D) {
                        const int malus = -std::min(
                            256 + 64 * node_depth,
                            1536
                        );
                        persistent.update_trust(pos, malus, tt_trust_bucket);
                    }
                    return singular_score;
                } else {
                    const bool cut_like = !pv_node && !move_is_none(tt_move);
                    move_extension = probed_tt_score >= beta
                        ? -3
                        : cut_like ? -2 : 0;
                }

                record_singular_decision(
                    singular_ctx,
                    move_extension,
                    false
                );
            }
            }
#endif
            if (lmr_quiet_candidate || lmr_capture_candidate)
                gives_check = ensure_gives_check();
            const int continuation_score = quiet_move
                ? history_tables.continuation_value_fast(
                        pos,
                        move,
                        prev2,
                        CONTINUATION_PLY2_SLOT
                    )
                    + history_tables.continuation_value_fast(
                        pos,
                        move,
                        prev4,
                        CONTINUATION_PLY4_SLOT
                    ) / 2
                    + history_tables.continuation_value_fast(
                        pos,
                        move,
                        prev8,
                        CONTINUATION_PLY8_SLOT
                    ) / 4
                : 0;
            const int pawn_hist_score = quiet_move
                ? history_tables.pawn_history_value_fast(pos, move) : 0;
            const int countermove_bonus = quiet_move
                ? history_tables.countermove_bonus_fast(pos, move, prev_move)
                : 0;
            const int move_see_bias_term = simple_capture
                ? history_tables.see_bias_value_fast(node_depth, move_see_value)
                : 0;
            LmrNodeContext lmr_node{};
            lmr_node.depth = node_depth;
            lmr_node.alpha = alpha;
            lmr_node.beta = beta;
            lmr_node.ply = ply;
            lmr_node.pv_node = pv_node;
            lmr_node.cut_node = !pv_node && !move_is_none(tt_move);
            lmr_node.all_node = !pv_node && move_is_none(tt_move);
            lmr_node.checked = checked;
            lmr_node.improving = improving;
            lmr_node.exclusion_search = exclusion_search;
            lmr_node.mate_window =
                (alpha > -VALUE_INF && is_mate_window(alpha)) ||
                (beta < VALUE_INF && is_mate_window(beta));
            lmr_node.tt_move_present = !move_is_none(tt_move);
            lmr_node.tt_move_is_capture =
                !move_is_none(tt_move) && move_is_capture(tt_move);
            lmr_node.static_eval = static_eval;
            lmr_node.move_extension = move_extension;
            lmr_node.next_ply_cutoff_count = search_stack[ply + 1].cutoff_count;
            lmr_node.parent_reduction_fp = ply > 0 ? search_stack[ply - 1].reduction_fp : 0;
            lmr_node.tt_depth = probe.hit ? probe.data.depth : 0;
            lmr_node.tt_bound = static_cast<int>(tt_bound);

            LmrMoveContext lmr_move{};
            lmr_move.move = move;
            lmr_move.move_index = move_index;
            lmr_move.reduction_index = quiet_move ? move_index : capture_index;
            lmr_move.is_tt_move = move == tt_move;
            lmr_move.quiet = quiet_move;
            lmr_move.capture = capture_move;
            lmr_move.simple_capture = simple_capture;
            lmr_move.bad_capture = bad_capture;
            lmr_move.gives_check = gives_check;
            lmr_move.recapture = is_recapture_move(move, ply);
            lmr_move.promotion = move_is_promotion(move);
            lmr_move.ordering_score = quiet_ordering_score;
            lmr_move.quiet_history_score = history_score + pawn_hist_score / 2;
            lmr_move.continuation_score = continuation_score;
            lmr_move.countermove_bonus = countermove_bonus;
            lmr_move.capture_history_score = capture_history_score;
            lmr_move.see_value = move_see_value;
            lmr_move.see_bias_term = move_see_bias_term;

            const LmrDecision lmr = limits.components.lmr
                ? decide_lmr(lmr_node, lmr_move)
                : LmrDecision{};
#if MAGNUS_LMR_OBS
            if (lmr_quiet_candidate || lmr_capture_candidate)
                record_lmr_considered(lmr, quiet_move, simple_capture, pv_node);
#endif
            ss.move_count = legal_count;
            ss.stat_score = lmr.stat_score;
            ss.reduction_fp = lmr.final_reduction_fp;
            ss.extension = std::max(0, move_extension);

            StateInfo st;
            const int move_alpha_before = alpha;
            const u64 move_nodes_before = nodes;
            set_move_history_context(pos, move, ply);
            make_search_move(pos, move, st);
            memory::tt_prefetch(mem.tt, memory::tt_key(pos, mem.tables));

            const int new_depth = stockfish_singular_child_depth(
                move_base_depth,
                move_extension
            );
            int score = 0;
            int searched_depth = new_depth;
            bool lmr_reduced_fail_high = false;
            if (!searched_first) {
                score = -pvs(pos, new_depth, -beta, -alpha, ply + 1, true);
                searched_first = true;
            } else {
                if (simple_capture) {
#if MAGNUS_CAPTURE_OBS
                    ++cap_obs.cap_lmr_late_simple_total;
                    if (node_depth >= 4 && simple_capture_count >= 3)
                        ++cap_obs.cap_lmr_eligible;
                    record_cap_lmr_considered(move_see_value, lmr.final_reduction);
#endif
                }

                if (lmr.eligible) {
                    const int reduced_depth = std::max(0, new_depth - lmr.final_reduction);
                    searched_depth = reduced_depth;
                    score = -pvs(pos, reduced_depth, -alpha - 1, -alpha, ply + 1, true);

                    if (score > alpha) {
                        lmr_reduced_fail_high = true;
#if MAGNUS_LMR_OBS
                        record_lmr_fail_high();
#endif
                        const int lmr_best_floor = std::max(best_score, alpha);
                        const int research_depth =
                            lmr_research_depth(lmr, new_depth, score, alpha, lmr_best_floor);
                        if (research_depth > reduced_depth) {
#if MAGNUS_LMR_OBS
                            record_lmr_research();
#endif
#if MAGNUS_SEARCHSTATS_OBS
                            ++stats.lmr_researches;
                            ++stats.lmr_by_ply[std::min(ply, MAX_PLY - 1)];
                            if (quiet_move)
                                ++stats.lmr_research_quiet;
                            if (capture_move)
                                ++stats.lmr_research_capture;
                            if (gives_check)
                                ++stats.lmr_research_check;
                            if (lmr.final_reduction <= 1)
                                ++stats.lmr_research_r1;
                            else if (lmr.final_reduction == 2)
                                ++stats.lmr_research_r2;
                            else
                                ++stats.lmr_research_r3_plus;
#endif
                            searched_depth = research_depth;
                            score = -pvs(
                                pos,
                                research_depth,
                                -alpha - 1,
                                -alpha,
                                ply + 1,
                                true
                            );
                        }
                    }
                } else {
                    score = -pvs(pos, new_depth, -alpha - 1, -alpha, ply + 1, true);
                }

#if MAGNUS_CAPTURE_OBS
                if (simple_capture && lmr.eligible && score > alpha)
                    record_cap_lmr_research(move_see_value, lmr.final_reduction);
#endif

                if (score > alpha && score < beta) {
                    // A null-window fail-high inside the PV must be confirmed by
                    // a full-window re-search before the score is trusted.
#if MAGNUS_SEARCHSTATS_OBS
                    ++stats.pvs_full_researches;
                    ++stats.pvs_full_by_ply[std::min(ply, MAX_PLY - 1)];
                    if (pv_node)
                        ++stats.pvs_full_pv_node;
                    else
                        ++stats.pvs_full_non_pv_node;
#endif
#if MAGNUS_LMR_OBS
                    if (lmr.eligible)
                        record_lmr_full_confirm();
#endif
                    score = -pvs(pos, searched_depth, -beta, -alpha, ply + 1, true);
                }
            }

            unmake_search_move(pos, move, st);
            if (stopped)
                return beta;

            if (move == tt_move) {
                tt_move_searched = true;
                tt_move_extension = move_extension;
            }

            record_singular_outcome(
                singular_bucket_index,
                move_extension,
                score,
                move_alpha_before,
                beta,
                nodes - move_nodes_before
            );

            if (lmr_reduced_fail_high && score > alpha && score < beta) {
                const int lmr_bonus_depth = std::max(1, node_depth / 4);
                if (capture_move)
                    history_tables.bonus_capture_fast(pos, move, lmr_bonus_depth);
                else {
                    history_tables.bonus_fast(pos, move, lmr_bonus_depth);
                    history_tables.bonus_pawn_history_fast(pos, move, lmr_bonus_depth);
                }
            }

            if (quiet_move)
                searched_quiets[searched_quiet_count++] = move;
#if MAGNUS_MOVEPICKER_OBS
            if (quiet_move)
                ++mp_obs.quiet_searched;
#endif
            if (capture_move) {
                searched_capture_see[searched_capture_count] = move_see_value;
                searched_captures[searched_capture_count++] = move;
            }

            if (score > best_score)
                best_score = score;

            if (score > alpha) {
                alpha = score;
                best_move = move;
                update_pv(ply, move);
                if (alpha >= beta) {
#if MAGNUS_MOVEPICKER_OBS
                    ++mp_obs.cutoffs_total;
                    ++mp_obs.cutoff_by_stage[static_cast<int>(stage_bucket)];
                    if (first_move_this && stage_bucket == MoveStageBucket::TT)
                        ++mp_obs.tt_first_cutoff;
                    if (first_good_capture_this)
                        ++mp_obs.first_good_capture_cutoff;
                    if (first_killer_this)
                        ++mp_obs.first_killer_cutoff;
                    if (first_quiet_this)
                        ++mp_obs.first_quiet_cutoff;
                    if (quiet_move && picker.last_quiet_in_skip_band())
                        ++mp_obs.late_quiet_fail_high;
                    if (quiet_move && picker.last_quiet_suppressed())
                        ++mp_obs.quiet_fail_high_after_skip_band;
#endif
#if MAGNUS_CAPTURE_OBS
                    if (capture_move)
                        record_main_capture_cutoff(
                            obs_capture_order, obs_see_value, obs_capture_hist
                        );
#endif
                    history_tables.reward_cutoff_fast(
                        pos,
                        move,
                        node_depth,
                        ply,
                        move_see_value,
                        prev_move,
                        prev2,
                        prev4,
                        prev8
                    );
                    // Near-miss bonus: first few quiets that almost cut get a small reward.
                    if (!move_is_capture(move) && searched_quiet_count > 1) {
                        const int near_bonus_depth = std::max(1, node_depth / 4);
                        for (int j = 0; j < std::min(3, searched_quiet_count); ++j)
                            history_tables.bonus_fast(pos, searched_quiets[j], near_bonus_depth);
                    }
                    const int fail_low_malus_depth = std::max(
                        1,
                        std::min(node_depth, searched_depth)
                    );
                    if (capture_move)
                        history_tables.penalize_captures_fast(
                            pos,
                            searched_captures,
                            searched_capture_count,
                            move,
                            fail_low_malus_depth
                        );
                    else {
                        history_tables.penalize_quiets_fast(
                            pos,
                            searched_quiets,
                            searched_quiet_count,
                            move,
                            fail_low_malus_depth
                        );
                        history_tables.penalize_continuation_quiets_fast(
                            pos,
                            searched_quiets,
                            searched_quiet_count,
                            move,
                            prev2,
                            prev4,
                            prev8,
                            fail_low_malus_depth
                        );
                    }
                    history_tables.penalize_see_bias_captures_fast(
                        searched_captures,
                        searched_capture_see,
                        searched_capture_count,
                        move,
                        fail_low_malus_depth
                    );
                    ++ss.cutoff_count;
                    cutoff = true;
                    break;
                }
                if (limits.tt_trust_stage >= TtTrustStage::B
                    && node_depth > 2
                    && node_depth < 14
                    && !is_decisive(score)) {
                    node_depth = stockfish_depth_after_alpha_improvement(
                        node_depth,
                        false
                    );
                }
            }
        }

#if MAGNUS_MOVEPICKER_OBS
        mp_obs.quiet_generated += static_cast<u64>(picker.quiet_generated());
        mp_obs.quiet_scored += static_cast<u64>(picker.quiet_scored());
        mp_obs.quiet_skipped_by_mp += static_cast<u64>(picker.quiet_suppressed());
#endif

        if (legal_count == 0) {
            const int score = exclusion_search
                ? alpha
                : (checked ? (-VALUE_MATE + ply) : draw_score(pos.side_to_move));
            if (!exclusion_search)
                save_tt(
                    pos,
                    limits.tt_trust_stage == TtTrustStage::A
                        ? legacy_tt_store_depth : node_depth,
                    ply,
                    score,
                    raw_eval,
                    0,
                    alpha0,
                    beta,
                    pv_node
                );
            return score;
        }

        if (!cutoff &&
            alpha > alpha0 &&
            best_move != 0) {
            const int hist_depth = std::max(1, node_depth - 1);
            if (move_is_capture(best_move)) {
                history_tables.bonus_capture_fast(pos, best_move, hist_depth);
                history_tables.penalize_captures_fast(pos, searched_captures, searched_capture_count, best_move, hist_depth);
            } else {
                history_tables.bonus_fast(pos, best_move, hist_depth);
                history_tables.bonus_pawn_history_fast(pos, best_move, hist_depth);
                history_tables.penalize_quiets_fast(pos, searched_quiets, searched_quiet_count, best_move, hist_depth);
                history_tables.bonus_continuation_fast(
                    pos,
                    prev2,
                    best_move,
                    hist_depth,
                    CONTINUATION_PLY2_SLOT
                );
                history_tables.bonus_continuation_fast(
                    pos,
                    prev4,
                    best_move,
                    std::max(1, hist_depth / 2),
                    CONTINUATION_PLY4_SLOT
                );
                history_tables.bonus_continuation_fast(
                    pos,
                    prev8,
                    best_move,
                    std::max(1, hist_depth / 4),
                    CONTINUATION_PLY8_SLOT
                );
                history_tables.penalize_continuation_quiets_fast(
                    pos,
                    searched_quiets,
                    searched_quiet_count,
                    best_move,
                    prev2,
                    prev4,
                    prev8,
                    hist_depth
                );
            }
        }
        if (!cutoff && searched_capture_count > 0) {
            const int fail_depth = std::max(1, node_depth - 1);
            history_tables.penalize_see_bias_captures_fast(
                searched_captures,
                searched_capture_see,
                searched_capture_count,
                Move(0),
                fail_depth
            );
        }

        if (limits.components.correction_history &&
            !exclusion_search &&
            !checked &&
            !cutoff &&
            best_move != 0 &&
            alpha > alpha0 &&
            alpha < beta &&
            !is_mate_window(alpha)) {
            update_correction_history(side, eval_info.keys, base_eval, alpha, node_depth);
        }

        int node_value = alpha;
        if (limits.tt_trust_stage >= TtTrustStage::B
            && best_score >= beta
            && !is_decisive(best_score)
            && !is_decisive(alpha)) {
            node_value = stockfish_fail_high_softbound(
                best_score,
                beta,
                node_depth
            );
        }
        node_value = std::min(node_value, tablebase_max_score);

        if (tt_move_searched
            && ply > 0
            && !exclusion_search
            && !is_decisive(node_value)) {
            TtTrustBucketCounters& counters =
                persistent.telemetry.bucket[tt_trust_bucket];
            ++counters.searched;
            counters.depth_sum += static_cast<u64>(std::max(0, node_depth));
            counters.single_extension += tt_move_extension >= 1;
            counters.double_extension += tt_move_extension >= 2;
            counters.triple_extension += tt_move_extension >= 3;
            if (move_is_none(best_move)) {
                ++counters.fail_low;
            } else {
                const int magnitude = std::clamp(
                    128 + 40 * node_depth,
                    256,
                    768
                );
                if (best_move == tt_move) {
                    ++counters.tt_best;
                    persistent.update_trust(pos, magnitude, tt_trust_bucket);
                } else {
                    ++counters.other_best;
                    persistent.update_trust(
                        pos,
                        -(magnitude * 9 / 8),
                        tt_trust_bucket
                    );
                }
            }
        }

        if (!exclusion_search)
            save_tt(
                pos,
                limits.tt_trust_stage == TtTrustStage::A
                    ? legacy_tt_store_depth : node_depth,
                ply,
                node_value,
                raw_eval,
                best_move,
                alpha0,
                beta,
                pv_node
            );
        return node_value;
    }

    struct RootMoveResult {
        int score = -VALUE_INF;
        bool improved_alpha = false;
        Move pv[MAX_PLY]{};
        int pv_length = 0;
    };

    [[nodiscard]] RootMoveResult search_root_move(
        const Position& root,
        Move move,
        int depth,
        int alpha,
        int beta,
        bool full_window,
        bool force_pv_capture = false
    ) noexcept {
        const int alpha_before = alpha;
        Position local_root = root;

        pv_length[1] = 0;
        StateInfo st;
        set_move_history_context(local_root, move, 0);
        make_search_move(local_root, move, st);
        memory::tt_prefetch(mem.tt, memory::tt_key(local_root, mem.tables));

        int score = 0;
        if (full_window) {
            score = -pvs(local_root, depth - 1, -beta, -alpha, 1, true);
        } else {
            score = -pvs(local_root, depth - 1, -alpha - 1, -alpha, 1, true);
            if (!stopped && score > alpha) {
#if MAGNUS_SEARCHSTATS_OBS
                ++stats.root_pvs_researches;
#endif
                score = -pvs(local_root, depth - 1, -beta, -alpha, 1, true);
            }
        }

        unmake_search_move(local_root, move, st);

        RootMoveResult result;
        if (stopped) {
            result.score = alpha_before;
            return result;
        }

        result.score = score;
        result.improved_alpha = score > alpha_before;
        if (force_pv_capture || result.improved_alpha)
            update_pv(0, move);
        result.pv_length = pv_length[0];
        if (result.pv_length > 0) {
            std::memcpy(
                result.pv,
                pv_table[0],
                static_cast<std::size_t>(result.pv_length) * sizeof(Move)
            );
        }
        if (result.pv_length <= 0 || result.pv[0] != move) {
            result.pv[0] = move;
            result.pv_length = 1;
        }
        return result;
    }

    [[nodiscard]] SearchResult search_root(
        const Position& root,
        int depth,
        Move hint_move,
        int alpha,
        int beta
    ) noexcept {
        // Root search mirrors PVS, but it also keeps the best move/result for UCI output.
        SearchResult result{};
        result.depth = depth;
        update_seldepth(0);
        rep_keys[0] = root.key;

        MoveList list{};
        generate_legal(root, mem, list);

        if (limits.root_move_count > 0) {
            MoveList filtered;
            filtered.size = 0;
            for (int i = 0; i < list.size; ++i) {
                const Move move = list.moves[i];
                bool allowed = false;
                for (int j = 0; j < limits.root_move_count; ++j) {
                    if (limits.root_moves[j] == move) {
                        allowed = true;
                        break;
                    }
                }
                if (allowed)
                    filtered.moves[filtered.size++] = move;
            }
            list = filtered;
        }

        const memory::TTProbe probe = memory::tt_probe(mem.tt, memory::tt_key(root, mem.tables));
        const Move tt_move = tt_move_from_probe(probe);
        const Move root_hint = move_is_none(tt_move) ? hint_move : tt_move;
        const bool checked = in_check(root);
        const StaticEvalInfo eval_info = resolve_static_eval(root, probe, 0, checked, false);
        const int raw_eval = eval_info.raw;
        const int base_eval = eval_info.base;
        const int alpha0 = alpha;

        ScoredMoveList scored;
        score_moves(root, list, scored, root_hint, 0, depth);
        apply_msv_root_order(root, scored, depth);
        apply_startpos_e4_root_order(root, scored);
        int best_score = -VALUE_INF;
        int best_selection_score = -VALUE_INF;
        result.best_move = 0;

        for (int i = 0; i < scored.size; ++i) {
            if (hit_hard_limit())
                break;

            const Move move = pick_next(scored, i);
#if MAGNUS_SEARCHSTATS_OBS
            ++stats.root_moves_searched;
#endif
            RootMsvActiveGuard msv_active(limits, move);
            const u64 move_nodes_before = nodes;
            const RootMoveResult move_result =
                search_root_move(root, move, depth, alpha, beta, i == 0);
            if (stopped)
                break;

            const int score = move_result.score;
            record_root_effort(
                move,
                nodes >= move_nodes_before ? nodes - move_nodes_before : 0,
                score
            );
            const int selection_score =
                score + root_opening_preference_bonus(limits, root, move);

            if (selection_score > best_selection_score) {
                if (!move_is_none(result.best_move) && result.best_move != move)
                    signal_best_move_change();
                best_selection_score = selection_score;
                best_score = score;
                result.best_move = move;
                result.pv_length = move_result.pv_length;
                if (result.pv_length > 0) {
                    std::memcpy(
                        result.pv,
                        move_result.pv,
                        static_cast<std::size_t>(result.pv_length) * sizeof(Move)
                    );
                }
            }

            if (move_result.improved_alpha) {
                alpha = score;
                if (alpha >= beta)
                    break;
            }
        }

        if (result.best_move == 0) {
            result.score = checked ? -VALUE_MATE : draw_score(root.side_to_move);
            result.best_move = 0;
            result.seldepth = 0;
            return result;
        }

        if (best_score == -VALUE_INF)
            best_score = alpha;

        if (!stopped &&
            limits.components.correction_history &&
            !checked &&
            !is_mate_window(best_score) &&
            best_score > alpha0 &&
            best_score < beta) {
            update_correction_history(
                static_cast<Color>(root.side_to_move),
                eval_info.keys,
                base_eval,
                best_score,
                depth
            );
        }

        result.score = best_score;
        result.nodes = nodes;
        result.tb_hits = tb_hits;
        result.seldepth = seldepth;
        if (result.pv_length > 0 && result.pv[0] == result.best_move) {
            std::memcpy(
                pv_table[0],
                result.pv,
                static_cast<std::size_t>(result.pv_length) * sizeof(Move)
            );
            pv_length[0] = result.pv_length;
        }
        save_tt(root, depth, 0, best_score, raw_eval, result.best_move, alpha0, beta, true);
        return result;
    }

    [[nodiscard]] SearchResult search_root(
        const Position& root,
        std::vector<RootLine>& root_lines,
        int pv_idx,
        int depth,
        Move hint_move,
        int alpha,
        int beta
    ) noexcept {
        // Root search mirrors PVS, but it also keeps the best move/result for UCI output.
        SearchResult result{};
        result.depth = depth;
        update_seldepth(0);
        rep_keys[0] = root.key;

        const memory::TTProbe probe = memory::tt_probe(mem.tt, memory::tt_key(root, mem.tables));
        const Move tt_move = tt_move_from_probe(probe);
        const bool checked = in_check(root);
        const StaticEvalInfo eval_info = resolve_static_eval(root, probe, 0, checked, false);
        const int raw_eval = eval_info.raw;
        const int base_eval = eval_info.base;
        const int alpha0 = alpha;

        if (pv_idx < 0 || pv_idx >= static_cast<int>(root_lines.size())) {
            result.score = checked ? -VALUE_MATE : draw_score(root.side_to_move);
            result.seldepth = 0;
            return result;
        }

        Move root_hint = root_lines[static_cast<std::size_t>(pv_idx)].move;
        if (pv_idx == 0) {
            const Move preferred = move_is_none(tt_move) ? hint_move : tt_move;
            if (root_line_index(root_lines, pv_idx, preferred) >= 0)
                root_hint = preferred;
        }

        MoveList list{};
        for (int i = pv_idx; i < static_cast<int>(root_lines.size()); ++i)
            list.moves[list.size++] = root_lines[static_cast<std::size_t>(i)].move;

        ScoredMoveList scored;
        score_moves(root, list, scored, root_hint, 0, depth);
        apply_msv_root_order(root, scored, depth);
        apply_startpos_e4_root_order(root, scored);
        int best_score = -VALUE_INF;
        int best_selection_score = -VALUE_INF;
        result.best_move = 0;

        for (int i = 0; i < scored.size; ++i) {
            if (hit_hard_limit())
                break;

            const Move move = pick_next(scored, i);
            const int line_index = root_line_index(root_lines, pv_idx, move);
            if (line_index < 0)
                continue;

#if MAGNUS_SEARCHSTATS_OBS
            ++stats.root_moves_searched;
#endif
            RootMsvActiveGuard msv_active(limits, move);
            const u64 move_nodes_before = nodes;
            const RootMoveResult move_result =
                search_root_move(root, move, depth, alpha, beta, i == 0, i == 0);
            if (stopped)
                break;

            const int score = move_result.score;
            record_root_effort(
                move,
                nodes >= move_nodes_before ? nodes - move_nodes_before : 0,
                score
            );
            RootLine& line = root_lines[static_cast<std::size_t>(line_index)];
            line.searched = true;
            line.depth = depth;
            line.score = score;
            line.selection_score =
                score + root_opening_preference_bonus(limits, root, move);
            line.bound = score_bound_from_window(score, alpha0, beta);
            line.seldepth = seldepth;
            line.pv_length = move_result.pv_length;
            std::memcpy(
                line.pv,
                move_result.pv,
                static_cast<std::size_t>(line.pv_length) * sizeof(Move)
            );

            const int selection_score =
                score + root_opening_preference_bonus(limits, root, move);
            if (selection_score > best_selection_score) {
                if (pv_idx == 0 &&
                    !move_is_none(result.best_move) &&
                    result.best_move != move) {
                    signal_best_move_change();
                }
                best_selection_score = selection_score;
                best_score = score;
                result.best_move = move;
            }

            if (move_result.improved_alpha) {
                alpha = score;
                if (alpha >= beta)
                    break;
            }
        }

        if (result.best_move == 0) {
            result.score = checked ? -VALUE_MATE : draw_score(root.side_to_move);
            result.best_move = 0;
            result.seldepth = 0;
            return result;
        }

        if (best_score == -VALUE_INF)
            best_score = alpha;

        stable_sort_root_lines(root_lines, pv_idx);
        promote_startpos_e4_root_line(
            limits,
            root,
            root_lines,
            pv_idx,
            static_cast<int>(root_lines.size())
        );
        RootLine& selected = root_lines[static_cast<std::size_t>(pv_idx)];
        if (selected.searched) {
            result.best_move = selected.move;
            best_score = selected.score;
        }

        if (!stopped &&
            limits.components.correction_history &&
            pv_idx == 0 &&
            !checked &&
            !is_mate_window(best_score) &&
            best_score > alpha0 &&
            best_score < beta) {
            update_correction_history(
                static_cast<Color>(root.side_to_move),
                eval_info.keys,
                base_eval,
                best_score,
                depth
            );
        }

        result.score = best_score;
        result.nodes = nodes;
        result.tb_hits = tb_hits;
        result.seldepth = seldepth;
        if (pv_idx == 0)
            save_tt(root, depth, 0, best_score, raw_eval, result.best_move, alpha0, beta, true);
        return result;
    }

    [[nodiscard]] bool root_pv_has_legal_ponder_move(
        const Position& root,
        Move best_move
    ) const noexcept {
        if (move_is_none(best_move) ||
            pv_length[0] < 2 ||
            pv_table[0][0] != best_move) {
            return false;
        }

        Position after_best{};
        position_copy_without_accumulators(after_best, root);
        do_move_copy(after_best, best_move, mem.tables);

        MoveList replies{};
        generate_legal(after_best, mem, replies);
        const Move ponder_move = pv_table[0][1];
        for (int i = 0; i < replies.size; ++i) {
            if (replies.moves[i] == ponder_move)
                return true;
        }

        return false;
    }

    [[nodiscard]] u64 recover_ponder_pv_full_window(
        const Position& root,
        SearchResult& result,
        int depth
    ) noexcept {
        if (!limits.recover_ponder_pv ||
            stopped ||
            depth < 2 ||
            move_is_none(result.best_move) ||
            root_pv_has_legal_ponder_move(root, result.best_move)) {
            return 0;
        }

        const u64 nodes_before = nodes;
        const RootMoveResult repair = search_root_move(
            root,
            result.best_move,
            depth,
            -VALUE_INF,
            VALUE_INF,
            true
        );
        const u64 node_delta = nodes >= nodes_before ? nodes - nodes_before : 0;

        if (!stopped) {
            result.score = repair.score;
            result.nodes = nodes;
            result.tb_hits = tb_hits;
            result.seldepth = seldepth;
        }

        return node_delta;
    }
};

struct IterationTimeState {
    std::array<int, 4> iter_values{};
    Move last_best_move = 0;
    int last_best_move_depth = 0;
    double previous_time_reduction = 0.85;
    double current_time_reduction = 1.0;
    double total_best_move_changes = 0.0;
    int root_legal_count = 0;
    int search_again_counter = 0;
    int last_depth_time_ms = 0;   // wall time consumed by the previous completed depth
    u64 last_depth_nodes = 0;     // nodes consumed by the previous completed depth
#if MAGNUS_SEARCHSTATS_OBS
    double last_falling_eval = 1.0;
    double last_instability = 1.0;
    double last_effort_factor = 1.0;
    double last_total_time = 0.0;
    int last_effective_depth = 0;
    bool last_stop_on_ponderhit = false;
    bool last_increase_depth = true;
#endif
};

void initialize_iteration_time_state(
    const SearchLimits& limits,
    IterationTimeState& state
) noexcept {
    const int initial_score =
        limits.time_state != nullptr &&
        limits.time_state->previous_score != VALUE_NONE
            ? limits.time_state->previous_score
            : 0;
    state.iter_values.fill(initial_score);
    if (limits.time_state != nullptr)
        state.previous_time_reduction =
            limits.time_state->previous_time_reduction;
}

[[nodiscard]] inline bool use_stockfish_style_time_management(
    const SearchLimits& limits
) noexcept {
    return limits.use_time_management &&
           !limits.infinite &&
           limits.soft_time_ms > 0 &&
           limits.hard_time_ms > 0;
}

[[nodiscard]] bool should_stop_after_iteration(
    Searcher& searcher,
    IterationTimeState& time_state,
    const SearchResult& current,
    int depth,
    int effective_depth,
    u64 total_nodes
) noexcept {
    const int iter_slot = depth & 3;
    const int older_score =
        depth > 4 ? time_state.iter_values[iter_slot] : current.score;
    const int previous_average_score =
        searcher.limits.time_state != nullptr &&
        searcher.limits.time_state->previous_average_score != VALUE_NONE
            ? searcher.limits.time_state->previous_average_score
            : current.score;

    if (!move_is_none(current.best_move) &&
        current.best_move != time_state.last_best_move) {
        time_state.last_best_move = current.best_move;
        time_state.last_best_move_depth = depth;
    }

    bool should_stop = false;

    // Dynamic signals continue to evolve while pondering. Once the useful
    // budget is consumed, defer the stop until ponderhit.
    if (searcher.stop_on_ponderhit && !searcher.pondering_active()) {
        searcher.stopped = true;
        searcher.hard_stop = true;
        return true;
    }

    // Fixed-time grace: simple depth-boundary stop for go movetime.
    // When the sf-style clock is disabled but a hard limit exists, avoid
    // launching a new depth with essentially zero remaining budget
    // (the hard-limit poll will kill it before a single move is searched).
    if (!use_stockfish_style_time_management(searcher.limits) &&
        searcher.limits.hard_time_ms > 0 &&
        time_state.last_depth_time_ms > 0) {
        const int elapsed = searcher.timed_elapsed_ms();
        const int remaining = searcher.limits.hard_time_ms - elapsed;
        // Only skip the next depth if the remaining budget is less than
        // 3 % of the current hard limit — i.e. practically zero.
        if (remaining < searcher.limits.hard_time_ms / 33)
            should_stop = true;
    }

    if (use_stockfish_style_time_management(searcher.limits)) {
        time_state.total_best_move_changes /= 2.0;
        if (searcher.limits.time_signals != nullptr) {
            time_state.total_best_move_changes +=
                searcher.limits.time_signals->best_move_changes.exchange(
                    0,
                    std::memory_order_relaxed
                );
        }

        // SPSA-tuned time management: falling-eval scaling, sigmoid depth factor,
        // and best-move instability combine to adjust soft-time allocation.
        double falling_eval =
            (11.85
             + 2.24 * double(previous_average_score - current.score)
             + 0.93 * double(older_score - current.score))
            / 100.0;
        falling_eval = std::clamp(falling_eval, 0.57, 1.70);

        constexpr double k = 0.51;
        const double center = double(time_state.last_best_move_depth) + 12.15;
        const double time_reduction =
            0.66 + 0.85 / (0.98 + std::exp(-k * (double(effective_depth) - center)));
        const double reduction =
            (1.43 + time_state.previous_time_reduction) / (2.28 * time_reduction);
        const double best_move_instability =
            1.02
            + 2.14 * time_state.total_best_move_changes
                / double(std::max(1, searcher.limits.thread_count));

        const u64 best_effort = searcher.root_effort(current.best_move);
        const u64 effort_per_100k =
            best_effort * 100000ULL / std::max<u64>(1, total_nodes);
        const double effort_factor = effort_per_100k >= 93340ULL ? 0.76 : 1.0;

        double total_time =
            double(searcher.limits.soft_time_ms)
            * falling_eval
            * reduction
            * best_move_instability
            * effort_factor;

        if (time_state.root_legal_count == 1)
            total_time = std::min(502.0, total_time);

        const double stop_time =
            std::min(total_time, double(searcher.limits.hard_time_ms));
        const double elapsed = double(searcher.timed_elapsed_ms());
        if (elapsed > stop_time) {
            if (searcher.pondering_active())
                searcher.stop_on_ponderhit = true;
            else
                should_stop = true;
        } else if (searcher.limits.time_signals != nullptr) {
            searcher.limits.time_signals->increase_depth.store(
                searcher.pondering_active() || elapsed <= total_time * 0.50,
                std::memory_order_relaxed
            );
        }
        time_state.current_time_reduction = time_reduction;
#if MAGNUS_SEARCHSTATS_OBS
        time_state.last_falling_eval = falling_eval;
        time_state.last_instability = best_move_instability;
        time_state.last_effort_factor = effort_factor;
        time_state.last_total_time = total_time;
        time_state.last_effective_depth = effective_depth;
        time_state.last_stop_on_ponderhit = searcher.stop_on_ponderhit;
        time_state.last_increase_depth =
            searcher.limits.time_signals == nullptr ||
            searcher.limits.time_signals->increase_depth.load(
                std::memory_order_relaxed
            );
#endif
    }

    time_state.iter_values[iter_slot] = current.score;
    return should_stop;
}

#if MAGNUS_SEARCHSTATS_OBS
void emit_searchstat_ply_top(
    std::ostream& out,
    const char* label,
    const std::array<u64, MAX_PLY>& current,
    const std::array<u64, MAX_PLY>& previous
) {
    struct Bucket {
        int ply = 0;
        u64 count = 0;
    };

    std::array<Bucket, 4> top{};
    for (int ply = 0; ply < MAX_PLY; ++ply) {
        const u64 count = current[static_cast<std::size_t>(ply)]
            - previous[static_cast<std::size_t>(ply)];
        if (count == 0)
            continue;

        for (int i = 0; i < static_cast<int>(top.size()); ++i) {
            if (count <= top[static_cast<std::size_t>(i)].count)
                continue;

            for (int j = static_cast<int>(top.size()) - 1; j > i; --j)
                top[static_cast<std::size_t>(j)] =
                    top[static_cast<std::size_t>(j - 1)];
            top[static_cast<std::size_t>(i)] = Bucket{ply, count};
            break;
        }
    }

    out << ' ' << label;
    bool any = false;
    for (const Bucket& bucket : top) {
        if (bucket.count == 0)
            continue;

        any = true;
        out << ' ' << bucket.ply << ':' << bucket.count;
    }
    if (!any)
        out << " none";
}

void emit_searchstats_line(
    std::ostream& out,
    int depth,
    bool partial,
    u64 depth_nodes,
    int aspiration_fail_low,
    int aspiration_fail_high,
    const IterationTimeState& time_state,
    const Searcher::SearchStats& before,
    const Searcher::SearchStats& after
) {
    out << "info string searchstats depth " << depth
        << " partial " << (partial ? 1 : 0)
        << " depthnodes " << depth_nodes
        << " aspiration_fail_low " << aspiration_fail_low
        << " aspiration_fail_high " << aspiration_fail_high
        << " tm_falling " << time_state.last_falling_eval
        << " tm_instability " << time_state.last_instability
        << " tm_effort " << time_state.last_effort_factor
        << " tm_total_ms " << time_state.last_total_time
        << " tm_effective_depth " << time_state.last_effective_depth
        << " tm_increase_depth " << (time_state.last_increase_depth ? 1 : 0)
        << " tm_stop_on_ponderhit " << (time_state.last_stop_on_ponderhit ? 1 : 0)
        << " root_moves_searched "
        << (after.root_moves_searched - before.root_moves_searched)
        << " root_pvs_researches "
        << (after.root_pvs_researches - before.root_pvs_researches)
        << " pvs_full_researches "
        << (after.pvs_full_researches - before.pvs_full_researches)
        << " pvs_pv "
        << (after.pvs_full_pv_node - before.pvs_full_pv_node)
        << " pvs_nonpv "
        << (after.pvs_full_non_pv_node - before.pvs_full_non_pv_node)
        << " lmr_researches "
        << (after.lmr_researches - before.lmr_researches)
        << " lmr_quiet "
        << (after.lmr_research_quiet - before.lmr_research_quiet)
        << " lmr_capture "
        << (after.lmr_research_capture - before.lmr_research_capture)
        << " lmr_check "
        << (after.lmr_research_check - before.lmr_research_check)
        << " lmr_r1 "
        << (after.lmr_research_r1 - before.lmr_research_r1)
        << " lmr_r2 "
        << (after.lmr_research_r2 - before.lmr_research_r2)
        << " lmr_r3plus "
        << (after.lmr_research_r3_plus - before.lmr_research_r3_plus);
    emit_searchstat_ply_top(
        out,
        "pvs_by_ply",
        after.pvs_full_by_ply,
        before.pvs_full_by_ply
    );
    emit_searchstat_ply_top(
        out,
        "lmr_by_ply",
        after.lmr_by_ply,
        before.lmr_by_ply
    );
    out << '\n';
}
#endif

struct RootWorker {
    SearchLimits limits;
    Searcher searcher;

    explicit RootWorker(
        memory::Memory& mem,
        WorkerPersistentState& persistent,
        const SearchLimits& base_limits
    )
        : limits(base_limits), searcher(mem, persistent, limits) {}
};


void reset_searcher_iteration(
    Searcher& searcher,
    Searcher::clock::time_point start_time,
    u64 base_nodes
) noexcept {
    searcher.nodes = 0;
    searcher.base_nodes = base_nodes;
    searcher.published_nodes = 0;
    searcher.tb_hits = 0;
    searcher.published_tb_hits = 0;
    searcher.seldepth = 0;
    searcher.nmp_min_ply = 0;
    searcher.singular_verification_nodes = 0;
    searcher.stopped = false;
    searcher.hard_stop = false;
    searcher.start_time = start_time;
    std::fill(std::begin(searcher.pv_length), std::end(searcher.pv_length), 0);
    std::fill(std::begin(searcher.move_stack), std::end(searcher.move_stack), Move(0));
    std::fill(
        std::begin(searcher.search_stack),
        std::end(searcher.search_stack),
        SearchStackEntry{}
    );
    std::fill(
        std::begin(searcher.static_eval_valid),
        std::end(searcher.static_eval_valid),
        false
    );
}


[[nodiscard]] bool use_root_aspiration(
    const SearchLimits& limits,
    int depth
) noexcept {
    return limits.components.aspiration && depth >= 2;
}

} // namespace

std::string move_to_uci(Move m) {
    if (move_is_none(m))
        return "0000";

    std::string s;
    s.reserve(5);
    s.push_back(static_cast<char>('a' + file_of(from_sq(m))));
    s.push_back(static_cast<char>('1' + rank_of(from_sq(m))));
    s.push_back(static_cast<char>('a' + file_of(to_sq(m))));
    s.push_back(static_cast<char>('1' + rank_of(to_sq(m))));

    if (move_is_promotion(m)) {
        switch (promo_piece(m)) {
            case KNIGHT: s.push_back('n'); break;
            case BISHOP: s.push_back('b'); break;
            case ROOK:   s.push_back('r'); break;
            case QUEEN:  s.push_back('q'); break;
            default: break;
        }
    }

    return s;
}

// Extracted common iterative-deepening helpers shared by single and lazy-SMP paths.
namespace {

struct IterativeWorkerResult {
    SearchResult best{};
    Move pv[MAX_PLY]{};
    int pv_length = 0;
    memory::Bound score_bound = memory::BOUND_EXACT;
    std::vector<RootLine> lines{};
    int best_average_score = VALUE_NONE;
    double time_reduction = 0.85;
};

[[nodiscard]] bool has_worker_best(
    const IterativeWorkerResult& result
) noexcept {
    return result.best.depth > 0 && !move_is_none(result.best.best_move);
}

void remember_exact_result(
    IterativeWorkerResult& exact,
    const IterativeWorkerResult& candidate
) {
    if (candidate.score_bound != memory::BOUND_EXACT ||
        !has_worker_best(candidate)) {
        return;
    }

    if (!has_worker_best(exact) ||
        candidate.best.depth >= exact.best.depth) {
        exact = candidate;
    }
}

void prefer_exact_result(
    IterativeWorkerResult& result,
    const IterativeWorkerResult& exact
) {
    if ((result.score_bound == memory::BOUND_EXACT && has_worker_best(result)) ||
        !has_worker_best(exact)) {
        return;
    }

    result = exact;
}

[[nodiscard]] bool pv_move_is_legal(
    const Position& pos,
    const memory::Memory& mem,
    Move move
) noexcept {
    if (move_is_none(move))
        return false;

    MoveList legal{};
    generate_legal(pos, mem, legal);
    for (int i = 0; i < legal.size; ++i)
        if (legal.moves[i] == move)
            return true;

    return false;
}

[[nodiscard]] int copy_pv_history_keys(
    const SearchLimits& limits,
    Key* keys,
    int capacity
) noexcept {
    if (keys == nullptr || capacity <= 0)
        return 0;

    const int count = std::clamp(
        limits.game_history_count,
        0,
        std::min(MAX_GAME_HISTORY, capacity)
    );
    for (int i = 0; i < count; ++i)
        keys[i] = limits.game_history_keys[i];
    return count;
}

bool append_pv_previous_key(
    Key* keys,
    int& key_count,
    int capacity,
    Key key
) noexcept {
    if (keys == nullptr || key_count < 0 || key_count >= capacity)
        return false;

    keys[key_count++] = key;
    return true;
}

[[nodiscard]] int pv_recent_key_occurrences(
    const Key* keys,
    int key_count,
    int halfmove_clock,
    Key key
) noexcept {
    if (keys == nullptr || key_count <= 0 || halfmove_clock <= 0)
        return 0;

    int matches = 0;
    const int recent_count = std::min(key_count, halfmove_clock);
    const int begin = key_count - recent_count;
    for (int i = begin; i < key_count; ++i)
        matches += keys[i] == key;
    return matches;
}

[[nodiscard]] bool pv_repetition_terminal(
    const Position& pos,
    const Key* previous_keys,
    int previous_key_count
) noexcept {
    return 1 + pv_recent_key_occurrences(
        previous_keys,
        previous_key_count,
        pos.halfmove_clock,
        pos.key
    ) >= 3;
}

[[nodiscard]] bool pv_position_is_terminal(
    const Position& pos,
    const memory::Memory& mem,
    const Key* previous_keys,
    int previous_key_count
) noexcept {
    if (pos.halfmove_clock >= 100)
        return true;

    if (pv_repetition_terminal(pos, previous_keys, previous_key_count))
        return true;

    MoveList legal{};
    generate_legal(pos, mem, legal);
    return legal.size == 0;
}

[[nodiscard]] int reconstruct_tt_pv(
    Position& root,
    memory::Memory& mem,
    const SearchLimits& limits,
    Move first_move,
    Move* out,
    int max_len
) noexcept {
    if (move_is_none(first_move) || out == nullptr || max_len <= 0)
        return 0;

    const int length_limit = std::clamp(max_len, 0, MAX_PLY);
    std::array<StateInfo, MAX_PLY> states{};
    std::array<Move, MAX_PLY> made_moves{};
    std::array<Key, MAX_PLY + MAX_GAME_HISTORY + 1> keys{};
    int key_count = copy_pv_history_keys(
        limits,
        keys.data(),
        static_cast<int>(keys.size())
    );
    if (pv_position_is_terminal(root, mem, keys.data(), key_count))
        return 0;
    append_pv_previous_key(
        keys.data(),
        key_count,
        static_cast<int>(keys.size()),
        root.key
    );

    int pv_length = 0;
    int made_count = 0;
    while (pv_length < length_limit) {
        const Move move = pv_length == 0
            ? first_move
            : [&]() noexcept {
                const memory::TTProbe probe = memory::tt_probe(mem.tt, memory::tt_key(root, mem.tables));
                return probe.hit
                    ? static_cast<Move>(probe.data.move)
                    : Move(0);
            }();

        if (!pv_move_is_legal(root, mem, move))
            break;

        const int made_index = made_count++;
        made_moves[static_cast<std::size_t>(made_index)] = move;
        make_move(
            root,
            move,
            mem.tables,
            states[static_cast<std::size_t>(made_index)]
        );

        const bool repetition_terminal =
            pv_repetition_terminal(root, keys.data(), key_count);
        if (repetition_terminal && pv_length > 0)
            break;

        out[pv_length] = move;
        ++pv_length;

        if (repetition_terminal ||
            pv_position_is_terminal(root, mem, keys.data(), key_count)) {
            break;
        }

        append_pv_previous_key(
            keys.data(),
            key_count,
            static_cast<int>(keys.size()),
            root.key
        );
    }

    for (int i = made_count - 1; i >= 0; --i) {
        unmake_move(
            root,
            made_moves[static_cast<std::size_t>(i)],
            mem.tables,
            states[static_cast<std::size_t>(i)]
        );
    }

    return pv_length;
}

[[nodiscard]] bool pv_contains_key(
    const std::array<Key, MAX_PLY + MAX_GAME_HISTORY + 2>& keys,
    int key_count,
    Key key
) noexcept {
    for (int i = 0; i < key_count; ++i)
        if (keys[static_cast<std::size_t>(i)] == key)
            return true;
    return false;
}

[[nodiscard]] int pv_key_occurrences(
    const std::array<Key, MAX_PLY + MAX_GAME_HISTORY + 2>& keys,
    int key_count,
    Key key
) noexcept {
    int count = 0;
    for (int i = 0; i < key_count; ++i)
        count += keys[static_cast<std::size_t>(i)] == key;
    return count;
}

[[nodiscard]] int validate_and_sanitize_pv(
    const Position& root,
    memory::Memory& mem,
    const SearchLimits& limits,
    Move* pv,
    int pv_length,
    std::ostream* diagnostic_out = nullptr
) noexcept {
    if (pv_length <= 0)
        return 0;

    Position pos{};
    position_copy_without_accumulators(pos, root);

    std::array<Key, MAX_PLY + MAX_GAME_HISTORY + 1> keys{};
    int key_count = copy_pv_history_keys(
        limits,
        keys.data(),
        static_cast<int>(keys.size())
    );

    if (pv_position_is_terminal(pos, mem, keys.data(), key_count))
        return 0;

    append_pv_previous_key(
        keys.data(),
        key_count,
        static_cast<int>(keys.size()),
        pos.key
    );

    const int length_limit = std::clamp(pv_length, 0, MAX_PLY);
    for (int i = 0; i < length_limit; ++i) {
        const Move move = pv[i];
        if (move_is_none(move)) {
            // Truncate at first null move.
            return i;
        }

        // Check legality in the current position.
        MoveList legal{};
        generate_legal(pos, mem, legal);
        bool ok = false;
        for (int j = 0; j < legal.size; ++j) {
            if (legal.moves[j] == move) {
                ok = true;
                break;
            }
        }

        if (!ok) {
            if (diagnostic_out != nullptr) {
                *diagnostic_out
                    << "info string PV_VALIDATE illegal move at index " << i
                    << " ply " << i
                    << " move " << move_to_uci(move)
                    << " pv";
                for (int k = 0; k < pv_length; ++k)
                    *diagnostic_out << ' ' << move_to_uci(pv[k]);
                *diagnostic_out << '\n';
            }
#if !defined(NDEBUG)
            // In debug builds, crash to make the bug visible.
            assert(false && "Illegal PV move detected");
#endif
            return i; // Truncate to the legal prefix.
        }

        StateInfo state{};
        make_move(pos, move, mem.tables, state);
        const int accepted_length = i + 1;
        const bool repetition_terminal =
            pv_repetition_terminal(pos, keys.data(), key_count);

        if (repetition_terminal && i > 0)
            return i;

        if (pos.halfmove_clock >= 100 || repetition_terminal) {
            return accepted_length;
        }

        append_pv_previous_key(
            keys.data(),
            key_count,
            static_cast<int>(keys.size()),
            pos.key
        );
    }

    return length_limit;
}

void extend_syzygy_pv(
    const Position& root,
    const memory::Memory& mem,
    const SearchLimits& limits,
    Move* pv,
    int& pv_length,
    int score
) {
    const int root_tb_score = Searcher::tablebase_score(
        static_cast<syzygy::Wdl>(limits.root_tb_wdl),
        0,
        limits.syzygy_50_move_rule
    );
    if (!limits.root_in_tb ||
        limits.syzygy_probe_limit <= 0 ||
        (std::abs(score) < VALUE_TB - MAX_PLY &&
         std::abs(root_tb_score) < VALUE_TB - MAX_PLY) ||
        std::abs(score) >= VALUE_MATE - MAX_PLY ||
        pv_length <= 0 ||
        move_is_none(pv[0])) {
        return;
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
    Position pos{};
    position_copy_without_accumulators(pos, root);

    std::array<Key, MAX_PLY + MAX_GAME_HISTORY + 2> keys{};
    int key_count = 0;
    const int history_count =
        std::min(limits.game_history_count, MAX_GAME_HISTORY);
    for (int i = 0; i < history_count; ++i)
        keys[static_cast<std::size_t>(key_count++)] = limits.game_history_keys[i];
    keys[static_cast<std::size_t>(key_count++)] = pos.key;

    StateInfo first_state{};
    make_move(pos, pv[0], mem.tables, first_state);
    keys[static_cast<std::size_t>(key_count++)] = pos.key;

    int kept_length = 1;
    while (kept_length < pv_length &&
           kept_length < MAX_PLY &&
           std::chrono::steady_clock::now() < deadline) {
        const Move candidate = pv[kept_length];
        syzygy::RootProbe probe{};
        if (!syzygy::rank_root_moves(
                pos,
                mem,
                limits.syzygy_probe_limit,
                limits.syzygy_50_move_rule,
                &candidate,
                1,
                probe,
                true,
                false
            )) {
            break;
        }

        Position next{};
        position_copy_without_accumulators(next, pos);
        do_move_copy(next, candidate, mem.tables);
        if ((limits.syzygy_50_move_rule && next.halfmove_clock >= 100) ||
            pv_contains_key(keys, key_count, next.key)) {
            break;
        }

        pos = next;
        keys[static_cast<std::size_t>(key_count++)] = pos.key;
        ++kept_length;
    }
    pv_length = kept_length;

    while (pv_length < MAX_PLY &&
           std::chrono::steady_clock::now() < deadline) {
        syzygy::RootProbe probe{};
        if (!syzygy::rank_root_moves(
                pos,
                mem,
                limits.syzygy_probe_limit,
                limits.syzygy_50_move_rule,
                nullptr,
                0,
                probe,
                true,
                true
            ) ||
            !probe.used_dtz ||
            probe.move_count <= 0) {
            break;
        }

        Move move = Move(0);
        Position next{};
        int best_tie_break = -VALUE_INF;
        for (int i = 0; i < probe.move_count; ++i) {
            Position candidate{};
            position_copy_without_accumulators(candidate, pos);
            do_move_copy(candidate, probe.moves[i], mem.tables);
            if ((limits.syzygy_50_move_rule && candidate.halfmove_clock >= 100) ||
                pv_key_occurrences(keys, key_count, candidate.key) >= 2) {
                continue;
            }

            MoveList replies{};
            generate_legal(candidate, mem, replies);
            int tie_break = -replies.size;
            for (int j = 0; j < replies.size; ++j)
                if (move_is_capture(replies.moves[j]))
                    tie_break -= 100;

            if (tie_break > best_tie_break) {
                best_tie_break = tie_break;
                move = probe.moves[i];
                next = candidate;
            }
        }
        if (move_is_none(move))
            break;

        pv[pv_length++] = move;
        pos = next;
        keys[static_cast<std::size_t>(key_count++)] = pos.key;
    }
}

void build_root_lines(
    Searcher& searcher,
    const Position& root,
    std::vector<RootLine>& lines,
    int depth
) {
    MoveList list{};
    generate_legal(root, searcher.mem, list);

    if (searcher.limits.root_move_count > 0) {
        MoveList filtered{};
        for (int i = 0; i < list.size; ++i) {
            const Move move = list.moves[i];
            bool allowed = false;
            for (int j = 0; j < searcher.limits.root_move_count; ++j) {
                if (searcher.limits.root_moves[j] == move) {
                    allowed = true;
                    break;
                }
            }
            if (allowed)
                filtered.moves[filtered.size++] = move;
        }
        list = filtered;
    }

    const memory::TTProbe probe =
        memory::tt_probe(searcher.mem.tt, memory::tt_key(root, searcher.mem.tables));
    const Move tt_move = searcher.tt_move_from_probe(probe);
    ScoredMoveList scored{};
    searcher.score_moves(root, list, scored, tt_move, 0, depth);
    searcher.apply_msv_root_order(root, scored, depth);
    searcher.apply_startpos_e4_root_order(root, scored);

    lines.clear();
    lines.reserve(static_cast<std::size_t>(scored.size));
    for (int i = 0; i < scored.size; ++i) {
        RootLine line{};
        line.move = searcher.pick_next(scored, i);
        line.pv[0] = line.move;
        line.pv_length = 1;
        lines.push_back(line);
    }
}

[[nodiscard]] SearchResult root_line_to_result(
    const RootLine& line,
    int depth,
    u64 nodes,
    u64 tb_hits
) noexcept {
    SearchResult result{};
    result.best_move = line.move;
    result.score = line.score;
    result.nodes = nodes;
    result.tb_hits = tb_hits;
    result.depth = depth;
    result.seldepth = line.seldepth;
    result.pv_length = line.pv_length;
    for (int i = 0; i < result.pv_length; ++i)
        result.pv[i] = line.pv[i];
    return result;
}

void complete_root_line_pvs_from_tt(
    const Position& root,
    Searcher& searcher,
    std::vector<RootLine>& lines,
    int line_count,
    int depth
) noexcept {
    if (depth <= 1)
        return;

    const int completed = std::clamp(
        line_count,
        0,
        static_cast<int>(lines.size())
    );

    for (int i = 0; i < completed; ++i) {
        RootLine& line = lines[static_cast<std::size_t>(i)];
        if (!line.searched ||
            move_is_none(line.move) ||
            line.bound != memory::BOUND_EXACT) {
            continue;
        }

        line.pv_length = validate_and_sanitize_pv(
            root,
            searcher.mem,
            searcher.limits,
            line.pv,
            line.pv_length
        );
        if (line.pv_length > 1 && line.pv[0] == line.move)
            continue;

        Position tt_pv_root{};
        position_copy_without_accumulators(tt_pv_root, root);

        Move rebuilt_pv[MAX_PLY]{};
        const int rebuilt_length = reconstruct_tt_pv(
            tt_pv_root,
            searcher.mem,
            searcher.limits,
            line.move,
            rebuilt_pv,
            std::min(depth, MAX_PLY)
        );
        if (rebuilt_length > line.pv_length &&
            rebuilt_pv[0] == line.move) {
            std::memcpy(
                line.pv,
                rebuilt_pv,
                static_cast<std::size_t>(rebuilt_length) * sizeof(Move)
            );
            line.pv_length = rebuilt_length;
        } else if (line.pv_length <= 0) {
            line.pv[0] = line.move;
            line.pv_length = 1;
        }
    }
}

void copy_result_to_root_line(
    RootLine& line,
    const SearchResult& result,
    const Move* pv,
    int pv_length,
    memory::Bound bound
) noexcept {
    line.move = result.best_move;
    line.score = result.score;
    line.bound = bound;
    line.depth = result.depth;
    line.seldepth = result.seldepth;
    line.pv_length = std::clamp(pv_length, 0, MAX_PLY);
    if (line.pv_length > 0 && pv != nullptr) {
        std::memcpy(
            line.pv,
            pv,
            static_cast<std::size_t>(line.pv_length) * sizeof(Move)
        );
    } else if (!move_is_none(result.best_move)) {
        line.pv[0] = result.best_move;
        line.pv_length = 1;
    }
    line.searched = !move_is_none(result.best_move);
}

void capture_completed_single_result(
    IterativeWorkerResult& result,
    const SearchResult& best,
    const Searcher& searcher,
    memory::Bound score_bound
) noexcept {
    result.best = best;
    result.score_bound = score_bound;
    result.lines.clear();
    result.pv_length = searcher.pv_length[0];
    result.best.pv_length = result.pv_length;
    for (int i = 0; i < result.pv_length; ++i) {
        result.pv[i] = searcher.pv_table[0][i];
        result.best.pv[i] = result.pv[i];
    }
}

void capture_completed_result(
    IterativeWorkerResult& result,
    const SearchResult& best,
    const std::vector<RootLine>& root_lines,
    int line_count,
    memory::Bound score_bound
) {
    result.best = best;
    result.score_bound = score_bound;
    result.lines.clear();
    const int copied_lines = std::clamp(
        line_count,
        0,
        static_cast<int>(root_lines.size())
    );
    result.lines.reserve(static_cast<std::size_t>(copied_lines));
    for (int i = 0; i < copied_lines; ++i) {
        const RootLine& line = root_lines[static_cast<std::size_t>(i)];
        if (!line.searched || move_is_none(line.move))
            break;
        result.lines.push_back(line);
    }

    result.pv_length = result.best.pv_length;
    if (!result.lines.empty()) {
        const RootLine& first = result.lines.front();
        result.score_bound = first.bound == memory::BOUND_NONE
            ? score_bound
            : first.bound;
        result.pv_length = first.pv_length;
        result.best.pv_length = result.pv_length;
    }
    result.best.pv_length = result.pv_length;
    for (int i = 0; i < result.pv_length; ++i) {
        result.pv[i] = result.lines.empty()
            ? result.best.pv[i]
            : result.lines.front().pv[i];
        result.best.pv[i] = result.pv[i];
    }
}

void emit_iteration_info(
    std::ostream& stream,
    memory::Memory& local_mem,
    const Position& local_root,
    const Searcher& searcher,
    const SearchResult& current,
    const Move* pv,
    int pv_length,
    Searcher::clock::time_point search_start,
    u64 nodes,
    u64 tb_hits,
    int depth,
    memory::Bound score_bound,
    int multipv_index
) {
    const double seconds =
        std::chrono::duration<double>(Searcher::clock::now() - search_start).count();
    const u64 nps = seconds > 0.0
        ? static_cast<u64>(static_cast<double>(nodes) / seconds)
        : 0ULL;
    const u64 time_ms = static_cast<u64>(seconds * 1000.0);
    const int hashfull = memory::tt_hashfull(local_mem.tt);

    stream << "info depth " << depth
           << " seldepth " << current.seldepth
           << " multipv " << multipv_index << ' ';
    append_uci_score(
        stream,
        current.score,
        local_root,
        score_bound
    );
    stream << " nodes " << nodes
           << " nps " << nps
           << " tbhits " << tb_hits
           << " hashfull " << hashfull
           << " time " << time_ms
           << " pv";

    Move fallback_pv[1]{};
    const Move* emitted_pv = pv;
    int emitted_pv_length = pv_length;
    if (emitted_pv_length <= 0 && !move_is_none(current.best_move)) {
        fallback_pv[0] = current.best_move;
        emitted_pv = fallback_pv;
        emitted_pv_length = 1;
    }

    // Validate PV legality before output.  In debug builds an illegal move
    // asserts; in release builds the PV is truncated to the legal prefix.
    Move validated_pv[MAX_PLY]{};
    if (emitted_pv_length > 0) {
        std::memcpy(
            validated_pv,
            emitted_pv,
            static_cast<std::size_t>(emitted_pv_length) * sizeof(Move)
        );
        emitted_pv_length = validate_and_sanitize_pv(
            local_root,
            local_mem,
            searcher.limits,
            validated_pv,
            emitted_pv_length,
            &stream
        );
        emitted_pv = validated_pv;
    }

    for (int i = 0; i < emitted_pv_length; ++i)
        stream << ' ' << move_to_uci(emitted_pv[i]);

    stream << '\n';
}

void emit_root_lines_info(
    std::ostream& stream,
    memory::Memory& local_mem,
    const Position& local_root,
    const Searcher& searcher,
    const std::vector<RootLine>& lines,
    int line_count,
    Searcher::clock::time_point search_start,
    u64 nodes,
    u64 tb_hits,
    int depth
) {
    const int emitted_count = std::clamp(
        line_count,
        0,
        static_cast<int>(lines.size())
    );
    for (int i = 0; i < emitted_count; ++i) {
        const RootLine& line = lines[static_cast<std::size_t>(i)];
        if (!line.searched || move_is_none(line.move))
            break;
        SearchResult current = root_line_to_result(line, depth, nodes, tb_hits);
        emit_iteration_info(
            stream,
            local_mem,
            local_root,
            searcher,
            current,
            line.pv,
            line.pv_length,
            search_start,
            nodes,
            tb_hits,
            depth,
            line.bound == memory::BOUND_NONE ? memory::BOUND_EXACT : line.bound,
            i + 1
        );
    }
}

[[nodiscard]] IterativeWorkerResult iterative_deepening_worker_single(
    Searcher& searcher,
    const Position& local_root,
    std::ostream* local_out,
    Searcher::clock::time_point search_start
) {
    IterativeWorkerResult result{};
    IterativeWorkerResult exact_result{};
    SearchResult best{};
    Move hint_move = 0;
    Position keyed_root = local_root;
    position_refresh_key(keyed_root, searcher.mem.tables);
    searcher.p2_accumulator_stack.reset();
    searcher.root_side_to_move = keyed_root.side_to_move;
    searcher.start_time = search_start;
    searcher.stop_on_ponderhit = false;
    u64 total_nodes = 0;
    u64 total_tb_hits = searcher.limits.shared_tb_hits == nullptr
        ? searcher.limits.root_tb_hits
        : 0;
    int prev_depth_end_ms = 0;
    Move fallback_best_move = 0;
#if MAGNUS_SEARCHSTATS_OBS
    u64 last_reported_nodes = 0;
#endif
    const int max_depth =
        std::clamp(searcher.limits.depth, 1, MAX_SEARCH_DEPTH);
    IterationTimeState time_state{};
    initialize_iteration_time_state(searcher.limits, time_state);
    if (searcher.limits.root_move_count > 0) {
        time_state.root_legal_count = searcher.limits.root_move_count;
        fallback_best_move = searcher.limits.root_moves[0];
    } else {
        MoveList root_moves{};
        generate_legal(keyed_root, searcher.mem, root_moves);
        time_state.root_legal_count = root_moves.size;
        if (root_moves.size > 0)
            fallback_best_move = root_moves.moves[0];
    }

    for (int depth = 1; depth <= max_depth; ++depth) {
        searcher.root_depth = depth;
        if (searcher.limits.time_signals != nullptr &&
            !searcher.limits.time_signals->increase_depth.load(
                std::memory_order_relaxed
            )) {
            ++time_state.search_again_counter;
        }

        SearchResult current{};
        u64 depth_nodes = 0;
        u64 depth_tb_hits = 0;
#if MAGNUS_SEARCHSTATS_OBS
        int aspiration_fail_low = 0;
        int aspiration_fail_high = 0;
        const Searcher::SearchStats stats_before = searcher.stats;
#endif

        int alpha = -VALUE_INF;
        int beta = VALUE_INF;
        int delta = ASPIRATION_DELTA;
        int depth_max_seldepth = 0;
        int failed_high_count = 0;
        int completed_effective_depth = depth;
        memory::Bound current_score_bound = memory::BOUND_EXACT;
        int current_bound_alpha = alpha;
        int current_bound_beta = beta;

        if (use_root_aspiration(searcher.limits, depth)) {
            alpha = std::max(-VALUE_INF, best.score - delta);
            beta = std::min(VALUE_INF, best.score + delta);
        }

        while (true) {
            if (searcher.stopped)
                break;

            reset_searcher_iteration(searcher, search_start, total_nodes + depth_nodes);

            const int attempt_alpha = alpha;
            const int attempt_beta = beta;
            const int effective_depth = searcher.limits.time_signals == nullptr
                ? depth
                : std::max(
                      1,
                      depth
                          - failed_high_count
                          - 3 * (time_state.search_again_counter + 1) / 4
                  );
            completed_effective_depth = effective_depth;
            current_bound_alpha = attempt_alpha;
            current_bound_beta = attempt_beta;
            current = searcher.search_root(
                keyed_root,
                effective_depth,
                hint_move,
                attempt_alpha,
                attempt_beta
            );
            current.depth = depth;
            depth_max_seldepth = std::max(depth_max_seldepth, searcher.seldepth);
            current.seldepth = depth_max_seldepth;
            searcher.publish_nodes();
            searcher.publish_tb_hits();
            depth_nodes += current.nodes;
            depth_tb_hits += current.tb_hits;
            current_score_bound = score_bound_from_window(
                current.score,
                attempt_alpha,
                attempt_beta
            );

            if (searcher.stopped || depth == 1)
                break;

            if (current.score <= attempt_alpha) {
                failed_high_count = 0;
#if MAGNUS_SEARCHSTATS_OBS
                ++aspiration_fail_low;
#endif
                alpha = std::max(-VALUE_INF, current.score - delta);
                beta = std::min(VALUE_INF, current.score + delta);
                delta *= 2;
                continue;
            }

            if (current.score >= attempt_beta) {
                ++failed_high_count;
#if MAGNUS_SEARCHSTATS_OBS
                ++aspiration_fail_high;
#endif
                alpha = std::max(-VALUE_INF, current.score - delta);
                beta = std::min(VALUE_INF, current.score + delta);
                delta *= 2;
                continue;
            }

            break;
        }

        if (searcher.limits.full_pv &&
            !searcher.stopped &&
            current_score_bound == memory::BOUND_EXACT &&
            searcher.pv_length[0] <= 1 &&
            !move_is_none(current.best_move)) {
            Position tt_pv_root{};
            position_copy_without_accumulators(tt_pv_root, keyed_root);

            Move rebuilt_pv[MAX_PLY]{};
            const int rebuilt_length = reconstruct_tt_pv(
                tt_pv_root,
                searcher.mem,
                searcher.limits,
                current.best_move,
                rebuilt_pv,
                std::min(completed_effective_depth, MAX_PLY)
            );
            if (rebuilt_length > searcher.pv_length[0] &&
                rebuilt_pv[0] == current.best_move) {
                std::memcpy(
                    searcher.pv_table[0],
                    rebuilt_pv,
                    static_cast<std::size_t>(rebuilt_length) * sizeof(Move)
                );
                searcher.pv_length[0] = rebuilt_length;
            }
        }

        if (!searcher.stopped && !move_is_none(current.best_move)) {
            const u64 recovery_tb_hits_before = searcher.tb_hits;
            const u64 recovery_nodes = searcher.recover_ponder_pv_full_window(
                keyed_root,
                current,
                completed_effective_depth
            );
            current.depth = depth;
            if (recovery_nodes > 0) {
                depth_nodes += recovery_nodes;
                depth_tb_hits += searcher.tb_hits >= recovery_tb_hits_before
                    ? searcher.tb_hits - recovery_tb_hits_before
                    : 0;
                depth_max_seldepth = std::max(depth_max_seldepth, searcher.seldepth);
                current.seldepth = depth_max_seldepth;
                searcher.publish_nodes();
                searcher.publish_tb_hits();
            }
        }

        if (searcher.limits.root_in_tb &&
            std::abs(current.score) < VALUE_MATE - MAX_PLY) {
            current.score = Searcher::tablebase_score(
                static_cast<syzygy::Wdl>(searcher.limits.root_tb_wdl),
                0,
                searcher.limits.syzygy_50_move_rule
            );
        }
        current_score_bound = score_bound_from_window(
            current.score,
            current_bound_alpha,
            current_bound_beta
        );

        total_nodes += depth_nodes;
        total_tb_hits += depth_tb_hits;
        {
            const int now_ms = searcher.timed_elapsed_ms();
            time_state.last_depth_time_ms = now_ms - prev_depth_end_ms;
            time_state.last_depth_nodes = depth_nodes;
            prev_depth_end_ms = now_ms;
        }
        const bool stopped_mid_depth = searcher.stopped && best.depth > 0;
        if (!searcher.stopped)
            searcher.completed_depth = depth;

        if (!searcher.stopped || best.depth == 0) {
            best = current;
            best.nodes = total_nodes;
            best.tb_hits = total_tb_hits;
            hint_move = current.best_move;
            capture_completed_single_result(result, best, searcher, current_score_bound);
            remember_exact_result(exact_result, result);
        }

        if (root_msv_enabled(searcher.limits) && !move_is_none(current.best_move)) {
            const bool msv_stopped = searcher.stopped || searcher.hit_hard_limit();
            const bool pv_valid =
                searcher.pv_length[0] > 2 &&
                searcher.pv_table[0][0] == current.best_move;
            root_msv_record_completed(
                searcher.limits,
                current.best_move,
                completed_effective_depth,
                current.seldepth,
                current.score,
                current_score_bound,
                msv_stopped,
                pv_valid,
                searcher.pv_length[0]
            );
        }

        const bool should_stop_for_time =
            !stopped_mid_depth &&
            should_stop_after_iteration(
                searcher,
                time_state,
                best,
                depth,
                completed_effective_depth,
                total_nodes
            );

        if (local_out != nullptr &&
            searcher.limits.report_info &&
            !stopped_mid_depth) {
            const u64 reported_nodes = searcher.limits.shared_nodes != nullptr
                ? searcher.limits.shared_nodes->load(std::memory_order_relaxed)
                : total_nodes;
            const u64 reported_tb_hits = searcher.limits.shared_tb_hits != nullptr
                ? searcher.limits.shared_tb_hits->load(std::memory_order_relaxed)
                : total_tb_hits;
#if MAGNUS_SEARCHSTATS_OBS
            const u64 reported_depth_nodes =
                reported_nodes >= last_reported_nodes
                    ? reported_nodes - last_reported_nodes
                    : depth_nodes;
            last_reported_nodes = reported_nodes;
#endif
            emit_iteration_info(
                *local_out,
                searcher.mem,
                keyed_root,
                searcher,
                current,
                searcher.pv_table[0],
                searcher.pv_length[0],
                search_start,
                reported_nodes,
                reported_tb_hits,
                depth,
                current_score_bound,
                1
            );
            searcher.emit_msv_info(*local_out, keyed_root, depth);
#if MAGNUS_SEARCHSTATS_OBS
            emit_searchstats_line(
                *local_out,
                depth,
                false,
                reported_depth_nodes,
                aspiration_fail_low,
                aspiration_fail_high,
                time_state,
                stats_before,
                searcher.stats
            );
#endif
        }

        if (stopped_mid_depth) {
            if (local_out != nullptr &&
                searcher.limits.report_info) {
                const u64 reported_nodes = searcher.limits.shared_nodes != nullptr
                    ? searcher.limits.shared_nodes->load(std::memory_order_relaxed)
                    : total_nodes;
                const u64 reported_tb_hits = searcher.limits.shared_tb_hits != nullptr
                    ? searcher.limits.shared_tb_hits->load(std::memory_order_relaxed)
                    : total_tb_hits;
                if (!move_is_none(current.best_move) && current.nodes > 0) {
                    emit_iteration_info(
                        *local_out,
                        searcher.mem,
                        keyed_root,
                        searcher,
                        current,
                        searcher.pv_table[0],
                        searcher.pv_length[0],
                        search_start,
                        reported_nodes,
                        reported_tb_hits,
                        depth,
                        // Partial root search is a best-so-far estimate, not exact.
                        memory::BOUND_LOWER,
                        1
                    );
                } else if (depth > best.depth + 1) {
                    emit_iteration_info(
                        *local_out,
                        searcher.mem,
                        keyed_root,
                        searcher,
                        best,
                        result.pv,
                        result.pv_length,
                        search_start,
                        reported_nodes,
                        reported_tb_hits,
                        best.depth,
                        result.score_bound,
                        1
                    );
                }
#if MAGNUS_SEARCHSTATS_OBS
                const u64 reported_depth_nodes =
                    reported_nodes >= last_reported_nodes
                        ? reported_nodes - last_reported_nodes
                        : depth_nodes;
                emit_searchstats_line(
                    *local_out,
                    depth,
                    true,
                    reported_depth_nodes,
                    aspiration_fail_low,
                    aspiration_fail_high,
                    time_state,
                    stats_before,
                    searcher.stats
                );
#endif
            }
            break;
        }

        if (should_stop_for_time)
            break;

        if (searcher.stop_after_completed_depth())
            break;
    }

    // Ponder busy-wait: after the depth loop finishes (e.g. MAX_PLY reached),
    // keep the search "alive" until stop or ponderhit arrives.
    if (searcher.limits.ponder && !searcher.stopped) {
        while (!searcher.stopped && searcher.pondering_active()) {
            if (searcher.hit_hard_limit())
                break;
            searcher.publish_nodes();
            searcher.publish_tb_hits();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (searcher.stop_on_ponderhit) {
            searcher.stopped = true;
            searcher.hard_stop = true;
        }
    }

    searcher.publish_nodes();
    searcher.publish_tb_hits();
    prefer_exact_result(result, exact_result);
    if (move_is_none(result.best.best_move) && !move_is_none(fallback_best_move)) {
        result.best.best_move = fallback_best_move;
        result.best.pv[0] = fallback_best_move;
        result.best.pv_length = 1;
        result.pv[0] = fallback_best_move;
        result.pv_length = 1;
    }
    result.best.nodes = searcher.limits.shared_nodes != nullptr
        ? searcher.limits.shared_nodes->load(std::memory_order_relaxed)
        : total_nodes;
    result.best.tb_hits = searcher.limits.shared_tb_hits != nullptr
        ? searcher.limits.shared_tb_hits->load(std::memory_order_relaxed)
        : total_tb_hits;

#if MAGNUS_CAPTURE_OBS
    if (local_out != nullptr && searcher.limits.report_info)
        searcher.emit_capture_observation(*local_out);
#endif
#if MAGNUS_MOVEPICKER_OBS
    if (local_out != nullptr && searcher.limits.report_info)
        searcher.emit_movepicker_observation(*local_out);
#endif
#if MAGNUS_SEARCH_OBS
    if (local_out != nullptr && searcher.limits.report_info)
        searcher.emit_search_observation(*local_out);
#endif
    if (local_out != nullptr &&
        searcher.limits.report_info &&
        searcher.limits.singular_telemetry) {
        searcher.emit_singular_telemetry(*local_out);
    }
#if MAGNUS_LMR_OBS
    if (local_out != nullptr && searcher.limits.report_info)
        searcher.emit_lmr_observation(*local_out);
#endif

    result.best_average_score = searcher.root_average_score(result.best.best_move);
    result.time_reduction = time_state.current_time_reduction;
    return result;
}

[[nodiscard]] IterativeWorkerResult iterative_deepening_worker(
    Searcher& searcher,
    const Position& local_root,
    std::ostream* local_out,
    Searcher::clock::time_point search_start
) {
    if (searcher.limits.multipv <= 1)
        return iterative_deepening_worker_single(
            searcher,
            local_root,
            local_out,
            search_start
        );

    IterativeWorkerResult result{};
    IterativeWorkerResult exact_result{};
    SearchResult best{};
    Move hint_move = 0;
    Position keyed_root = local_root;
    position_refresh_key(keyed_root, searcher.mem.tables);
    searcher.p2_accumulator_stack.reset();
    searcher.root_side_to_move = keyed_root.side_to_move;
    searcher.start_time = search_start;
    searcher.stop_on_ponderhit = false;
    u64 total_nodes = 0;
    u64 total_tb_hits = searcher.limits.shared_tb_hits == nullptr
        ? searcher.limits.root_tb_hits
        : 0;
    int prev_depth_end_ms = 0;  // for fixed-time per-depth cost tracking
    Move fallback_best_move = 0;
#if MAGNUS_SEARCHSTATS_OBS
    u64 last_reported_nodes = 0;
#endif
    const int max_depth =
        std::clamp(searcher.limits.depth, 1, MAX_SEARCH_DEPTH);
    IterationTimeState time_state{};
    initialize_iteration_time_state(searcher.limits, time_state);
    std::vector<RootLine> root_lines{};
    build_root_lines(searcher, keyed_root, root_lines, 1);
    time_state.root_legal_count = static_cast<int>(root_lines.size());
    if (!root_lines.empty())
        fallback_best_move = root_lines.front().move;

    for (int depth = 1; depth <= max_depth; ++depth) {
        searcher.root_depth = depth;
        if (searcher.limits.time_signals != nullptr &&
            !searcher.limits.time_signals->increase_depth.load(
                std::memory_order_relaxed
            )) {
            ++time_state.search_again_counter;
        }

        SearchResult current{};
        u64 depth_nodes = 0;
        u64 depth_tb_hits = 0;
#if MAGNUS_SEARCHSTATS_OBS
        int aspiration_fail_low = 0;
        int aspiration_fail_high = 0;
        const Searcher::SearchStats stats_before = searcher.stats;
#endif

        memory::Bound current_score_bound = memory::BOUND_EXACT;
        int completed_line_count = 0;
        int completed_effective_depth = depth;

        for (RootLine& line : root_lines) {
            line.previous_score = line.score;
            line.score = -VALUE_INF;
            line.selection_score = -VALUE_INF;
            line.bound = memory::BOUND_NONE;
            line.depth = 0;
            line.seldepth = 0;
            line.searched = false;
            line.pv[0] = line.move;
            line.pv_length = move_is_none(line.move) ? 0 : 1;
        }

        if (root_lines.empty()) {
            current.depth = depth;
            current.score = searcher.in_check(keyed_root)
                ? -VALUE_MATE
                : searcher.draw_score(keyed_root.side_to_move);
            current.seldepth = 0;
            current_score_bound = memory::BOUND_EXACT;
        } else {
            const int requested_multipv = std::clamp(
                searcher.limits.multipv,
                1,
                static_cast<int>(root_lines.size())
            );

            for (int pv_idx = 0; pv_idx < requested_multipv; ++pv_idx) {
                if (searcher.stopped)
                    break;

                int alpha = -VALUE_INF;
                int beta = VALUE_INF;
                int delta = ASPIRATION_DELTA;
                int line_max_seldepth = 0;
                int failed_high_count = 0;

                const int previous_score =
                    root_lines[static_cast<std::size_t>(pv_idx)].previous_score;
                const int aspiration_base = has_root_line_score(previous_score)
                    ? previous_score
                    : (has_root_line_score(best.score) ? best.score : 0);

                if (use_root_aspiration(searcher.limits, depth) &&
                    has_root_line_score(aspiration_base)) {
                    alpha = std::max(-VALUE_INF, aspiration_base - delta);
                    beta = std::min(VALUE_INF, aspiration_base + delta);
                }

                while (true) {
                    if (searcher.stopped)
                        break;

                    reset_searcher_iteration(
                        searcher,
                        search_start,
                        total_nodes + depth_nodes
                    );

                    const int attempt_alpha = alpha;
                    const int attempt_beta = beta;
                    const int effective_depth = searcher.limits.time_signals == nullptr
                        ? depth
                        : std::max(
                              1,
                              depth
                                  - failed_high_count
                                  - 3 * (time_state.search_again_counter + 1) / 4
                          );
                    if (pv_idx == 0)
                        completed_effective_depth = effective_depth;
                    const Move line_hint = pv_idx == 0
                        ? hint_move
                        : root_lines[static_cast<std::size_t>(pv_idx)].move;

                    current = searcher.search_root(
                        keyed_root,
                        root_lines,
                        pv_idx,
                        effective_depth,
                        line_hint,
                        attempt_alpha,
                        attempt_beta
                    );
                    current.depth = depth;
                    line_max_seldepth = std::max(line_max_seldepth, searcher.seldepth);
                    current.seldepth = line_max_seldepth;
                    if (pv_idx < static_cast<int>(root_lines.size()))
                        root_lines[static_cast<std::size_t>(pv_idx)].seldepth =
                            line_max_seldepth;

                    searcher.publish_nodes();
                    searcher.publish_tb_hits();
                    depth_nodes += current.nodes;
                    depth_tb_hits += current.tb_hits;
                    current_score_bound = score_bound_from_window(
                        current.score,
                        attempt_alpha,
                        attempt_beta
                    );
                    if (pv_idx < static_cast<int>(root_lines.size()))
                        root_lines[static_cast<std::size_t>(pv_idx)].bound =
                            current_score_bound;

                    if (searcher.stopped || depth == 1)
                        break;

                    if (current.score <= attempt_alpha) {
                        failed_high_count = 0;
#if MAGNUS_SEARCHSTATS_OBS
                        ++aspiration_fail_low;
#endif
                        alpha = std::max(-VALUE_INF, current.score - delta);
                        beta = std::min(VALUE_INF, current.score + delta);
                        delta *= 2;
                        continue;
                    }

                    if (current.score >= attempt_beta) {
                        ++failed_high_count;
#if MAGNUS_SEARCHSTATS_OBS
                        ++aspiration_fail_high;
#endif
                        alpha = std::max(-VALUE_INF, current.score - delta);
                        beta = std::min(VALUE_INF, current.score + delta);
                        delta *= 2;
                        continue;
                    }

                    break;
                }

                if (pv_idx < static_cast<int>(root_lines.size()) &&
                    root_lines[static_cast<std::size_t>(pv_idx)].searched) {
                    completed_line_count = std::max(completed_line_count, pv_idx + 1);
                    stable_sort_root_lines(root_lines, 0, completed_line_count);
                    promote_startpos_e4_root_line(
                        searcher.limits,
                        keyed_root,
                        root_lines,
                        0,
                        completed_line_count
                    );
                }

                if (searcher.stopped)
                    break;
            }

            if (completed_line_count > 0) {
                complete_root_line_pvs_from_tt(
                    keyed_root,
                    searcher,
                    root_lines,
                    completed_line_count,
                    depth
                );
                current = root_line_to_result(
                    root_lines.front(),
                    depth,
                    depth_nodes,
                    depth_tb_hits
                );
                current_score_bound = root_lines.front().bound == memory::BOUND_NONE
                    ? memory::BOUND_EXACT
                    : root_lines.front().bound;
            }
        }

        if (searcher.limits.full_pv &&
            !searcher.stopped &&
            completed_line_count > 0 &&
            current_score_bound == memory::BOUND_EXACT &&
            root_lines.front().pv_length <= 1 &&
            !move_is_none(current.best_move)) {
            Position tt_pv_root{};
            position_copy_without_accumulators(tt_pv_root, keyed_root);

            Move rebuilt_pv[MAX_PLY]{};
            const int rebuilt_length = reconstruct_tt_pv(
                tt_pv_root,
                searcher.mem,
                searcher.limits,
                current.best_move,
                rebuilt_pv,
                std::min(completed_effective_depth, MAX_PLY)
            );
            if (rebuilt_length > root_lines.front().pv_length &&
                rebuilt_pv[0] == current.best_move) {
                std::memcpy(
                    root_lines.front().pv,
                    rebuilt_pv,
                    static_cast<std::size_t>(rebuilt_length) * sizeof(Move)
                );
                root_lines.front().pv_length = rebuilt_length;
                current = root_line_to_result(
                    root_lines.front(),
                    depth,
                    depth_nodes,
                    depth_tb_hits
                );
            }
        }

        if (!searcher.stopped &&
            completed_line_count > 0 &&
            !move_is_none(current.best_move)) {
            const u64 recovery_tb_hits_before = searcher.tb_hits;
            const int score_before_recovery = current.score;
            const memory::Bound bound_before_recovery = current_score_bound;
            const u64 recovery_nodes = searcher.recover_ponder_pv_full_window(
                keyed_root,
                current,
                completed_effective_depth
            );
            current.depth = depth;
            if (recovery_nodes > 0) {
                depth_nodes += recovery_nodes;
                depth_tb_hits += searcher.tb_hits >= recovery_tb_hits_before
                    ? searcher.tb_hits - recovery_tb_hits_before
                    : 0;
                current.seldepth = std::max(current.seldepth, searcher.seldepth);
                searcher.publish_nodes();
                searcher.publish_tb_hits();
                if (searcher.limits.multipv > 1) {
                    current.score = score_before_recovery;
                    current_score_bound = bound_before_recovery;
                }
                copy_result_to_root_line(
                    root_lines.front(),
                    current,
                    searcher.pv_table[0],
                    searcher.pv_length[0],
                    searcher.limits.multipv > 1
                        ? bound_before_recovery
                        : memory::BOUND_EXACT
                );
                complete_root_line_pvs_from_tt(
                    keyed_root,
                    searcher,
                    root_lines,
                    completed_line_count,
                    depth
                );
                current = root_line_to_result(
                    root_lines.front(),
                    depth,
                    depth_nodes,
                    depth_tb_hits
                );
            }
        }

        if (searcher.limits.root_in_tb &&
            completed_line_count > 0 &&
            std::abs(current.score) < VALUE_MATE - MAX_PLY) {
            current.score = Searcher::tablebase_score(
                static_cast<syzygy::Wdl>(searcher.limits.root_tb_wdl),
                0,
                searcher.limits.syzygy_50_move_rule
            );
            root_lines.front().score = current.score;
            current_score_bound = memory::BOUND_EXACT;
            root_lines.front().bound = current_score_bound;
        } else if (completed_line_count > 0) {
            current_score_bound = root_lines.front().bound == memory::BOUND_NONE
                ? current_score_bound
                : root_lines.front().bound;
        }

        total_nodes += depth_nodes;
        total_tb_hits += depth_tb_hits;
        {
            const int now_ms = searcher.timed_elapsed_ms();
            time_state.last_depth_time_ms = now_ms - prev_depth_end_ms;
            time_state.last_depth_nodes = depth_nodes;
            prev_depth_end_ms = now_ms;
        }
        const bool stopped_mid_depth = searcher.stopped && best.depth > 0;
        if (!searcher.stopped)
            searcher.completed_depth = depth;

        if (!searcher.stopped || best.depth == 0) {
            best = current;
            best.nodes = total_nodes;
            best.tb_hits = total_tb_hits;
            hint_move = current.best_move;
            if (completed_line_count > 0) {
                root_lines.front().score = best.score;
                root_lines.front().depth = best.depth;
                capture_completed_result(
                    result,
                    best,
                    root_lines,
                    completed_line_count,
                    current_score_bound
                );
            } else {
                result.best = best;
                result.score_bound = current_score_bound;
                result.pv_length = 0;
                result.lines.clear();
            }
            remember_exact_result(exact_result, result);
        }

        if (root_msv_enabled(searcher.limits) &&
            completed_line_count > 0 &&
            !move_is_none(current.best_move)) {
            const bool msv_stopped = searcher.stopped || searcher.hit_hard_limit();
            const bool pv_valid =
                root_lines.front().pv_length > 2 &&
                root_lines.front().pv[0] == current.best_move;
            root_msv_record_completed(
                searcher.limits,
                current.best_move,
                depth,
                current.seldepth,
                current.score,
                current_score_bound,
                msv_stopped,
                pv_valid,
                root_lines.front().pv_length
            );
        }

        const bool should_stop_for_time =
            !stopped_mid_depth &&
            should_stop_after_iteration(
                searcher,
                time_state,
                best,
                completed_effective_depth,
                completed_effective_depth,
                total_nodes
            );

        if (local_out != nullptr &&
            searcher.limits.report_info &&
            !stopped_mid_depth) {
            const u64 reported_nodes = searcher.limits.shared_nodes != nullptr
                ? searcher.limits.shared_nodes->load(std::memory_order_relaxed)
                : total_nodes;
            const u64 reported_tb_hits = searcher.limits.shared_tb_hits != nullptr
                ? searcher.limits.shared_tb_hits->load(std::memory_order_relaxed)
                : total_tb_hits;
#if MAGNUS_SEARCHSTATS_OBS
            const u64 reported_depth_nodes =
                reported_nodes >= last_reported_nodes
                    ? reported_nodes - last_reported_nodes
                    : depth_nodes;
            last_reported_nodes = reported_nodes;
#endif
            if (completed_line_count > 0) {
                emit_root_lines_info(
                    *local_out,
                    searcher.mem,
                    keyed_root,
                    searcher,
                    root_lines,
                    completed_line_count,
                    search_start,
                    reported_nodes,
                    reported_tb_hits,
                    depth
                );
            } else {
                emit_iteration_info(
                    *local_out,
                    searcher.mem,
                    keyed_root,
                    searcher,
                    current,
                    current.pv,
                    current.pv_length,
                    search_start,
                    reported_nodes,
                    reported_tb_hits,
                    depth,
                    current_score_bound,
                    1
                );
            }
            searcher.emit_msv_info(*local_out, keyed_root, depth);
#if MAGNUS_SEARCHSTATS_OBS
            emit_searchstats_line(
                *local_out,
                depth,
                false,
                reported_depth_nodes,
                aspiration_fail_low,
                aspiration_fail_high,
                time_state,
                stats_before,
                searcher.stats
            );
#endif
        }

        if (stopped_mid_depth) {
            if (local_out != nullptr &&
                searcher.limits.report_info) {
                const u64 reported_nodes = searcher.limits.shared_nodes != nullptr
                    ? searcher.limits.shared_nodes->load(std::memory_order_relaxed)
                    : total_nodes;
                const u64 reported_tb_hits = searcher.limits.shared_tb_hits != nullptr
                    ? searcher.limits.shared_tb_hits->load(std::memory_order_relaxed)
                    : total_tb_hits;
                if (!move_is_none(current.best_move) && current.nodes > 0) {
                    if (completed_line_count > 0) {
                        emit_root_lines_info(
                            *local_out,
                            searcher.mem,
                            keyed_root,
                            searcher,
                            root_lines,
                            completed_line_count,
                            search_start,
                            reported_nodes,
                            reported_tb_hits,
                            depth
                        );
                    } else {
                        emit_iteration_info(
                            *local_out,
                            searcher.mem,
                            keyed_root,
                            searcher,
                            current,
                            current.pv,
                            current.pv_length,
                            search_start,
                            reported_nodes,
                            reported_tb_hits,
                            depth,
                            // Partial root search is a best-so-far estimate, not exact.
                            memory::BOUND_LOWER,
                            1
                        );
                    }
                } else if (depth > best.depth + 1) {
                    if (!result.lines.empty()) {
                        emit_root_lines_info(
                            *local_out,
                            searcher.mem,
                            keyed_root,
                            searcher,
                            result.lines,
                            static_cast<int>(result.lines.size()),
                            search_start,
                            reported_nodes,
                            reported_tb_hits,
                            best.depth
                        );
                    } else {
                        emit_iteration_info(
                            *local_out,
                            searcher.mem,
                            keyed_root,
                            searcher,
                            best,
                            result.pv,
                            result.pv_length,
                            search_start,
                            reported_nodes,
                            reported_tb_hits,
                            best.depth,
                            result.score_bound,
                            1
                        );
                    }
                }
#if MAGNUS_SEARCHSTATS_OBS
                const u64 reported_depth_nodes =
                    reported_nodes >= last_reported_nodes
                        ? reported_nodes - last_reported_nodes
                        : depth_nodes;
                emit_searchstats_line(
                    *local_out,
                    depth,
                    true,
                    reported_depth_nodes,
                    aspiration_fail_low,
                    aspiration_fail_high,
                    time_state,
                    stats_before,
                    searcher.stats
                );
#endif
            }
            break;
        }

        if (should_stop_for_time)
            break;

        if (searcher.stop_after_completed_depth())
            break;
    }

    // Ponder busy-wait: after the depth loop finishes (e.g. MAX_PLY reached),
    // keep the search "alive" until stop or ponderhit arrives.
    if (searcher.limits.ponder && !searcher.stopped) {
        while (!searcher.stopped && searcher.pondering_active()) {
            if (searcher.hit_hard_limit())
                break;
            searcher.publish_nodes();
            searcher.publish_tb_hits();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (searcher.stop_on_ponderhit) {
            searcher.stopped = true;
            searcher.hard_stop = true;
        }
    }

    searcher.publish_nodes();
    searcher.publish_tb_hits();
    prefer_exact_result(result, exact_result);
    if (move_is_none(result.best.best_move) && !move_is_none(fallback_best_move)) {
        result.best.best_move = fallback_best_move;
        result.best.pv[0] = fallback_best_move;
        result.best.pv_length = 1;
        result.pv[0] = fallback_best_move;
        result.pv_length = 1;
    }
    result.best.nodes = searcher.limits.shared_nodes != nullptr
        ? searcher.limits.shared_nodes->load(std::memory_order_relaxed)
        : total_nodes;
    result.best.tb_hits = searcher.limits.shared_tb_hits != nullptr
        ? searcher.limits.shared_tb_hits->load(std::memory_order_relaxed)
        : total_tb_hits;

#if MAGNUS_CAPTURE_OBS
    if (local_out != nullptr && searcher.limits.report_info)
        searcher.emit_capture_observation(*local_out);
#endif
#if MAGNUS_MOVEPICKER_OBS
    if (local_out != nullptr && searcher.limits.report_info)
        searcher.emit_movepicker_observation(*local_out);
#endif
#if MAGNUS_SEARCH_OBS
    if (local_out != nullptr && searcher.limits.report_info)
        searcher.emit_search_observation(*local_out);
#endif
    if (local_out != nullptr &&
        searcher.limits.report_info &&
        searcher.limits.singular_telemetry) {
        searcher.emit_singular_telemetry(*local_out);
    }
#if MAGNUS_LMR_OBS
    if (local_out != nullptr && searcher.limits.report_info)
        searcher.emit_lmr_observation(*local_out);
#endif

    result.best_average_score = searcher.root_average_score(result.best.best_move);
    result.time_reduction = time_state.current_time_reduction;
    return result;
}

} // namespace

[[nodiscard]] SearchResult iterative_deepening_single(
    const Position& root,
    memory::Memory& mem,
    SearchSessionState& session_state,
    const SearchLimits& limits,
    std::ostream* out,
    Searcher::clock::time_point search_start
) {
    memory::memory_new_search(mem);

    SearchLimits local_limits = limits;
    SearchTimeSignals time_signals{};
    if (local_limits.use_time_management)
        local_limits.time_signals = &time_signals;
    Searcher searcher(mem, session_state.worker(0), local_limits);
    IterativeWorkerResult result =
        iterative_deepening_worker(searcher, root, out, search_start);
    const int searched_pv_length = result.pv_length;
    extend_syzygy_pv(
        root,
        mem,
        local_limits,
        result.pv,
        result.pv_length,
        result.best.score
    );
    result.best.pv_length = result.pv_length;
    for (int i = 0; i < result.pv_length; ++i)
        result.best.pv[i] = result.pv[i];
    if (!result.lines.empty()) {
        copy_result_to_root_line(
            result.lines.front(),
            result.best,
            result.pv,
            result.pv_length,
            result.score_bound
        );
    }

    if (out != nullptr &&
        local_limits.report_info &&
        result.pv_length > searched_pv_length) {
        if (!result.lines.empty()) {
            emit_root_lines_info(
                *out,
                mem,
                root,
                searcher,
                result.lines,
                static_cast<int>(result.lines.size()),
                search_start,
                result.best.nodes,
                result.best.tb_hits,
                result.best.depth
            );
        } else {
            emit_iteration_info(
                *out,
                mem,
                root,
                searcher,
                result.best,
                result.pv,
                result.pv_length,
                search_start,
                result.best.nodes,
                result.best.tb_hits,
                result.best.depth,
                result.score_bound,
                1
            );
        }
    }

    if (local_limits.use_time_management &&
        local_limits.time_state != nullptr &&
        result.best.depth > 0 &&
        !move_is_none(result.best.best_move)) {
        local_limits.time_state->previous_score = result.best.score;
        local_limits.time_state->previous_average_score =
            result.best_average_score == VALUE_NONE
                ? result.best.score
                : result.best_average_score;
        local_limits.time_state->previous_time_reduction = result.time_reduction;
    }

    return result.best;
}

[[nodiscard]] SearchResult iterative_deepening_lazy_smp(
    const Position& root,
    memory::Memory& mem,
    SearchSessionState& session_state,
    const SearchLimits& limits,
    std::ostream* out,
    Searcher::clock::time_point search_start
) {
    memory::memory_new_search(mem);

    const auto select_lazy_smp_best_index = [](
        const std::vector<IterativeWorkerResult>& results
    ) noexcept {
        int best_index = 0;
        int min_score = VALUE_INF;

        for (const IterativeWorkerResult& result : results) {
            if (result.best.depth <= 0 || move_is_none(result.best.best_move))
                continue;
            min_score = std::min(min_score, result.best.score);
        }

        if (min_score == VALUE_INF)
            return best_index;

        std::vector<std::pair<Move, long long>> vote_map;
        vote_map.reserve(results.size());

        const auto vote_value = [min_score](const IterativeWorkerResult& result) noexcept {
            if (result.best.depth <= 0 || move_is_none(result.best.best_move))
                return 0LL;
            return static_cast<long long>(result.best.score - min_score + 10)
                * static_cast<long long>(result.best.depth);
        };

        for (const IterativeWorkerResult& result : results) {
            if (result.best.depth <= 0 || move_is_none(result.best.best_move))
                continue;

            const long long votes = vote_value(result);
            const Move move = result.best.best_move;
            auto it = std::find_if(
                vote_map.begin(),
                vote_map.end(),
                [move](const auto& entry) noexcept { return entry.first == move; }
            );
            if (it == vote_map.end())
                vote_map.emplace_back(move, votes);
            else
                it->second += votes;
        }

        const auto move_votes = [&vote_map](Move move) noexcept {
            const auto it = std::find_if(
                vote_map.begin(),
                vote_map.end(),
                [move](const auto& entry) noexcept { return entry.first == move; }
            );
            return it == vote_map.end() ? 0LL : it->second;
        };

        for (int i = 0; i < static_cast<int>(results.size()); ++i) {
            const IterativeWorkerResult& candidate = results[static_cast<std::size_t>(i)];
            const IterativeWorkerResult& incumbent = results[static_cast<std::size_t>(best_index)];

            if (candidate.best.depth <= 0 || move_is_none(candidate.best.best_move))
                continue;
            if (incumbent.best.depth <= 0 || move_is_none(incumbent.best.best_move)) {
                best_index = i;
                continue;
            }

            const long long candidate_move_votes = move_votes(candidate.best.best_move);
            const long long incumbent_move_votes = move_votes(incumbent.best.best_move);

            if (candidate_move_votes > incumbent_move_votes ||
                (candidate_move_votes == incumbent_move_votes &&
                 candidate.best.depth > incumbent.best.depth) ||
                (candidate_move_votes == incumbent_move_votes &&
                 candidate.best.depth == incumbent.best.depth &&
                 vote_value(candidate) > vote_value(incumbent)) ||
                (candidate_move_votes == incumbent_move_votes &&
                 candidate.best.depth == incumbent.best.depth &&
                 vote_value(candidate) == vote_value(incumbent) &&
                 candidate.best.score > incumbent.best.score)) {
                best_index = i;
            }
        }

        return best_index;
    };

    SearchLimits main_limits = limits;
    SearchTimeSignals time_signals{};
    std::atomic<bool> shared_stop{false};
    std::atomic<u64> shared_nodes{0};
    std::atomic<u64> shared_tb_hits{limits.root_tb_hits};
    main_limits.thread_id = 0;
    main_limits.thread_count = std::max(1, limits.thread_count);
    main_limits.report_info = true;
    main_limits.external_stop = limits.stop;
    main_limits.stop = &shared_stop;
    main_limits.shared_nodes = &shared_nodes;
    main_limits.shared_tb_hits = &shared_tb_hits;
    if (main_limits.use_time_management)
        main_limits.time_signals = &time_signals;
    if (main_limits.multipv > 1) {
        main_limits.use_msv_smp = false;
        main_limits.msv_info = false;
    }
    RootMsvShared root_msv_state{};
    if (main_limits.use_msv_smp && main_limits.thread_count > 1) {
        root_msv_seed_moves(root, mem, main_limits, root_msv_state);
        main_limits.root_msv = &root_msv_state;
    } else {
        main_limits.root_msv = nullptr;
    }

    Searcher main_searcher(mem, session_state.worker(0), main_limits);
    const int worker_count = main_limits.thread_count;
    std::vector<IterativeWorkerResult> results(static_cast<std::size_t>(worker_count));
    std::vector<std::unique_ptr<RootWorker>> helpers;
    std::vector<std::thread> helper_threads;
    helpers.reserve(static_cast<std::size_t>(std::max(0, worker_count - 1)));
    helper_threads.reserve(static_cast<std::size_t>(std::max(0, worker_count - 1)));

    for (int i = 1; i < worker_count; ++i) {
        SearchLimits helper_limits = main_limits;
        helper_limits.thread_id = i;
        helper_limits.report_info = false;
        helper_limits.singular_telemetry = false;
        helper_limits.soft_time_ms = 0;
        helper_limits.hard_time_ms = 0;
        helper_limits.infinite = true;

        auto helper = std::make_unique<RootWorker>(
            mem,
            session_state.worker(static_cast<std::size_t>(i)),
            helper_limits
        );
        RootWorker* helper_ptr = helper.get();
        helper_threads.emplace_back([&, i, helper_ptr]() {
            results[static_cast<std::size_t>(i)] =
                iterative_deepening_worker(
                    helper_ptr->searcher,
                    root,
                    nullptr,
                    search_start
                );
        });
        helpers.push_back(std::move(helper));
    }

    results[0] = iterative_deepening_worker(
        main_searcher,
        root,
        out,
        search_start
    );
    shared_stop.store(true, std::memory_order_release);

    for (std::thread& thread : helper_threads)
        thread.join();

    const int best_index = select_lazy_smp_best_index(results);
    IterativeWorkerResult best = results[static_cast<std::size_t>(best_index)];

    const bool selected_needs_ponder_recovery =
        main_limits.recover_ponder_pv &&
        best.best.depth >= 2 &&
        !move_is_none(best.best.best_move) &&
        (best.pv_length < 2 || best.pv[0] != best.best.best_move);
    const bool timed_search =
        !main_limits.infinite && main_limits.hard_time_ms > 0;
    const int wall_elapsed_before_recovery = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            Searcher::clock::now() - search_start
        ).count()
    );
    const bool recovery_has_time =
        !timed_search || wall_elapsed_before_recovery < main_limits.hard_time_ms;
    if (selected_needs_ponder_recovery && recovery_has_time) {
        std::atomic<bool> recovery_stop{false};
        SearchLimits recovery_limits = main_limits;
        recovery_limits.thread_count = 1;
        recovery_limits.thread_id = 0;
        recovery_limits.report_info = false;
        recovery_limits.singular_telemetry = false;
        recovery_limits.soft_time_ms = 0;
        recovery_limits.use_time_management = false;
        recovery_limits.ponder = false;
        if (!timed_search) {
            recovery_limits.hard_time_ms = 0;
            recovery_limits.infinite = true;
        }
        recovery_limits.stop = &recovery_stop;
        recovery_limits.external_stop = limits.stop;
        recovery_limits.shared_nodes = &shared_nodes;
        recovery_limits.shared_tb_hits = &shared_tb_hits;
        recovery_limits.use_msv_smp = false;
        recovery_limits.msv_info = false;
        recovery_limits.root_msv = nullptr;
        recovery_limits.time_signals = nullptr;

        Searcher recovery_searcher(mem, session_state.worker(0), recovery_limits);
        recovery_searcher.p2_accumulator_stack.reset();
        recovery_searcher.completed_depth = best.best.depth;
        recovery_searcher.root_depth = best.best.depth;
        reset_searcher_iteration(
            recovery_searcher,
            timed_search ? search_start : Searcher::clock::now(),
            0
        );
        SearchResult recovered = best.best;
        const int score_before_recovery = recovered.score;
        const memory::Bound bound_before_recovery = best.score_bound;
        (void)recovery_searcher.recover_ponder_pv_full_window(
            root,
            recovered,
            recovered.depth
        );
        if (!recovery_searcher.stopped &&
            recovery_searcher.pv_length[0] > 0 &&
            recovery_searcher.pv_table[0][0] == recovered.best_move) {
            recovery_searcher.publish_nodes();
            recovery_searcher.publish_tb_hits();
            if (main_limits.multipv > 1)
                recovered.score = score_before_recovery;
            best.best = recovered;
            best.pv_length = recovery_searcher.pv_length[0];
            for (int i = 0; i < best.pv_length; ++i) {
                best.pv[i] = recovery_searcher.pv_table[0][i];
                best.best.pv[i] = best.pv[i];
            }
            best.best.pv_length = best.pv_length;
            best.score_bound = main_limits.multipv > 1
                ? bound_before_recovery
                : memory::BOUND_EXACT;
            if (!best.lines.empty()) {
                copy_result_to_root_line(
                    best.lines.front(),
                    best.best,
                    best.pv,
                    best.pv_length,
                    best.score_bound
                );
            }
        }
    }

    best.best.nodes = shared_nodes.load(std::memory_order_relaxed);
    best.best.tb_hits = shared_tb_hits.load(std::memory_order_relaxed);
    const int searched_pv_length = best.pv_length;
    extend_syzygy_pv(
        root,
        mem,
        main_limits,
        best.pv,
        best.pv_length,
        best.best.score
    );
    best.best.pv_length = best.pv_length;
    for (int i = 0; i < best.pv_length; ++i)
        best.best.pv[i] = best.pv[i];
    if (!best.lines.empty()) {
        copy_result_to_root_line(
            best.lines.front(),
            best.best,
            best.pv,
            best.pv_length,
            best.score_bound
        );
    }

    if ((best_index != 0 || best.pv_length > searched_pv_length) &&
        out != nullptr) {
        if (!best.lines.empty()) {
            emit_root_lines_info(
                *out,
                mem,
                root,
                main_searcher,
                best.lines,
                static_cast<int>(best.lines.size()),
                search_start,
                best.best.nodes,
                best.best.tb_hits,
                best.best.depth
            );
        } else {
            emit_iteration_info(
                *out,
                mem,
                root,
                main_searcher,
                best.best,
                best.pv,
                best.pv_length,
                search_start,
                best.best.nodes,
                best.best.tb_hits,
                best.best.depth,
                best.score_bound,
                1
            );
        }
    }

    if (main_limits.use_time_management &&
        main_limits.time_state != nullptr &&
        best.best.depth > 0 &&
        !move_is_none(best.best.best_move)) {
        main_limits.time_state->previous_score = best.best.score;
        main_limits.time_state->previous_average_score =
            best.best_average_score == VALUE_NONE
                ? best.best.score
                : best.best_average_score;
        main_limits.time_state->previous_time_reduction = results[0].time_reduction;
    }

    return best.best;
}

SearchResult iterative_deepening(
    const Position& root,
    memory::Memory& mem,
    SearchSessionState& session_state,
    const SearchLimits& limits,
    std::ostream* out
) {
    const auto search_start =
        limits.start_time.time_since_epoch().count() != 0
            ? limits.start_time
            : Searcher::clock::now();
    SearchLimits prepared_limits = limits;
    const std::size_t active_workers = static_cast<std::size_t>(
        std::max(1, prepared_limits.thread_count)
    );
    session_state.ensure_workers(active_workers);
    (void) session_state.begin_search(active_workers);
    syzygy::RootProbe root_probe{};
    if (syzygy::rank_root_moves(
            root,
            mem,
            prepared_limits.syzygy_probe_limit,
            prepared_limits.syzygy_50_move_rule,
            prepared_limits.root_moves,
            prepared_limits.root_move_count,
            root_probe
        )) {
        prepared_limits.root_move_count = root_probe.move_count;
        for (int i = 0; i < root_probe.move_count; ++i)
            prepared_limits.root_moves[i] = root_probe.moves[i];
        prepared_limits.root_in_tb = true;
        prepared_limits.root_tb_wdl = static_cast<int>(root_probe.wdl);
        prepared_limits.root_tb_hits = 1;
    }

    SearchResult result{};
    if (prepared_limits.thread_count <= 1) {
        result = iterative_deepening_single(
            root,
            mem,
            session_state,
            prepared_limits,
            out,
            search_start
        );
    } else {
        result = iterative_deepening_lazy_smp(
            root,
            mem,
            session_state,
            prepared_limits,
            out,
            search_start
        );
    }

    return result;
}

} // namespace magnus::search
