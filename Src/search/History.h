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

/* ===== ANNOTATED: 繁體中文註釋已添加 =====
 * 本檔案是 MagnusChessX Thinking 西洋棋引擎的一部分。
 * 詳細說明請參閱對應的 .cpp 實作檔案。
 */


#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>

#include "HistoryContext.h"
#include "board/MoveGen.h"
#include "board/Position.h"
#include "Search.h"
#include "Types.h"

namespace magnus::search {

#ifndef MAGNUS_SEE_BIAS_BONUS_DIV
#define MAGNUS_SEE_BIAS_BONUS_DIV 10
#endif

#if MAGNUS_SEE_BIAS_BONUS_DIV <= 0
#error "MAGNUS_SEE_BIAS_BONUS_DIV must be positive"
#endif

#ifndef MAGNUS_SEE_BIAS_MED_LOW_BONUS_DIV
#define MAGNUS_SEE_BIAS_MED_LOW_BONUS_DIV 24
#endif

#if MAGNUS_SEE_BIAS_MED_LOW_BONUS_DIV <= 0
#error "MAGNUS_SEE_BIAS_MED_LOW_BONUS_DIV must be positive"
#endif

#ifndef MAGNUS_SEE_BIAS_FAIL_SCALE
#define MAGNUS_SEE_BIAS_FAIL_SCALE 4
#endif

#if MAGNUS_SEE_BIAS_FAIL_SCALE <= 0
#error "MAGNUS_SEE_BIAS_FAIL_SCALE must be positive"
#endif

#ifndef MAGNUS_SEE_BIAS_BAD_THRESHOLD
#define MAGNUS_SEE_BIAS_BAD_THRESHOLD -50
#endif

#ifndef MAGNUS_SEE_BIAS_EQ_THRESHOLD
#define MAGNUS_SEE_BIAS_EQ_THRESHOLD 50
#endif

#ifndef MAGNUS_SEE_BIAS_GOOD_BIG_THRESHOLD
#define MAGNUS_SEE_BIAS_GOOD_BIG_THRESHOLD 200
#endif

#ifndef MAGNUS_COUNTERMOVE_BONUS
#define MAGNUS_COUNTERMOVE_BONUS 4096
#endif

#ifndef MAGNUS_QUIET_HISTORY_WEIGHT
#define MAGNUS_QUIET_HISTORY_WEIGHT 1
#endif

#ifndef MAGNUS_COUNTERMOVE_WEIGHT_NUM
#define MAGNUS_COUNTERMOVE_WEIGHT_NUM 1
#endif

#ifndef MAGNUS_COUNTERMOVE_WEIGHT_DEN
#define MAGNUS_COUNTERMOVE_WEIGHT_DEN 1
#endif

#ifndef MAGNUS_CONT1_WEIGHT_NUM
#define MAGNUS_CONT1_WEIGHT_NUM 1
#endif

#ifndef MAGNUS_CONT1_WEIGHT_DEN
#define MAGNUS_CONT1_WEIGHT_DEN 1
#endif

#ifndef MAGNUS_CONT2_WEIGHT_NUM
#define MAGNUS_CONT2_WEIGHT_NUM 1
#endif

#ifndef MAGNUS_CONT2_WEIGHT_DEN
#define MAGNUS_CONT2_WEIGHT_DEN 4
#endif

#ifndef MAGNUS_CONT8_WEIGHT_NUM
#define MAGNUS_CONT8_WEIGHT_NUM 1
#endif

#ifndef MAGNUS_CONT8_WEIGHT_DEN
#define MAGNUS_CONT8_WEIGHT_DEN 8
#endif

#ifndef MAGNUS_CAPTURE_HISTORY_WEIGHT
#define MAGNUS_CAPTURE_HISTORY_WEIGHT 1
#endif

#ifndef MAGNUS_CAPTURE_IMM_SEE_WEIGHT
#define MAGNUS_CAPTURE_IMM_SEE_WEIGHT 1
#endif

#ifndef MAGNUS_CAPTURE_SEE_BIAS_WEIGHT
#define MAGNUS_CAPTURE_SEE_BIAS_WEIGHT 1
#endif

enum class SeeClass : std::uint8_t {
    LossBig = 0,
    LossSmall,
    Equal,
    GainSmall,
    GainBig,
    Promo,
    Check,
    Count
};

enum class DepthClass : std::uint8_t {
    Shallow = 0,
    MediumLow,
    MediumHigh,
    Deep,
    Count
};

enum class SeeScalePreset : std::uint8_t {
    Weak = 0,
    Medium,
    Strong
};

#ifndef MAGNUS_SEE_TERM_PRESET
#define MAGNUS_SEE_TERM_PRESET 1
#endif

#if MAGNUS_SEE_TERM_PRESET == 0
constexpr SeeScalePreset HISTORY_ORDERING_SEE_TERM_PRESET = SeeScalePreset::Weak;
#elif MAGNUS_SEE_TERM_PRESET == 1
constexpr SeeScalePreset HISTORY_ORDERING_SEE_TERM_PRESET = SeeScalePreset::Medium;
#elif MAGNUS_SEE_TERM_PRESET == 2
constexpr SeeScalePreset HISTORY_ORDERING_SEE_TERM_PRESET = SeeScalePreset::Strong;
#else
#error "MAGNUS_SEE_TERM_PRESET must be 0 (Weak), 1 (Medium), or 2 (Strong)"
#endif

struct KillerTable {
    Move table[MAX_PLY][2]{};
};

struct QuietHistoryTable {
    i16 value[COLOR_NB][SQ_NB][SQ_NB]{};
};

struct CaptureHistoryTable {
    i16 value[COLOR_NB][PIECE_TYPE_NB][SQ_NB][PIECE_TYPE_NB]{};
};

struct CounterMoveTable {
    Move value[COLOR_NB][SQ_NB][SQ_NB]{};
};

struct ContinuationHistoryTable {
    i16 value[COLOR_NB][PIECE_TYPE_NB][SQ_NB][PIECE_TYPE_NB][SQ_NB]{};
};

struct SeeBiasTable {
    i16 value[static_cast<int>(DepthClass::Count)][static_cast<int>(SeeClass::Count)]{};
};

// Pawn history: indexed by pawn-structure hash, piece type and target square.
// Captures tactical patterns that recur in similar pawn formations.
constexpr int PAWN_HISTORY_SIZE = 512;
constexpr int CONTINUATION_HISTORY_SLOTS = 3;
constexpr int CONTINUATION_PLY2_SLOT = 0;
constexpr int CONTINUATION_PLY4_SLOT = 1;
constexpr int CONTINUATION_PLY8_SLOT = 2;
constexpr i32 HISTORY_VALUE_LIMIT = 32767;
constexpr i32 SEE_BIAS_VALUE_LIMIT = 2048;

inline void update_history_value(
    i16& value,
    int bonus,
    i32 limit = HISTORY_VALUE_LIMIT
) noexcept {
    const i32 bounded_limit = std::clamp<i32>(limit, 1, 32767);
    const i32 bounded_bonus = std::clamp<i32>(bonus, -bounded_limit, bounded_limit);
    const i32 current = static_cast<i32>(value);
    const i32 gravity = (current * std::abs(bounded_bonus)) / bounded_limit;
    value = static_cast<i16>(std::clamp(
        current + bounded_bonus - gravity,
        -bounded_limit,
        bounded_limit
    ));
}

[[nodiscard]] inline int pawn_history_index(const Position& pos) noexcept {
    const Key pawn_key =
        pieces(pos, WHITE, PAWN)
        ^ std::rotl(pieces(pos, BLACK, PAWN), 7);
    return static_cast<int>(pawn_key % PAWN_HISTORY_SIZE);
}

struct PawnHistoryTable {
    i16 value[PAWN_HISTORY_SIZE][PIECE_TYPE_NB][SQ_NB]{};
};

[[nodiscard]] DepthClass depth_class(int depth) noexcept;
[[nodiscard]] SeeClass classify_see(int see_value, bool gives_check, bool is_promotion) noexcept;
[[nodiscard]] SeeClass classify_see_bias(int see_value) noexcept;
[[nodiscard]] int history_bonus(int depth) noexcept;
[[nodiscard]] int history_penalty(int depth) noexcept;
[[nodiscard]] int see_immediate_term(int see_value, SeeScalePreset preset) noexcept;

/*
HistoryTables separates quiet and capture experience, while keeping killers and
countermove hints in one place for move ordering and light pruning signals.
*/
/*
 * HistoryTables — 歷史啟發式查表集合
 *
 * 包含以下子表（全部以 [Color][PieceType][Square] 或類似維度索引）：
 *   quiet_history     — 安靜著法歷史（主歷史表）
 *   capture_history   — 捕獲著法歷史
 *   continuation[3]   — 延續歷史（按 2/4/8 ply 距離分槽）
 *   countermove       — 反著表（對前一步的直接回應）
 *   pawn_history      — 兵結構歷史（基於兵 hash）
 *   see_bias          — SEE 偏差表（按深度類別×SEE類別調整捕獲排序）
 *   killers           — 殺手著法表 [MAX_PLY][2]
 *
 * 更新機制：
 *   bonus  — 著法導致截斷時增加權重（增量 = depth²）
 *   penalty — 被搜索但未截斷的著法減少權重（減量 = depth²×4）
 *   所有更新使用有界重力公式，越接近上限時同方向更新越小
 */
struct HistoryTables {
    KillerTable killers{};
    QuietHistoryTable quiet{};
    CaptureHistoryTable capture{};
    CounterMoveTable countermove{};
    ContinuationHistoryTable continuation[CONTINUATION_HISTORY_SLOTS]{};
    SeeBiasTable see_bias{};
    PawnHistoryTable pawn_history{};

    void clear() noexcept;

    [[nodiscard]] inline Move killer_fast(int ply, int slot) const noexcept {
        return killers.table[ply][slot];
    }
    [[nodiscard]] inline i32 quiet_value_fast(const Position& pos, Move move) const noexcept {
        if (move_is_capture(move))
            return 0;

        const Color side = static_cast<Color>(pos.side_to_move);
        return quiet.value[side][from_sq(move)][to_sq(move)];
    }
    [[nodiscard]] inline i32 capture_value_fast(const Position& pos, Move move) const noexcept {
        if (!move_is_capture(move))
            return 0;

        const Color side = static_cast<Color>(pos.side_to_move);
        const PieceType mover = piece_type_on(pos, from_sq(move));
        const PieceType captured = move_is_ep(move) ? PAWN : piece_type_on(pos, to_sq(move));
        return capture.value[side][mover][to_sq(move)][captured];
    }
    [[nodiscard]] inline i32 see_bias_value_fast(int depth, int see_value) const noexcept {
        const int dc = static_cast<int>(depth_class(depth));
        const int sc = static_cast<int>(classify_see_bias(see_value));
        const i32 raw = static_cast<i32>(see_bias.value[dc][sc]);
        return std::clamp(raw / 4, -96, 96);
    }
    [[nodiscard]] inline i32 pawn_history_value_fast(const Position& pos, Move move) const noexcept {
        if (move_is_capture(move))
            return 0;
        const int idx = pawn_history_index(pos);
        const PieceType pt = piece_type_on(pos, from_sq(move));
        if (!is_ok(pt))
            return 0;
        return static_cast<i32>(pawn_history.value[idx][pt][to_sq(move)]);
    }
    inline void bonus_pawn_history_fast(const Position& pos, Move move, int depth) noexcept {
        if (move_is_capture(move))
            return;
        const int idx = pawn_history_index(pos);
        const PieceType pt = piece_type_on(pos, from_sq(move));
        if (!is_ok(pt))
            return;
        i16& h = pawn_history.value[idx][pt][to_sq(move)];
        update_history_value(h, history_bonus(depth));
    }
    inline void penalty_pawn_history_fast(const Position& pos, Move move, int depth) noexcept {
        if (move_is_capture(move))
            return;
        const int idx = pawn_history_index(pos);
        const PieceType pt = piece_type_on(pos, from_sq(move));
        if (!is_ok(pt))
            return;
        i16& h = pawn_history.value[idx][pt][to_sq(move)];
        update_history_value(h, -history_penalty(depth));
    }
    [[nodiscard]] inline Move countermove_fast(
        const Position& pos,
        Move prev_move
    ) const noexcept {
        if (move_is_none(prev_move))
            return Move(0);

        const Color side = static_cast<Color>(pos.side_to_move);
        return countermove.value[side][from_sq(prev_move)][to_sq(prev_move)];
    }
    [[nodiscard]] inline i32 countermove_bonus_fast(
        const Position& pos,
        Move move,
        Move prev_move
    ) const noexcept {
        if (move_is_capture(move) || move_is_none(prev_move))
            return 0;

        return countermove_fast(pos, prev_move) == move ? MAGNUS_COUNTERMOVE_BONUS : 0;
    }
    [[nodiscard]] inline i32 continuation_value_fast(
        const Position& pos,
        Move move,
        ContinuationHistoryContext previous,
        int slot
    ) const noexcept {
        if (move_is_capture(move) || !previous.valid())
            return 0;

        const Color side = static_cast<Color>(pos.side_to_move);
        const PieceType cur_piece = piece_type_on(pos, from_sq(move));
        if (!is_ok(cur_piece))
            return 0;

        const int idx = std::clamp(slot, 0, CONTINUATION_HISTORY_SLOTS - 1);
        return continuation[idx]
            .value[side][previous.piece][previous.to][cur_piece][to_sq(move)];
    }
    [[nodiscard]] inline i32 quiet_ordering_score_fast(
        const Position& pos,
        Move move,
        Move prev_move,
        ContinuationHistoryContext prev2,
        ContinuationHistoryContext prev4,
        ContinuationHistoryContext prev8
    ) const noexcept {
        if (move_is_capture(move))
            return 0;

        i32 score = 0;
        score += MAGNUS_QUIET_HISTORY_WEIGHT * quiet_value_fast(pos, move);
        score += (MAGNUS_COUNTERMOVE_WEIGHT_NUM * countermove_bonus_fast(pos, move, prev_move))
            / MAGNUS_COUNTERMOVE_WEIGHT_DEN;
        score += (MAGNUS_CONT1_WEIGHT_NUM
            * continuation_value_fast(pos, move, prev2, CONTINUATION_PLY2_SLOT))
            / MAGNUS_CONT1_WEIGHT_DEN;
        score += (MAGNUS_CONT2_WEIGHT_NUM
            * continuation_value_fast(pos, move, prev4, CONTINUATION_PLY4_SLOT))
            / MAGNUS_CONT2_WEIGHT_DEN;
        score += (MAGNUS_CONT8_WEIGHT_NUM
            * continuation_value_fast(pos, move, prev8, CONTINUATION_PLY8_SLOT))
            / MAGNUS_CONT8_WEIGHT_DEN;
        score += pawn_history_value_fast(pos, move);
        return score;
    }
    [[nodiscard]] inline i32 capture_ordering_score_fast(
        const Position& pos,
        Move move,
        int depth,
        int see_value
    ) const noexcept {
        if (!move_is_capture(move))
            return 0;

        i32 score = 0;
        score += MAGNUS_CAPTURE_HISTORY_WEIGHT * capture_value_fast(pos, move);
        score += MAGNUS_CAPTURE_IMM_SEE_WEIGHT
            * see_immediate_term(see_value, HISTORY_ORDERING_SEE_TERM_PRESET);
        score += MAGNUS_CAPTURE_SEE_BIAS_WEIGHT * see_bias_value_fast(depth, see_value);
        return score;
    }

    inline void bonus_fast(const Position& pos, Move move, int depth) noexcept {
        if (move_is_capture(move))
            return;

        const Color side = static_cast<Color>(pos.side_to_move);
        i16& h = quiet.value[side][from_sq(move)][to_sq(move)];
        update_history_value(h, history_bonus(depth));
    }
    inline void penalty_fast(const Position& pos, Move move, int depth) noexcept {
        if (move_is_capture(move))
            return;

        const Color side = static_cast<Color>(pos.side_to_move);
        i16& h = quiet.value[side][from_sq(move)][to_sq(move)];
        update_history_value(h, -history_penalty(depth));
    }
    inline void bonus_capture_fast(const Position& pos, Move move, int depth) noexcept {
        if (!move_is_capture(move))
            return;

        const Color side = static_cast<Color>(pos.side_to_move);
        const PieceType mover = piece_type_on(pos, from_sq(move));
        const PieceType captured = move_is_ep(move) ? PAWN : piece_type_on(pos, to_sq(move));
        i16& h = capture.value[side][mover][to_sq(move)][captured];
        update_history_value(h, history_bonus(depth));
    }
    inline void penalty_capture_fast(const Position& pos, Move move, int depth) noexcept {
        if (!move_is_capture(move))
            return;

        const Color side = static_cast<Color>(pos.side_to_move);
        const PieceType mover = piece_type_on(pos, from_sq(move));
        const PieceType captured = move_is_ep(move) ? PAWN : piece_type_on(pos, to_sq(move));
        i16& h = capture.value[side][mover][to_sq(move)][captured];
        update_history_value(h, -history_penalty(depth));
    }
    inline void bonus_see_bias_fast(int depth, int see_value) noexcept {
        const DepthClass dc_class = depth_class(depth);
        const int dc = static_cast<int>(dc_class);
        const int sc = static_cast<int>(classify_see_bias(see_value));
        i16& h = see_bias.value[dc][sc];
        int bonus_div = MAGNUS_SEE_BIAS_BONUS_DIV;
        if (dc_class == DepthClass::MediumLow)
            bonus_div = MAGNUS_SEE_BIAS_MED_LOW_BONUS_DIV;
        const i32 bonus = std::max(1, history_bonus(depth) / bonus_div);
        update_history_value(h, bonus, SEE_BIAS_VALUE_LIMIT);
    }
    inline void penalty_see_bias_fast(int depth, int see_value) noexcept {
        const int dc = static_cast<int>(depth_class(depth));
        const int sc = static_cast<int>(classify_see_bias(see_value));
        i16& h = see_bias.value[dc][sc];
        const i32 penalty =
            history_bonus(depth) / (MAGNUS_SEE_BIAS_BONUS_DIV * MAGNUS_SEE_BIAS_FAIL_SCALE);
        if (penalty <= 0)
            return;
        update_history_value(h, -penalty, SEE_BIAS_VALUE_LIMIT);
    }
    inline void set_countermove_fast(
        const Position& pos,
        Move prev_move,
        Move reply
    ) noexcept {
        if (move_is_none(prev_move) || move_is_none(reply) || move_is_capture(reply))
            return;

        const Color side = static_cast<Color>(pos.side_to_move);
        countermove.value[side][from_sq(prev_move)][to_sq(prev_move)] = reply;
    }
    inline void bonus_continuation_fast(
        const Position& pos,
        ContinuationHistoryContext previous,
        Move move,
        int depth,
        int slot
    ) noexcept {
        if (move_is_capture(move) || !previous.valid())
            return;

        const Color side = static_cast<Color>(pos.side_to_move);
        const PieceType cur_piece = piece_type_on(pos, from_sq(move));
        if (!is_ok(cur_piece))
            return;

        const int idx = std::clamp(slot, 0, CONTINUATION_HISTORY_SLOTS - 1);
        i16& h = continuation[idx]
            .value[side][previous.piece][previous.to][cur_piece][to_sq(move)];
        update_history_value(h, history_bonus(depth));
    }
    inline void penalty_continuation_fast(
        const Position& pos,
        ContinuationHistoryContext previous,
        Move move,
        int depth,
        int slot
    ) noexcept {
        if (move_is_capture(move) || !previous.valid())
            return;

        const Color side = static_cast<Color>(pos.side_to_move);
        const PieceType cur_piece = piece_type_on(pos, from_sq(move));
        if (!is_ok(cur_piece))
            return;

        const int idx = std::clamp(slot, 0, CONTINUATION_HISTORY_SLOTS - 1);
        i16& h = continuation[idx]
            .value[side][previous.piece][previous.to][cur_piece][to_sq(move)];
        update_history_value(h, -history_penalty(depth));
    }

    inline void penalize_quiets_fast(
        const Position& pos,
        const Move* quiets,
        int count,
        Move excluded_move,
        int depth
    ) noexcept {
        for (int i = 0; i < count; ++i)
            if (quiets[i] != excluded_move) {
                penalty_fast(pos, quiets[i], depth);
                penalty_pawn_history_fast(pos, quiets[i], depth);
            }
    }
    inline void penalize_captures_fast(
        const Position& pos,
        const Move* caps,
        int count,
        Move excluded_move,
        int depth
    ) noexcept {
        for (int i = 0; i < count; ++i)
            if (caps[i] != excluded_move)
                penalty_capture_fast(pos, caps[i], depth);
    }
    inline void penalize_continuation_quiets_fast(
        const Position& pos,
        const Move* quiets,
        int count,
        Move excluded_move,
        ContinuationHistoryContext prev2,
        ContinuationHistoryContext prev4,
        ContinuationHistoryContext prev8,
        int depth
    ) noexcept {
        for (int i = 0; i < count; ++i) {
            if (quiets[i] == excluded_move)
                continue;

            penalty_continuation_fast(pos, prev2, quiets[i], depth, CONTINUATION_PLY2_SLOT);
            penalty_continuation_fast(
                pos,
                prev4,
                quiets[i],
                std::max(1, depth / 2),
                CONTINUATION_PLY4_SLOT
            );
            penalty_continuation_fast(
                pos,
                prev8,
                quiets[i],
                std::max(1, depth / 4),
                CONTINUATION_PLY8_SLOT
            );
        }
    }
    inline void penalize_see_bias_captures_fast(
        const Move* caps,
        const int* cap_see_values,
        int count,
        Move excluded_move,
        int depth
    ) noexcept {
        bool penalized_bad = false;
        bool penalized_eq = false;
        bool penalized_good_small = false;
        bool penalized_good_big = false;

        for (int i = 0; i < count; ++i) {
            if (caps[i] == excluded_move)
                continue;

            const SeeClass sc = classify_see_bias(cap_see_values[i]);
            bool* touched = nullptr;
            if (sc == SeeClass::LossSmall)
                touched = &penalized_bad;
            else if (sc == SeeClass::Equal)
                touched = &penalized_eq;
            else if (sc == SeeClass::GainSmall)
                touched = &penalized_good_small;
            else if (sc == SeeClass::GainBig)
                touched = &penalized_good_big;
            else
                continue;

            if (*touched)
                continue;

            penalty_see_bias_fast(depth, cap_see_values[i]);
            *touched = true;
        }
    }

    inline void reward_cutoff_fast(
        const Position& pos,
        Move move,
        int depth,
        int ply,
        int capture_see_value = 0,
        Move prev_move = Move(0),
        ContinuationHistoryContext prev2 = {},
        ContinuationHistoryContext prev4 = {},
        ContinuationHistoryContext prev8 = {}
    ) noexcept {
        if (move_is_capture(move)) {
            bonus_capture_fast(pos, move, depth);
            bonus_see_bias_fast(depth, capture_see_value);
            return;
        }

        if (killers.table[ply][0] != move) {
            killers.table[ply][1] = killers.table[ply][0];
            killers.table[ply][0] = move;
        }

        bonus_fast(pos, move, depth);
        bonus_pawn_history_fast(pos, move, depth);
        set_countermove_fast(pos, prev_move, move);
        bonus_continuation_fast(pos, prev2, move, depth, CONTINUATION_PLY2_SLOT);
        bonus_continuation_fast(
            pos,
            prev4,
            move,
            std::max(1, depth / 2),
            CONTINUATION_PLY4_SLOT
        );
        bonus_continuation_fast(
            pos,
            prev8,
            move,
            std::max(1, depth / 4),
            CONTINUATION_PLY8_SLOT
        );
    }
};

} // namespace magnus::search
