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

#include "See.h"

#include <algorithm>

#include "board/Attack.h"
#include "board/MoveGen.h"
#include "board/Position.h"

/*
 * SEE (靜態交換評估) 實作 — Static Exchange Evaluation
 *
 * 演算法：在不實際走棋的情況下，模擬目標格上的連續兌子交換。
 * 使用 LVA (Least Valuable Attacker) 順序：兵→馬→象→車→后→王。
 * 每次交換後更新攻擊者集合（滑子攻擊線可能被打開）。
 *
 * 兩個核心函數：
 *   see_value_fast() — 完整計算捕獲的淨收益（用於著法排序）
 *   see_ge_fast()    — 判斷捕獲是否達到指定閾值（含提前退出，用於剪枝）
 *
 * 棋子價值表（厘兵）：兵=100, 馬=320, 象=330, 車=500, 后=900, 王=20000
 */
namespace magnus::search {

namespace {

// SEE 棋子價值表 — 國王設為 20000 確保捕獲國王總是最優先/最昂貴的交換
constexpr int see_piece_value[PIECE_TYPE_NB] = {
    100, 320, 330, 500, 900, 20000
};
// 最大交換步數 — 防止極端情況下的無限循環
constexpr int SEE_MAX_SWAPS = 64;

[[nodiscard]] inline Bitboard lsb_bit(Bitboard bb) noexcept {
    return bb & (0ULL - bb);
}

[[nodiscard]] inline bool pick_lva_attacker(
    Bitboard side_attackers,
    const Bitboard piece_bb[PIECE_NB],
    PieceType& attacker,
    Bitboard& from_set
) noexcept {
    Bitboard by_pt = side_attackers & piece_bb[PAWN];
    if (by_pt) {
        attacker = PAWN;
        from_set = lsb_bit(by_pt);
        return true;
    }

    by_pt = side_attackers & piece_bb[KNIGHT];
    if (by_pt) {
        attacker = KNIGHT;
        from_set = lsb_bit(by_pt);
        return true;
    }

    by_pt = side_attackers & piece_bb[BISHOP];
    if (by_pt) {
        attacker = BISHOP;
        from_set = lsb_bit(by_pt);
        return true;
    }

    by_pt = side_attackers & piece_bb[ROOK];
    if (by_pt) {
        attacker = ROOK;
        from_set = lsb_bit(by_pt);
        return true;
    }

    by_pt = side_attackers & piece_bb[QUEEN];
    if (by_pt) {
        attacker = QUEEN;
        from_set = lsb_bit(by_pt);
        return true;
    }

    by_pt = side_attackers & piece_bb[KING];
    if (by_pt) {
        attacker = KING;
        from_set = lsb_bit(by_pt);
        return true;
    }

    return false;
}

} // namespace

int see_value(
    const Position& pos,
    const memory::Memory& mem,
    Move move
) noexcept {
    if (!move_is_capture(move))
        return 0;

    return see_value_fast(pos, mem, move);
}

int see_value_fast(
    const Position& pos,
    const memory::Memory& mem,
    Move move
) noexcept {
    if (!move_is_capture(move))
        return 0;

    const Color us = static_cast<Color>(pos.side_to_move);
    const Square from = from_sq(move);
    const Square to = to_sq(move);
    const auto& piece_bb = pos.piece_bb;
    const auto& color_bb = pos.color_bb;

    PieceType moving = piece_type_on(pos, from);
    if (!is_ok(moving))
        return 0;

    const PieceType captured = move_is_ep(move)
        ? PAWN
        : piece_type_on(pos, to);
    if (!is_ok(captured))
        return 0;

    int gain[SEE_MAX_SWAPS]{};
    int depth = 0;
    gain[0] = see_piece_value[captured];

    if (move_is_promotion(move)) {
        const PieceType promo = promo_piece(move);
        if (!is_ok(promo))
            return 0;
        gain[0] += see_piece_value[promo] - see_piece_value[PAWN];
        moving = promo;
    }

    PieceType next_victim = moving;

    Bitboard occupied = pos.occupied ^ bb_of(from);
    if (move_is_ep(move)) {
        const Square cap_sq = (us == WHITE) ? (to - 8) : (to + 8);
        occupied ^= bb_of(cap_sq);
    } else {
        occupied ^= bb_of(to);
    }

    const Bitboard bishop_like = piece_bb[BISHOP] | piece_bb[QUEEN];
    const Bitboard rook_like = piece_bb[ROOK] | piece_bb[QUEEN];
    Bitboard attackers = attackers_to(pos, mem, to, occupied);
    Color side = static_cast<Color>(us ^ 1);

    while (true) {
        const Bitboard side_attackers = (attackers & occupied) & color_bb[side];
        if (side_attackers == 0ULL || depth + 1 >= SEE_MAX_SWAPS)
            break;

        PieceType attacker = PIECE_TYPE_NONE;
        Bitboard from_set = 0ULL;
        if (!pick_lva_attacker(side_attackers, piece_bb, attacker, from_set))
            break;

        const int prev = depth;
        ++depth;
        gain[depth] = see_piece_value[next_victim] - gain[prev];
        next_victim = attacker;

        occupied ^= from_set;

        if (attacker == PAWN || attacker == BISHOP || attacker == QUEEN)
            attackers |= bishop_attacks(mem, to, occupied) & bishop_like;
        if (attacker == ROOK || attacker == QUEEN)
            attackers |= rook_attacks(mem, to, occupied) & rook_like;

        if (attacker == KING) {
            const Color opponent = static_cast<Color>(side ^ 1);
            if ((attackers & occupied & color_bb[opponent]) != 0ULL) {
                --depth;
                break;
            }
        }

        side = static_cast<Color>(side ^ 1);
    }

    while (depth > 0) {
        gain[depth - 1] = -std::max(-gain[depth - 1], gain[depth]);
        --depth;
    }

    return gain[0];
}

bool see_ge(
    const Position& pos,
    const memory::Memory& mem,
    Move move,
    int threshold
) noexcept {
    if (!move_is_capture(move))
        return threshold <= 0;

    return see_ge_fast(pos, mem, move, threshold);
}

bool see_ge_fast(
    const Position& pos,
    const memory::Memory& mem,
    Move move,
    int threshold
) noexcept {
    if (!move_is_capture(move))
        return threshold <= 0;

    const Color us = static_cast<Color>(pos.side_to_move);
    const Square from = from_sq(move);
    const Square to = to_sq(move);
    const auto& piece_bb = pos.piece_bb;
    const auto& color_bb = pos.color_bb;
    const PieceType moving = piece_type_on(pos, from);
    if (!is_ok(moving))
        return false;

    const PieceType captured = move_is_ep(move)
        ? PAWN
        : piece_type_on(pos, to);
    if (!is_ok(captured))
        return false;

    int balance = see_piece_value[captured] - threshold;
    PieceType next_victim = moving;

    if (move_is_promotion(move)) {
        const PieceType promo = promo_piece(move);
        if (!is_ok(promo))
            return false;
        balance += see_piece_value[promo] - see_piece_value[PAWN];
        next_victim = promo;
    }

    if (balance < 0)
        return false;

    balance -= see_piece_value[next_victim];
    if (balance >= 0)
        return true;

    Bitboard occupied = pos.occupied ^ bb_of(from);
    if (move_is_ep(move)) {
        const Square cap_sq = (us == WHITE) ? (to - 8) : (to + 8);
        occupied ^= bb_of(cap_sq);
    } else {
        occupied ^= bb_of(to);
    }

    const Bitboard bishop_like = piece_bb[BISHOP] | piece_bb[QUEEN];
    const Bitboard rook_like = piece_bb[ROOK] | piece_bb[QUEEN];
    Bitboard attackers = attackers_to(pos, mem, to, occupied);

    Color side = static_cast<Color>(us ^ 1);
    while (true) {
        const Bitboard side_attackers = (attackers & occupied) & color_bb[side];
        if (side_attackers == 0ULL)
            break;

        PieceType attacker = PIECE_TYPE_NONE;
        Bitboard from_set = 0ULL;
        if (!pick_lva_attacker(side_attackers, piece_bb, attacker, from_set))
            break;

        occupied ^= from_set;

        if (attacker == PAWN || attacker == BISHOP || attacker == QUEEN)
            attackers |= bishop_attacks(mem, to, occupied) & bishop_like;
        if (attacker == ROOK || attacker == QUEEN)
            attackers |= rook_attacks(mem, to, occupied) & rook_like;

        balance = see_piece_value[attacker] - balance;
        side = static_cast<Color>(side ^ 1);

        if (balance >= 0) {
            const Bitboard remaining_attackers = attackers & occupied;
            if (attacker == KING && (remaining_attackers & color_bb[side]) != 0ULL)
                side = static_cast<Color>(side ^ 1);
            break;
        }
    }

    return side != us;
}

} // namespace magnus::search
