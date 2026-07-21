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
 * SEE (Static Exchange Evaluation) Implementation
 *
 * Algorithm: simulates successive piece exchanges on a target square without
 * actually making moves on the board. Uses LVA (Least Valuable Attacker) order:
 * Pawn → Knight → Bishop → Rook → Queen → King.
 * The attacker set is updated after each exchange (slider attack lines may open).
 *
 * Two core functions:
 *   see_value_fast() — computes the full net gain of a capture (for move ordering)
 *   see_ge_fast()    — tests whether a capture meets a threshold (with early exit, for pruning)
 *
 * Piece value table (centipawns): Pawn=100, Knight=320, Bishop=330, Rook=500, Queen=900, King=20000
 */
namespace magnus::search {

namespace {

// SEE piece value table — King set to 20000 to ensure capturing the king is always the highest-priority / most expensive exchange
constexpr int see_piece_value[PIECE_TYPE_NB] = {
    100, 320, 330, 500, 900, 20000
};
// Maximum exchange depth — prevents infinite loops in extreme cases
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
    Bitboard pinners[COLOR_NB]{};
    Bitboard pinned[COLOR_NB]{};
    bool pin_info_ready[COLOR_NB]{};
    Bitboard attackers = attackers_to(pos, mem, to, occupied);
    Color side = static_cast<Color>(us ^ 1);

    while (true) {
        Bitboard side_attackers = (attackers & occupied) & color_bb[side];
        if (side_attackers != 0ULL) {
            if (!pin_info_ready[side]) {
                pinners_and_pinned_bb(pos, mem, side, pinners[side], pinned[side]);
                pin_info_ready[side] = true;
            }

            // A pinned recapturer is unavailable while its original pinner remains.
            if ((pinners[side] & occupied) != 0ULL)
                side_attackers &= ~pinned[side];
        }
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
    Bitboard pinners[COLOR_NB]{};
    Bitboard pinned[COLOR_NB]{};
    bool pin_info_ready[COLOR_NB]{};
    Bitboard attackers = attackers_to(pos, mem, to, occupied);

    Color side = static_cast<Color>(us ^ 1);
    while (true) {
        Bitboard side_attackers = (attackers & occupied) & color_bb[side];
        if (side_attackers != 0ULL) {
            if (!pin_info_ready[side]) {
                pinners_and_pinned_bb(pos, mem, side, pinners[side], pinned[side]);
                pin_info_ready[side] = true;
            }

            // A pinned recapturer is unavailable while its original pinner remains.
            if ((pinners[side] & occupied) != 0ULL)
                side_attackers &= ~pinned[side];
        }
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

        if (attacker == KING) {
            const Color opponent = static_cast<Color>(side ^ 1);
            const bool destination_attacked =
                (attackers & occupied & color_bb[opponent]) != 0ULL;

            // Pinned pieces still attack squares for king-move legality.  Stop
            // here instead of passing them through the recapturer pin filter.
            return destination_attacked ? side != us : side == us;
        }

        balance = -balance - 1 - see_piece_value[attacker];
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
