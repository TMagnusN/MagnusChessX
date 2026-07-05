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

#include "MoveGen.h"
#include "Attack.h"
#include "Position.h"

#include <bit>
#include <cstdlib>

/*
This file builds legal move lists from bitboards plus precomputed attack tables.
It first derives side-to-move context (checkers, pins, masks),
then uses specialized generators for evasions, non-evasions, and the final
legality filter.
*/

/* ===== 繁體中文註釋 =====
 * 本檔案是 MagnusChessX Thinking 西洋棋引擎的一部分。
 * 實作詳情請參閱對應的 .h 標頭檔案。
 */

namespace magnus {

namespace {

[[nodiscard]] constexpr Color query_opposite(Color color) noexcept {
    return color == WHITE ? BLACK : WHITE;
}

[[nodiscard]] inline Square query_lsb_sq(Bitboard bitboard) noexcept {
    return static_cast<Square>(std::countr_zero(bitboard));
}

[[nodiscard]] constexpr bool query_more_than_one(Bitboard bitboard) noexcept {
    return (bitboard & (bitboard - 1)) != 0ULL;
}

[[nodiscard]] inline Bitboard query_pieces_bb(
    const Position& pos,
    Color color,
    PieceType piece_type
) noexcept {
    return pos.color_bb[color] & pos.piece_bb[piece_type];
}

[[nodiscard]] inline Bitboard query_diagonal_sliders(
    const Position& pos,
    Color color
) noexcept {
    return query_pieces_bb(pos, color, BISHOP) |
           query_pieces_bb(pos, color, QUEEN);
}

[[nodiscard]] inline Bitboard query_orthogonal_sliders(
    const Position& pos,
    Color color
) noexcept {
    return query_pieces_bb(pos, color, ROOK) |
           query_pieces_bb(pos, color, QUEEN);
}

Bitboard query_attacks_by_color_on_occupancy(
    const Position& pos,
    const memory::Memory& mem,
    Color by,
    Bitboard occupied
) noexcept {
    Bitboard attacks = 0ULL;
    Bitboard pawns = query_pieces_bb(pos, by, PAWN);
    Bitboard knights = query_pieces_bb(pos, by, KNIGHT);
    Bitboard bishops = query_pieces_bb(pos, by, BISHOP);
    Bitboard rooks = query_pieces_bb(pos, by, ROOK);
    Bitboard queens = query_pieces_bb(pos, by, QUEEN);
    Bitboard kings = query_pieces_bb(pos, by, KING);

    while (pawns) {
        const Square sq = query_lsb_sq(pawns);
        pawns &= pawns - 1;
        attacks |= pawn_attacks(mem, by, sq);
    }
    while (knights) {
        const Square sq = query_lsb_sq(knights);
        knights &= knights - 1;
        attacks |= knight_attacks(mem, sq);
    }
    while (bishops) {
        const Square sq = query_lsb_sq(bishops);
        bishops &= bishops - 1;
        attacks |= bishop_attacks_fast(mem, sq, occupied);
    }
    while (rooks) {
        const Square sq = query_lsb_sq(rooks);
        rooks &= rooks - 1;
        attacks |= rook_attacks_fast(mem, sq, occupied);
    }
    while (queens) {
        const Square sq = query_lsb_sq(queens);
        queens &= queens - 1;
        attacks |= queen_attacks_fast(mem, sq, occupied);
    }
    while (kings) {
        const Square sq = query_lsb_sq(kings);
        kings &= kings - 1;
        attacks |= king_attacks(mem, sq);
    }

    return attacks;
}

void query_compute_pinners_and_pinned(
    const Position& pos,
    const memory::Memory& mem,
    Color us,
    Bitboard& pinners,
    Bitboard& pinned
) noexcept {
    const Color them = query_opposite(us);
    const Square king_sq = pos.king_sq[us];
    const Bitboard own = pos.color_bb[us];

    pinners =
        (rook_xray_attacks(mem, king_sq, pos.occupied, own) &
         query_orthogonal_sliders(pos, them)) |
        (bishop_xray_attacks(mem, king_sq, pos.occupied, own) &
         query_diagonal_sliders(pos, them));

    pinned = 0ULL;
    Bitboard remaining = pinners;

    while (remaining) {
        const Square pinner_sq = query_lsb_sq(remaining);
        remaining &= remaining - 1;
        pinned |= between_bb(mem, king_sq, pinner_sq) & own;
    }
}

} // namespace

Bitboard attackers_to_color(
    const Position& pos,
    const memory::Memory& mem,
    Square sq,
    Color by,
    Bitboard occupied
) noexcept {
    const Bitboard bishops =
        query_pieces_bb(pos, by, BISHOP) | query_pieces_bb(pos, by, QUEEN);
    const Bitboard rooks =
        query_pieces_bb(pos, by, ROOK) | query_pieces_bb(pos, by, QUEEN);

    return (pawn_attacks(mem, query_opposite(by), sq) &
            query_pieces_bb(pos, by, PAWN)) |
           (knight_attacks(mem, sq) & query_pieces_bb(pos, by, KNIGHT)) |
           (king_attacks(mem, sq) & query_pieces_bb(pos, by, KING)) |
           (bishop_attacks_fast(mem, sq, occupied) & bishops) |
           (rook_attacks_fast(mem, sq, occupied) & rooks);
}

Bitboard attackers_to(
    const Position& pos,
    const memory::Memory& mem,
    Square sq,
    Bitboard occupied
) noexcept {
    return attackers_to_color(pos, mem, sq, WHITE, occupied) |
           attackers_to_color(pos, mem, sq, BLACK, occupied);
}

Bitboard checkers_bb(
    const Position& pos,
    const memory::Memory& mem,
    Color us
) noexcept {
    return attackers_to_color(
        pos,
        mem,
        pos.king_sq[us],
        query_opposite(us),
        pos.occupied
    );
}

Bitboard pinners_bb(
    const Position& pos,
    const memory::Memory& mem,
    Color us
) noexcept {
    Bitboard pinners = 0ULL;
    Bitboard pinned = 0ULL;
    query_compute_pinners_and_pinned(pos, mem, us, pinners, pinned);
    return pinners;
}

Bitboard pinned_bb(
    const Position& pos,
    const memory::Memory& mem,
    Color us
) noexcept {
    Bitboard pinners = 0ULL;
    Bitboard pinned = 0ULL;
    query_compute_pinners_and_pinned(pos, mem, us, pinners, pinned);
    return pinned;
}

Bitboard danger_bb(
    const Position& pos,
    const memory::Memory& mem,
    Color by
) noexcept {
    const Color us = query_opposite(by);
    const Bitboard occupied = pos.occupied ^ bb_of(pos.king_sq[us]);
    return query_attacks_by_color_on_occupancy(pos, mem, by, occupied);
}

void init_gen_info(
    GenInfo& info,
    const Position& pos,
    const memory::Memory& mem
) noexcept {
    info.us = static_cast<Color>(pos.side_to_move);
    info.them = query_opposite(info.us);
    info.king_sq = pos.king_sq[info.us];
    info.ep_sq = pos.ep_sq;
    info.occupied = pos.occupied;
    info.us_occ = pos.color_bb[info.us];
    info.them_occ = pos.color_bb[info.them];
    info.checkers = checkers_bb(pos, mem, info.us);
    query_compute_pinners_and_pinned(
        pos,
        mem,
        info.us,
        info.pinners,
        info.pinned
    );
    info.danger = 0ULL;
    info.in_check = info.checkers != 0ULL;
    info.double_check = query_more_than_one(info.checkers);

    if (!info.in_check) {
        info.capture_mask = info.them_occ;
        info.push_mask = ~info.occupied;
        return;
    }

    if (info.double_check) {
        info.capture_mask = 0ULL;
        info.push_mask = 0ULL;
        return;
    }

    const Square checker_sq = query_lsb_sq(info.checkers);
    const Bitboard checker_mask = bb_of(checker_sq);
    const Bitboard slider_checkers =
        query_diagonal_sliders(pos, info.them) |
        query_orthogonal_sliders(pos, info.them);

    info.capture_mask = checker_mask;
    info.push_mask = (checker_mask & slider_checkers)
        ? between_bb(mem, info.king_sq, checker_sq)
        : 0ULL;
}

namespace {

[[nodiscard]] constexpr Color opposite(Color color) noexcept {
    return color == WHITE ? BLACK : WHITE;
}

[[nodiscard]] inline Bitboard pieces_bb(
    const Position& pos,
    Color color,
    PieceType piece_type
) noexcept {
    return pos.color_bb[color] & pos.piece_bb[piece_type];
}

[[nodiscard]] inline Bitboard diagonal_sliders(
    const Position& pos,
    Color color
) noexcept {
    return pieces_bb(pos, color, BISHOP) | pieces_bb(pos, color, QUEEN);
}

[[nodiscard]] inline Bitboard orthogonal_sliders(
    const Position& pos,
    Color color
) noexcept {
    return pieces_bb(pos, color, ROOK) | pieces_bb(pos, color, QUEEN);
}

[[nodiscard]] inline Bitboard pin_mask_for(
    const memory::Memory& mem,
    const GenInfo& info,
    Square from
) noexcept {
    return (info.pinned & bb_of(from))
        ? line_bb(mem, info.king_sq, from)
        : ~0ULL;
}

[[nodiscard]] inline bool square_attacked_by_them(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Square sq
) noexcept {
    const Bitboard occupied_without_king =
        info.occupied ^ bb_of(info.king_sq);
    return attackers_to_color(
        pos,
        mem,
        sq,
        info.them,
        occupied_without_king
    ) != 0ULL;
}

[[nodiscard]] inline bool legal_slow(
    Position& pos,
    const memory::Memory& mem,
    Move move
) noexcept {
    if (move_is_none(move))
        return false;

    const Color us = static_cast<Color>(pos.side_to_move);
    const Color them = opposite(us);

    StateInfo state{};
    make_move(pos, move, mem.tables, state);

    const Square king_sq = pos.king_sq[us];
    const bool is_legal =
        attackers_to_color(pos, mem, king_sq, them, pos.occupied) == 0ULL;

    unmake_move(pos, move, mem.tables, state);
    return is_legal;
}

[[nodiscard]] inline bool legal_fast_impl(
    Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move move
) noexcept {
    if (move_is_none(move))
        return false;

    const Square from = from_sq(move);
    const Piece moving = piece_on(pos, from);
    if (moving == PIECE_NONE)
        return false;

    if (type_of(moving) == KING)
        return true;

    if (move_is_ep(move))
        return legal_slow(pos, mem, move);

    if ((info.pinned & bb_of(from)) == 0ULL)
        return true;

    return (line_bb(mem, info.king_sq, from) & bb_of(to_sq(move))) != 0ULL;
}

[[nodiscard]] inline bool needs_legal_filter(
    const Position&,
    const GenInfo& info,
    Move move
) noexcept {
    if (move_is_none(move) || move_is_ep(move))
        return true;

    const Square from = from_sq(move);
    return from == info.king_sq || (info.pinned & bb_of(from)) != 0ULL;
}

} // namespace

namespace {


inline Square lsb_sq(Bitboard bb) noexcept {
    return static_cast<Square>(std::countr_zero(bb));
}

inline Move* append_moves_from_mask(
    Square from,
    Bitboard mask,
    Bitboard them_occ,
    Move* out
) noexcept {
    while (mask) {
        const Square to = lsb_sq(mask);
        mask &= mask - 1;

        const u16 flag = (them_occ & bb_of(to)) ? MOVE_CAPTURE : MOVE_QUIET;
        *out++ = make_move(from, to, flag);
    }

    return out;
}

inline Move* append_capture_moves_from_mask(
    Square from,
    Bitboard mask,
    Move* out
) noexcept {
    while (mask) {
        const Square to = lsb_sq(mask);
        mask &= mask - 1;
        *out++ = make_move(from, to, MOVE_CAPTURE);
    }

    return out;
}

inline Move* append_quiet_moves_from_mask(
    Square from,
    Bitboard mask,
    Move* out
) noexcept {
    while (mask) {
        const Square to = lsb_sq(mask);
        mask &= mask - 1;
        *out++ = make_move(from, to, MOVE_QUIET);
    }

    return out;
}

inline Move* append_promotion_moves(
    Square from,
    Square to,
    bool is_capture,
    Move* out
) noexcept {
    if (is_capture) {
        *out++ = make_move(from, to, MOVE_CAP_PROMO_N);
        *out++ = make_move(from, to, MOVE_CAP_PROMO_B);
        *out++ = make_move(from, to, MOVE_CAP_PROMO_R);
        *out++ = make_move(from, to, MOVE_CAP_PROMO_Q);
    } else {
        *out++ = make_move(from, to, MOVE_PROMO_N);
        *out++ = make_move(from, to, MOVE_PROMO_B);
        *out++ = make_move(from, to, MOVE_PROMO_R);
        *out++ = make_move(from, to, MOVE_PROMO_Q);
    }
    return out;
}

enum class PieceMoveMode {
    NonEvasions,
    CaptureNonEvasions,
    QuietNonEvasions,
    Evasions,
    CaptureEvasions,
    QuietEvasions
};

template<PieceType Pt>
inline Bitboard piece_attacks_for(
    const memory::Memory& mem,
    Square from,
    Bitboard occupied
) noexcept {
    static_assert(
        Pt == KNIGHT || Pt == BISHOP || Pt == ROOK || Pt == QUEEN,
        "piece_attacks_for only supports non-king, non-pawn pieces"
    );

    if constexpr (Pt == KNIGHT)
        return knight_attacks(mem, from);
    else if constexpr (Pt == BISHOP)
        return bishop_attacks_fast(mem, from, occupied);
    else if constexpr (Pt == ROOK)
        return rook_attacks_fast(mem, from, occupied);
    else
        return queen_attacks_fast(mem, from, occupied);
}

template<PieceMoveMode Mode>
inline Bitboard filter_piece_move_mask(
    const memory::Memory& mem,
    const GenInfo& info,
    Square from,
    Bitboard mask
) noexcept {
    if constexpr (Mode == PieceMoveMode::NonEvasions) {
        return mask & ~info.us_occ & pin_mask_for(mem, info, from);
    } else if constexpr (Mode == PieceMoveMode::CaptureNonEvasions) {
        return mask & info.them_occ;
    } else if constexpr (Mode == PieceMoveMode::QuietNonEvasions) {
        return mask & ~info.occupied;
    } else {
        mask &= pin_mask_for(mem, info, from);

        if constexpr (Mode == PieceMoveMode::Evasions) {
            return mask & ~info.us_occ & (info.capture_mask | info.push_mask);
        } else if constexpr (Mode == PieceMoveMode::CaptureEvasions) {
            return mask & info.capture_mask & info.them_occ;
        } else {
            return mask & info.push_mask & ~info.occupied;
        }
    }
}

template<PieceMoveMode Mode>
inline Move* append_piece_moves_for_mode(
    Square from,
    Bitboard mask,
    Bitboard them_occ,
    Move* out
) noexcept {
    if constexpr (Mode == PieceMoveMode::NonEvasions || Mode == PieceMoveMode::Evasions) {
        return append_moves_from_mask(from, mask, them_occ, out);
    } else if constexpr (
        Mode == PieceMoveMode::CaptureNonEvasions ||
        Mode == PieceMoveMode::CaptureEvasions
    ) {
        return append_capture_moves_from_mask(from, mask, out);
    } else {
        return append_quiet_moves_from_mask(from, mask, out);
    }
}

template<PieceType Pt, PieceMoveMode Mode>
inline Move* generate_piece_moves(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    Bitboard pieces = pieces_bb(pos, info.us, Pt);

    while (pieces) {
        const Square from = lsb_sq(pieces);
        pieces &= pieces - 1;

        Bitboard mask = piece_attacks_for<Pt>(mem, from, info.occupied);
        mask = filter_piece_move_mask<Mode>(mem, info, from, mask);
        out = append_piece_moves_for_mode<Mode>(from, mask, info.them_occ, out);
    }

    return out;
}

template<PieceMoveMode Mode>
inline Move* generate_non_king_non_pawn_moves(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    out = generate_piece_moves<KNIGHT, Mode>(pos, mem, info, out);
    out = generate_piece_moves<BISHOP, Mode>(pos, mem, info, out);
    out = generate_piece_moves<ROOK, Mode>(pos, mem, info, out);
    out = generate_piece_moves<QUEEN, Mode>(pos, mem, info, out);
    return out;
}

template<Color Us>
inline constexpr int pawn_push_delta_v = (Us == WHITE ? 8 : -8);

template<Color Us>
inline constexpr int pawn_left_capture_delta_v = (Us == WHITE ? 7 : -9);

template<Color Us>
inline constexpr int pawn_right_capture_delta_v = (Us == WHITE ? 9 : -7);

template<Color Us>
inline constexpr int pawn_promotion_rank_v = (Us == WHITE ? 6 : 1);

template<Color Us>
inline constexpr int pawn_double_push_rank_v = (Us == WHITE ? 1 : 6);

constexpr Bitboard FILE_A_BB = 0x0101010101010101ULL;
constexpr Bitboard FILE_H_BB = 0x8080808080808080ULL;
constexpr Bitboard RANK_2_BB = 0x000000000000FF00ULL;
constexpr Bitboard RANK_3_BB = 0x0000000000FF0000ULL;
constexpr Bitboard RANK_6_BB = 0x0000FF0000000000ULL;
constexpr Bitboard RANK_7_BB = 0x00FF000000000000ULL;

template<Color Us>
inline constexpr Bitboard pawn_promotion_from_rank_bb_v =
    Us == WHITE ? RANK_7_BB : RANK_2_BB;

template<Color Us>
inline constexpr Bitboard pawn_double_push_mid_rank_bb_v =
    Us == WHITE ? RANK_3_BB : RANK_6_BB;

template<Color Us>
inline Bitboard pawn_shift_push(Bitboard pawns) noexcept {
    if constexpr (Us == WHITE)
        return pawns << 8;
    else
        return pawns >> 8;
}

template<Color Us>
inline Bitboard pawn_shift_double_push(Bitboard pawns) noexcept {
    if constexpr (Us == WHITE)
        return pawns << 16;
    else
        return pawns >> 16;
}

template<Color Us>
inline Bitboard pawn_shift_left_capture(Bitboard pawns) noexcept {
    if constexpr (Us == WHITE)
        return (pawns & ~FILE_A_BB) << 7;
    else
        return (pawns & ~FILE_A_BB) >> 9;
}

template<Color Us>
inline Bitboard pawn_shift_right_capture(Bitboard pawns) noexcept {
    if constexpr (Us == WHITE)
        return (pawns & ~FILE_H_BB) << 9;
    else
        return (pawns & ~FILE_H_BB) >> 7;
}

template<PieceMoveMode Mode>
inline constexpr bool pawn_mode_allows_quiets_v =
    Mode == PieceMoveMode::NonEvasions ||
    Mode == PieceMoveMode::QuietNonEvasions ||
    Mode == PieceMoveMode::Evasions ||
    Mode == PieceMoveMode::QuietEvasions;

template<PieceMoveMode Mode>
inline constexpr bool pawn_mode_allows_captures_v =
    Mode == PieceMoveMode::NonEvasions ||
    Mode == PieceMoveMode::CaptureNonEvasions ||
    Mode == PieceMoveMode::Evasions ||
    Mode == PieceMoveMode::CaptureEvasions;

template<PieceMoveMode Mode>
inline constexpr bool pawn_mode_needs_evasion_filters_v =
    Mode == PieceMoveMode::Evasions ||
    Mode == PieceMoveMode::CaptureEvasions ||
    Mode == PieceMoveMode::QuietEvasions;

inline Move* append_pawn_move_or_promotion(
    Square from,
    Square to,
    bool is_capture,
    bool is_promotion,
    Move* out
) noexcept {
    if (is_promotion)
        return append_promotion_moves(from, to, is_capture, out);

    *out++ = make_move(from, to, is_capture ? MOVE_CAPTURE : MOVE_QUIET);
    return out;
}

template<PieceMoveMode Mode>
inline bool pawn_push_allowed(
    const GenInfo& info,
    Bitboard pin_mask,
    Bitboard target_bb
) noexcept {
    if constexpr (pawn_mode_needs_evasion_filters_v<Mode>)
        return (info.push_mask & target_bb) && (pin_mask & target_bb);
    else
        return true;
}

template<PieceMoveMode Mode>
inline bool pawn_capture_allowed(
    const GenInfo& info,
    Bitboard pin_mask,
    Bitboard target_bb
) noexcept {
    if ((info.them_occ & target_bb) == 0ULL)
        return false;

    if constexpr (pawn_mode_needs_evasion_filters_v<Mode>)
        return (info.capture_mask & target_bb) && (pin_mask & target_bb);
    else
        return true;
}

template<Color Us, PieceMoveMode Mode>
inline Move* generate_pawn_moves_scalar_for_color(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Bitboard pawns,
    Move* out
) noexcept {
    while (pawns) {
        const Square from = lsb_sq(pawns);
        pawns &= pawns - 1;

        const int r = rank_of(from);
        const int f = file_of(from);
        const Bitboard pin_mask = pawn_mode_needs_evasion_filters_v<Mode>
            ? pin_mask_for(mem, info, from)
            : ~0ULL;

        if constexpr (pawn_mode_allows_quiets_v<Mode>) {
            const int one_idx = static_cast<int>(from) + pawn_push_delta_v<Us>;
            if (one_idx >= 0 && one_idx < SQ_NB) {
                const Square one = static_cast<Square>(one_idx);
                const Bitboard one_bb = bb_of(one);
                if ((info.occupied & one_bb) == 0ULL) {
                    if (pawn_push_allowed<Mode>(info, pin_mask, one_bb)) {
                        out = append_pawn_move_or_promotion(
                            from,
                            one,
                            false,
                            r == pawn_promotion_rank_v<Us>,
                            out
                        );
                    }

                    if (r == pawn_double_push_rank_v<Us>) {
                        const Square two =
                            static_cast<Square>(static_cast<int>(from) + 2 * pawn_push_delta_v<Us>);
                        const Bitboard two_bb = bb_of(two);
                        if ((info.occupied & two_bb) == 0ULL &&
                            pawn_push_allowed<Mode>(info, pin_mask, two_bb)) {
                            *out++ = make_move(from, two, MOVE_DOUBLE_PUSH);
                        }
                    }
                }
            }
        }

        if constexpr (pawn_mode_allows_captures_v<Mode>) {
            if (f > 0) {
                const Square to = static_cast<Square>(
                    static_cast<int>(from) + pawn_left_capture_delta_v<Us>
                );
                const Bitboard to_bb = bb_of(to);
                if (pawn_capture_allowed<Mode>(info, pin_mask, to_bb)) {
                    out = append_pawn_move_or_promotion(
                        from,
                        to,
                        true,
                        r == pawn_promotion_rank_v<Us>,
                        out
                    );
                }
            }

            if (f < 7) {
                const Square to = static_cast<Square>(
                    static_cast<int>(from) + pawn_right_capture_delta_v<Us>
                );
                const Bitboard to_bb = bb_of(to);
                if (pawn_capture_allowed<Mode>(info, pin_mask, to_bb)) {
                    out = append_pawn_move_or_promotion(
                        from,
                        to,
                        true,
                        r == pawn_promotion_rank_v<Us>,
                        out
                    );
                }
            }
        }
    }

    return out;
}

template<Color Us, int Delta, u16 Flag>
inline Move* append_pawn_targets(Bitboard targets, Move* out) noexcept {
    while (targets) {
        const Square to = lsb_sq(targets);
        targets &= targets - 1;
        *out++ = make_move(static_cast<Square>(to - Delta), to, Flag);
    }
    return out;
}

template<Color Us, int Delta>
inline Move* append_pawn_promotions_from_targets(
    Bitboard targets,
    bool is_capture,
    Move* out
) noexcept {
    while (targets) {
        const Square to = lsb_sq(targets);
        targets &= targets - 1;
        out = append_promotion_moves(
            static_cast<Square>(to - Delta),
            to,
            is_capture,
            out
        );
    }
    return out;
}

template<Color Us, PieceMoveMode Mode>
inline Move* generate_unpinned_pawn_moves_for_color(
    const GenInfo& info,
    Bitboard pawns,
    Move* out
) noexcept {
    constexpr int PushDelta = pawn_push_delta_v<Us>;
    constexpr int LeftDelta = pawn_left_capture_delta_v<Us>;
    constexpr int RightDelta = pawn_right_capture_delta_v<Us>;

    const Bitboard empty = ~info.occupied;
    const Bitboard promotion_pawns = pawns & pawn_promotion_from_rank_bb_v<Us>;
    const Bitboard non_promotion_pawns = pawns & ~pawn_promotion_from_rank_bb_v<Us>;

    if constexpr (pawn_mode_allows_quiets_v<Mode>) {
        Bitboard one = pawn_shift_push<Us>(non_promotion_pawns) & empty;
        Bitboard two = pawn_shift_push<Us>(one & pawn_double_push_mid_rank_bb_v<Us>) & empty;

        if constexpr (pawn_mode_needs_evasion_filters_v<Mode>) {
            one &= info.push_mask;
            two &= info.push_mask;
        }

        out = append_pawn_targets<Us, PushDelta, MOVE_QUIET>(one, out);
        out = append_pawn_targets<Us, PushDelta * 2, MOVE_DOUBLE_PUSH>(two, out);

        Bitboard promotions = pawn_shift_push<Us>(promotion_pawns) & empty;
        if constexpr (pawn_mode_needs_evasion_filters_v<Mode>)
            promotions &= info.push_mask;

        out = append_pawn_promotions_from_targets<Us, PushDelta>(
            promotions,
            false,
            out
        );
    }

    if constexpr (pawn_mode_allows_captures_v<Mode>) {
        Bitboard left = pawn_shift_left_capture<Us>(non_promotion_pawns) & info.them_occ;
        Bitboard right = pawn_shift_right_capture<Us>(non_promotion_pawns) & info.them_occ;

        if constexpr (pawn_mode_needs_evasion_filters_v<Mode>) {
            left &= info.capture_mask;
            right &= info.capture_mask;
        }

        out = append_pawn_targets<Us, LeftDelta, MOVE_CAPTURE>(left, out);
        out = append_pawn_targets<Us, RightDelta, MOVE_CAPTURE>(right, out);

        Bitboard left_promotions = pawn_shift_left_capture<Us>(promotion_pawns) & info.them_occ;
        Bitboard right_promotions = pawn_shift_right_capture<Us>(promotion_pawns) & info.them_occ;

        if constexpr (pawn_mode_needs_evasion_filters_v<Mode>) {
            left_promotions &= info.capture_mask;
            right_promotions &= info.capture_mask;
        }

        out = append_pawn_promotions_from_targets<Us, LeftDelta>(
            left_promotions,
            true,
            out
        );
        out = append_pawn_promotions_from_targets<Us, RightDelta>(
            right_promotions,
            true,
            out
        );
    }

    return out;
}

template<Color Us, PieceMoveMode Mode>
inline Move* generate_pawn_moves_for_color(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    const Bitboard pawns = pieces_bb(pos, Us, PAWN);
    const Bitboard pinned = pawns & info.pinned;
    const Bitboard unpinned = pawns & ~info.pinned;

    out = generate_unpinned_pawn_moves_for_color<Us, Mode>(info, unpinned, out);
    return generate_pawn_moves_scalar_for_color<Us, Mode>(
        pos,
        mem,
        info,
        pinned,
        out
    );
}

template<PieceMoveMode Mode>
inline Move* generate_pawn_moves(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    return info.us == WHITE
        ? generate_pawn_moves_for_color<WHITE, Mode>(pos, mem, info, out)
        : generate_pawn_moves_for_color<BLACK, Mode>(pos, mem, info, out);
}

inline Move* generate_pawn_evasions(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    return generate_pawn_moves<PieceMoveMode::Evasions>(pos, mem, info, out);
}

template<MoveFlag QuietFlag, MoveFlag CaptureFlag>
inline Move* append_safe_king_moves_from_mask(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Bitboard mask,
    Move* out
) noexcept {
    const Square from = info.king_sq;

    while (mask) {
        const Square to = lsb_sq(mask);
        mask &= mask - 1;

        if (square_attacked_by_them(pos, mem, info, to))
            continue;

        const u16 flag = (info.them_occ & bb_of(to)) ? CaptureFlag : QuietFlag;
        *out++ = make_move(from, to, flag);
    }

    return out;
}

inline Move* generate_king_non_evasions(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    Bitboard mask = king_attacks(mem, info.king_sq);
    mask &= ~info.us_occ;

    return append_safe_king_moves_from_mask<MOVE_QUIET, MOVE_CAPTURE>(
        pos,
        mem,
        info,
        mask,
        out
    );
}

inline Move* generate_pawn_non_evasions(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    return generate_pawn_moves<PieceMoveMode::NonEvasions>(pos, mem, info, out);
}

inline Move* generate_king_captures(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    Bitboard mask = king_attacks(mem, info.king_sq);
    mask &= info.them_occ;
    return append_safe_king_moves_from_mask<MOVE_QUIET, MOVE_CAPTURE>(
        pos,
        mem,
        info,
        mask,
        out
    );
}

inline Move* generate_pawn_captures(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    return generate_pawn_moves<PieceMoveMode::CaptureNonEvasions>(pos, mem, info, out);
}

inline Move* generate_ep_non_evasions(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    if (info.ep_sq == NO_SQ)
        return out;

    const Bitboard pawns = pieces_bb(pos, info.us, PAWN);
    Bitboard ep_attackers = pawn_attacks(mem, opposite(info.us), info.ep_sq) & pawns;

    while (ep_attackers) {
        const Square from = lsb_sq(ep_attackers);
        ep_attackers &= ep_attackers - 1;

        // 普通 pin line 过滤先保留；
        // 但 EP 真正是否合法，后面还要靠 legal()/copy-make 再判一次。
        if (!(pin_mask_for(mem, info, from) & bb_of(info.ep_sq)))
            continue;

        *out++ = make_move(from, info.ep_sq, MOVE_EP);
    }

    return out;
}

inline Move* generate_capture_non_evasions_with_info(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    out = generate_king_captures(pos, mem, info, out);
    out = generate_non_king_non_pawn_moves<PieceMoveMode::CaptureNonEvasions>(pos, mem, info, out);
    out = generate_pawn_captures(pos, mem, info, out);
    out = generate_ep_non_evasions(pos, mem, info, out);
    return out;
}

inline Move* generate_ep_evasions(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    if (info.ep_sq == NO_SQ || info.double_check)
        return out;

    // EP 的目标格是 ep_sq，但被吃掉的兵在 cap_sq
    const Square cap_sq = (info.us == WHITE) ? (info.ep_sq - 8) : (info.ep_sq + 8);
    const Bitboard ep_to_bb = bb_of(info.ep_sq);
    const Bitboard cap_bb   = bb_of(cap_sq);

    // EP 能解将的两种情况：
    // 1) 吃掉的那只兵本身就是 checker
    // 2) 落到 ep_sq 这个格正好是 block square
    if (!(info.capture_mask & cap_bb) && !(info.push_mask & ep_to_bb))
        return out;

    const Bitboard pawns = pieces_bb(pos, info.us, PAWN);
    Bitboard ep_attackers = pawn_attacks(mem, opposite(info.us), info.ep_sq) & pawns;

    while (ep_attackers) {
        const Square from = lsb_sq(ep_attackers);
        ep_attackers &= ep_attackers - 1;

        if (!(pin_mask_for(mem, info, from) & ep_to_bb))
            continue;

        *out++ = make_move(from, info.ep_sq, MOVE_EP);
    }

    return out;
}

inline Move* generate_castling_non_evasions(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    // 王在将军中绝不能易位
    if (info.in_check)
        return out;

    if (info.us == WHITE) {
        // White O-O: e1 -> g1, rook h1 -> f1
        if ((pos.castling_rights & WHITE_OO) &&
            piece_on(pos, 4) == W_KING &&
            piece_on(pos, 7) == W_ROOK &&
            !(info.occupied & (bb_of(5) | bb_of(6))) &&
            !square_attacked_by_them(pos, mem, info, 5) &&
            !square_attacked_by_them(pos, mem, info, 6)) {
            *out++ = make_move(4, 6, MOVE_OO);
        }

        // White O-O-O: e1 -> c1, rook a1 -> d1
        if ((pos.castling_rights & WHITE_OOO) &&
            piece_on(pos, 4) == W_KING &&
            piece_on(pos, 0) == W_ROOK &&
            !(info.occupied & (bb_of(1) | bb_of(2) | bb_of(3))) &&
            !square_attacked_by_them(pos, mem, info, 3) &&
            !square_attacked_by_them(pos, mem, info, 2)) {
            *out++ = make_move(4, 2, MOVE_OOO);
        }
    } else {
        // Black O-O: e8 -> g8, rook h8 -> f8
        if ((pos.castling_rights & BLACK_OO) &&
            piece_on(pos, 60) == B_KING &&
            piece_on(pos, 63) == B_ROOK &&
            !(info.occupied & (bb_of(61) | bb_of(62))) &&
            !square_attacked_by_them(pos, mem, info, 61) &&
            !square_attacked_by_them(pos, mem, info, 62)) {
            *out++ = make_move(60, 62, MOVE_OO);
        }

        // Black O-O-O: e8 -> c8, rook a8 -> d8
        if ((pos.castling_rights & BLACK_OOO) &&
            piece_on(pos, 60) == B_KING &&
            piece_on(pos, 56) == B_ROOK &&
            !(info.occupied & (bb_of(57) | bb_of(58) | bb_of(59))) &&
            !square_attacked_by_them(pos, mem, info, 59) &&
            !square_attacked_by_them(pos, mem, info, 58)) {
            *out++ = make_move(60, 58, MOVE_OOO);
        }
    }

    return out;
}

inline Move* generate_king_quiet_non_evasions(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    Bitboard mask = king_attacks(mem, info.king_sq);
    mask &= ~info.occupied;
    return append_safe_king_moves_from_mask<MOVE_QUIET, MOVE_CAPTURE>(
        pos,
        mem,
        info,
        mask,
        out
    );
}

inline Move* generate_pawn_quiet_non_evasions(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    return generate_pawn_moves<PieceMoveMode::QuietNonEvasions>(pos, mem, info, out);
}

inline Move* generate_king_capture_evasions(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    Bitboard mask = king_attacks(mem, info.king_sq);
    mask &= info.them_occ;
    return append_safe_king_moves_from_mask<MOVE_QUIET, MOVE_CAPTURE>(
        pos,
        mem,
        info,
        mask,
        out
    );
}

inline Move* generate_king_quiet_evasions(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    Bitboard mask = king_attacks(mem, info.king_sq);
    mask &= ~info.occupied;
    return append_safe_king_moves_from_mask<MOVE_QUIET, MOVE_CAPTURE>(
        pos,
        mem,
        info,
        mask,
        out
    );
}

inline Move* generate_pawn_capture_evasions(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    return generate_pawn_moves<PieceMoveMode::CaptureEvasions>(pos, mem, info, out);
}

inline Move* generate_pawn_quiet_evasions(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    return generate_pawn_moves<PieceMoveMode::QuietEvasions>(pos, mem, info, out);
}

inline Move* generate_capture_evasions_with_info(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    out = generate_king_capture_evasions(pos, mem, info, out);

    if (info.double_check)
        return out;

    out = generate_non_king_non_pawn_moves<PieceMoveMode::CaptureEvasions>(pos, mem, info, out);
    out = generate_pawn_capture_evasions(pos, mem, info, out);
    out = generate_ep_evasions(pos, mem, info, out);
    return out;
}

inline Move* generate_quiet_evasions_with_info(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    out = generate_king_quiet_evasions(pos, mem, info, out);

    if (info.double_check)
        return out;

    out = generate_non_king_non_pawn_moves<PieceMoveMode::QuietEvasions>(pos, mem, info, out);
    out = generate_pawn_quiet_evasions(pos, mem, info, out);
    return out;
}

inline Move* generate_quiet_non_evasions_with_info(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    out = generate_king_quiet_non_evasions(pos, mem, info, out);
    out = generate_non_king_non_pawn_moves<PieceMoveMode::QuietNonEvasions>(pos, mem, info, out);
    out = generate_pawn_quiet_non_evasions(pos, mem, info, out);
    out = generate_castling_non_evasions(pos, mem, info, out);
    return out;
}

inline Move* generate_non_evasions_with_info(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    out = generate_king_non_evasions(pos, mem, info, out);
    out = generate_non_king_non_pawn_moves<PieceMoveMode::NonEvasions>(pos, mem, info, out);
    out = generate_pawn_non_evasions(pos, mem, info, out);
    out = generate_ep_non_evasions(pos, mem, info, out);
    out = generate_castling_non_evasions(pos, mem, info, out);
    return out;
}

inline Move* generate_evasions_with_info(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    out = generate_king_non_evasions(pos, mem, info, out);

    if (info.double_check)
        return out;

    out = generate_non_king_non_pawn_moves<PieceMoveMode::Evasions>(pos, mem, info, out);
    out = generate_pawn_evasions(pos, mem, info, out);
    out = generate_ep_evasions(pos, mem, info, out);
    return out;
}

inline Move* generate_pseudo_legal_with_info(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    return info.in_check
        ? generate_evasions_with_info(pos, mem, info, out)
        : generate_non_evasions_with_info(pos, mem, info, out);
}

} // namespace

namespace {

[[nodiscard]] inline bool pseudo_legal_castle_fast(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move move
) noexcept {
    if (info.in_check)
        return false;

    const Square from = from_sq(move);
    const Square to = to_sq(move);

    if (info.us == WHITE) {
        if (move_flag(move) == MOVE_OO) {
            return from == 4 &&
                to == 6 &&
                piece_on(pos, 4) == W_KING &&
                piece_on(pos, 7) == W_ROOK &&
                (pos.castling_rights & WHITE_OO) != 0 &&
                !(info.occupied & (bb_of(5) | bb_of(6))) &&
                !square_attacked_by_them(pos, mem, info, 5) &&
                !square_attacked_by_them(pos, mem, info, 6);
        }

        if (move_flag(move) == MOVE_OOO) {
            return from == 4 &&
                to == 2 &&
                piece_on(pos, 4) == W_KING &&
                piece_on(pos, 0) == W_ROOK &&
                (pos.castling_rights & WHITE_OOO) != 0 &&
                !(info.occupied & (bb_of(1) | bb_of(2) | bb_of(3))) &&
                !square_attacked_by_them(pos, mem, info, 3) &&
                !square_attacked_by_them(pos, mem, info, 2);
        }

        return false;
    }

    if (move_flag(move) == MOVE_OO) {
        return from == 60 &&
            to == 62 &&
            piece_on(pos, 60) == B_KING &&
            piece_on(pos, 63) == B_ROOK &&
            (pos.castling_rights & BLACK_OO) != 0 &&
            !(info.occupied & (bb_of(61) | bb_of(62))) &&
            !square_attacked_by_them(pos, mem, info, 61) &&
            !square_attacked_by_them(pos, mem, info, 62);
    }

    if (move_flag(move) == MOVE_OOO) {
        return from == 60 &&
            to == 58 &&
            piece_on(pos, 60) == B_KING &&
            piece_on(pos, 56) == B_ROOK &&
            (pos.castling_rights & BLACK_OOO) != 0 &&
            !(info.occupied & (bb_of(57) | bb_of(58) | bb_of(59))) &&
            !square_attacked_by_them(pos, mem, info, 59) &&
            !square_attacked_by_them(pos, mem, info, 58);
    }

    return false;
}

[[nodiscard]] inline bool pseudo_legal_king_step_fast(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move move
) noexcept {
    if (move_is_castle(move) ||
        move_is_promotion(move) ||
        move_is_ep(move) ||
        move_is_double_push(move)) {
        return false;
    }

    const Square from = from_sq(move);
    const Square to = to_sq(move);
    const Bitboard to_bb = bb_of(to);

    Bitboard mask = king_attacks(mem, from);
    mask &= ~info.us_occ;
    if ((mask & to_bb) == 0ULL)
        return false;
    if (square_attacked_by_them(pos, mem, info, to))
        return false;

    const bool capture = (info.them_occ & to_bb) != 0ULL;
    return capture ? move_flag(move) == MOVE_CAPTURE
                   : move_flag(move) == MOVE_QUIET;
}

[[nodiscard]] inline bool pseudo_legal_piece_step_fast(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move move,
    PieceType piece_type
) noexcept {
    if (move_is_castle(move) ||
        move_is_promotion(move) ||
        move_is_ep(move) ||
        move_is_double_push(move)) {
        return false;
    }

    const Square from = from_sq(move);
    const Square to = to_sq(move);
    const Bitboard to_bb = bb_of(to);
    if ((info.us_occ & to_bb) != 0ULL)
        return false;

    Bitboard mask = 0ULL;
    switch (piece_type) {
        case KNIGHT:
            mask = knight_attacks(mem, from);
            break;
        case BISHOP:
            mask = bishop_attacks_fast(mem, from, pos.occupied);
            break;
        case ROOK:
            mask = rook_attacks_fast(mem, from, pos.occupied);
            break;
        case QUEEN:
            mask = queen_attacks_fast(mem, from, pos.occupied);
            break;
        default:
            return false;
    }

    mask &= ~info.us_occ;
    if (info.in_check) {
        if (info.double_check)
            return false;
        mask &= info.capture_mask | info.push_mask;
        mask &= pin_mask_for(mem, info, from);
    }

    if ((mask & to_bb) == 0ULL)
        return false;

    const bool capture = (info.them_occ & to_bb) != 0ULL;
    return capture ? move_flag(move) == MOVE_CAPTURE
                   : move_flag(move) == MOVE_QUIET;
}

[[nodiscard]] inline bool pseudo_legal_white_pawn_fast(
    const memory::Memory& mem,
    const GenInfo& info,
    Move move
) noexcept {
    const Square from = from_sq(move);
    const Square to = to_sq(move);
    const Bitboard to_bb = bb_of(to);
    const Bitboard pin_mask = pin_mask_for(mem, info, from);

    switch (move_flag(move)) {
        case MOVE_QUIET:
            if (rank_of(from) == 6 || to != from + 8)
                return false;
            if ((info.occupied & to_bb) != 0ULL)
                return false;
            if (!info.in_check)
                return true;
            return !info.double_check &&
                (info.push_mask & to_bb) != 0ULL &&
                (pin_mask & to_bb) != 0ULL;

        case MOVE_DOUBLE_PUSH:
            if (rank_of(from) != 1 || to != from + 16)
                return false;
            if ((info.occupied & (bb_of(from + 8) | to_bb)) != 0ULL)
                return false;
            if (!info.in_check)
                return true;
            return !info.double_check &&
                (info.push_mask & to_bb) != 0ULL &&
                (pin_mask & to_bb) != 0ULL;

        case MOVE_CAPTURE:
            if (rank_of(from) == 6)
                return false;
            if ((info.them_occ & to_bb) == 0ULL)
                return false;
            if ((pawn_attacks(mem, WHITE, from) & to_bb) == 0ULL)
                return false;
            if (!info.in_check)
                return true;
            return !info.double_check &&
                (info.capture_mask & to_bb) != 0ULL &&
                (pin_mask & to_bb) != 0ULL;

        case MOVE_PROMO_N:
        case MOVE_PROMO_B:
        case MOVE_PROMO_R:
        case MOVE_PROMO_Q:
            if (rank_of(from) != 6 || to != from + 8)
                return false;
            if ((info.occupied & to_bb) != 0ULL)
                return false;
            if (!info.in_check)
                return true;
            return !info.double_check &&
                (info.push_mask & to_bb) != 0ULL &&
                (pin_mask & to_bb) != 0ULL;

        case MOVE_CAP_PROMO_N:
        case MOVE_CAP_PROMO_B:
        case MOVE_CAP_PROMO_R:
        case MOVE_CAP_PROMO_Q:
            if (rank_of(from) != 6)
                return false;
            if ((info.them_occ & to_bb) == 0ULL)
                return false;
            if ((pawn_attacks(mem, WHITE, from) & to_bb) == 0ULL)
                return false;
            if (!info.in_check)
                return true;
            return !info.double_check &&
                (info.capture_mask & to_bb) != 0ULL &&
                (pin_mask & to_bb) != 0ULL;

        case MOVE_EP:
            if (info.ep_sq == NO_SQ || to != info.ep_sq)
                return false;
            if ((pawn_attacks(mem, WHITE, from) & to_bb) == 0ULL)
                return false;
            if ((pin_mask & to_bb) == 0ULL)
                return false;
            if (!info.in_check)
                return true;
            if (info.double_check)
                return false;
            return (info.capture_mask & bb_of(to - 8)) != 0ULL ||
                (info.push_mask & to_bb) != 0ULL;

        default:
            return false;
    }
}

[[nodiscard]] inline bool pseudo_legal_black_pawn_fast(
    const memory::Memory& mem,
    const GenInfo& info,
    Move move
) noexcept {
    const Square from = from_sq(move);
    const Square to = to_sq(move);
    const Bitboard to_bb = bb_of(to);
    const Bitboard pin_mask = pin_mask_for(mem, info, from);

    switch (move_flag(move)) {
        case MOVE_QUIET:
            if (rank_of(from) == 1 || to != from - 8)
                return false;
            if ((info.occupied & to_bb) != 0ULL)
                return false;
            if (!info.in_check)
                return true;
            return !info.double_check &&
                (info.push_mask & to_bb) != 0ULL &&
                (pin_mask & to_bb) != 0ULL;

        case MOVE_DOUBLE_PUSH:
            if (rank_of(from) != 6 || to != from - 16)
                return false;
            if ((info.occupied & (bb_of(from - 8) | to_bb)) != 0ULL)
                return false;
            if (!info.in_check)
                return true;
            return !info.double_check &&
                (info.push_mask & to_bb) != 0ULL &&
                (pin_mask & to_bb) != 0ULL;

        case MOVE_CAPTURE:
            if (rank_of(from) == 1)
                return false;
            if ((info.them_occ & to_bb) == 0ULL)
                return false;
            if ((pawn_attacks(mem, BLACK, from) & to_bb) == 0ULL)
                return false;
            if (!info.in_check)
                return true;
            return !info.double_check &&
                (info.capture_mask & to_bb) != 0ULL &&
                (pin_mask & to_bb) != 0ULL;

        case MOVE_PROMO_N:
        case MOVE_PROMO_B:
        case MOVE_PROMO_R:
        case MOVE_PROMO_Q:
            if (rank_of(from) != 1 || to != from - 8)
                return false;
            if ((info.occupied & to_bb) != 0ULL)
                return false;
            if (!info.in_check)
                return true;
            return !info.double_check &&
                (info.push_mask & to_bb) != 0ULL &&
                (pin_mask & to_bb) != 0ULL;

        case MOVE_CAP_PROMO_N:
        case MOVE_CAP_PROMO_B:
        case MOVE_CAP_PROMO_R:
        case MOVE_CAP_PROMO_Q:
            if (rank_of(from) != 1)
                return false;
            if ((info.them_occ & to_bb) == 0ULL)
                return false;
            if ((pawn_attacks(mem, BLACK, from) & to_bb) == 0ULL)
                return false;
            if (!info.in_check)
                return true;
            return !info.double_check &&
                (info.capture_mask & to_bb) != 0ULL &&
                (pin_mask & to_bb) != 0ULL;

        case MOVE_EP:
            if (info.ep_sq == NO_SQ || to != info.ep_sq)
                return false;
            if ((pawn_attacks(mem, BLACK, from) & to_bb) == 0ULL)
                return false;
            if ((pin_mask & to_bb) == 0ULL)
                return false;
            if (!info.in_check)
                return true;
            if (info.double_check)
                return false;
            return (info.capture_mask & bb_of(to + 8)) != 0ULL ||
                (info.push_mask & to_bb) != 0ULL;

        default:
            return false;
    }
}

} // namespace

bool pseudo_legal(
    const Position& pos,
    const memory::Memory& mem,
    Move move
) noexcept {
    GenInfo info{};
    init_gen_info(info, pos, mem);
    return pseudo_legal_fast(pos, mem, info, move);
}

bool pseudo_legal_fast(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move move
) noexcept {
    if (move_is_none(move))
        return false;

    const Square from = from_sq(move);
    const Square to = to_sq(move);
    if (!is_ok(from) || !is_ok(to))
        return false;

    const Piece moving = piece_on(pos, from);
    if (moving == PIECE_NONE || color_of(moving) != info.us)
        return false;

    switch (type_of(moving)) {
        case KING:
            return move_is_castle(move)
                ? pseudo_legal_castle_fast(pos, mem, info, move)
                : pseudo_legal_king_step_fast(pos, mem, info, move);

        case KNIGHT:
        case BISHOP:
        case ROOK:
        case QUEEN:
            return pseudo_legal_piece_step_fast(
                pos,
                mem,
                info,
                move,
                type_of(moving)
            );

        case PAWN:
            return info.us == WHITE
                ? pseudo_legal_white_pawn_fast(mem, info, move)
                : pseudo_legal_black_pawn_fast(mem, info, move);

        default:
            return false;
    }
}

bool legal(
    Position& pos,
    const memory::Memory& mem,
    Move move
) noexcept {
    return legal_slow(pos, mem, move);
}

bool legal(
    const Position& pos,
    const memory::Memory& mem,
    Move move
) noexcept {
    Position work{};
    position_copy_without_accumulators(work, pos);
    return legal(work, mem, move);
}

bool legal_fast(
    Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move move
) noexcept {
    return legal_fast_impl(pos, mem, info, move);
}

bool move_gives_check(
    const Position& pos,
    const memory::Memory& mem,
    Move move
) noexcept {
    if (move_is_none(move))
        return false;

    const Square from = from_sq(move);
    const Square to = to_sq(move);
    if (!is_ok(from) || !is_ok(to))
        return false;

    const Color us = static_cast<Color>(pos.side_to_move);
    const Color them = opposite(us);
    const Square enemy_king = pos.king_sq[them];
    if (!is_ok(enemy_king))
        return false;

    const Piece moving = piece_on(pos, from);
    if (moving == PIECE_NONE || color_of(moving) != us)
        return false;

    Bitboard occupied = pos.occupied;
    Bitboard pawns = pieces_bb(pos, us, PAWN);
    Bitboard knights = pieces_bb(pos, us, KNIGHT);
    Bitboard kings = pieces_bb(pos, us, KING);
    Bitboard bishop_like = diagonal_sliders(pos, us);
    Bitboard rook_like = orthogonal_sliders(pos, us);

    const Bitboard from_bb = bb_of(from);
    const Bitboard to_bb = bb_of(to);
    const PieceType moving_type = type_of(moving);

    occupied ^= from_bb;

    switch (moving_type) {
        case PAWN:
            pawns ^= from_bb;
            break;
        case KNIGHT:
            knights ^= from_bb;
            break;
        case BISHOP:
            bishop_like ^= from_bb;
            break;
        case ROOK:
            rook_like ^= from_bb;
            break;
        case QUEEN:
            bishop_like ^= from_bb;
            rook_like ^= from_bb;
            break;
        case KING:
            kings ^= from_bb;
            break;
        default:
            return false;
    }

    if (move_flag(move) == MOVE_OO || move_flag(move) == MOVE_OOO) {
        occupied |= to_bb;
        kings |= to_bb;

        const Square rook_from =
            us == WHITE
                ? (move_flag(move) == MOVE_OO ? 7 : 0)
                : (move_flag(move) == MOVE_OO ? 63 : 56);
        const Square rook_to =
            us == WHITE
                ? (move_flag(move) == MOVE_OO ? 5 : 3)
                : (move_flag(move) == MOVE_OO ? 61 : 59);
        const Bitboard rook_from_bb = bb_of(rook_from);
        const Bitboard rook_to_bb = bb_of(rook_to);

        occupied ^= rook_from_bb;
        occupied |= rook_to_bb;
        rook_like ^= rook_from_bb;
        rook_like |= rook_to_bb;
    } else if (move_is_ep(move)) {
        const Square captured_sq = us == WHITE ? to - 8 : to + 8;
        occupied ^= bb_of(captured_sq);
        occupied |= to_bb;
        pawns |= to_bb;
    } else if (move_is_promotion(move)) {
        occupied |= to_bb;

        switch (promo_piece(move)) {
            case KNIGHT:
                knights |= to_bb;
                break;
            case BISHOP:
                bishop_like |= to_bb;
                break;
            case ROOK:
                rook_like |= to_bb;
                break;
            case QUEEN:
                bishop_like |= to_bb;
                rook_like |= to_bb;
                break;
            default:
                return false;
        }
    } else {
        occupied |= to_bb;

        switch (moving_type) {
            case PAWN:
                pawns |= to_bb;
                break;
            case KNIGHT:
                knights |= to_bb;
                break;
            case BISHOP:
                bishop_like |= to_bb;
                break;
            case ROOK:
                rook_like |= to_bb;
                break;
            case QUEEN:
                bishop_like |= to_bb;
                rook_like |= to_bb;
                break;
            case KING:
                kings |= to_bb;
                break;
            default:
                return false;
        }
    }

    const bool gives_check =
        ((pawn_attacks(mem, them, enemy_king) & pawns) != 0ULL) ||
        ((knight_attacks(mem, enemy_king) & knights) != 0ULL) ||
        ((king_attacks(mem, enemy_king) & kings) != 0ULL) ||
        ((bishop_attacks_fast(mem, enemy_king, occupied) & bishop_like) != 0ULL) ||
        ((rook_attacks_fast(mem, enemy_king, occupied) & rook_like) != 0ULL);

#if MAGNUS_VERIFY_GIVES_CHECK
    Position next{};
    position_copy_without_accumulators(next, pos);
    do_move_copy(next, move, mem.tables);
    const bool slow_gives_check =
        checkers_bb(next, mem, static_cast<Color>(next.side_to_move)) != 0ULL;
    if (gives_check != slow_gives_check)
        std::abort();
#endif

    return gives_check;
}

/*
 * è‘—æ³•ç”Ÿæˆå¯¦ä½œ
 * generate_pseudo_captures/quiets() â€” ç”Ÿæˆå½åˆæ³•æ•ç²/å®‰éœè‘—æ³•
 * generate_legal() â€” ç”Ÿæˆå®Œæ•´åˆæ³•è‘—æ³•åˆ—è¡¨
 */
Move* generate_captures(
    Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept {
    // In non-check nodes, avoid full pseudo generation and build capture
    // candidates directly for qsearch.
    GenInfo info{};
    init_gen_info(info, pos, mem);

    Move* end = generate_pseudo_captures_only(pos, mem, info, out);
    Move* write = out;

    for (Move* cur = out; cur != end; ++cur) {
        if (needs_legal_filter(pos, info, *cur) &&
            !legal_fast_impl(pos, mem, info, *cur))
            continue;

        if (write != cur)
            *write = *cur;
        ++write;
    }

    return write;
}

Move* generate_captures(
    const Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept {
    Position work{};
    position_copy_without_accumulators(work, pos);
    return generate_captures(work, mem, out);
}

Move* generate_quiets(
    const Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept {
    GenInfo info{};
    init_gen_info(info, pos, mem);

    Move* end = generate_pseudo_quiets(pos, mem, info, out);
    Move* write = out;
    Position work{};
    position_copy_without_accumulators(work, pos);

    for (Move* cur = out; cur != end; ++cur) {
        if (needs_legal_filter(work, info, *cur) &&
            !legal_fast_impl(work, mem, info, *cur))
            continue;

        if (write != cur)
            *write = *cur;
        ++write;
    }

    return write;
}

Move* generate_non_evasions(
    const Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept {
    GenInfo info{};
    init_gen_info(info, pos, mem);

    if (info.in_check)
        return generate_evasions_with_info(pos, mem, info, out);

    return generate_non_evasions_with_info(pos, mem, info, out);
}

Move* generate_evasions(
    const Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept {
    GenInfo info{};
    init_gen_info(info, pos, mem);

    if (!info.in_check)
        return generate_non_evasions_with_info(pos, mem, info, out);

    return generate_evasions_with_info(pos, mem, info, out);
}

Move* generate_pseudo_legal(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    return generate_pseudo_legal_with_info(pos, mem, info, out);
}

Move* generate_pseudo_legal(
    const Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept {
    GenInfo info{};
    init_gen_info(info, pos, mem);
    return generate_pseudo_legal_with_info(pos, mem, info, out);
}

Move* generate_pseudo_captures(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    return info.in_check
        ? generate_evasions_with_info(pos, mem, info, out)
        : generate_capture_non_evasions_with_info(pos, mem, info, out);
}

Move* generate_pseudo_captures(
    const Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept {
    GenInfo info{};
    init_gen_info(info, pos, mem);
    return generate_pseudo_captures(pos, mem, info, out);
}

Move* generate_pseudo_captures_only(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    return info.in_check
        ? generate_capture_evasions_with_info(pos, mem, info, out)
        : generate_capture_non_evasions_with_info(pos, mem, info, out);
}

Move* generate_pseudo_quiets(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept {
    return info.in_check
        ? generate_quiet_evasions_with_info(pos, mem, info, out)
        : generate_quiet_non_evasions_with_info(pos, mem, info, out);
}

Move* generate_legal(
    Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept {
    // Generate pseudo-legal moves first, then keep only the moves that leave
    // the king safe according to the fast legality checker.
    GenInfo info{};
    init_gen_info(info, pos, mem);

    Move* end = generate_pseudo_legal_with_info(pos, mem, info, out);
    Move* write = out;

    for (Move* cur = out; cur != end; ++cur) {
        if (needs_legal_filter(pos, info, *cur) &&
            !legal_fast_impl(pos, mem, info, *cur))
            continue;

        if (write != cur)
            *write = *cur;
        ++write;
    }

    return write;
}

Move* generate_legal(
    const Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept {
    Position work{};
    position_copy_without_accumulators(work, pos);
    return generate_legal(work, mem, out);
}

Move* generate(
    const Position& pos,
    const memory::Memory& mem,
    Move* out,
    GenType type
) noexcept {
    switch (type) {
        case GEN_CAPTURES:     return generate_captures(pos, mem, out);
        case GEN_QUIETS:       return generate_quiets(pos, mem, out);
        case GEN_NON_EVASIONS: return generate_non_evasions(pos, mem, out);
        case GEN_EVASIONS:     return generate_evasions(pos, mem, out);
        case GEN_PSEUDO_LEGAL: return generate_pseudo_legal(pos, mem, out);
        case GEN_LEGAL:        return generate_legal(pos, mem, out);
        default:               return out;
    }
}

} // namespace magnus
