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

#include <array>

#include "Types.h"

namespace magnus {

// Canonical board state. Field order is intentionally stable because Position
// is copied frequently and owns aligned NNUE accumulator storage.
struct Position {
    int side_to_move = WHITE;
    Square ep_sq = NO_SQ;
    int castling_rights = 0;
    int halfmove_clock = 0;
    int fullmove_number = 1;

    Square king_sq[COLOR_NB]{ NO_SQ, NO_SQ };

    Bitboard color_bb[COLOR_NB]{};
    Bitboard piece_bb[PIECE_NB]{};
    Bitboard occupied = 0ULL;
    std::array<std::array<u8, PIECE_NB>, COLOR_NB> piece_counts{};
    u8 non_king_material = 0;
    u8 mnue_phase_units = 0;
    Key material_signature = 0ULL;

    Key key = 0;

    int board[SQ_NB];
};

// Reversible metadata captured before make_move().
struct StateInfo {
    int castling_rights = 0;
    Square ep_sq = NO_SQ;
    int halfmove_clock = 0;
    int fullmove_number = 1;
    Key key = 0ULL;
    Piece captured = PIECE_NONE;
    Square captured_sq = NO_SQ;
};

inline int us(const Position& pos) noexcept {
    return pos.side_to_move;
}

inline int them(const Position& pos) noexcept {
    return pos.side_to_move ^ 1;
}

inline Bitboard pieces(const Position& pos) noexcept {
    return pos.occupied;
}

inline Bitboard pieces(const Position& pos, Color color) noexcept {
    return pos.color_bb[color];
}

inline Bitboard pieces_of_type(
    const Position& pos,
    PieceType piece_type
) noexcept {
    return pos.piece_bb[piece_type];
}

inline Bitboard pieces(
    const Position& pos,
    Color color,
    PieceType piece_type
) noexcept {
    return pos.color_bb[color] & pos.piece_bb[piece_type];
}

inline int piece_count(
    const Position& pos,
    Color color,
    PieceType piece_type
) noexcept {
    return static_cast<int>(pos.piece_counts[color][piece_type]);
}

inline int non_king_material(const Position& pos) noexcept {
    return static_cast<int>(pos.non_king_material);
}

inline int mnue_phase_units(const Position& pos) noexcept {
    return static_cast<int>(pos.mnue_phase_units);
}

inline Key packed_material_signature(const Position& pos) noexcept {
    return pos.material_signature;
}

inline Square king_square(const Position& pos, Color color) noexcept {
    return pos.king_sq[color];
}

inline bool has_ep(const Position& pos) noexcept {
    return pos.ep_sq != NO_SQ;
}

inline Piece piece_on(const Position& pos, Square sq) noexcept {
    return static_cast<Piece>(pos.board[sq]);
}

inline Color color_on(const Position& pos, Square sq) noexcept {
    const Piece piece = static_cast<Piece>(pos.board[sq]);
    return piece == PIECE_NONE ? COLOR_NONE : color_of(piece);
}

inline PieceType piece_type_on(const Position& pos, Square sq) noexcept {
    const Piece piece = static_cast<Piece>(pos.board[sq]);
    return piece == PIECE_NONE ? PIECE_TYPE_NONE : type_of(piece);
}

inline bool empty_on(const Position& pos, Square sq) noexcept {
    return pos.board[sq] == PIECE_NONE;
}

inline bool occupied_on(const Position& pos, Square sq) noexcept {
    return pos.board[sq] != PIECE_NONE;
}

struct Tables;

void position_clear(Position& pos) noexcept;
void position_copy_without_accumulators(Position& dst, const Position& src) noexcept;
void position_recompute_occupied(Position& pos) noexcept;
void position_refresh_king_squares(Position& pos) noexcept;
[[nodiscard]] Key position_compute_key(const Position& pos, const Tables& tables) noexcept;
void position_refresh_key(Position& pos, const Tables& tables) noexcept;

void position_put_piece(
    Position& pos,
    Color color,
    PieceType piece_type,
    Square sq
) noexcept;

void position_remove_piece(
    Position& pos,
    Color color,
    PieceType piece_type,
    Square sq
) noexcept;

void position_move_piece(
    Position& pos,
    Color color,
    PieceType piece_type,
    Square from,
    Square to
) noexcept;

bool position_has_valid_kings(const Position& pos) noexcept;
bool position_board_matches_bitboards(const Position& pos) noexcept;

void make_move(
    Position& pos,
    Move m,
    const Tables& tables,
    StateInfo& st
) noexcept;

void unmake_move(
    Position& pos,
    Move m,
    const Tables& tables,
    const StateInfo& st
) noexcept;

// Copy-make helpers used by perft and legality validation paths. The overload
// with Tables keeps the incremental Zobrist key up to date as well.
void do_move_copy(Position& pos, Move m) noexcept;
void do_move_copy(Position& pos, Move m, const Tables& tables) noexcept;

} // namespace magnus
