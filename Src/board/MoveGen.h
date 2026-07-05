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

#include <cassert>

#include "Memory.h"
#include "Position.h"
#include "Types.h"

namespace magnus {

/*
    16-bit move layout

    bits  0..5   : to
    bits  6..11  : from
    bits 12..15  : flag
*/
enum MoveFlag : u16 {
    MOVE_QUIET        = 0,
    MOVE_DOUBLE_PUSH  = 1,
    MOVE_OO           = 2,
    MOVE_OOO          = 3,
    MOVE_CAPTURE      = 4,
    MOVE_EP           = 5,

    MOVE_PROMO_N      = 8,
    MOVE_PROMO_B      = 9,
    MOVE_PROMO_R      = 10,
    MOVE_PROMO_Q      = 11,

    MOVE_CAP_PROMO_N  = 12,
    MOVE_CAP_PROMO_B  = 13,
    MOVE_CAP_PROMO_R  = 14,
    MOVE_CAP_PROMO_Q  = 15
};

constexpr Square from_sq(Move move) noexcept {
    return static_cast<Square>((move >> 6) & 63);
}

constexpr Square to_sq(Move move) noexcept {
    return static_cast<Square>(move & 63);
}

constexpr u16 move_flag(Move move) noexcept {
    return static_cast<u16>((move >> 12) & 15);
}

constexpr Move make_move(
    Square from,
    Square to,
    u16 flag = MOVE_QUIET
) noexcept {
    return static_cast<Move>(
        (to & 63) | ((from & 63) << 6) | ((flag & 15) << 12)
    );
}

constexpr bool move_is_none(Move move) noexcept {
    return move == 0;
}

constexpr bool move_is_capture(Move move) noexcept {
    const u16 flag = move_flag(move);
    return flag == MOVE_CAPTURE || flag == MOVE_EP ||
           flag >= MOVE_CAP_PROMO_N;
}

constexpr bool move_is_promotion(Move move) noexcept {
    return move_flag(move) >= MOVE_PROMO_N;
}

constexpr bool move_is_underpromotion(Move move) noexcept {
    const u16 flag = move_flag(move);
    return flag == MOVE_PROMO_N || flag == MOVE_PROMO_B ||
           flag == MOVE_PROMO_R || flag == MOVE_CAP_PROMO_N ||
           flag == MOVE_CAP_PROMO_B || flag == MOVE_CAP_PROMO_R;
}

constexpr bool move_is_castle(Move move) noexcept {
    const u16 flag = move_flag(move);
    return flag == MOVE_OO || flag == MOVE_OOO;
}

constexpr bool move_is_ep(Move move) noexcept {
    return move_flag(move) == MOVE_EP;
}

constexpr bool move_is_double_push(Move move) noexcept {
    return move_flag(move) == MOVE_DOUBLE_PUSH;
}

constexpr PieceType promo_piece(Move move) noexcept {
    switch (move_flag(move)) {
        case MOVE_PROMO_N:
        case MOVE_CAP_PROMO_N: return KNIGHT;
        case MOVE_PROMO_B:
        case MOVE_CAP_PROMO_B: return BISHOP;
        case MOVE_PROMO_R:
        case MOVE_CAP_PROMO_R: return ROOK;
        case MOVE_PROMO_Q:
        case MOVE_CAP_PROMO_Q: return QUEEN;
        default: return QUEEN;
    }
}

constexpr int MAX_MOVES = 384;

struct MoveList {
    Move moves[MAX_MOVES];
    int size = 0;
};

struct ScoredMove {
    Move move = 0;
    i32 score = 0;
    i32 see_value = 0;
};

struct ScoredMoveList {
    ScoredMove moves[MAX_MOVES];
    int size = 0;
};

inline void movelist_clear(MoveList& list) noexcept {
    list.size = 0;
}

inline void scored_movelist_clear(ScoredMoveList& list) noexcept {
    list.size = 0;
}

inline void movelist_push(MoveList& list, Move move) noexcept {
    assert(list.size >= 0 && list.size < MAX_MOVES);
    list.moves[list.size++] = move;
}

inline void scored_movelist_push(
    ScoredMoveList& list,
    Move move,
    i32 score
) noexcept {
    assert(list.size >= 0 && list.size < MAX_MOVES);
    list.moves[list.size++] = {move, score};
}

enum GenType : int {
    GEN_CAPTURES = 0,
    GEN_QUIETS,
    GEN_NON_EVASIONS,
    GEN_EVASIONS,
    GEN_PSEUDO_LEGAL,
    GEN_LEGAL
};

struct GenInfo {
    Color us = WHITE;
    Color them = BLACK;

    Square king_sq = NO_SQ;
    Square ep_sq = NO_SQ;

    Bitboard occupied = 0ULL;
    Bitboard us_occ = 0ULL;
    Bitboard them_occ = 0ULL;

    Bitboard checkers = 0ULL;
    Bitboard pinners = 0ULL;
    Bitboard pinned = 0ULL;
    Bitboard danger = 0ULL;

    Bitboard capture_mask = 0ULL;
    Bitboard push_mask = 0ULL;

    bool in_check = false;
    bool double_check = false;
};


Bitboard attackers_to(
    const Position& pos,
    const memory::Memory& mem,
    Square sq,
    Bitboard occupied
) noexcept;

Bitboard attackers_to_color(
    const Position& pos,
    const memory::Memory& mem,
    Square sq,
    Color by,
    Bitboard occupied
) noexcept;

Bitboard pinned_bb(
    const Position& pos,
    const memory::Memory& mem,
    Color us
) noexcept;

Bitboard pinners_bb(
    const Position& pos,
    const memory::Memory& mem,
    Color us
) noexcept;

Bitboard checkers_bb(
    const Position& pos,
    const memory::Memory& mem,
    Color us
) noexcept;

Bitboard danger_bb(
    const Position& pos,
    const memory::Memory& mem,
    Color by
) noexcept;

void init_gen_info(
    GenInfo& info,
    const Position& pos,
    const memory::Memory& mem
) noexcept;


bool pseudo_legal(
    const Position& pos,
    const memory::Memory& mem,
    Move m
) noexcept;

bool pseudo_legal_fast(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move m
) noexcept;

bool legal(
    Position& pos,
    const memory::Memory& mem,
    Move m
) noexcept;

bool legal(
    const Position& pos,
    const memory::Memory& mem,
    Move m
) noexcept;

bool legal_fast(
    Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move m
) noexcept;

bool move_gives_check(
    const Position& pos,
    const memory::Memory& mem,
    Move m
) noexcept;


Move* generate_captures(
    Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept;

Move* generate_captures(
    const Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept;

Move* generate_quiets(
    const Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept;

Move* generate_non_evasions(
    const Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept;

Move* generate_evasions(
    const Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept;

Move* generate_pseudo_legal(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept;

Move* generate_pseudo_legal(
    const Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept;

Move* generate_pseudo_captures(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept;

Move* generate_pseudo_captures(
    const Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept;

Move* generate_pseudo_captures_only(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept;

Move* generate_pseudo_quiets(
    const Position& pos,
    const memory::Memory& mem,
    const GenInfo& info,
    Move* out
) noexcept;

Move* generate_legal(
    Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept;

Move* generate_legal(
    const Position& pos,
    const memory::Memory& mem,
    Move* out
) noexcept;

inline void generate_captures(
    Position& pos,
    const memory::Memory& mem,
    MoveList& list
) noexcept {
    Move* end = generate_captures(pos, mem, list.moves);
    list.size = static_cast<int>(end - list.moves);
}

inline void generate_captures(
    const Position& pos,
    const memory::Memory& mem,
    MoveList& list
) noexcept {
    Move* end = generate_captures(pos, mem, list.moves);
    list.size = static_cast<int>(end - list.moves);
}

inline void generate_quiets(
    const Position& pos,
    const memory::Memory& mem,
    MoveList& list
) noexcept {
    Move* end = generate_quiets(pos, mem, list.moves);
    list.size = static_cast<int>(end - list.moves);
}

inline void generate_non_evasions(
    const Position& pos,
    const memory::Memory& mem,
    MoveList& list
) noexcept {
    Move* end = generate_non_evasions(pos, mem, list.moves);
    list.size = static_cast<int>(end - list.moves);
}

inline void generate_evasions(
    const Position& pos,
    const memory::Memory& mem,
    MoveList& list
) noexcept {
    Move* end = generate_evasions(pos, mem, list.moves);
    list.size = static_cast<int>(end - list.moves);
}

inline void generate_pseudo_legal(
    const Position& pos,
    const memory::Memory& mem,
    MoveList& list
) noexcept {
    Move* end = generate_pseudo_legal(pos, mem, list.moves);
    list.size = static_cast<int>(end - list.moves);
}

inline void generate_legal(
    Position& pos,
    const memory::Memory& mem,
    MoveList& list
) noexcept {
    Move* end = generate_legal(pos, mem, list.moves);
    list.size = static_cast<int>(end - list.moves);
}

inline void generate_legal(
    const Position& pos,
    const memory::Memory& mem,
    MoveList& list
) noexcept {
    Move* end = generate_legal(pos, mem, list.moves);
    list.size = static_cast<int>(end - list.moves);
}

Move* generate(
    const Position& pos,
    const memory::Memory& mem,
    Move* out,
    GenType type
) noexcept;

inline void generate(
    const Position& pos,
    const memory::Memory& mem,
    MoveList& list,
    GenType type
) noexcept {
    Move* end = generate(pos, mem, list.moves, type);
    list.size = static_cast<int>(end - list.moves);
}

} // namespace magnus