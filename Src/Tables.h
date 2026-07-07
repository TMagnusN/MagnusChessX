/*
MIT License

Copyright (c) 2026 Magnus🦄

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

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "Types.h"

namespace magnus {

/*
This file owns the static lookup tables that are expensive to rebuild during
search: leaper attacks, geometry masks, distance tables, and Zobrist keys.
Geometry / leaper tables are built at compile time via consteval factories;
only Zobrist keys still need runtime PRNG setup.
*/

struct ZobristTables {
    // Piece-square Zobrist keys indexed by color, piece type, and square.
    Key piece[COLOR_NB][PIECE_NB][SQ_NB]{};
    Key castling[16]{};
    Key ep_file[8]{};
    Key rule50[16]{};
    Key side = 0;
};

constexpr std::size_t CUCKOO_REPETITION_SIZE = 8192;

struct CuckooRepetitionTables {
    Key keys[CUCKOO_REPETITION_SIZE]{};
    Move moves[CUCKOO_REPETITION_SIZE]{};
    bool valid = false;
};

[[nodiscard]] constexpr std::size_t cuckoo_repetition_h1(Key key) noexcept {
    return static_cast<std::size_t>((key >> 32) & 0x1FFFULL);
}

[[nodiscard]] constexpr std::size_t cuckoo_repetition_h2(Key key) noexcept {
    return static_cast<std::size_t>((key >> 48) & 0x1FFFULL);
}

[[nodiscard]] constexpr Move encode_cuckoo_repetition_move(
    Square from,
    Square to
) noexcept {
    return static_cast<Move>(
        (to & 63) | ((from & 63) << 6)
    );
}

// Small constexpr math helpers used while building geometry tables.
constexpr int abs_i(int x) noexcept {
    return x < 0 ? -x : x;
}

constexpr int sign_i(int x) noexcept {
    return (x > 0) - (x < 0);
}

constexpr int chebyshev_distance(Square a, Square b) noexcept {
    const int df = abs_i(file_of(a) - file_of(b));
    const int dr = abs_i(rank_of(a) - rank_of(b));
    return df > dr ? df : dr;
}

constexpr int manhattan_distance(Square a, Square b) noexcept {
    const int df = abs_i(file_of(a) - file_of(b));
    const int dr = abs_i(rank_of(a) - rank_of(b));
    return df + dr;
}

[[nodiscard]] constexpr bool attacks_on_empty_board(
    PieceType piece_type,
    Square from,
    Square to
) noexcept {
    if (from == to)
        return false;

    const int df = abs_i(file_of(from) - file_of(to));
    const int dr = abs_i(rank_of(from) - rank_of(to));

    switch (piece_type) {
        case KNIGHT:
            return (df == 1 && dr == 2) || (df == 2 && dr == 1);
        case BISHOP:
            return df == dr;
        case ROOK:
            return df == 0 || dr == 0;
        case QUEEN:
            return df == 0 || dr == 0 || df == dr;
        case KING:
            return df <= 1 && dr <= 1;
        default:
            return false;
    }
}

inline u64 splitmix64(u64& x) noexcept {
    u64 z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline u64 mix64(u64 x) noexcept {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

constexpr Bitboard king_attack_mask(Square sq) noexcept {
    Bitboard b = 0;
    const int f = file_of(sq);
    const int r = rank_of(sq);

    for (int df = -1; df <= 1; ++df) {
        for (int dr = -1; dr <= 1; ++dr) {
            if (df == 0 && dr == 0) continue;
            const int nf = f + df;
            const int nr = r + dr;
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8)
                b |= bb_of(nr * 8 + nf);
        }
    }

    return b;
}

constexpr Bitboard knight_attack_mask(Square sq) noexcept {
    Bitboard b = 0;
    const int f = file_of(sq);
    const int r = rank_of(sq);

    constexpr int D[8][2] = {
        { 1,  2}, { 2,  1}, { 2, -1}, { 1, -2},
        {-1, -2}, {-2, -1}, {-2,  1}, {-1,  2}
    };

    for (int i = 0; i < 8; ++i) {
        const int nf = f + D[i][0];
        const int nr = r + D[i][1];
        if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8)
            b |= bb_of(nr * 8 + nf);
    }

    return b;
}

constexpr Bitboard pawn_attack_mask(Color c, Square sq) noexcept {
    Bitboard b = 0;
    const int f = file_of(sq);
    const int r = rank_of(sq);

    if (c == WHITE) {
        if (f > 0 && r < 7) b |= bb_of((r + 1) * 8 + (f - 1));
        if (f < 7 && r < 7) b |= bb_of((r + 1) * 8 + (f + 1));
    } else {
        if (f > 0 && r > 0) b |= bb_of((r - 1) * 8 + (f - 1));
        if (f < 7 && r > 0) b |= bb_of((r - 1) * 8 + (f + 1));
    }

    return b;
}

constexpr bool aligned_line(Square a, Square b) noexcept {
    const int df = file_of(b) - file_of(a);
    const int dr = rank_of(b) - rank_of(a);
    return df == 0 || dr == 0 || abs_i(df) == abs_i(dr);
}

constexpr Bitboard compute_line(Square a, Square b) noexcept {
    if (a == b) return bb_of(a);
    if (!aligned_line(a, b)) return 0ULL;

    const int df = sign_i(file_of(b) - file_of(a));
    const int dr = sign_i(rank_of(b) - rank_of(a));

    int f = file_of(a);
    int r = rank_of(a);
    while (true) {
        const int prev_f = f - df;
        const int prev_r = r - dr;
        if (prev_f < 0 || prev_f >= 8 || prev_r < 0 || prev_r >= 8)
            break;
        f = prev_f;
        r = prev_r;
    }

    Bitboard mask = 0ULL;
    while (f >= 0 && f < 8 && r >= 0 && r < 8) {
        const Square sq = static_cast<Square>(r * 8 + f);
        mask |= bb_of(sq);
        f += df;
        r += dr;
    }

    return mask;
}

constexpr Bitboard compute_between(Square a, Square b) noexcept {
    if (a == b) return 0ULL;
    if (!aligned_line(a, b)) return 0ULL;

    const int df = sign_i(file_of(b) - file_of(a));
    const int dr = sign_i(rank_of(b) - rank_of(a));
    const int step = dr * 8 + df;

    Bitboard mask = 0ULL;
    int sq = a + step;

    while (sq != b) {
        mask |= bb_of(sq);
        sq += step;
    }

    return mask;
}

/*
 * Consteval factories — every geometry/lookup table is precomputed here so
 * Tables can be populated with zero runtime overhead. Each function is called
 * exactly once as a default member initializer of Tables; the resulting
 * std::array data lives directly in .rodata.
 *
 * These MUST appear before the Tables struct definition because they are
 * used in its default member initializers.
 */
consteval std::array<Bitboard, SQ_NB> make_king_attacks() noexcept {
    std::array<Bitboard, SQ_NB> arr{};
    for (int sq = 0; sq < SQ_NB; ++sq)
        arr[sq] = king_attack_mask(static_cast<Square>(sq));
    return arr;
}

consteval std::array<Bitboard, SQ_NB> make_knight_attacks() noexcept {
    std::array<Bitboard, SQ_NB> arr{};
    for (int sq = 0; sq < SQ_NB; ++sq)
        arr[sq] = knight_attack_mask(static_cast<Square>(sq));
    return arr;
}

consteval std::array<std::array<Bitboard, SQ_NB>, COLOR_NB> make_pawn_attacks() noexcept {
    std::array<std::array<Bitboard, SQ_NB>, COLOR_NB> arr{};
    for (int c = 0; c < COLOR_NB; ++c)
        for (int sq = 0; sq < SQ_NB; ++sq)
            arr[c][sq] = pawn_attack_mask(static_cast<Color>(c), static_cast<Square>(sq));
    return arr;
}

consteval std::array<std::array<Bitboard, SQ_NB>, SQ_NB> make_between() noexcept {
    std::array<std::array<Bitboard, SQ_NB>, SQ_NB> arr{};
    for (int a = 0; a < SQ_NB; ++a)
        for (int b = 0; b < SQ_NB; ++b)
            arr[a][b] = compute_between(static_cast<Square>(a), static_cast<Square>(b));
    return arr;
}

consteval std::array<std::array<Bitboard, SQ_NB>, SQ_NB> make_line() noexcept {
    std::array<std::array<Bitboard, SQ_NB>, SQ_NB> arr{};
    for (int a = 0; a < SQ_NB; ++a)
        for (int b = 0; b < SQ_NB; ++b)
            arr[a][b] = compute_line(static_cast<Square>(a), static_cast<Square>(b));
    return arr;
}

consteval std::array<std::array<u8, SQ_NB>, SQ_NB> make_chebyshev() noexcept {
    std::array<std::array<u8, SQ_NB>, SQ_NB> arr{};
    for (int a = 0; a < SQ_NB; ++a)
        for (int b = 0; b < SQ_NB; ++b)
            arr[a][b] = static_cast<u8>(chebyshev_distance(static_cast<Square>(a), static_cast<Square>(b)));
    return arr;
}

consteval std::array<std::array<u8, SQ_NB>, SQ_NB> make_manhattan() noexcept {
    std::array<std::array<u8, SQ_NB>, SQ_NB> arr{};
    for (int a = 0; a < SQ_NB; ++a)
        for (int b = 0; b < SQ_NB; ++b)
            arr[a][b] = static_cast<u8>(manhattan_distance(static_cast<Square>(a), static_cast<Square>(b)));
    return arr;
}

/*
 * Tables — 靜態查表集合（編譯期初始化，搜尋期間唯讀）
 * 包含：國王/騎士/兵攻擊遮罩、between/line 幾何遮罩、
 * Chebyshev/Manhattan 距離表、Zobrist 雜湊鍵表
 * 幾何/跳子表在編譯期通過 consteval 工廠函數一次性構建
 */
struct Tables {
    // Leaper attack masks and board geometry helpers.
    // All geometry/lookup tables are built at compile time via consteval
    // factory functions so there is zero runtime setup overhead here.
    std::array<Bitboard, SQ_NB> king_attacks   = make_king_attacks();
    std::array<Bitboard, SQ_NB> knight_attacks  = make_knight_attacks();
    std::array<std::array<Bitboard, SQ_NB>, COLOR_NB> pawn_attacks = make_pawn_attacks();

    std::array<std::array<Bitboard, SQ_NB>, SQ_NB> between  = make_between();  // Squares strictly between two aligned squares.
    std::array<std::array<Bitboard, SQ_NB>, SQ_NB> line     = make_line();     // Full rank/file/diagonal through two aligned squares.

    std::array<std::array<u8, SQ_NB>, SQ_NB> chebyshev  = make_chebyshev();
    std::array<std::array<u8, SQ_NB>, SQ_NB> manhattan  = make_manhattan();

    ZobristTables zobrist{};
    CuckooRepetitionTables cuckoo_repetition{};
    bool initialized = false;
};

// Zobrist keys are still initialized at runtime because they depend on a
// runtime-supplied seed (needed for TT collision diversity). If the seed is
// changed to a constexpr default, the PRNG chain below can be consteval too.
inline void tables_init_zobrist(ZobristTables& z, u64 seed = 0xC0FFEE1234567890ULL) noexcept {
    u64 x = seed;

    for (int c = 0; c < COLOR_NB; ++c)
        for (int pt = 0; pt < PIECE_NB; ++pt)
            for (int sq = 0; sq < SQ_NB; ++sq)
                z.piece[c][pt][sq] = splitmix64(x);

    for (int i = 0; i < 16; ++i)
        z.castling[i] = splitmix64(x);

    for (int i = 0; i < 8; ++i)
        z.ep_file[i] = splitmix64(x);

    z.side = splitmix64(x);

    for (int i = 0; i < 16; ++i)
        z.rule50[i] = splitmix64(x);
}

inline bool cuckoo_repetition_insert(
    CuckooRepetitionTables& cuckoo,
    Key key,
    Move move
) noexcept {
    std::size_t slot = cuckoo_repetition_h1(key);

    for (std::size_t n = 0; n < CUCKOO_REPETITION_SIZE * 4; ++n) {
        std::swap(cuckoo.keys[slot], key);
        std::swap(cuckoo.moves[slot], move);

        if (move == Move(0))
            return true;

        const std::size_t h1 = cuckoo_repetition_h1(key);
        const std::size_t h2 = cuckoo_repetition_h2(key);
        slot = slot == h1 ? h2 : h1;
    }

    return false;
}

inline bool tables_init_cuckoo_repetition(Tables& t) noexcept {
    CuckooRepetitionTables& cuckoo = t.cuckoo_repetition;
    cuckoo = CuckooRepetitionTables{};

    constexpr PieceType pieces[] = {
        KNIGHT,
        BISHOP,
        ROOK,
        QUEEN,
        KING
    };

    int count = 0;
    for (const PieceType piece_type : pieces) {
        for (int color = 0; color < COLOR_NB; ++color) {
            for (int from = 0; from < SQ_NB; ++from) {
                for (int to = from + 1; to < SQ_NB; ++to) {
                    if (!attacks_on_empty_board(
                            piece_type,
                            static_cast<Square>(from),
                            static_cast<Square>(to))) {
                        continue;
                    }

                    const Key key =
                        t.zobrist.piece[color][piece_type][from] ^
                        t.zobrist.piece[color][piece_type][to] ^
                        t.zobrist.side;
                    const Move move = encode_cuckoo_repetition_move(
                        static_cast<Square>(from),
                        static_cast<Square>(to)
                    );

                    if (!cuckoo_repetition_insert(cuckoo, key, move)) {
                        assert(false && "Failed to initialize cuckoo repetition table");
                        cuckoo.valid = false;
                        return false;
                    }

                    ++count;
                }
            }
        }
    }

    cuckoo.valid = count == 3668;
    assert(cuckoo.valid && "Unexpected cuckoo repetition table entry count");
    return cuckoo.valid;
}

// Geometry tables are now consteval-populated (see make_* factories above),
// so only Zobrist still needs a runtime initialization step.
inline void tables_init(Tables& t, u64 zobrist_seed = 0xC0FFEE1234567890ULL) noexcept {
    if (!t.initialized) {
        tables_init_zobrist(t.zobrist, zobrist_seed);
        tables_init_cuckoo_repetition(t);
        t.initialized = true;
    }
}

} // namespace magnus
