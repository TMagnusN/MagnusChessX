/*
MIT License

Copyright (c) 2026 Magnus

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

#include "mnue/MnueX1Features.h"

#include "Memory.h"
#include "board/Attack.h"
#include "board/Position.h"

#include <bit>
#include <cassert>
#include <cstdint>
#include <ostream>
#include <span>

namespace magnus::mnue::x1 {
namespace {

[[nodiscard]] constexpr Square relative_square(
    Color perspective,
    Square square
) noexcept {
    return perspective == WHITE ? square : square ^ 56;
}

[[nodiscard]] constexpr int king_zone16(Square square) noexcept {
    return (rank_of(square) / 2) * 4 + file_of(square) / 2;
}

[[nodiscard]] Square safe_king_square(
    const Position& pos,
    Color color
) noexcept {
    const Square square = king_square(pos, color);
    return square == NO_SQ ? 0 : square;
}

[[nodiscard]] Bitboard pawn_attackers(
    const Position& pos,
    const memory::Memory& mem,
    Square square,
    Color by
) noexcept {
    return pawn_attacks(mem, ~by, square) & pieces(pos, by, PAWN);
}

[[nodiscard]] Bitboard knight_attackers(
    const Position& pos,
    const memory::Memory& mem,
    Square square,
    Color by
) noexcept {
    return knight_attacks(mem, square) & pieces(pos, by, KNIGHT);
}

[[nodiscard]] Bitboard diagonal_attackers(
    const Position& pos,
    const memory::Memory& mem,
    Square square,
    Color by
) noexcept {
    return bishop_attacks_fast(mem, square, pos.occupied)
        & (pieces(pos, by, BISHOP) | pieces(pos, by, QUEEN));
}

[[nodiscard]] Bitboard orthogonal_attackers(
    const Position& pos,
    const memory::Memory& mem,
    Square square,
    Color by
) noexcept {
    return rook_attacks_fast(mem, square, pos.occupied)
        & (pieces(pos, by, ROOK) | pieces(pos, by, QUEEN));
}

[[nodiscard]] Bitboard king_attackers(
    const Position& pos,
    const memory::Memory& mem,
    Square square,
    Color by
) noexcept {
    return king_attacks(mem, square) & pieces(pos, by, KING);
}

[[nodiscard]] bool has_any_attacker(
    const Position& pos,
    const memory::Memory& mem,
    Square square,
    Color by
) noexcept {
    return (
        pawn_attackers(pos, mem, square, by)
        | knight_attackers(pos, mem, square, by)
        | diagonal_attackers(pos, mem, square, by)
        | orthogonal_attackers(pos, mem, square, by)
        | king_attackers(pos, mem, square, by)
    ) != 0ULL;
}

} // namespace

int input_bucket(
    const Position& pos,
    Color perspective
) noexcept {
    return king_zone16(relative_square(
        perspective,
        safe_king_square(pos, perspective)
    ));
}

int output_bucket(const Position& pos) noexcept {
    const int count = std::popcount(
        static_cast<std::uint64_t>(pos.occupied)
    );

    if (count <= 5) return 0;
    if (count <= 8) return 1;
    if (count <= 11) return 2;
    if (count <= 14) return 3;
    if (count <= 17) return 4;
    if (count <= 20) return 5;
    if (count <= 24) return 6;
    return 7;
}

u8 tactical_status(
    const Position& pos,
    const memory::Memory& mem,
    Square square
) noexcept {
    const Piece piece = piece_on(pos, square);
    if (piece == PIECE_NONE)
        return 0;

    const Color victim = color_of(piece);
    const Color enemy = ~victim;
    u8 status = 0;

    if (pawn_attackers(pos, mem, square, enemy) != 0ULL)
        status = static_cast<u8>(status | EnemyPawn);
    if (knight_attackers(pos, mem, square, enemy) != 0ULL)
        status = static_cast<u8>(status | EnemyKnight);
    if (diagonal_attackers(pos, mem, square, enemy) != 0ULL)
        status = static_cast<u8>(status | EnemyDiagonal);
    if (orthogonal_attackers(pos, mem, square, enemy) != 0ULL)
        status = static_cast<u8>(status | EnemyOrthogonal);
    if (king_attackers(pos, mem, square, enemy) != 0ULL)
        status = static_cast<u8>(status | EnemyKing);
    if (has_any_attacker(pos, mem, square, victim))
        status = static_cast<u8>(status | Defended);

    return status;
}

int piece_feature_index(
    const Position& pos,
    Color perspective,
    Piece piece,
    Square square
) noexcept {
    if (piece == PIECE_NONE || type_of(piece) == KING)
        return -1;

    const int relative_color = color_of(piece) == perspective ? 0 : 1;
    const int piece_type = static_cast<int>(type_of(piece));
    const int relative_sq = relative_square(perspective, square);

    return (((input_bucket(pos, perspective) * Layout::RelativeColors
              + relative_color)
             * Layout::NonKingPieceTypes
             + piece_type)
            * Layout::Squares)
        + relative_sq;
}

int tactical_feature_index(
    Color perspective,
    Piece piece,
    Square square,
    u8 status
) noexcept {
    if (piece == PIECE_NONE)
        return -1;

    const int relative_color = color_of(piece) == perspective ? 0 : 1;
    const int piece_type = static_cast<int>(type_of(piece));
    const int relative_class =
        relative_color * Layout::PieceTypes + piece_type;
    const int relative_sq = relative_square(perspective, square);

    return Layout::PieceInputSize
        + ((relative_class * Layout::Squares + relative_sq)
           * Layout::TacticalStates)
        + static_cast<int>(status);
}

std::size_t collect_features(
    const Position& pos,
    const memory::Memory& mem,
    Color perspective,
    FeatureList& output
) noexcept {
    std::size_t count = 0;
    Bitboard occupied = pos.occupied;

    while (occupied != 0ULL) {
        const Square square = static_cast<Square>(
            std::countr_zero(static_cast<std::uint64_t>(occupied))
        );
        occupied &= occupied - 1;

        const Piece piece = piece_on(pos, square);
        assert(piece != PIECE_NONE);

        const int piece_index =
            piece_feature_index(pos, perspective, piece, square);
        if (piece_index >= 0) {
            assert(count < output.size());
            output[count++] = static_cast<u16>(piece_index);
        }

        const int tactical_index = tactical_feature_index(
            perspective,
            piece,
            square,
            tactical_status(pos, mem, square)
        );
        assert(tactical_index >= Layout::PieceInputSize);
        assert(tactical_index < Layout::InputSize);
        assert(count < output.size());
        output[count++] = static_cast<u16>(tactical_index);
    }

    assert(count <= Layout::MaxActive);
    return count;
}

void collect_feature_pairs(
    const Position& pos,
    const memory::Memory& mem,
    PerspectiveFeatureLists& output,
    PerspectiveFeatureCounts& counts
) noexcept {
    counts = {};
    const std::array<int, COLOR_NB> buckets{{
        input_bucket(pos, WHITE),
        input_bucket(pos, BLACK)
    }};

    Bitboard occupied = pos.occupied;
    while (occupied != 0ULL) {
        const Square square = static_cast<Square>(
            std::countr_zero(static_cast<std::uint64_t>(occupied))
        );
        occupied &= occupied - 1;

        const Piece piece = piece_on(pos, square);
        assert(piece != PIECE_NONE);
        const u8 status = tactical_status(pos, mem, square);

        for (int perspective_index = WHITE;
             perspective_index <= BLACK;
             ++perspective_index) {
            const Color perspective =
                static_cast<Color>(perspective_index);
            FeatureList& features =
                output[static_cast<std::size_t>(perspective_index)];
            std::size_t& count =
                counts[static_cast<std::size_t>(perspective_index)];

            if (type_of(piece) != KING) {
                const int relative_color =
                    color_of(piece) == perspective ? 0 : 1;
                const int relative_sq =
                    relative_square(perspective, square);
                const int piece_index =
                    (((buckets[static_cast<std::size_t>(perspective_index)]
                       * Layout::RelativeColors + relative_color)
                      * Layout::NonKingPieceTypes
                      + static_cast<int>(type_of(piece)))
                     * Layout::Squares)
                    + relative_sq;
                assert(count < features.size());
                features[count++] = static_cast<u16>(piece_index);
            }

            const int tactical_index = tactical_feature_index(
                perspective,
                piece,
                square,
                status
            );
            assert(tactical_index >= Layout::PieceInputSize);
            assert(tactical_index < Layout::InputSize);
            assert(count < features.size());
            features[count++] = static_cast<u16>(tactical_index);
        }
    }

    assert(counts[WHITE] <= Layout::MaxActive);
    assert(counts[BLACK] <= Layout::MaxActive);
}

void debug_dump_features(
    const Position& pos,
    const memory::Memory& mem,
    std::ostream& output
) {
    output << "mnue x1 features\n";
    output << "output_bucket " << output_bucket(pos) << '\n';

    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const Color perspective = static_cast<Color>(perspective_index);
        FeatureList features{};
        const std::size_t count =
            collect_features(pos, mem, perspective, features);
        std::sort(features.begin(), features.begin() + count);

        output << "perspective " << perspective_index
            << " input_bucket " << input_bucket(pos, perspective)
            << " count " << count
            << '\n';
        output << "indices";
        for (const u16 index : std::span(features.data(), count))
            output << ' ' << index;
        output << '\n';
    }
}

} // namespace magnus::mnue::x1
