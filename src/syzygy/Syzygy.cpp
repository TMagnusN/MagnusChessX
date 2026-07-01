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

#include "Syzygy.h"

#include <algorithm>
#include <bit>
#include <string>

#include "Memory.h"
#include "board/MoveGen.h"
#include "board/Position.h"

extern "C" {
#include <tbprobe.h>
}

namespace magnus::syzygy {

namespace {

struct ProbePosition {
    u64 white = 0;
    u64 black = 0;
    u64 kings = 0;
    u64 queens = 0;
    u64 rooks = 0;
    u64 bishops = 0;
    u64 knights = 0;
    u64 pawns = 0;
    unsigned rule50 = 0;
    unsigned castling = 0;
    unsigned ep = 0;
    bool white_to_move = true;
};

[[nodiscard]] ProbePosition convert_position(const Position& pos) noexcept {
    return {
        .white = pieces(pos, WHITE),
        .black = pieces(pos, BLACK),
        .kings = pieces_of_type(pos, KING),
        .queens = pieces_of_type(pos, QUEEN),
        .rooks = pieces_of_type(pos, ROOK),
        .bishops = pieces_of_type(pos, BISHOP),
        .knights = pieces_of_type(pos, KNIGHT),
        .pawns = pieces_of_type(pos, PAWN),
        .rule50 = static_cast<unsigned>(std::clamp(pos.halfmove_clock, 0, 255)),
        .castling = static_cast<unsigned>(pos.castling_rights),
        .ep = has_ep(pos) ? static_cast<unsigned>(pos.ep_sq) : 0U,
        .white_to_move = pos.side_to_move == WHITE
    };
}

[[nodiscard]] int effective_limit(int probe_limit) noexcept {
    return std::min(
        std::clamp(probe_limit, MIN_PROBE_LIMIT, MAX_PROBE_LIMIT),
        max_cardinality()
    );
}

[[nodiscard]] bool position_is_probeable(
    const Position& pos,
    int probe_limit
) noexcept {
    const int limit = effective_limit(probe_limit);
    return limit > 0 &&
           pos.castling_rights == NO_CASTLING &&
           std::popcount(pos.occupied) <= limit;
}

[[nodiscard]] Wdl wdl_from_fathom(unsigned value) noexcept {
    switch (value) {
        case TB_LOSS: return Wdl::Loss;
        case TB_BLESSED_LOSS: return Wdl::BlessedLoss;
        case TB_CURSED_WIN: return Wdl::CursedWin;
        case TB_WIN: return Wdl::Win;
        default: return Wdl::Draw;
    }
}

[[nodiscard]] PieceType promotion_from_fathom(unsigned promotion) noexcept {
    switch (promotion) {
        case TB_PROMOTES_QUEEN: return QUEEN;
        case TB_PROMOTES_ROOK: return ROOK;
        case TB_PROMOTES_BISHOP: return BISHOP;
        case TB_PROMOTES_KNIGHT: return KNIGHT;
        default: return PIECE_TYPE_NONE;
    }
}

[[nodiscard]] Move match_root_move(
    const MoveList& legal_moves,
    TbMove tb_move
) noexcept {
    const Square from = static_cast<Square>(TB_MOVE_FROM(tb_move));
    const Square to = static_cast<Square>(TB_MOVE_TO(tb_move));
    const PieceType promotion =
        promotion_from_fathom(TB_MOVE_PROMOTES(tb_move));

    for (int i = 0; i < legal_moves.size; ++i) {
        const Move move = legal_moves.moves[i];
        if (from_sq(move) != from || to_sq(move) != to)
            continue;
        if (promotion == PIECE_TYPE_NONE)
            return move_is_promotion(move) ? Move(0) : move;
        if (move_is_promotion(move) && promo_piece(move) == promotion)
            return move;
    }

    return Move(0);
}

[[nodiscard]] bool move_is_allowed(
    Move move,
    const Move* allowed_moves,
    int allowed_move_count
) noexcept {
    if (allowed_move_count <= 0)
        return true;

    for (int i = 0; i < allowed_move_count; ++i)
        if (allowed_moves[i] == move)
            return true;

    return false;
}

[[nodiscard]] Wdl wdl_from_rank(int rank, bool use_rule50) noexcept {
    if (!use_rule50)
        return rank > 0 ? Wdl::Win : (rank < 0 ? Wdl::Loss : Wdl::Draw);
    if (rank >= 1000)
        return Wdl::Win;
    if (rank > 0)
        return Wdl::CursedWin;
    if (rank == 0)
        return Wdl::Draw;
    if (rank > -1000)
        return Wdl::BlessedLoss;
    return Wdl::Loss;
}

} // namespace

bool init(std::string_view path) noexcept {
    const std::string owned_path(path);
    return tb_init(owned_path.c_str());
}

void shutdown() noexcept {
    tb_free();
}

int max_cardinality() noexcept {
    return static_cast<int>(TB_LARGEST);
}

bool probe_wdl(
    const Position& pos,
    int probe_limit,
    Wdl& result
) noexcept {
    if (!position_is_probeable(pos, probe_limit))
        return false;

    const ProbePosition tb = convert_position(pos);
    const unsigned value = tb_probe_wdl(
        tb.white,
        tb.black,
        tb.kings,
        tb.queens,
        tb.rooks,
        tb.bishops,
        tb.knights,
        tb.pawns,
        0,
        tb.castling,
        tb.ep,
        tb.white_to_move
    );
    if (value == TB_RESULT_FAILED)
        return false;

    result = wdl_from_fathom(value);
    return true;
}

bool rank_root_moves(
    const Position& pos,
    const memory::Memory& mem,
    int probe_limit,
    bool use_rule50,
    const Move* allowed_moves,
    int allowed_move_count,
    RootProbe& result,
    bool rank_against_all_legal,
    bool rank_dtz
) noexcept {
    result = {};
    if (!position_is_probeable(pos, probe_limit))
        return false;

    const ProbePosition tb = convert_position(pos);
    TbRootMoves ranked{};
    int success = tb_probe_root_dtz(
        tb.white,
        tb.black,
        tb.kings,
        tb.queens,
        tb.rooks,
        tb.bishops,
        tb.knights,
        tb.pawns,
        tb.rule50,
        tb.castling,
        tb.ep,
        tb.white_to_move,
        false,
        use_rule50,
        rank_dtz,
        &ranked
    );
    result.used_dtz = success != 0;
    if (!success) {
        success = tb_probe_root_wdl(
            tb.white,
            tb.black,
            tb.kings,
            tb.queens,
            tb.rooks,
            tb.bishops,
            tb.knights,
            tb.pawns,
            tb.rule50,
            tb.castling,
            tb.ep,
            tb.white_to_move,
            use_rule50,
            &ranked
        );
    }
    if (!success || ranked.size == 0)
        return false;

    MoveList legal_moves{};
    generate_legal(pos, mem, legal_moves);

    int best_rank = -32'000;
    for (unsigned i = 0; i < ranked.size; ++i) {
        const Move move = match_root_move(legal_moves, ranked.moves[i].move);
        if (move_is_none(move)) {
            continue;
        }
        if (!rank_against_all_legal &&
            !move_is_allowed(move, allowed_moves, allowed_move_count)) {
            continue;
        }
        best_rank = std::max(best_rank, ranked.moves[i].tbRank);
    }
    if (best_rank == -32'000)
        return false;

    for (unsigned i = 0;
         i < ranked.size && result.move_count < 256;
         ++i) {
        if (ranked.moves[i].tbRank != best_rank)
            continue;

        const Move move = match_root_move(legal_moves, ranked.moves[i].move);
        if (move_is_none(move) ||
            !move_is_allowed(move, allowed_moves, allowed_move_count)) {
            continue;
        }

        result.moves[result.move_count++] = move;
    }
    if (result.move_count == 0)
        return false;

    result.rank = best_rank;
    result.wdl = wdl_from_rank(best_rank, use_rule50);
    return true;
}

} // namespace magnus::syzygy
