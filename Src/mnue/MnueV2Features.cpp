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

#include "mnue/MnueV2Features.h"
#include "mnue/MnueV2Telemetry.h"

#include "Memory.h"
#include "board/Attack.h"
#include "board/Position.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <sstream>

namespace magnus::mnue::v2 {
namespace {

constexpr int kPieceTypes = 6;
constexpr int kSquares = 64;

constexpr int kPositionKingBuckets = 16;

constexpr int kAttackStates = 64;
constexpr int kAttackStatusCount =
    2 * kPieceTypes * kSquares * kAttackStates;
constexpr int kAttackVictimClasses = 2 * kPieceTypes;
constexpr int kAttackEdgeCount = 90'048;
constexpr int kAttackRelationStates = 16;
constexpr int kAttackRelationCount =
    2 * kPieceTypes * kSquares * kAttackRelationStates;
constexpr int kAttackPressureBins = 9;
constexpr int kAttackPressureCount =
    2 * kPieceTypes * 16 * kAttackPressureBins;

constexpr int kStructurePawnStates = 256;
constexpr int kStructurePawnCount =
    2 * kSquares * kStructurePawnStates;
constexpr int kStructureFileCount = 8 * 4;
constexpr int kStructureIslandCount = 2 * 9;
constexpr int kStructureCenterCount = 9;
constexpr int kStructureShelterCount = 2 * 16 * 9;
constexpr int kStructureOutpostCount = 2 * 2 * kSquares * 4;
constexpr int kStructureComplexCount = 2 * 9 * 9;
constexpr int kStructureBlockerCount = 2 * kSquares * 3;

static_assert(
    kAttackStatusCount + kAttackEdgeCount + kAttackRelationCount
        + kAttackPressureCount
    == Layout::AttackFeatureCount
);
static_assert(
    kStructurePawnCount + kStructureFileCount + kStructureIslandCount
        + kStructureCenterCount + kStructureShelterCount
        + kStructureOutpostCount + kStructureComplexCount
        + kStructureBlockerCount
    == Layout::StructureFeatureCount
);

enum TacticalStatus : u8 {
    EnemyPawn = 1u << 0,
    EnemyKnight = 1u << 1,
    EnemyDiagonal = 1u << 2,
    EnemyOrthogonal = 1u << 3,
    EnemyKing = 1u << 4,
    Defended = 1u << 5
};

template<std::size_t Capacity>
[[nodiscard]] bool add_feature(
    FeatureList<Capacity>& output,
    u32 index,
    u32 vocabulary
) noexcept {
    assert(index < vocabulary);
    if (index >= vocabulary || output.count >= output.indices.size())
        return false;
    output.indices[output.count++] = index;
    return true;
}

template<std::size_t Capacity>
void sort_and_deduplicate(FeatureList<Capacity>& output) noexcept {
    CycleScope cycles(CycleKind::Deduplication);
    auto begin = output.indices.begin();
    auto end = begin + static_cast<std::ptrdiff_t>(output.count);
    std::sort(begin, end);
    end = std::unique(begin, end);
    output.count = static_cast<std::size_t>(end - begin);
}

[[nodiscard]] constexpr Square relative_square(
    Color perspective,
    Square square
) noexcept {
    return perspective == WHITE ? square : square ^ 56;
}

[[nodiscard]] constexpr int relative_color(
    Color perspective,
    Color color
) noexcept {
    return static_cast<int>(color) ^ static_cast<int>(perspective);
}

[[nodiscard]] constexpr int king_zone16(Square square) noexcept {
    return (rank_of(square) / 2) * 4 + file_of(square) / 2;
}

[[nodiscard]] constexpr Bitboard file_mask(int file) noexcept {
    return 0x0101010101010101ULL << file;
}

[[nodiscard]] Bitboard attacks_from(
    const Position& pos,
    const memory::Memory& mem,
    Piece piece,
    Square square
) noexcept {
    const Color color = color_of(piece);
    switch (type_of(piece)) {
        case PAWN:
            if (rank_of(square) < 1 || rank_of(square) > 6)
                return 0ULL;
            return pawn_attacks(mem, color, square);
        case KNIGHT:
            return knight_attacks(mem, square);
        case BISHOP:
            return bishop_attacks_fast(mem, square, pos.occupied);
        case ROOK:
            return rook_attacks_fast(mem, square, pos.occupied);
        case QUEEN:
            return queen_attacks_fast(mem, square, pos.occupied);
        case KING:
            return king_attacks(mem, square);
        default:
            return 0ULL;
    }
}

[[nodiscard]] Bitboard empty_board_attacks(
    const memory::Memory& mem,
    int relative_color_value,
    int piece_type,
    Square square
) noexcept {
    switch (piece_type) {
        case PAWN:
            if (rank_of(square) < 1 || rank_of(square) > 6)
                return 0ULL;
            return pawn_attacks(
                mem,
                static_cast<Color>(relative_color_value),
                square
            );
        case KNIGHT:
            return knight_attacks(mem, square);
        case BISHOP:
            return bishop_rays(square);
        case ROOK:
            return rook_rays(square);
        case QUEEN:
            return queen_rays(square);
        case KING:
            return king_attacks(mem, square);
        default:
            return 0ULL;
    }
}

struct EdgeIndex {
    std::array<
        std::array<std::array<u32, kSquares>, kPieceTypes>,
        COLOR_NB
    > base{};
    u32 size = 0;
};

[[nodiscard]] const EdgeIndex& edge_index_table(
    const memory::Memory& mem
) noexcept {
    // Empty-board leaper/ray geometry is independent of the selected slider
    // backend. Tables are immutable after engine initialisation.
    static const EdgeIndex table = [&mem]() {
        EdgeIndex result{};
        for (int color = 0; color < COLOR_NB; ++color) {
            for (int piece_type = 0; piece_type < kPieceTypes; ++piece_type) {
                for (int square = 0; square < kSquares; ++square) {
                    result.base[color][piece_type][square] = result.size;
                    const Bitboard attacks = empty_board_attacks(
                        mem,
                        color,
                        piece_type,
                        square
                    );
                    result.size += static_cast<u32>(
                        std::popcount(static_cast<u64>(attacks))
                        * kAttackVictimClasses
                    );
                }
            }
        }
        assert(result.size == kAttackEdgeCount);
        return result;
    }();
    return table;
}

[[nodiscard]] u32 position_index(
    int bucket,
    int color,
    int piece_type,
    int square
) noexcept {
    return static_cast<u32>(
        (((bucket * 2 + color) * kPieceTypes + piece_type) * kSquares)
        + square
    );
}

[[nodiscard]] u32 attack_status_index(
    int color,
    int piece_type,
    int square,
    u8 status
) noexcept {
    return static_cast<u32>(
        (((color * kPieceTypes + piece_type) * kSquares + square)
         * kAttackStates)
        + static_cast<int>(status)
    );
}

[[nodiscard]] u32 attack_edge_index(
    const memory::Memory& mem,
    int attacker_color,
    int attacker_type,
    int from,
    int victim_color,
    int victim_type,
    int to
) noexcept {
    const Bitboard attacks = empty_board_attacks(
        mem,
        attacker_color,
        attacker_type,
        from
    );
    const Bitboard before =
        to == 0 ? 0ULL : (Bitboard(1) << to) - 1ULL;
    const u32 slot = static_cast<u32>(
        std::popcount(static_cast<u64>(attacks & before))
    );
    return static_cast<u32>(kAttackStatusCount)
        + edge_index_table(mem).base[attacker_color][attacker_type][from]
        + slot * kAttackVictimClasses
        + static_cast<u32>(victim_color * kPieceTypes + victim_type);
}

[[nodiscard]] u32 attack_relation_index(
    int color,
    int piece_type,
    int square,
    u8 flags
) noexcept {
    return static_cast<u32>(kAttackStatusCount + kAttackEdgeCount)
        + static_cast<u32>(
            (((color * kPieceTypes + piece_type) * kSquares + square)
             * kAttackRelationStates)
            + static_cast<int>(flags)
        );
}

[[nodiscard]] u32 attack_pressure_index(
    int color,
    int piece_type,
    int zone,
    int count
) noexcept {
    return static_cast<u32>(
        kAttackStatusCount + kAttackEdgeCount + kAttackRelationCount
        + (((color * kPieceTypes + piece_type) * 16 + zone)
           * kAttackPressureBins)
        + std::min(count, kAttackPressureBins - 1)
    );
}

[[nodiscard]] bool pinned_to_king(
    const Position& pos,
    Piece piece,
    Square square,
    Square king
) noexcept {
    const Color own = color_of(piece);
    if (type_of(piece) == KING
        || king == NO_SQ
        || color_of(piece_on(pos, king)) != own) {
        return false;
    }

    const int df = file_of(king) - file_of(square);
    const int dr = rank_of(king) - rank_of(square);
    if (df != 0 && dr != 0 && std::abs(df) != std::abs(dr))
        return false;

    const int sf = (df > 0) - (df < 0);
    const int sr = (dr > 0) - (dr < 0);
    int file = file_of(square) - sf;
    int rank = rank_of(square) - sr;
    while (file >= 0 && file < 8 && rank >= 0 && rank < 8) {
        const Square target = rank * 8 + file;
        const Piece attacker = piece_on(pos, target);
        if (attacker != PIECE_NONE) {
            if (color_of(attacker) == ~own) {
                const PieceType type = type_of(attacker);
                return sf == 0 || sr == 0
                    ? type == ROOK || type == QUEEN
                    : type == BISHOP || type == QUEEN;
            }
            return false;
        }
        file -= sf;
        rank -= sr;
    }
    return false;
}

[[nodiscard]] int file_bits(Bitboard pawns) noexcept {
    int mask = 0;
    for (int file = 0; file < 8; ++file) {
        if ((pawns & file_mask(file)) != 0ULL)
            mask |= 1 << file;
    }
    return mask;
}

[[nodiscard]] int pawn_islands(int mask) noexcept {
    return std::popcount(static_cast<unsigned>(mask & ~(mask << 1)));
}

[[nodiscard]] Bitboard ahead_mask(
    Color color,
    Square square,
    bool adjacent
) noexcept {
    Bitboard mask = 0ULL;
    const int file = file_of(square);
    const int rank = rank_of(square);
    for (int df = adjacent ? -1 : 0; df <= (adjacent ? 1 : 0); ++df) {
        const int target_file = file + df;
        if (target_file < 0 || target_file >= 8)
            continue;
        if (color == WHITE) {
            for (int target_rank = rank + 1; target_rank < 8; ++target_rank)
                mask |= bb_of(target_rank * 8 + target_file);
        } else {
            for (int target_rank = rank - 1; target_rank >= 0; --target_rank)
                mask |= bb_of(target_rank * 8 + target_file);
        }
    }
    return mask;
}

[[nodiscard]] Bitboard pawn_attacks_from_geometry(
    Color color,
    Square square
) noexcept {
    if (rank_of(square) < 1 || rank_of(square) > 6)
        return 0ULL;
    const int target_rank = rank_of(square) + (color == WHITE ? 1 : -1);
    Bitboard attacks = 0ULL;
    for (const int df : {-1, 1}) {
        const int target_file = file_of(square) + df;
        if (target_file >= 0 && target_file < 8)
            attacks |= bb_of(target_rank * 8 + target_file);
    }
    return attacks;
}

[[nodiscard]] u8 pawn_flags(
    const Position& pos,
    Color color,
    Square square
) noexcept {
    const Bitboard own = pieces(pos, color, PAWN);
    const Bitboard enemy = pieces(pos, ~color, PAWN);
    const int file = file_of(square);
    Bitboard adjacent_files = 0ULL;
    if (file > 0)
        adjacent_files |= file_mask(file - 1);
    if (file < 7)
        adjacent_files |= file_mask(file + 1);

    const Bitboard same_file = file_mask(file);
    const bool isolated = (own & adjacent_files) == 0ULL;
    const bool doubled =
        std::popcount(static_cast<u64>(own & same_file)) > 1;
    const bool passed =
        (enemy & ahead_mask(color, square, true)) == 0ULL;
    const int enemy_ahead = std::popcount(
        static_cast<u64>(enemy & ahead_mask(color, square, true))
    );
    const int own_support = std::popcount(
        static_cast<u64>(
            own & ahead_mask(color, square, true) & adjacent_files
        )
    );
    const bool candidate = !passed && own_support >= enemy_ahead;
    const bool connected =
        (own & pawn_attacks_from_geometry(~color, square)) != 0ULL;
    const bool chain =
        (own & pawn_attacks_from_geometry(color, square)) != 0ULL;

    const Square next = color == WHITE ? square + 8 : square - 8;
    const bool has_next = on_board(next);
    const bool blocked = has_next && occupied_on(pos, next);
    const bool backward = !isolated && !connected && !chain && !passed
        && has_next
        && (pawn_attacks_from_geometry(~color, next) & enemy) != 0ULL;

    return static_cast<u8>(
        static_cast<u8>(isolated)
        | (static_cast<u8>(doubled) << 1)
        | (static_cast<u8>(passed) << 2)
        | (static_cast<u8>(candidate) << 3)
        | (static_cast<u8>(connected) << 4)
        | (static_cast<u8>(chain) << 5)
        | (static_cast<u8>(backward) << 6)
        | (static_cast<u8>(blocked) << 7)
    );
}

[[nodiscard]] u32 structure_pawn_index(
    int color,
    int square,
    u8 flags
) noexcept {
    return static_cast<u32>(
        ((color * kSquares + square) * kStructurePawnStates)
        + static_cast<int>(flags)
    );
}

[[nodiscard]] u32 structure_file_index(int file, int state) noexcept {
    return static_cast<u32>(kStructurePawnCount + file * 4 + state);
}

[[nodiscard]] u32 structure_island_index(int color, int count) noexcept {
    return static_cast<u32>(
        kStructurePawnCount + kStructureFileCount
        + color * 9 + std::min(count, 8)
    );
}

[[nodiscard]] u32 structure_center_index(int count) noexcept {
    return static_cast<u32>(
        kStructurePawnCount + kStructureFileCount + kStructureIslandCount
        + std::min(count, 8)
    );
}

[[nodiscard]] u32 structure_shelter_index(
    int color,
    int zone,
    int count
) noexcept {
    return static_cast<u32>(
        kStructurePawnCount + kStructureFileCount + kStructureIslandCount
        + kStructureCenterCount
        + ((color * 16 + zone) * 9)
        + std::min(count, 8)
    );
}

[[nodiscard]] u32 structure_outpost_index(
    int color,
    int piece,
    int square,
    int flags
) noexcept {
    return static_cast<u32>(
        kStructurePawnCount + kStructureFileCount + kStructureIslandCount
        + kStructureCenterCount + kStructureShelterCount
        + (((color * 2 + piece) * kSquares + square) * 4)
        + flags
    );
}

[[nodiscard]] u32 structure_complex_index(
    int color,
    int light,
    int dark
) noexcept {
    return static_cast<u32>(
        kStructurePawnCount + kStructureFileCount + kStructureIslandCount
        + kStructureCenterCount + kStructureShelterCount
        + kStructureOutpostCount
        + ((color * 9 + std::min(light, 8)) * 9)
        + std::min(dark, 8)
    );
}

[[nodiscard]] u32 structure_blocker_index(
    int color,
    int square,
    int state
) noexcept {
    return static_cast<u32>(
        kStructurePawnCount + kStructureFileCount + kStructureIslandCount
        + kStructureCenterCount + kStructureShelterCount
        + kStructureOutpostCount + kStructureComplexCount
        + ((color * kSquares + square) * 3)
        + state
    );
}

[[nodiscard]] Bitboard flip_vertical(Bitboard board) noexcept {
#if defined(_MSC_VER)
    return _byteswap_uint64(static_cast<u64>(board));
#else
    return __builtin_bswap64(static_cast<u64>(board));
#endif
}

template<std::size_t Capacity>
void dump_list(
    const char* name,
    Color perspective,
    const FeatureList<Capacity>& list,
    std::string (*decoder)(u32),
    std::ostream& output
) {
    output << name << " perspective " << static_cast<int>(perspective)
        << " count " << list.count << '\n';
    for (std::size_t i = 0; i < list.count; ++i) {
        output << "  " << list.indices[i]
            << " " << decoder(list.indices[i]) << '\n';
    }
}

} // namespace

u32 compute_position_slot(
    const Position& pos,
    Color perspective,
    Square square
) noexcept {
    const Piece piece = piece_on(pos, square);
    if (piece == PIECE_NONE)
        return Layout::NoFeature;
    const Square own_king = king_square(pos, perspective);
    if (own_king == NO_SQ)
        return Layout::NoFeature;
    return position_index(
        king_zone16(relative_square(perspective, own_king)),
        relative_color(perspective, color_of(piece)),
        static_cast<int>(type_of(piece)),
        relative_square(perspective, square)
    );
}

Bitboard compute_piece_attacks(
    const Position& pos,
    const memory::Memory& mem,
    Square source
) noexcept {
    const Piece piece = piece_on(pos, source);
    if (piece == PIECE_NONE)
        return 0ULL;
    return attacks_from(pos, mem, piece, source);
}

bool build_attack_graph(
    const Position& pos,
    const memory::Memory& mem,
    AttackGraph& graph
) noexcept {
    graph = {};
    Bitboard occupied = pos.occupied;
    while (occupied != 0ULL) {
        const Square source = static_cast<Square>(
            std::countr_zero(static_cast<u64>(occupied))
        );
        occupied &= occupied - 1;
        const Bitboard attacks =
            compute_piece_attacks(pos, mem, source);
        graph.attacks_from[source] = attacks;
        Bitboard targets = attacks;
        while (targets != 0ULL) {
            const Square target = static_cast<Square>(
                std::countr_zero(static_cast<u64>(targets))
            );
            targets &= targets - 1;
            graph.attackers_to[target] |= bb_of(source);
        }
    }
    return true;
}

u32 compute_attack_slot(
    const Position& pos,
    const memory::Memory& mem,
    const AttackGraph& graph,
    Color perspective,
    std::size_t slot
) noexcept {
    constexpr std::array<int, kPieceTypes> PieceValue{
        1, 3, 3, 5, 9, 100
    };
    constexpr std::size_t StatusBegin = 0;
    constexpr std::size_t RelationBegin =
        StatusBegin + Layout::AttackStatusSlotCount;
    constexpr std::size_t EdgeBegin =
        RelationBegin + Layout::AttackRelationSlotCount;
    constexpr std::size_t PressureBegin =
        EdgeBegin + Layout::AttackEdgeSlotCount;

    if (slot < RelationBegin) {
        const Square square = static_cast<Square>(slot);
        const Piece piece = piece_on(pos, square);
        if (piece == PIECE_NONE)
            return Layout::NoFeature;
        const Color own = color_of(piece);
        Bitboard attackers = graph.attackers_to[square];
        u8 status = 0;
        while (attackers != 0ULL) {
            const Square source = static_cast<Square>(
                std::countr_zero(static_cast<u64>(attackers))
            );
            attackers &= attackers - 1;
            const Piece attacker = piece_on(pos, source);
            if (attacker == PIECE_NONE)
                continue;
            if (color_of(attacker) == own) {
                status = static_cast<u8>(status | Defended);
                continue;
            }
            switch (type_of(attacker)) {
                case PAWN:
                    status = static_cast<u8>(status | EnemyPawn);
                    break;
                case KNIGHT:
                    status = static_cast<u8>(status | EnemyKnight);
                    break;
                case BISHOP:
                    status = static_cast<u8>(status | EnemyDiagonal);
                    break;
                case ROOK:
                    status = static_cast<u8>(
                        status | EnemyOrthogonal
                    );
                    break;
                case QUEEN:
                    status = static_cast<u8>(
                        status | EnemyDiagonal | EnemyOrthogonal
                    );
                    break;
                case KING:
                    status = static_cast<u8>(status | EnemyKing);
                    break;
                default:
                    break;
            }
        }
        return attack_status_index(
            relative_color(perspective, own),
            static_cast<int>(type_of(piece)),
            relative_square(perspective, square),
            status
        );
    }

    if (slot < EdgeBegin) {
        const Square square = static_cast<Square>(
            slot - RelationBegin
        );
        const Piece piece = piece_on(pos, square);
        if (piece == PIECE_NONE)
            return Layout::NoFeature;
        const Color own = color_of(piece);
        Bitboard attackers = graph.attackers_to[square];
        int enemy_count = 0;
        int own_count = 0;
        bool lower_value = false;
        while (attackers != 0ULL) {
            const Square source = static_cast<Square>(
                std::countr_zero(static_cast<u64>(attackers))
            );
            attackers &= attackers - 1;
            const Piece attacker = piece_on(pos, source);
            if (attacker == PIECE_NONE)
                continue;
            if (color_of(attacker) == own) {
                ++own_count;
            } else {
                ++enemy_count;
                lower_value = lower_value
                    || PieceValue[type_of(attacker)]
                        < PieceValue[type_of(piece)];
            }
        }
        const bool hanging = enemy_count != 0 && own_count == 0;
        const bool overloaded = enemy_count >= 2 && own_count == 1;
        const bool pinned = pinned_to_king(
            pos,
            piece,
            square,
            king_square(pos, own)
        );
        const u8 flags = static_cast<u8>(
            static_cast<u8>(hanging)
            | (static_cast<u8>(overloaded) << 1)
            | (static_cast<u8>(pinned) << 2)
            | (static_cast<u8>(lower_value) << 3)
        );
        return attack_relation_index(
            relative_color(perspective, own),
            static_cast<int>(type_of(piece)),
            relative_square(perspective, square),
            flags
        );
    }

    if (slot < PressureBegin) {
        const std::size_t edge = slot - EdgeBegin;
        const Square source = static_cast<Square>(edge / 64);
        const Square target = static_cast<Square>(edge % 64);
        const Piece attacker = piece_on(pos, source);
        const Piece victim = piece_on(pos, target);
        if (attacker == PIECE_NONE
            || victim == PIECE_NONE
            || (graph.attacks_from[source] & bb_of(target)) == 0ULL) {
            return Layout::NoFeature;
        }
        return attack_edge_index(
            mem,
            relative_color(perspective, color_of(attacker)),
            static_cast<int>(type_of(attacker)),
            relative_square(perspective, source),
            relative_color(perspective, color_of(victim)),
            static_cast<int>(type_of(victim)),
            relative_square(perspective, target)
        );
    }

    const std::size_t pressure = slot - PressureBegin;
    if (pressure >= Layout::AttackPressureSlotCount)
        return Layout::NoFeature;
    const int relative = static_cast<int>(pressure / kPieceTypes);
    const int piece_type = static_cast<int>(pressure % kPieceTypes);
    const Color absolute = static_cast<Color>(
        relative ^ static_cast<int>(perspective)
    );
    const Square enemy_king = king_square(pos, ~perspective);
    if (enemy_king == NO_SQ)
        return Layout::NoFeature;
    const Bitboard ring =
        king_attacks(mem, enemy_king) | bb_of(enemy_king);
    int count = 0;
    Bitboard sources = pieces(
        pos,
        absolute,
        static_cast<PieceType>(piece_type)
    );
    while (sources != 0ULL) {
        const Square source = static_cast<Square>(
            std::countr_zero(static_cast<u64>(sources))
        );
        sources &= sources - 1;
        count += std::popcount(static_cast<u64>(
            graph.attacks_from[source] & ring
        ));
    }
    return attack_pressure_index(
        relative,
        piece_type,
        king_zone16(relative_square(perspective, enemy_king)),
        count
    );
}

void compute_attack_slot_pair(
    const Position& pos,
    const memory::Memory& mem,
    const AttackGraph& graph,
    std::size_t slot,
    std::array<u32, COLOR_NB>& output
) noexcept {
    constexpr std::array<int, kPieceTypes> PieceValue{
        1, 3, 3, 5, 9, 100
    };
    constexpr std::size_t RelationBegin =
        Layout::AttackStatusSlotCount;
    constexpr std::size_t EdgeBegin =
        RelationBegin + Layout::AttackRelationSlotCount;
    constexpr std::size_t PressureBegin =
        EdgeBegin + Layout::AttackEdgeSlotCount;

    if (slot < RelationBegin) {
        const Square square = static_cast<Square>(slot);
        const Piece piece = piece_on(pos, square);
        if (piece == PIECE_NONE) {
            output.fill(Layout::NoFeature);
            return;
        }
        const Color own = color_of(piece);
        u8 status = 0;
        Bitboard attackers = graph.attackers_to[square];
        while (attackers != 0ULL) {
            const Square source = static_cast<Square>(
                std::countr_zero(static_cast<u64>(attackers))
            );
            attackers &= attackers - 1;
            const Piece attacker = piece_on(pos, source);
            if (attacker == PIECE_NONE)
                continue;
            if (color_of(attacker) == own) {
                status = static_cast<u8>(status | Defended);
                continue;
            }
            switch (type_of(attacker)) {
                case PAWN:
                    status = static_cast<u8>(status | EnemyPawn);
                    break;
                case KNIGHT:
                    status = static_cast<u8>(status | EnemyKnight);
                    break;
                case BISHOP:
                    status = static_cast<u8>(
                        status | EnemyDiagonal
                    );
                    break;
                case ROOK:
                    status = static_cast<u8>(
                        status | EnemyOrthogonal
                    );
                    break;
                case QUEEN:
                    status = static_cast<u8>(
                        status | EnemyDiagonal | EnemyOrthogonal
                    );
                    break;
                case KING:
                    status = static_cast<u8>(status | EnemyKing);
                    break;
                default:
                    break;
            }
        }
        for (int perspective = WHITE;
             perspective <= BLACK;
             ++perspective) {
            const Color color = static_cast<Color>(perspective);
            output[perspective] = attack_status_index(
                relative_color(color, own),
                static_cast<int>(type_of(piece)),
                relative_square(color, square),
                status
            );
        }
        return;
    }

    if (slot < EdgeBegin) {
        const Square square = static_cast<Square>(
            slot - RelationBegin
        );
        const Piece piece = piece_on(pos, square);
        if (piece == PIECE_NONE) {
            output.fill(Layout::NoFeature);
            return;
        }
        const Color own = color_of(piece);
        int enemy_count = 0;
        int own_count = 0;
        bool lower_value = false;
        Bitboard attackers = graph.attackers_to[square];
        while (attackers != 0ULL) {
            const Square source = static_cast<Square>(
                std::countr_zero(static_cast<u64>(attackers))
            );
            attackers &= attackers - 1;
            const Piece attacker = piece_on(pos, source);
            if (attacker == PIECE_NONE)
                continue;
            if (color_of(attacker) == own) {
                ++own_count;
            } else {
                ++enemy_count;
                lower_value = lower_value
                    || PieceValue[type_of(attacker)]
                        < PieceValue[type_of(piece)];
            }
        }
        const u8 flags = static_cast<u8>(
            static_cast<u8>(enemy_count != 0 && own_count == 0)
            | (
                static_cast<u8>(
                    enemy_count >= 2 && own_count == 1
                ) << 1
            )
            | (
                static_cast<u8>(
                    pinned_to_king(
                        pos,
                        piece,
                        square,
                        king_square(pos, own)
                    )
                ) << 2
            )
            | (static_cast<u8>(lower_value) << 3)
        );
        for (int perspective = WHITE;
             perspective <= BLACK;
             ++perspective) {
            const Color color = static_cast<Color>(perspective);
            output[perspective] = attack_relation_index(
                relative_color(color, own),
                static_cast<int>(type_of(piece)),
                relative_square(color, square),
                flags
            );
        }
        return;
    }

    if (slot < PressureBegin) {
        const std::size_t edge = slot - EdgeBegin;
        const Square source = static_cast<Square>(edge / 64);
        const Square target = static_cast<Square>(edge % 64);
        const Piece attacker = piece_on(pos, source);
        const Piece victim = piece_on(pos, target);
        if (attacker == PIECE_NONE
            || victim == PIECE_NONE
            || (graph.attacks_from[source] & bb_of(target)) == 0ULL) {
            output.fill(Layout::NoFeature);
            return;
        }
        for (int perspective = WHITE;
             perspective <= BLACK;
             ++perspective) {
            const Color color = static_cast<Color>(perspective);
            output[perspective] = attack_edge_index(
                mem,
                relative_color(color, color_of(attacker)),
                static_cast<int>(type_of(attacker)),
                relative_square(color, source),
                relative_color(color, color_of(victim)),
                static_cast<int>(type_of(victim)),
                relative_square(color, target)
            );
        }
        return;
    }

    output[WHITE] = compute_attack_slot(
        pos,
        mem,
        graph,
        WHITE,
        slot
    );
    output[BLACK] = compute_attack_slot(
        pos,
        mem,
        graph,
        BLACK,
        slot
    );
}

void compute_attack_summary_pairs(
    const Position& pos,
    const AttackGraph& graph,
    Square square,
    std::array<u32, COLOR_NB>& status_output,
    std::array<u32, COLOR_NB>& relation_output
) noexcept {
    constexpr std::array<int, kPieceTypes> PieceValue{
        1, 3, 3, 5, 9, 100
    };
    const Piece piece = piece_on(pos, square);
    if (piece == PIECE_NONE) {
        status_output.fill(Layout::NoFeature);
        relation_output.fill(Layout::NoFeature);
        return;
    }
    const Color own = color_of(piece);
    u8 status = 0;
    int enemy_count = 0;
    int own_count = 0;
    bool lower_value = false;
    Bitboard attackers = graph.attackers_to[square];
    while (attackers != 0ULL) {
        const Square source = static_cast<Square>(
            std::countr_zero(static_cast<u64>(attackers))
        );
        attackers &= attackers - 1;
        const Piece attacker = piece_on(pos, source);
        if (attacker == PIECE_NONE)
            continue;
        if (color_of(attacker) == own) {
            ++own_count;
            status = static_cast<u8>(status | Defended);
            continue;
        }
        ++enemy_count;
        lower_value = lower_value
            || PieceValue[type_of(attacker)]
                < PieceValue[type_of(piece)];
        switch (type_of(attacker)) {
            case PAWN:
                status = static_cast<u8>(status | EnemyPawn);
                break;
            case KNIGHT:
                status = static_cast<u8>(status | EnemyKnight);
                break;
            case BISHOP:
                status = static_cast<u8>(status | EnemyDiagonal);
                break;
            case ROOK:
                status = static_cast<u8>(
                    status | EnemyOrthogonal
                );
                break;
            case QUEEN:
                status = static_cast<u8>(
                    status | EnemyDiagonal | EnemyOrthogonal
                );
                break;
            case KING:
                status = static_cast<u8>(status | EnemyKing);
                break;
            default:
                break;
        }
    }
    const u8 flags = static_cast<u8>(
        static_cast<u8>(enemy_count != 0 && own_count == 0)
        | (
            static_cast<u8>(
                enemy_count >= 2 && own_count == 1
            ) << 1
        )
        | (
            static_cast<u8>(
                pinned_to_king(
                    pos,
                    piece,
                    square,
                    king_square(pos, own)
                )
            ) << 2
        )
        | (static_cast<u8>(lower_value) << 3)
    );
    for (int perspective = WHITE;
         perspective <= BLACK;
         ++perspective) {
        const Color color = static_cast<Color>(perspective);
        const int relative_piece_color =
            relative_color(color, own);
        const int relative_sq =
            relative_square(color, square);
        const int piece_type =
            static_cast<int>(type_of(piece));
        status_output[perspective] = attack_status_index(
            relative_piece_color,
            piece_type,
            relative_sq,
            status
        );
        relation_output[perspective] = attack_relation_index(
            relative_piece_color,
            piece_type,
            relative_sq,
            flags
        );
    }
}

u32 compute_structure_slot(
    const Position& pos,
    Color perspective,
    std::size_t slot
) noexcept {
    constexpr std::size_t PawnBegin = 0;
    constexpr std::size_t FileBegin =
        PawnBegin + Layout::StructurePawnSlotCount;
    constexpr std::size_t IslandBegin =
        FileBegin + Layout::StructureFileSlotCount;
    constexpr std::size_t CenterBegin =
        IslandBegin + Layout::StructureIslandSlotCount;
    constexpr std::size_t ShelterBegin =
        CenterBegin + Layout::StructureCenterSlotCount;
    constexpr std::size_t OutpostBegin =
        ShelterBegin + Layout::StructureShelterSlotCount;
    constexpr std::size_t ComplexBegin =
        OutpostBegin + Layout::StructureOutpostSlotCount;
    constexpr std::size_t BlockerBegin =
        ComplexBegin + Layout::StructureComplexSlotCount;

    const std::array<Bitboard, COLOR_NB> pawns{{
        pieces(pos, WHITE, PAWN),
        pieces(pos, BLACK, PAWN)
    }};
    std::array<Bitboard, COLOR_NB> relative_pawns{};
    if (perspective == WHITE) {
        relative_pawns = pawns;
    } else {
        relative_pawns[WHITE] = flip_vertical(pawns[BLACK]);
        relative_pawns[BLACK] = flip_vertical(pawns[WHITE]);
    }

    if (slot < FileBegin) {
        const Color color = static_cast<Color>(slot / 64);
        const Square square = static_cast<Square>(slot % 64);
        if ((pawns[color] & bb_of(square)) == 0ULL)
            return Layout::NoFeature;
        return structure_pawn_index(
            relative_color(perspective, color),
            relative_square(perspective, square),
            pawn_flags(pos, color, square)
        );
    }
    if (slot < IslandBegin) {
        const int file = static_cast<int>(slot - FileBegin);
        const Bitboard mask = file_mask(file);
        const int state =
            static_cast<int>((relative_pawns[WHITE] & mask) != 0ULL)
            | (
                static_cast<int>(
                    (relative_pawns[BLACK] & mask) != 0ULL
                ) << 1
            );
        return structure_file_index(file, state);
    }
    if (slot < CenterBegin) {
        const int color = static_cast<int>(slot - IslandBegin);
        return structure_island_index(
            color,
            pawn_islands(file_bits(relative_pawns[color]))
        );
    }
    if (slot < ShelterBegin) {
        constexpr std::array<Square, 8> Center{
            27, 28, 35, 36, 19, 20, 43, 44
        };
        int count = 0;
        for (const Square square : Center) {
            count += static_cast<int>(
                ((relative_pawns[WHITE] | relative_pawns[BLACK])
                 & bb_of(square))
                != 0ULL
            );
        }
        return structure_center_index(count);
    }
    if (slot < OutpostBegin) {
        const int relative = static_cast<int>(slot - ShelterBegin);
        const Color absolute = static_cast<Color>(
            relative ^ static_cast<int>(perspective)
        );
        const Square absolute_king = king_square(pos, absolute);
        if (absolute_king == NO_SQ)
            return Layout::NoFeature;
        const Square king = relative_square(absolute, absolute_king);
        int shelter = 0;
        for (int df = -1; df <= 1; ++df) {
            for (int distance = 1; distance <= 3; ++distance) {
                const int file = file_of(king) + df;
                const int rank = rank_of(king)
                    + (relative == WHITE ? distance : -distance);
                if (file >= 0 && file < 8 && rank >= 0 && rank < 8
                    && (relative_pawns[relative]
                        & bb_of(rank * 8 + file)) != 0ULL) {
                    ++shelter;
                }
            }
        }
        return structure_shelter_index(
            relative,
            king_zone16(king),
            shelter
        );
    }
    if (slot < ComplexBegin) {
        const Square square = static_cast<Square>(slot - OutpostBegin);
        const Piece piece = piece_on(pos, square);
        if (piece == PIECE_NONE
            || (type_of(piece) != KNIGHT
                && type_of(piece) != BISHOP)) {
            return Layout::NoFeature;
        }
        const Color color = color_of(piece);
        const bool cannot_be_chased =
            (pieces(pos, ~color, PAWN)
             & ahead_mask(~color, square, true))
            == 0ULL;
        const bool pawn_supported =
            (pawn_attacks_from_geometry(~color, square)
             & pieces(pos, color, PAWN))
            != 0ULL;
        const int flags = static_cast<int>(cannot_be_chased)
            | (static_cast<int>(pawn_supported) << 1);
        return structure_outpost_index(
            relative_color(perspective, color),
            static_cast<int>(type_of(piece)) - 1,
            relative_square(perspective, square),
            flags
        );
    }
    if (slot < BlockerBegin) {
        const int color = static_cast<int>(slot - ComplexBegin);
        int light = 0;
        int dark = 0;
        Bitboard remaining = relative_pawns[color];
        while (remaining != 0ULL) {
            const Square square = static_cast<Square>(
                std::countr_zero(static_cast<u64>(remaining))
            );
            remaining &= remaining - 1;
            if (((file_of(square) + rank_of(square)) & 1) == 0)
                ++light;
            else
                ++dark;
        }
        return structure_complex_index(color, light, dark);
    }

    const std::size_t blocker = slot - BlockerBegin;
    if (blocker >= Layout::StructureBlockerSlotCount)
        return Layout::NoFeature;
    const Color color = static_cast<Color>(blocker / 64);
    const Square square = static_cast<Square>(blocker % 64);
    if ((pawns[color] & bb_of(square)) == 0ULL
        || (pawn_flags(pos, color, square) & (1u << 2)) == 0) {
        return Layout::NoFeature;
    }
    const Square next = color == WHITE ? square + 8 : square - 8;
    int state = 0;
    if (on_board(next) && occupied_on(pos, next)) {
        state = 1 + static_cast<int>(
            color_on(pos, next) != color
        );
    }
    return structure_blocker_index(
        relative_color(perspective, color),
        relative_square(perspective, square),
        state
    );
}

void compute_structure_slot_pair(
    const Position& pos,
    std::size_t slot,
    std::array<u32, COLOR_NB>& output
) noexcept {
    constexpr std::size_t FileBegin =
        Layout::StructurePawnSlotCount;
    constexpr std::size_t IslandBegin =
        FileBegin + Layout::StructureFileSlotCount;
    constexpr std::size_t ShelterBegin =
        IslandBegin
        + Layout::StructureIslandSlotCount
        + Layout::StructureCenterSlotCount;
    constexpr std::size_t OutpostBegin =
        ShelterBegin + Layout::StructureShelterSlotCount;
    constexpr std::size_t ComplexBegin =
        OutpostBegin + Layout::StructureOutpostSlotCount;
    constexpr std::size_t BlockerBegin =
        ComplexBegin + Layout::StructureComplexSlotCount;

    if (slot < FileBegin) {
        const Color color = static_cast<Color>(slot / 64);
        const Square square = static_cast<Square>(slot % 64);
        if ((pieces(pos, color, PAWN) & bb_of(square)) == 0ULL) {
            output.fill(Layout::NoFeature);
            return;
        }
        const u8 flags = pawn_flags(pos, color, square);
        for (int perspective = WHITE;
             perspective <= BLACK;
             ++perspective) {
            const Color view = static_cast<Color>(perspective);
            output[perspective] = structure_pawn_index(
                relative_color(view, color),
                relative_square(view, square),
                flags
            );
        }
        return;
    }

    if (slot >= OutpostBegin && slot < ComplexBegin) {
        const Square square =
            static_cast<Square>(slot - OutpostBegin);
        const Piece piece = piece_on(pos, square);
        if (piece == PIECE_NONE
            || (
                type_of(piece) != KNIGHT
                && type_of(piece) != BISHOP
            )) {
            output.fill(Layout::NoFeature);
            return;
        }
        const Color color = color_of(piece);
        const bool cannot_be_chased =
            (pieces(pos, ~color, PAWN)
             & ahead_mask(~color, square, true))
            == 0ULL;
        const bool pawn_supported =
            (pawn_attacks_from_geometry(~color, square)
             & pieces(pos, color, PAWN))
            != 0ULL;
        const int flags = static_cast<int>(cannot_be_chased)
            | (static_cast<int>(pawn_supported) << 1);
        for (int perspective = WHITE;
             perspective <= BLACK;
             ++perspective) {
            const Color view = static_cast<Color>(perspective);
            output[perspective] = structure_outpost_index(
                relative_color(view, color),
                static_cast<int>(type_of(piece)) - 1,
                relative_square(view, square),
                flags
            );
        }
        return;
    }

    if (slot >= BlockerBegin) {
        const std::size_t blocker = slot - BlockerBegin;
        if (blocker >= Layout::StructureBlockerSlotCount) {
            output.fill(Layout::NoFeature);
            return;
        }
        const Color color = static_cast<Color>(blocker / 64);
        const Square square = static_cast<Square>(blocker % 64);
        if ((pieces(pos, color, PAWN) & bb_of(square)) == 0ULL
            || (pawn_flags(pos, color, square) & (1u << 2)) == 0) {
            output.fill(Layout::NoFeature);
            return;
        }
        const Square next = color == WHITE
            ? square + 8
            : square - 8;
        int state = 0;
        if (on_board(next) && occupied_on(pos, next)) {
            state = 1 + static_cast<int>(
                color_on(pos, next) != color
            );
        }
        for (int perspective = WHITE;
             perspective <= BLACK;
             ++perspective) {
            const Color view = static_cast<Color>(perspective);
            output[perspective] = structure_blocker_index(
                relative_color(view, color),
                relative_square(view, square),
                state
            );
        }
        return;
    }

    output[WHITE] = compute_structure_slot(pos, WHITE, slot);
    output[BLACK] = compute_structure_slot(pos, BLACK, slot);
}

bool debug_attack_graph_invariant(
    const Position& pos,
    const memory::Memory& mem,
    const AttackGraph& graph,
    std::ostream& output
) noexcept {
    AttackGraph rebuilt{};
    (void)build_attack_graph(pos, mem, rebuilt);
    if (rebuilt.attacks_from != graph.attacks_from
        || rebuilt.attackers_to != graph.attackers_to) {
        output << "info string MNUEv2 AttackGraph rebuild mismatch\n";
        return false;
    }
    for (int source = 0; source < SQ_NB; ++source) {
        for (int target = 0; target < SQ_NB; ++target) {
            const bool forward =
                (graph.attacks_from[source] & bb_of(target)) != 0ULL;
            const bool transpose =
                (graph.attackers_to[target] & bb_of(source)) != 0ULL;
            if (forward != transpose) {
                output << "info string MNUEv2 AttackGraph transpose "
                    << source << ' ' << target << '\n';
                return false;
            }
        }
    }
    return true;
}

bool encode_position_features(
    const Position& pos,
    Color perspective,
    PositionFeatureList& output
) noexcept {
    CycleScope cycles(CycleKind::PositionGeneration);
    output.count = 0;
    const Square own_king = king_square(pos, perspective);
    if (own_king == NO_SQ)
        return false;
    const int bucket = king_zone16(relative_square(perspective, own_king));

    Bitboard occupied = pos.occupied;
    while (occupied != 0ULL) {
        const Square square = static_cast<Square>(
            std::countr_zero(static_cast<u64>(occupied))
        );
        occupied &= occupied - 1;
        const Piece piece = piece_on(pos, square);
        const u32 index = position_index(
            bucket,
            relative_color(perspective, color_of(piece)),
            static_cast<int>(type_of(piece)),
            relative_square(perspective, square)
        );
        if (!add_feature(
                output,
                index,
                Layout::PositionFeatureCount
            )) {
            return false;
        }
    }
    sort_and_deduplicate(output);
    return true;
}

bool encode_attack_features_pair(
    const Position& pos,
    const memory::Memory& mem,
    std::array<AttackFeatureList, COLOR_NB>& output
) noexcept {
    CycleScope cycles(CycleKind::AttackGeneration);
    constexpr std::array<int, kPieceTypes> PieceValue{
        1, 3, 3, 5, 9, 100
    };
    std::array<Square, 32> squares{};
    std::array<Piece, 32> board_pieces{};
    std::array<Bitboard, 32> attacks{};
    std::size_t piece_count = 0;
    Bitboard remaining = pos.occupied;
    while (remaining != 0ULL) {
        const Square square = static_cast<Square>(
            std::countr_zero(static_cast<u64>(remaining))
        );
        remaining &= remaining - 1;
        assert(piece_count < squares.size());
        if (piece_count >= squares.size())
            return false;
        const Piece piece = piece_on(pos, square);
        squares[piece_count] = square;
        board_pieces[piece_count] = piece;
        attacks[piece_count] =
            attacks_from(pos, mem, piece, square);
        ++piece_count;
    }

    const auto encode = [&](Color perspective) {
        AttackFeatureList& perspective_output = output[perspective];
        perspective_output.count = 0;
        for (std::size_t piece_index = 0;
             piece_index < piece_count;
             ++piece_index) {
            const Square square = squares[piece_index];
            const Piece piece = board_pieces[piece_index];
            const Color own = color_of(piece);
            const int color =
                relative_color(perspective, color_of(piece));
            const int piece_type = static_cast<int>(type_of(piece));
            const int relative_sq =
                relative_square(perspective, square);
            const Bitboard square_bit = bb_of(square);

            u8 status = 0;
            int attacker_count = 0;
            int defender_count = 0;
            bool lower_value_attacker = false;
            for (std::size_t source_index = 0;
                 source_index < piece_count;
                 ++source_index) {
                if ((attacks[source_index] & square_bit) == 0ULL)
                    continue;
                const Piece attacker = board_pieces[source_index];
                if (color_of(attacker) == own) {
                    ++defender_count;
                    status = static_cast<u8>(status | Defended);
                    continue;
                }
                ++attacker_count;
                const PieceType attacker_type = type_of(attacker);
                lower_value_attacker =
                    lower_value_attacker
                    || PieceValue[attacker_type]
                        < PieceValue[type_of(piece)];
                switch (attacker_type) {
                    case PAWN:
                        status = static_cast<u8>(status | EnemyPawn);
                        break;
                    case KNIGHT:
                        status = static_cast<u8>(status | EnemyKnight);
                        break;
                    case BISHOP:
                        status = static_cast<u8>(
                            status | EnemyDiagonal
                        );
                        break;
                    case ROOK:
                        status = static_cast<u8>(
                            status | EnemyOrthogonal
                        );
                        break;
                    case QUEEN:
                        status = static_cast<u8>(
                            status
                            | EnemyDiagonal
                            | EnemyOrthogonal
                        );
                        break;
                    case KING:
                        status = static_cast<u8>(status | EnemyKing);
                        break;
                    default:
                        break;
                }
            }
            const bool hanging =
                attacker_count != 0 && defender_count == 0;
            const bool overloaded =
                attacker_count >= 2 && defender_count == 1;
            const bool pinned = pinned_to_king(
                pos,
                piece,
                square,
                king_square(pos, own)
            );
            const u8 relation = static_cast<u8>(
                static_cast<u8>(hanging)
                | (static_cast<u8>(overloaded) << 1)
                | (static_cast<u8>(pinned) << 2)
                | (static_cast<u8>(lower_value_attacker) << 3)
            );

            if (!add_feature(
                    perspective_output,
                    attack_status_index(
                        color,
                        piece_type,
                        relative_sq,
                        status
                    ),
                    Layout::AttackFeatureCount
                )
                || !add_feature(
                    perspective_output,
                    attack_relation_index(
                        color,
                        piece_type,
                        relative_sq,
                        relation
                    ),
                    Layout::AttackFeatureCount
                )) {
                return false;
            }

            Bitboard targets = attacks[piece_index] & pos.occupied;
            while (targets != 0ULL) {
                const Square target = static_cast<Square>(
                    std::countr_zero(static_cast<u64>(targets))
                );
                targets &= targets - 1;
                const Piece victim = piece_on(pos, target);
                if (!add_feature(
                        perspective_output,
                        attack_edge_index(
                            mem,
                            color,
                            piece_type,
                            relative_sq,
                            relative_color(
                                perspective,
                                color_of(victim)
                            ),
                            static_cast<int>(type_of(victim)),
                            relative_square(perspective, target)
                        ),
                        Layout::AttackFeatureCount
                    )) {
                    return false;
                }
            }
        }

        const Color enemy = ~perspective;
        const Square enemy_king = king_square(pos, enemy);
        if (enemy_king == NO_SQ)
            return false;
        const Square relative_enemy_king =
            relative_square(perspective, enemy_king);
        const int zone = king_zone16(relative_enemy_king);
        const Bitboard ring =
            king_attacks(mem, enemy_king) | bb_of(enemy_king);
        std::array<std::array<int, kPieceTypes>, COLOR_NB> counts{};
        for (std::size_t piece_index = 0;
             piece_index < piece_count;
             ++piece_index) {
            const Piece piece = board_pieces[piece_index];
            counts[
                relative_color(perspective, color_of(piece))
            ][type_of(piece)] += std::popcount(static_cast<u64>(
                attacks[piece_index] & ring
            ));
        }

        for (int color = 0; color < COLOR_NB; ++color) {
            for (int piece_type = 0;
                 piece_type < kPieceTypes;
                 ++piece_type) {
                if (!add_feature(
                        perspective_output,
                        attack_pressure_index(
                            color,
                            piece_type,
                            zone,
                            counts[color][piece_type]
                        ),
                        Layout::AttackFeatureCount
                    )) {
                    return false;
                }
            }
        }
        sort_and_deduplicate(perspective_output);
        return true;
    };

    return encode(WHITE) && encode(BLACK);
}

bool encode_attack_features(
    const Position& pos,
    const memory::Memory& mem,
    Color perspective,
    AttackFeatureList& output
) noexcept {
    std::array<AttackFeatureList, COLOR_NB> pair{};
    if (!encode_attack_features_pair(pos, mem, pair))
        return false;
    output = pair[perspective];
    return true;
}

bool encode_structure_features(
    const Position& pos,
    Color perspective,
    StructureFeatureList& output
) noexcept {
    CycleScope cycles(CycleKind::StructureGeneration);
    output.count = 0;
    const std::array<Bitboard, COLOR_NB> pawns{{
        pieces(pos, WHITE, PAWN),
        pieces(pos, BLACK, PAWN)
    }};

    for (int color = WHITE; color <= BLACK; ++color) {
        Bitboard remaining = pawns[color];
        while (remaining != 0ULL) {
            const Square square = static_cast<Square>(
                std::countr_zero(static_cast<u64>(remaining))
            );
            remaining &= remaining - 1;
            if (!add_feature(
                    output,
                    structure_pawn_index(
                        relative_color(
                            perspective,
                            static_cast<Color>(color)
                        ),
                        relative_square(perspective, square),
                        pawn_flags(
                            pos,
                            static_cast<Color>(color),
                            square
                        )
                    ),
                    Layout::StructureFeatureCount
                )) {
                return false;
            }
        }
    }

    std::array<Bitboard, COLOR_NB> relative_pawns{};
    if (perspective == WHITE) {
        relative_pawns = pawns;
    } else {
        relative_pawns[WHITE] = flip_vertical(pawns[BLACK]);
        relative_pawns[BLACK] = flip_vertical(pawns[WHITE]);
    }

    for (int file = 0; file < 8; ++file) {
        const Bitboard mask = file_mask(file);
        const int state =
            static_cast<int>((relative_pawns[WHITE] & mask) != 0ULL)
            | (static_cast<int>(
                (relative_pawns[BLACK] & mask) != 0ULL
            ) << 1);
        if (!add_feature(
                output,
                structure_file_index(file, state),
                Layout::StructureFeatureCount
            )) {
            return false;
        }
    }

    if (!add_feature(
            output,
            structure_island_index(
                WHITE,
                pawn_islands(file_bits(relative_pawns[WHITE]))
            ),
            Layout::StructureFeatureCount
        )
        || !add_feature(
            output,
            structure_island_index(
                BLACK,
                pawn_islands(file_bits(relative_pawns[BLACK]))
            ),
            Layout::StructureFeatureCount
        )) {
        return false;
    }

    constexpr std::array<Square, 8> Center{
        27, 28, 35, 36, 19, 20, 43, 44
    };
    int center_count = 0;
    for (const Square square : Center) {
        center_count += static_cast<int>(
            ((relative_pawns[WHITE] | relative_pawns[BLACK])
             & bb_of(square))
            != 0ULL
        );
    }
    if (!add_feature(
            output,
            structure_center_index(center_count),
            Layout::StructureFeatureCount
        )) {
        return false;
    }

    for (int relative = WHITE; relative <= BLACK; ++relative) {
        const Color absolute = static_cast<Color>(
            relative ^ static_cast<int>(perspective)
        );
        const Square absolute_king = king_square(pos, absolute);
        if (absolute_king == NO_SQ)
            return false;
        // Match bulletformat::ChessBoard exactly: ksq and opp_ksq are each
        // normalised from that king's own colour, not from the currently
        // encoded perspective.
        const Square king = relative_square(absolute, absolute_king);
        const int zone = king_zone16(king);
        int shelter = 0;
        for (int df = -1; df <= 1; ++df) {
            for (int distance = 1; distance <= 3; ++distance) {
                const int file = file_of(king) + df;
                const int rank = rank_of(king)
                    + (relative == WHITE ? distance : -distance);
                if (file >= 0 && file < 8 && rank >= 0 && rank < 8
                    && (relative_pawns[relative]
                        & bb_of(rank * 8 + file)) != 0ULL) {
                    ++shelter;
                }
            }
        }
        if (!add_feature(
                output,
                structure_shelter_index(relative, zone, shelter),
                Layout::StructureFeatureCount
            )) {
            return false;
        }
    }

    Bitboard occupied = pos.occupied;
    while (occupied != 0ULL) {
        const Square square = static_cast<Square>(
            std::countr_zero(static_cast<u64>(occupied))
        );
        occupied &= occupied - 1;
        const Piece piece = piece_on(pos, square);
        const PieceType piece_type = type_of(piece);
        if (piece_type != KNIGHT && piece_type != BISHOP)
            continue;
        const Color color = color_of(piece);
        const Bitboard enemy_pawns = pieces(pos, ~color, PAWN);
        const Bitboard own_pawns = pieces(pos, color, PAWN);
        const bool cannot_be_chased =
            (enemy_pawns & ahead_mask(~color, square, true)) == 0ULL;
        const bool pawn_supported =
            (pawn_attacks_from_geometry(~color, square) & own_pawns) != 0ULL;
        const int flags = static_cast<int>(cannot_be_chased)
            | (static_cast<int>(pawn_supported) << 1);
        if (!add_feature(
                output,
                structure_outpost_index(
                    relative_color(perspective, color),
                    static_cast<int>(piece_type) - 1,
                    relative_square(perspective, square),
                    flags
                ),
                Layout::StructureFeatureCount
            )) {
            return false;
        }
    }

    for (int color = WHITE; color <= BLACK; ++color) {
        int light = 0;
        int dark = 0;
        Bitboard remaining = relative_pawns[color];
        while (remaining != 0ULL) {
            const Square square = static_cast<Square>(
                std::countr_zero(static_cast<u64>(remaining))
            );
            remaining &= remaining - 1;
            if (((file_of(square) + rank_of(square)) & 1) == 0)
                ++light;
            else
                ++dark;
        }
        if (!add_feature(
                output,
                structure_complex_index(color, light, dark),
                Layout::StructureFeatureCount
            )) {
            return false;
        }
    }

    for (int color = WHITE; color <= BLACK; ++color) {
        Bitboard remaining = pawns[color];
        while (remaining != 0ULL) {
            const Square square = static_cast<Square>(
                std::countr_zero(static_cast<u64>(remaining))
            );
            remaining &= remaining - 1;
            if ((pawn_flags(pos, static_cast<Color>(color), square)
                 & (1u << 2)) == 0) {
                continue;
            }
            const Square next =
                color == WHITE ? square + 8 : square - 8;
            int state = 0;
            if (on_board(next) && occupied_on(pos, next)) {
                state = 1 + static_cast<int>(
                    color_on(pos, next) != static_cast<Color>(color)
                );
            }
            if (!add_feature(
                    output,
                    structure_blocker_index(
                        relative_color(
                            perspective,
                            static_cast<Color>(color)
                        ),
                        relative_square(perspective, square),
                        state
                    ),
                    Layout::StructureFeatureCount
                )) {
                return false;
            }
        }
    }

    sort_and_deduplicate(output);
    return true;
}

bool encode_all_features(
    const Position& pos,
    const memory::Memory& mem,
    EncodedFeatures& output
) noexcept {
    CycleScope cycles(CycleKind::FeatureReconstruction);
    if (!encode_attack_features_pair(pos, mem, output.attack))
        return false;
    for (int perspective = WHITE; perspective <= BLACK; ++perspective) {
        const Color color = static_cast<Color>(perspective);
        if (!encode_position_features(
                pos,
                color,
                output.position[perspective]
            )
            || !encode_structure_features(
                pos,
                color,
                output.structure[perspective]
            )) {
            return false;
        }
    }
    return true;
}

int material_units(const Position& pos) noexcept {
    constexpr std::array<int, kPieceTypes> Value{1, 3, 3, 5, 9, 0};
    int total = 0;
    for (int color = WHITE; color <= BLACK; ++color) {
        for (int piece_type = PAWN; piece_type <= QUEEN; ++piece_type) {
            total += piece_count(
                pos,
                static_cast<Color>(color),
                static_cast<PieceType>(piece_type)
            ) * Value[piece_type];
        }
    }
    return total;
}

int material_bucket_from_units(int units) noexcept {
    constexpr std::array<int, Layout::OutputBuckets - 1> Upper{
        7, 11, 13, 17, 21, 27, 33, 41, 50, 59, 69
    };
    return static_cast<int>(
        std::lower_bound(Upper.begin(), Upper.end(), units) - Upper.begin()
    );
}

int material_bucket(const Position& pos) noexcept {
    return material_bucket_from_units(material_units(pos));
}

bool debug_bucket_selftest(std::ostream& output) {
    constexpr std::array<int, Layout::OutputBuckets - 1> Upper{
        7, 11, 13, 17, 21, 27, 33, 41, 50, 59, 69
    };
    bool ok = true;
    int previous = -1;
    std::array<bool, Layout::OutputBuckets> reached{};
    for (int units = 0; units <= 90; ++units) {
        const int bucket = material_bucket_from_units(units);
        ok = ok
            && bucket >= 0
            && bucket < Layout::OutputBuckets
            && bucket >= previous;
        if (bucket >= 0 && bucket < Layout::OutputBuckets)
            reached[static_cast<std::size_t>(bucket)] = true;
        previous = bucket;
    }
    for (std::size_t index = 0; index < Upper.size(); ++index) {
        const int boundary = Upper[index];
        ok = ok
            && material_bucket_from_units(boundary)
                == static_cast<int>(index)
            && material_bucket_from_units(boundary + 1)
                == static_cast<int>(index + 1);
        if (boundary > 0) {
            ok = ok
                && material_bucket_from_units(boundary - 1)
                    == static_cast<int>(index);
        }
    }
    ok = ok && std::all_of(
        reached.begin(),
        reached.end(),
        [](bool value) { return value; }
    );
    output << "mnue v2 bucket selftest"
        << " boundaries " << Upper.size()
        << " all_reachable "
        << (std::all_of(
            reached.begin(),
            reached.end(),
            [](bool value) { return value; }
        ) ? 1 : 0)
        << " monotonic " << (ok ? 1 : 0)
        << " ok " << (ok ? 1 : 0)
        << '\n';
    return ok;
}

std::string decode_position_feature(u32 index) {
    if (index >= Layout::PositionFeatureCount)
        return "invalid";
    const int square = static_cast<int>(index % kSquares);
    u32 value = index / kSquares;
    const int piece = static_cast<int>(value % kPieceTypes);
    value /= kPieceTypes;
    const int color = static_cast<int>(value % 2);
    const int bucket = static_cast<int>(value / 2);
    std::ostringstream out;
    out << "position king_bucket=" << bucket
        << " rel_colour=" << color
        << " piece=" << piece
        << " square=" << square;
    return out.str();
}

std::string decode_attack_feature(u32 index) {
    if (index >= Layout::AttackFeatureCount)
        return "invalid";
    std::ostringstream out;
    if (index < kAttackStatusCount)
        out << "attack tactical_status raw_index=" << index;
    else if (index < kAttackStatusCount + kAttackEdgeCount)
        out << "attack occupied_edge raw_index="
            << index - kAttackStatusCount;
    else if (index
             < kAttackStatusCount + kAttackEdgeCount
                 + kAttackRelationCount) {
        out << "attack relation_flags raw_index="
            << index - kAttackStatusCount - kAttackEdgeCount;
    } else {
        out << "attack king_pressure raw_index="
            << index - kAttackStatusCount - kAttackEdgeCount
                - kAttackRelationCount;
    }
    return out.str();
}

std::string decode_structure_feature(u32 index) {
    if (index >= Layout::StructureFeatureCount)
        return "invalid";
    const char* family = "passed_blocker";
    if (index < kStructurePawnCount)
        family = "pawn_state";
    else if (index < kStructurePawnCount + kStructureFileCount)
        family = "file_state";
    else if (index
             < kStructurePawnCount + kStructureFileCount
                 + kStructureIslandCount)
        family = "pawn_islands";
    else if (index
             < kStructurePawnCount + kStructureFileCount
                 + kStructureIslandCount + kStructureCenterCount)
        family = "centre_openness";
    else if (index
             < kStructurePawnCount + kStructureFileCount
                 + kStructureIslandCount + kStructureCenterCount
                 + kStructureShelterCount)
        family = "king_shelter";
    else if (index
             < kStructurePawnCount + kStructureFileCount
                 + kStructureIslandCount + kStructureCenterCount
                 + kStructureShelterCount + kStructureOutpostCount)
        family = "outpost";
    else if (index
             < Layout::StructureFeatureCount - kStructureBlockerCount)
        family = "colour_complex";
    std::ostringstream out;
    out << "structure " << family << " raw_index=" << index;
    return out.str();
}

void debug_dump_features(
    const Position& pos,
    const memory::Memory& mem,
    std::ostream& output
) {
    EncodedFeatures features{};
    if (!encode_all_features(pos, mem, features)) {
        output << "mnue v2 feature encoding failed\n";
        return;
    }
    output << "mnue v2 features\n";
    output << "material_units " << material_units(pos) << '\n';
    output << "material_bucket " << material_bucket(pos) << '\n';
    for (int perspective = WHITE; perspective <= BLACK; ++perspective) {
        const Color color = static_cast<Color>(perspective);
        dump_list(
            "Position",
            color,
            features.position[perspective],
            &decode_position_feature,
            output
        );
        dump_list(
            "Attack",
            color,
            features.attack[perspective],
            &decode_attack_feature,
            output
        );
        dump_list(
            "Structure",
            color,
            features.structure[perspective],
            &decode_structure_feature,
            output
        );
    }
}

} // namespace magnus::mnue::v2
