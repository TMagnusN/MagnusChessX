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

#include "mnue/MnueX2K16PawnNetwork.h"

#include "Memory.h"
#include "board/MoveGen.h"
#include "board/Position.h"
#include "mnue/Mnue.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace magnus::mnue::x2k16 {
namespace {

using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;

constexpr u32 kMagic = 0x45554E4Du;
constexpr u32 kVersion = 3;
constexpr u32 kHeaderBytes = 112;
constexpr u32 kScale = 400;
constexpr u32 kQa = 255;
constexpr u32 kPieceQuant = 64;
constexpr u32 kPieceRescale = 4;
constexpr u32 kAttackQuant = 64;
constexpr u32 kAttackRescale = 4;
constexpr u32 kPawnPairQuant = 128;
constexpr u32 kPawnPairRescale = 2;
constexpr u32 kL1Quant = 64;

struct FileHeader {
    u32 magic;
    u32 version;
    u32 arch;
    u32 header_bytes;
    u32 piece_input_size;
    u32 attack_input_size;
    u32 pawn_pair_input_size;
    u32 piece_hidden_size;
    u32 attack_hidden_size;
    u32 pawn_pair_hidden_size;
    u32 merged_hidden_size;
    u32 input_buckets;
    u32 output_buckets;
    u32 l1_size;
    u32 l2_size;
    u32 scale;
    u32 qa;
    u32 piece_quant;
    u32 piece_rescale;
    u32 attack_quant;
    u32 attack_rescale;
    u32 pawn_pair_quant;
    u32 pawn_pair_rescale;
    u32 l1_quant;
    u32 feature_version;
    u32 flags;
    u32 reserved0;
    u32 reserved1;
};

static_assert(sizeof(FileHeader) == kHeaderBytes);

struct Network {
    bool is_loaded = false;
    std::size_t file_bytes = 0;
    std::string file_path{};
    std::string error{};

    std::vector<std::int8_t> piece_l0w{};
    std::vector<std::int8_t> attack_l0w{};
    std::vector<std::int8_t> pawn_pair_l0w{};
    std::vector<i16> l0b{};
    std::vector<std::int8_t> l1w{};
    std::vector<float> l1b{};
    std::vector<float> l2w{};
    std::vector<float> l2b{};
    std::vector<float> l3w{};
    std::vector<float> l3b{};

    [[nodiscard]] bool valid() const noexcept {
        return piece_l0w.size()
                == static_cast<std::size_t>(Layout::PieceInputSize)
                    * Layout::PieceHiddenSize
            && attack_l0w.size()
                == static_cast<std::size_t>(Layout::AttackInputSize)
                    * Layout::AttackHiddenSize
            && pawn_pair_l0w.size()
                == static_cast<std::size_t>(Layout::PawnPairInputSize)
                    * Layout::PawnPairHiddenSize
            && l0b.size()
                == static_cast<std::size_t>(Layout::MergedHiddenSize)
            && l1w.size()
                == static_cast<std::size_t>(Layout::OutputBuckets)
                    * Layout::L1Size * Layout::HeadInputSize
            && l1b.size()
                == static_cast<std::size_t>(Layout::OutputBuckets)
                    * Layout::L1Size
            && l2w.size()
                == static_cast<std::size_t>(Layout::OutputBuckets)
                    * Layout::L2Size * Layout::L1Size
            && l2b.size()
                == static_cast<std::size_t>(Layout::OutputBuckets)
                    * Layout::L2Size
            && l3w.size()
                == static_cast<std::size_t>(Layout::OutputBuckets)
                    * Layout::L2Size
            && l3b.size()
                == static_cast<std::size_t>(Layout::OutputBuckets);
    }
};

Network g_network{};

template<typename T>
[[nodiscard]] bool read_exact(
    std::istream& input,
    T* destination,
    std::size_t count
) {
    input.read(
        reinterpret_cast<char*>(destination),
        static_cast<std::streamsize>(sizeof(T) * count)
    );
    return static_cast<bool>(input);
}

[[nodiscard]] constexpr std::uintmax_t payload_bytes() noexcept {
    return static_cast<std::uintmax_t>(Layout::PieceInputSize)
            * Layout::PieceHiddenSize * sizeof(std::int8_t)
        + static_cast<std::uintmax_t>(Layout::AttackInputSize)
            * Layout::AttackHiddenSize * sizeof(std::int8_t)
        + static_cast<std::uintmax_t>(Layout::PawnPairInputSize)
            * Layout::PawnPairHiddenSize * sizeof(std::int8_t)
        + static_cast<std::uintmax_t>(Layout::MergedHiddenSize) * sizeof(i16)
        + static_cast<std::uintmax_t>(Layout::OutputBuckets)
            * Layout::L1Size * Layout::HeadInputSize * sizeof(std::int8_t)
        + static_cast<std::uintmax_t>(Layout::OutputBuckets)
            * Layout::L1Size * sizeof(float)
        + static_cast<std::uintmax_t>(Layout::OutputBuckets)
            * Layout::L2Size * Layout::L1Size * sizeof(float)
        + static_cast<std::uintmax_t>(Layout::OutputBuckets)
            * Layout::L2Size * sizeof(float)
        + static_cast<std::uintmax_t>(Layout::OutputBuckets)
            * Layout::L2Size * sizeof(float)
        + static_cast<std::uintmax_t>(Layout::OutputBuckets) * sizeof(float);
}

[[nodiscard]] constexpr std::uintmax_t file_bytes() noexcept {
    return kHeaderBytes + payload_bytes();
}

static_assert(payload_bytes() == 47636512);
static_assert(file_bytes() == 47636624);

void clear(Network& network) noexcept {
    network = {};
}

[[nodiscard]] constexpr int file_of_sq(Square square) noexcept {
    return square & 7;
}

[[nodiscard]] constexpr int rank_of_sq(Square square) noexcept {
    return square >> 3;
}

[[nodiscard]] constexpr Bitboard bit(Square square) noexcept {
    return 1ULL << square;
}

[[nodiscard]] constexpr bool mirror_for_king(Square relative_king) noexcept {
    return file_of_sq(relative_king) >= 4;
}

[[nodiscard]] constexpr Square relative_square(
    Color perspective,
    Square square,
    bool mirror
) noexcept {
    const Square vertical = perspective == WHITE ? square : square ^ 56;
    return mirror ? vertical ^ 7 : vertical;
}

[[nodiscard]] constexpr Square relative_square_no_mirror(
    Color perspective,
    Square square
) noexcept {
    return perspective == WHITE ? square : square ^ 56;
}

[[nodiscard]] constexpr int king_bucket16(Square relative_king) noexcept {
    const int mirrored_file = std::min(
        file_of_sq(relative_king),
        7 - file_of_sq(relative_king)
    );
    const int rank_band = std::min(rank_of_sq(relative_king), 3);
    return rank_band * 4 + mirrored_file;
}

[[nodiscard]] Square safe_king_square(
    const Position& pos,
    Color color
) noexcept {
    const Square square = king_square(pos, color);
    return square == NO_SQ ? 0 : square;
}

[[nodiscard]] constexpr int piece_feature_index(
    int bucket,
    int relative_color,
    PieceType piece_type,
    Square square
) noexcept {
    return (((bucket * Layout::RelativeColors + relative_color)
             * Layout::PieceTypes + static_cast<int>(piece_type))
            * Layout::Squares)
        + square;
}

constexpr std::array<std::pair<int, int>, 8> kKnightSteps{{
    { 1,  2}, { 2,  1}, { 2, -1}, { 1, -2},
    {-1, -2}, {-2, -1}, {-2,  1}, {-1,  2}
}};

constexpr std::array<std::pair<int, int>, 8> kKingSteps{{
    { 1,  1}, { 1,  0}, { 1, -1}, { 0,  1},
    { 0, -1}, {-1,  1}, {-1,  0}, {-1, -1}
}};

constexpr std::array<std::pair<int, int>, 4> kDiagonals{{
    { 1,  1}, { 1, -1}, {-1,  1}, {-1, -1}
}};

constexpr std::array<std::pair<int, int>, 4> kOrthogonals{{
    { 1, 0}, {-1, 0}, {0, 1}, {0, -1}
}};

template<std::size_t Count>
[[nodiscard]] Bitboard leaper_attacks(
    Square square,
    const std::array<std::pair<int, int>, Count>& steps
) noexcept {
    const int file = file_of_sq(square);
    const int rank = rank_of_sq(square);
    Bitboard attacks = 0ULL;

    for (const auto& [df, dr] : steps) {
        const int target_file = file + df;
        const int target_rank = rank + dr;
        if (target_file >= 0 && target_file < 8
            && target_rank >= 0 && target_rank < 8) {
            attacks |= bit(target_rank * 8 + target_file);
        }
    }
    return attacks;
}

[[nodiscard]] Bitboard pawn_attacks_x2(
    int color,
    Square square
) noexcept {
    const int file = file_of_sq(square);
    const int rank = rank_of_sq(square);
    if (rank < 1 || rank > 6)
        return 0ULL;

    const int target_rank = rank + (color == WHITE ? 1 : -1);
    if (target_rank < 0 || target_rank >= 8)
        return 0ULL;

    Bitboard attacks = 0ULL;
    for (const int df : {-1, 1}) {
        const int target_file = file + df;
        if (target_file >= 0 && target_file < 8)
            attacks |= bit(target_rank * 8 + target_file);
    }
    return attacks;
}

template<std::size_t Count>
[[nodiscard]] Bitboard slider_attacks(
    Square square,
    Bitboard occupied,
    const std::array<std::pair<int, int>, Count>& directions
) noexcept {
    const int file = file_of_sq(square);
    const int rank = rank_of_sq(square);
    Bitboard attacks = 0ULL;

    for (const auto& [df, dr] : directions) {
        int target_file = file + df;
        int target_rank = rank + dr;
        while (target_file >= 0 && target_file < 8
               && target_rank >= 0 && target_rank < 8) {
            const Square target = target_rank * 8 + target_file;
            const Bitboard target_bit = bit(target);
            attacks |= target_bit;
            if ((occupied & target_bit) != 0ULL)
                break;
            target_file += df;
            target_rank += dr;
        }
    }
    return attacks;
}

[[nodiscard]] Bitboard empty_board_attacks(
    int relative_color,
    PieceType piece_type,
    Square square
) noexcept {
    switch (piece_type) {
    case PAWN: return pawn_attacks_x2(relative_color, square);
    case KNIGHT: return leaper_attacks(square, kKnightSteps);
    case BISHOP: return slider_attacks(square, 0ULL, kDiagonals);
    case ROOK: return slider_attacks(square, 0ULL, kOrthogonals);
    case QUEEN:
        return slider_attacks(square, 0ULL, kDiagonals)
            | slider_attacks(square, 0ULL, kOrthogonals);
    case KING: return leaper_attacks(square, kKingSteps);
    default: return 0ULL;
    }
}

[[nodiscard]] Bitboard occupied_attacks(
    Piece piece,
    Square square,
    Bitboard occupied
) noexcept {
    const int color = static_cast<int>(color_of(piece));
    switch (type_of(piece)) {
    case PAWN: return pawn_attacks_x2(color, square) & occupied;
    case KNIGHT: return leaper_attacks(square, kKnightSteps) & occupied;
    case BISHOP:
        return slider_attacks(square, occupied, kDiagonals) & occupied;
    case ROOK:
        return slider_attacks(square, occupied, kOrthogonals) & occupied;
    case QUEEN:
        return (
            slider_attacks(square, occupied, kDiagonals)
            | slider_attacks(square, occupied, kOrthogonals)
        ) & occupied;
    case KING: return leaper_attacks(square, kKingSteps) & occupied;
    default: return 0ULL;
    }
}

struct AttackIndex {
    std::array<
        std::array<
            std::array<std::size_t, Layout::Squares>,
            Layout::PieceTypes
        >,
        Layout::RelativeColors
    > base{};
    std::size_t size = 0;
};

[[nodiscard]] const AttackIndex& attack_index_table() noexcept {
    static const AttackIndex table = [] {
        AttackIndex result{};
        for (int relative_color = 0;
             relative_color < Layout::RelativeColors;
             ++relative_color) {
            for (int piece_type = 0;
                 piece_type < Layout::PieceTypes;
                 ++piece_type) {
                for (int square = 0; square < Layout::Squares; ++square) {
                    result.base[static_cast<std::size_t>(relative_color)]
                        [static_cast<std::size_t>(piece_type)]
                        [static_cast<std::size_t>(square)] = result.size;
                    const Bitboard attacks = empty_board_attacks(
                        relative_color,
                        static_cast<PieceType>(piece_type),
                        square
                    );
                    result.size += static_cast<std::size_t>(
                        std::popcount(static_cast<std::uint64_t>(attacks))
                    ) * Layout::VictimClasses;
                }
            }
        }
        return result;
    }();
    return table;
}

[[nodiscard]] int target_slot_base(
    int attacker_color,
    PieceType attacker_type,
    Square attacker_square,
    Square victim_square
) noexcept {
    const Bitboard empty_attacks = empty_board_attacks(
        attacker_color,
        attacker_type,
        attacker_square
    );
    if ((empty_attacks & bit(victim_square)) == 0ULL)
        return -1;
    const Bitboard before =
        victim_square == 0 ? 0ULL : ((1ULL << victim_square) - 1ULL);
    const std::size_t target_slot = static_cast<std::size_t>(
        std::popcount(static_cast<std::uint64_t>(empty_attacks & before))
    );
    const std::size_t base =
        attack_index_table()
            .base[static_cast<std::size_t>(attacker_color)]
                 [static_cast<std::size_t>(attacker_type)]
                 [static_cast<std::size_t>(attacker_square)]
        + target_slot * Layout::VictimClasses;
    return static_cast<int>(base);
}

[[nodiscard]] constexpr int orientation_index(
    Color perspective,
    bool mirror
) noexcept {
    return static_cast<int>(perspective) * 2 + (mirror ? 1 : 0);
}

struct LookupTables {
    std::array<
        std::array<
            std::array<
                std::array<
                    std::array<u16, Layout::Squares>,
                    Layout::PieceTypes
                >,
                Layout::RelativeColors
            >,
            Layout::Squares
        >,
        COLOR_NB
    > piece_row_by_king{};

    std::array<
        std::array<
            std::array<
                std::array<
                    std::array<i32, Layout::Squares>,
                    Layout::Squares
                >,
                Layout::PieceTypes
            >,
            Layout::RelativeColors
        >,
        4
    > edge_base{};

    std::array<
        std::array<
            std::array<i16, Layout::RelativeColors>,
            Layout::Squares
        >,
        COLOR_NB
    > pawn_token{};

    std::array<
        std::array<i16, Layout::PawnTokens>,
        Layout::PawnTokens
    > pawn_pair_row{};
};

[[nodiscard]] int pawn_square48(Square relative_square) noexcept {
    const int rank = rank_of_sq(relative_square);
    if (rank < 1 || rank > 6)
        return -1;
    return (rank - 1) * 8 + file_of_sq(relative_square);
}

[[nodiscard]] constexpr int pawn_pair_index(int a, int b) noexcept {
    const int hi = a >= b ? a : b;
    const int lo = a >= b ? b : a;
    return hi * (hi - 1) / 2 + lo;
}

[[nodiscard]] const LookupTables& lookup_tables() noexcept {
    static const LookupTables tables = [] {
        LookupTables result{};

        for (int perspective_index = WHITE;
             perspective_index <= BLACK;
             ++perspective_index) {
            const Color perspective =
                static_cast<Color>(perspective_index);
            for (int king = 0; king < Layout::Squares; ++king) {
                const Square relative_king =
                    relative_square(perspective, king, false);
                const bool mirror = mirror_for_king(relative_king);
                const int bucket = king_bucket16(relative_king);
                for (int relative_color = 0;
                     relative_color < Layout::RelativeColors;
                     ++relative_color) {
                    for (int piece_type = 0;
                         piece_type < Layout::PieceTypes;
                         ++piece_type) {
                        for (int square = 0;
                             square < Layout::Squares;
                             ++square) {
                            const Square relative_sq =
                                relative_square(perspective, square, mirror);
                            result.piece_row_by_king
                                [static_cast<std::size_t>(perspective_index)]
                                [static_cast<std::size_t>(king)]
                                [static_cast<std::size_t>(relative_color)]
                                [static_cast<std::size_t>(piece_type)]
                                [static_cast<std::size_t>(square)] =
                                    static_cast<u16>(piece_feature_index(
                                        bucket,
                                        relative_color,
                                        static_cast<PieceType>(piece_type),
                                        relative_sq
                                    ));
                        }
                    }
                }
            }
        }

        for (auto& by_color : result.edge_base)
            for (auto& by_piece : by_color)
                for (auto& by_from : by_piece)
                    for (auto& by_to : by_from)
                        by_to.fill(-1);

        for (int perspective_index = WHITE;
             perspective_index <= BLACK;
             ++perspective_index) {
            const Color perspective =
                static_cast<Color>(perspective_index);
            for (const bool mirror : {false, true}) {
                const int orientation =
                    orientation_index(perspective, mirror);
                for (int relative_color = 0;
                     relative_color < Layout::RelativeColors;
                     ++relative_color) {
                    for (int piece_type = 0;
                         piece_type < Layout::PieceTypes;
                         ++piece_type) {
                        for (int from = 0;
                             from < Layout::Squares;
                             ++from) {
                            const Square relative_from =
                                relative_square(perspective, from, mirror);
                            for (int to = 0;
                                 to < Layout::Squares;
                                 ++to) {
                                const Square relative_to =
                                    relative_square(perspective, to, mirror);
                                result.edge_base
                                    [static_cast<std::size_t>(orientation)]
                                    [static_cast<std::size_t>(relative_color)]
                                    [static_cast<std::size_t>(piece_type)]
                                    [static_cast<std::size_t>(from)]
                                    [static_cast<std::size_t>(to)] =
                                        target_slot_base(
                                            relative_color,
                                            static_cast<PieceType>(piece_type),
                                            relative_from,
                                            relative_to
                                        );
                            }
                        }
                    }
                }
            }
        }

        for (auto& by_square : result.pawn_token)
            for (auto& by_color : by_square)
                by_color.fill(-1);

        for (int perspective_index = WHITE;
             perspective_index <= BLACK;
             ++perspective_index) {
            const Color perspective =
                static_cast<Color>(perspective_index);
            for (int square = 0; square < Layout::Squares; ++square) {
                const Square relative_sq =
                    relative_square_no_mirror(perspective, square);
                const int pawn_sq = pawn_square48(relative_sq);
                for (int relative_color = 0;
                     relative_color < Layout::RelativeColors;
                     ++relative_color) {
                    if (pawn_sq >= 0) {
                        result.pawn_token
                            [static_cast<std::size_t>(perspective_index)]
                            [static_cast<std::size_t>(square)]
                            [static_cast<std::size_t>(relative_color)] =
                                static_cast<i16>(
                                    relative_color * Layout::PawnSquares
                                    + pawn_sq
                                );
                    }
                }
            }
        }

        for (auto& row : result.pawn_pair_row)
            row.fill(-1);
        for (int a = 0; a < Layout::PawnTokens; ++a) {
            for (int b = 0; b < Layout::PawnTokens; ++b) {
                if (a == b)
                    continue;
                const int file_a = (a % Layout::PawnSquares) % 8;
                const int file_b = (b % Layout::PawnSquares) % 8;
                if (std::abs(file_a - file_b) <= 1) {
                    result.pawn_pair_row
                        [static_cast<std::size_t>(a)]
                        [static_cast<std::size_t>(b)] =
                            static_cast<i16>(pawn_pair_index(a, b));
                }
            }
        }

        return result;
    }();
    return tables;
}

[[nodiscard]] std::string header_error(const FileHeader& header) {
    const auto mismatch = [](const char* name, u32 got, u32 expected) {
        return std::string("MNUE-X2-K16-pawn-Q8-A384 header mismatch: ")
            + name + " got " + std::to_string(got)
            + " expected " + std::to_string(expected);
    };
    if (header.magic != kMagic)
        return mismatch("magic", header.magic, kMagic);
    if (header.version != kVersion)
        return mismatch("version", header.version, kVersion);
    if (header.arch != Layout::ArchId)
        return mismatch("arch", header.arch, Layout::ArchId);
    if (header.header_bytes != kHeaderBytes)
        return mismatch("header_bytes", header.header_bytes, kHeaderBytes);
    if (header.piece_input_size != Layout::PieceInputSize) {
        return mismatch(
            "piece_input_size",
            header.piece_input_size,
            Layout::PieceInputSize
        );
    }
    if (header.attack_input_size != Layout::AttackInputSize) {
        return mismatch(
            "attack_input_size",
            header.attack_input_size,
            Layout::AttackInputSize
        );
    }
    if (header.pawn_pair_input_size != Layout::PawnPairInputSize) {
        return mismatch(
            "pawn_pair_input_size",
            header.pawn_pair_input_size,
            Layout::PawnPairInputSize
        );
    }
    if (header.piece_hidden_size != Layout::PieceHiddenSize) {
        return mismatch(
            "piece_hidden_size",
            header.piece_hidden_size,
            Layout::PieceHiddenSize
        );
    }
    if (header.attack_hidden_size != Layout::AttackHiddenSize) {
        return mismatch(
            "attack_hidden_size",
            header.attack_hidden_size,
            Layout::AttackHiddenSize
        );
    }
    if (header.pawn_pair_hidden_size != Layout::PawnPairHiddenSize) {
        return mismatch(
            "pawn_pair_hidden_size",
            header.pawn_pair_hidden_size,
            Layout::PawnPairHiddenSize
        );
    }
    if (header.merged_hidden_size != Layout::MergedHiddenSize) {
        return mismatch(
            "merged_hidden_size",
            header.merged_hidden_size,
            Layout::MergedHiddenSize
        );
    }
    if (header.input_buckets != Layout::InputBuckets)
        return mismatch("input_buckets", header.input_buckets, Layout::InputBuckets);
    if (header.output_buckets != Layout::OutputBuckets) {
        return mismatch(
            "output_buckets",
            header.output_buckets,
            Layout::OutputBuckets
        );
    }
    if (header.l1_size != Layout::L1Size)
        return mismatch("l1_size", header.l1_size, Layout::L1Size);
    if (header.l2_size != Layout::L2Size)
        return mismatch("l2_size", header.l2_size, Layout::L2Size);
    if (header.scale != kScale)
        return mismatch("scale", header.scale, kScale);
    if (header.qa != kQa)
        return mismatch("qa", header.qa, kQa);
    if (header.piece_quant != kPieceQuant)
        return mismatch("piece_quant", header.piece_quant, kPieceQuant);
    if (header.piece_rescale != kPieceRescale)
        return mismatch("piece_rescale", header.piece_rescale, kPieceRescale);
    if (header.attack_quant != kAttackQuant)
        return mismatch("attack_quant", header.attack_quant, kAttackQuant);
    if (header.attack_rescale != kAttackRescale)
        return mismatch("attack_rescale", header.attack_rescale, kAttackRescale);
    if (header.pawn_pair_quant != kPawnPairQuant) {
        return mismatch(
            "pawn_pair_quant",
            header.pawn_pair_quant,
            kPawnPairQuant
        );
    }
    if (header.pawn_pair_rescale != kPawnPairRescale) {
        return mismatch(
            "pawn_pair_rescale",
            header.pawn_pair_rescale,
            kPawnPairRescale
        );
    }
    if (header.l1_quant != kL1Quant)
        return mismatch("l1_quant", header.l1_quant, kL1Quant);
    if (header.feature_version != Layout::FeatureVersion) {
        return mismatch(
            "feature_version",
            header.feature_version,
            Layout::FeatureVersion
        );
    }
    if (header.flags != 0)
        return mismatch("flags", header.flags, 0);
    if (header.reserved0 != 0)
        return mismatch("reserved0", header.reserved0, 0);
    if (header.reserved1 != 0)
        return mismatch("reserved1", header.reserved1, 0);
    return {};
}

void resize_tensors(Network& network) {
    network.piece_l0w.resize(
        static_cast<std::size_t>(Layout::PieceInputSize)
        * Layout::PieceHiddenSize
    );
    network.attack_l0w.resize(
        static_cast<std::size_t>(Layout::AttackInputSize)
        * Layout::AttackHiddenSize
    );
    network.pawn_pair_l0w.resize(
        static_cast<std::size_t>(Layout::PawnPairInputSize)
        * Layout::PawnPairHiddenSize
    );
    network.l0b.resize(Layout::MergedHiddenSize);
    network.l1w.resize(
        static_cast<std::size_t>(Layout::OutputBuckets)
        * Layout::L1Size * Layout::HeadInputSize
    );
    network.l1b.resize(
        static_cast<std::size_t>(Layout::OutputBuckets) * Layout::L1Size
    );
    network.l2w.resize(
        static_cast<std::size_t>(Layout::OutputBuckets)
        * Layout::L2Size * Layout::L1Size
    );
    network.l2b.resize(
        static_cast<std::size_t>(Layout::OutputBuckets) * Layout::L2Size
    );
    network.l3w.resize(
        static_cast<std::size_t>(Layout::OutputBuckets) * Layout::L2Size
    );
    network.l3b.resize(Layout::OutputBuckets);
}

[[nodiscard]] bool read_payload(std::istream& input, Network& network) {
    return read_exact(input, network.piece_l0w.data(), network.piece_l0w.size())
        && read_exact(
            input,
            network.attack_l0w.data(),
            network.attack_l0w.size()
        )
        && read_exact(
            input,
            network.pawn_pair_l0w.data(),
            network.pawn_pair_l0w.size()
        )
        && read_exact(input, network.l0b.data(), network.l0b.size())
        && read_exact(input, network.l1w.data(), network.l1w.size())
        && read_exact(input, network.l1b.data(), network.l1b.size())
        && read_exact(input, network.l2w.data(), network.l2w.size())
        && read_exact(input, network.l2b.data(), network.l2b.size())
        && read_exact(input, network.l3w.data(), network.l3w.size())
        && read_exact(input, network.l3b.data(), network.l3b.size());
}

using PieceAccumulator =
    std::array<i32, static_cast<std::size_t>(Layout::PieceHiddenSize)>;
using AttackAccumulator =
    std::array<i32, static_cast<std::size_t>(Layout::AttackHiddenSize)>;
using PawnPairAccumulator =
    std::array<i32, static_cast<std::size_t>(Layout::PawnPairHiddenSize)>;
using MergedAccumulator =
    std::array<i32, static_cast<std::size_t>(Layout::MergedHiddenSize)>;
using Pairwise =
    std::array<u8, static_cast<std::size_t>(Layout::PairwiseSize)>;
using BackendInput =
    std::array<u8, static_cast<std::size_t>(Layout::HeadInputSize)>;
using Hidden1 = std::array<float, static_cast<std::size_t>(Layout::L1Size)>;
using Hidden2 = std::array<float, static_cast<std::size_t>(Layout::L2Size)>;

struct X2K16PawnAccumulator {
    alignas(64) std::array<PieceAccumulator, COLOR_NB> piece{};
    alignas(64) std::array<AttackAccumulator, COLOR_NB> attack{};
    alignas(64) std::array<PawnPairAccumulator, COLOR_NB> pawn{};
    bool piece_valid = false;
    bool pawn_valid = false;
    bool attack_valid = false;
    Key position_hash = 0ULL;
};

struct EvaluationProfile {
    i64 feature_gen_time_us = 0;
    i64 piece_l0_time_us = 0;
    i64 attack_l0_time_us = 0;
    i64 pawn_l0_time_us = 0;
    i64 merge_activation_time_us = 0;
    i64 head_time_us = 0;
    i64 total_eval_time_us = 0;
    int active_piece_rows = 0;
    int active_attack_rows = 0;
    int active_pawn_rows = 0;
    int raw = 0;
    int searchcp = 0;
};

struct FeatureLists {
    std::array<std::vector<int>, COLOR_NB> piece{};
    std::array<std::vector<int>, COLOR_NB> attack{};
    std::array<std::vector<int>, COLOR_NB> pawn_pair{};
};

using Clock = std::chrono::steady_clock;

[[nodiscard]] i64 elapsed_us(
    Clock::time_point begin,
    Clock::time_point end
) noexcept {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        end - begin
    ).count();
}

void sort_unique_feature_lists(FeatureLists& lists) {
    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        auto sort_unique = [](std::vector<int>& values) {
            std::sort(values.begin(), values.end());
            values.erase(
                std::unique(values.begin(), values.end()),
                values.end()
            );
        };
        sort_unique(lists.piece[static_cast<std::size_t>(perspective_index)]);
        sort_unique(lists.attack[static_cast<std::size_t>(perspective_index)]);
        sort_unique(
            lists.pawn_pair[static_cast<std::size_t>(perspective_index)]
        );
    }
}

[[nodiscard]] int active_rows(
    const std::array<std::vector<int>, COLOR_NB>& rows
) noexcept {
    return static_cast<int>(rows[WHITE].size() + rows[BLACK].size());
}

template<std::size_t N>
[[nodiscard]] i64 checksum_accumulator(
    const std::array<i32, N>& accumulator
) noexcept {
    i64 checksum = 0;
    for (std::size_t index = 0; index < N; ++index) {
        checksum += static_cast<i64>(accumulator[index])
            * static_cast<i64>(index + 1);
    }
    return checksum;
}

template<std::size_t N>
[[nodiscard]] i64 checksum_branch(
    const std::array<std::array<i32, N>, COLOR_NB>& branch
) noexcept {
    return checksum_accumulator(branch[WHITE])
        ^ (checksum_accumulator(branch[BLACK]) * 1000003LL);
}

enum class RowAddBackend {
    Scalar,
    Avx2
};

[[nodiscard]] constexpr bool avx2_rowadd_compiled() noexcept {
#if defined(__AVX2__)
    return true;
#else
    return false;
#endif
}

[[nodiscard]] constexpr RowAddBackend active_rowadd_backend() noexcept {
#if defined(__AVX2__)
    return RowAddBackend::Avx2;
#else
    return RowAddBackend::Scalar;
#endif
}

[[nodiscard]] const char* rowadd_backend_name(
    RowAddBackend backend
) noexcept {
    if (backend == RowAddBackend::Avx2 && avx2_rowadd_compiled())
        return "avx2-lut-full-rebuild";
    return "scalar-lut-full-rebuild";
}

void add_i8_row_to_i32_acc_scalar(
    i32* accumulator,
    const std::int8_t* row,
    int lanes,
    int scale
) noexcept {
    for (int column = 0; column < lanes; ++column) {
        accumulator[column] +=
            static_cast<i32>(row[column]) * static_cast<i32>(scale);
    }
}

void sub_i8_row_from_i32_acc_scalar(
    i32* accumulator,
    const std::int8_t* row,
    int lanes,
    int scale
) noexcept {
    for (int column = 0; column < lanes; ++column) {
        accumulator[column] -=
            static_cast<i32>(row[column]) * static_cast<i32>(scale);
    }
}

#if defined(__AVX2__)
[[nodiscard]] __m256i scale_i32_avx2(
    __m256i values,
    int scale,
    __m256i scale_vector
) noexcept {
    if (scale == 4)
        return _mm256_slli_epi32(values, 2);
    if (scale == 2)
        return _mm256_slli_epi32(values, 1);
    return _mm256_mullo_epi32(values, scale_vector);
}

void add_i8_row_to_i32_acc_avx2(
    i32* accumulator,
    const std::int8_t* row,
    int lanes,
    int scale
) noexcept {
    const __m256i scale_vector = _mm256_set1_epi32(scale);
    int column = 0;
    for (; column + 32 <= lanes; column += 32) {
        for (int offset = 0; offset < 32; offset += 8) {
            const int lane = column + offset;
            const __m128i bytes = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(row + lane)
            );
            __m256i values = _mm256_cvtepi8_epi32(bytes);
            values = scale_i32_avx2(values, scale, scale_vector);
            const __m256i old_acc = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(accumulator + lane)
            );
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(accumulator + lane),
                _mm256_add_epi32(old_acc, values)
            );
        }
    }

    for (; column < lanes; ++column) {
        accumulator[column] +=
            static_cast<i32>(row[column]) * static_cast<i32>(scale);
    }
}

void sub_i8_row_from_i32_acc_avx2(
    i32* accumulator,
    const std::int8_t* row,
    int lanes,
    int scale
) noexcept {
    const __m256i scale_vector = _mm256_set1_epi32(scale);
    int column = 0;
    for (; column + 32 <= lanes; column += 32) {
        for (int offset = 0; offset < 32; offset += 8) {
            const int lane = column + offset;
            const __m128i bytes = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(row + lane)
            );
            __m256i values = _mm256_cvtepi8_epi32(bytes);
            values = scale_i32_avx2(values, scale, scale_vector);
            const __m256i old_acc = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(accumulator + lane)
            );
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(accumulator + lane),
                _mm256_sub_epi32(old_acc, values)
            );
        }
    }

    for (; column < lanes; ++column) {
        accumulator[column] -=
            static_cast<i32>(row[column]) * static_cast<i32>(scale);
    }
}
#endif

void add_i8_row_to_i32_acc(
    RowAddBackend backend,
    i32* accumulator,
    const std::int8_t* row,
    int lanes,
    int scale
) noexcept {
#if defined(__AVX2__)
    if (backend == RowAddBackend::Avx2) {
        add_i8_row_to_i32_acc_avx2(accumulator, row, lanes, scale);
        return;
    }
#else
    (void)backend;
#endif
    add_i8_row_to_i32_acc_scalar(accumulator, row, lanes, scale);
}

void sub_i8_row_from_i32_acc(
    RowAddBackend backend,
    i32* accumulator,
    const std::int8_t* row,
    int lanes,
    int scale
) noexcept {
#if defined(__AVX2__)
    if (backend == RowAddBackend::Avx2) {
        sub_i8_row_from_i32_acc_avx2(accumulator, row, lanes, scale);
        return;
    }
#else
    (void)backend;
#endif
    sub_i8_row_from_i32_acc_scalar(accumulator, row, lanes, scale);
}

void add_piece_row(
    PieceAccumulator& accumulator,
    int feature,
    RowAddBackend backend = RowAddBackend::Scalar
) noexcept {
    const std::int8_t* row =
        g_network.piece_l0w.data()
        + static_cast<std::size_t>(feature) * Layout::PieceHiddenSize;
    add_i8_row_to_i32_acc(
        backend,
        accumulator.data(),
        row,
        Layout::PieceHiddenSize,
        static_cast<int>(kPieceRescale)
    );
}

void sub_piece_row(
    PieceAccumulator& accumulator,
    int feature,
    RowAddBackend backend = RowAddBackend::Scalar
) noexcept {
    const std::int8_t* row =
        g_network.piece_l0w.data()
        + static_cast<std::size_t>(feature) * Layout::PieceHiddenSize;
    sub_i8_row_from_i32_acc(
        backend,
        accumulator.data(),
        row,
        Layout::PieceHiddenSize,
        static_cast<int>(kPieceRescale)
    );
}

void add_attack_row(
    AttackAccumulator& accumulator,
    int feature,
    RowAddBackend backend = RowAddBackend::Scalar
) noexcept {
    const std::int8_t* row =
        g_network.attack_l0w.data()
        + static_cast<std::size_t>(feature) * Layout::AttackHiddenSize;
    add_i8_row_to_i32_acc(
        backend,
        accumulator.data(),
        row,
        Layout::AttackHiddenSize,
        static_cast<int>(kAttackRescale)
    );
}

void add_pawn_pair_row(
    PawnPairAccumulator& accumulator,
    int feature,
    RowAddBackend backend = RowAddBackend::Scalar
) noexcept {
    const std::int8_t* row =
        g_network.pawn_pair_l0w.data()
        + static_cast<std::size_t>(feature) * Layout::PawnPairHiddenSize;
    add_i8_row_to_i32_acc(
        backend,
        accumulator.data(),
        row,
        Layout::PawnPairHiddenSize,
        static_cast<int>(kPawnPairRescale)
    );
}

void sub_pawn_pair_row(
    PawnPairAccumulator& accumulator,
    int feature,
    RowAddBackend backend = RowAddBackend::Scalar
) noexcept {
    const std::int8_t* row =
        g_network.pawn_pair_l0w.data()
        + static_cast<std::size_t>(feature) * Layout::PawnPairHiddenSize;
    sub_i8_row_from_i32_acc(
        backend,
        accumulator.data(),
        row,
        Layout::PawnPairHiddenSize,
        static_cast<int>(kPawnPairRescale)
    );
}

void clear_accumulator_arrays(
    std::array<PieceAccumulator, COLOR_NB>& piece_accumulators,
    std::array<AttackAccumulator, COLOR_NB>& attack_accumulators,
    std::array<PawnPairAccumulator, COLOR_NB>& pawn_pair_accumulators
) noexcept {
    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const std::size_t p = static_cast<std::size_t>(perspective_index);
        piece_accumulators[p].fill(0);
        attack_accumulators[p].fill(0);
        pawn_pair_accumulators[p].fill(0);
    }
}

void clear_accumulator(X2K16PawnAccumulator& accumulator) noexcept {
    clear_accumulator_arrays(
        accumulator.piece,
        accumulator.attack,
        accumulator.pawn
    );
    accumulator.piece_valid = false;
    accumulator.pawn_valid = false;
    accumulator.attack_valid = false;
    accumulator.position_hash = 0ULL;
}

void add_feature_lists_to_accumulators(
    const FeatureLists& lists,
    std::array<PieceAccumulator, COLOR_NB>& piece_accumulators,
    std::array<AttackAccumulator, COLOR_NB>& attack_accumulators,
    std::array<PawnPairAccumulator, COLOR_NB>& pawn_pair_accumulators,
    EvaluationProfile* profile = nullptr,
    RowAddBackend backend = RowAddBackend::Scalar
) noexcept {
    const auto piece_begin = Clock::now();
    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const std::size_t p = static_cast<std::size_t>(perspective_index);
        for (const int row : lists.piece[p])
            add_piece_row(piece_accumulators[p], row, backend);
    }
    const auto piece_end = Clock::now();

    const auto attack_begin = Clock::now();
    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const std::size_t p = static_cast<std::size_t>(perspective_index);
        for (const int row : lists.attack[p])
            add_attack_row(attack_accumulators[p], row, backend);
    }
    const auto attack_end = Clock::now();

    const auto pawn_begin = Clock::now();
    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const std::size_t p = static_cast<std::size_t>(perspective_index);
        for (const int row : lists.pawn_pair[p])
            add_pawn_pair_row(pawn_pair_accumulators[p], row, backend);
    }
    const auto pawn_end = Clock::now();

    if (profile != nullptr) {
        profile->piece_l0_time_us += elapsed_us(piece_begin, piece_end);
        profile->attack_l0_time_us += elapsed_us(attack_begin, attack_end);
        profile->pawn_l0_time_us += elapsed_us(pawn_begin, pawn_end);
        profile->active_piece_rows += active_rows(lists.piece);
        profile->active_attack_rows += active_rows(lists.attack);
        profile->active_pawn_rows += active_rows(lists.pawn_pair);
    }
}

template<typename AddRow, typename SubRow>
void apply_sorted_row_delta(
    const std::vector<int>& before,
    const std::vector<int>& after,
    AddRow add_row,
    SubRow sub_row
) noexcept {
    std::size_t old_index = 0;
    std::size_t new_index = 0;
    while (old_index < before.size() || new_index < after.size()) {
        if (new_index >= after.size()
            || (old_index < before.size()
                && before[old_index] < after[new_index])) {
            sub_row(before[old_index]);
            ++old_index;
        } else if (old_index >= before.size()
                   || after[new_index] < before[old_index]) {
            add_row(after[new_index]);
            ++new_index;
        } else {
            ++old_index;
            ++new_index;
        }
    }
}

void reset_accumulator_lut(
    X2K16PawnAccumulator& accumulator,
    const Position& pos,
    RowAddBackend backend = RowAddBackend::Scalar
) noexcept;

[[nodiscard]] std::array<bool, COLOR_NB> perspective_mirrors(
    const Position& pos
) noexcept {
    return {{
        mirror_for_king(relative_square(
            WHITE,
            safe_king_square(pos, WHITE),
            false
        )),
        mirror_for_king(relative_square(
            BLACK,
            safe_king_square(pos, BLACK),
            false
        ))
    }};
}

[[maybe_unused]] void add_piece_feature_rows(
    const Position& pos,
    X2K16PawnAccumulator& accumulator,
    Piece piece,
    Square square,
    RowAddBackend backend
) noexcept {
    if (piece == PIECE_NONE || square == NO_SQ)
        return;
    const LookupTables& tables = lookup_tables();
    const PieceType piece_type = type_of(piece);
    const Color piece_color = color_of(piece);
    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const Color perspective =
            static_cast<Color>(perspective_index);
        const std::size_t p = static_cast<std::size_t>(perspective_index);
        const int relative_color = piece_color == perspective ? 0 : 1;
        const Square king = safe_king_square(pos, perspective);
        const int row = tables.piece_row_by_king[p]
            [static_cast<std::size_t>(king)]
            [static_cast<std::size_t>(relative_color)]
            [static_cast<std::size_t>(piece_type)]
            [static_cast<std::size_t>(square)];
        add_piece_row(accumulator.piece[p], row, backend);
    }
}

[[maybe_unused]] void remove_piece_feature_rows(
    const Position& pos,
    X2K16PawnAccumulator& accumulator,
    Piece piece,
    Square square,
    RowAddBackend backend
) noexcept {
    if (piece == PIECE_NONE || square == NO_SQ)
        return;
    const LookupTables& tables = lookup_tables();
    const PieceType piece_type = type_of(piece);
    const Color piece_color = color_of(piece);
    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const Color perspective =
            static_cast<Color>(perspective_index);
        const std::size_t p = static_cast<std::size_t>(perspective_index);
        const int relative_color = piece_color == perspective ? 0 : 1;
        const Square king = safe_king_square(pos, perspective);
        const int row = tables.piece_row_by_king[p]
            [static_cast<std::size_t>(king)]
            [static_cast<std::size_t>(relative_color)]
            [static_cast<std::size_t>(piece_type)]
            [static_cast<std::size_t>(square)];
        sub_piece_row(accumulator.piece[p], row, backend);
    }
}

[[nodiscard]] FeatureLists collect_features_formula(const Position& pos) {
    FeatureLists lists{};
    const std::array<bool, COLOR_NB> mirrors = perspective_mirrors(pos);
    const std::array<Square, COLOR_NB> kings{{
        safe_king_square(pos, WHITE),
        safe_king_square(pos, BLACK)
    }};
    std::array<std::vector<int>, COLOR_NB> pawn_tokens{};

    Bitboard occupied = pos.occupied;
    while (occupied != 0ULL) {
        const Square square = static_cast<Square>(
            std::countr_zero(static_cast<std::uint64_t>(occupied))
        );
        occupied &= occupied - 1;

        const Piece piece = piece_on(pos, square);
        if (piece == PIECE_NONE)
            continue;
        const PieceType piece_type = type_of(piece);
        const Color piece_color = color_of(piece);

        for (int perspective_index = WHITE;
             perspective_index <= BLACK;
             ++perspective_index) {
            const Color perspective =
                static_cast<Color>(perspective_index);
            const std::size_t p =
                static_cast<std::size_t>(perspective_index);
            const int relative_color =
                piece_color == perspective ? 0 : 1;
            const Square relative_king =
                relative_square(perspective, kings[p], false);
            const int bucket = king_bucket16(relative_king);
            const Square relative_sq =
                relative_square(perspective, square, mirrors[p]);
            lists.piece[p].push_back(piece_feature_index(
                bucket,
                relative_color,
                piece_type,
                relative_sq
            ));

            if (piece_type == PAWN) {
                const Square relative_pawn =
                    relative_square_no_mirror(perspective, square);
                const int pawn_sq = pawn_square48(relative_pawn);
                if (pawn_sq >= 0) {
                    pawn_tokens[p].push_back(
                        relative_color * Layout::PawnSquares + pawn_sq
                    );
                }
            }
        }

        Bitboard targets = occupied_attacks(piece, square, pos.occupied);
        while (targets != 0ULL) {
            const Square target = static_cast<Square>(
                std::countr_zero(static_cast<std::uint64_t>(targets))
            );
            targets &= targets - 1;

            const Piece victim = piece_on(pos, target);
            if (victim == PIECE_NONE)
                continue;
            const PieceType victim_type = type_of(victim);
            const Color victim_color = color_of(victim);

            for (int perspective_index = WHITE;
                 perspective_index <= BLACK;
                 ++perspective_index) {
                const Color perspective =
                    static_cast<Color>(perspective_index);
                const std::size_t p =
                    static_cast<std::size_t>(perspective_index);
                const int attacker_relative =
                    piece_color == perspective ? 0 : 1;
                const int victim_relative =
                    victim_color == perspective ? 0 : 1;
                const Square relative_from =
                    relative_square(perspective, square, mirrors[p]);
                const Square relative_to =
                    relative_square(perspective, target, mirrors[p]);
                const int base = target_slot_base(
                    attacker_relative,
                    piece_type,
                    relative_from,
                    relative_to
                );
                if (base >= 0) {
                    lists.attack[p].push_back(
                        base
                        + victim_relative * Layout::PieceTypes
                        + static_cast<int>(victim_type)
                    );
                }
            }
        }
    }

    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        std::vector<int>& tokens =
            pawn_tokens[static_cast<std::size_t>(perspective_index)];
        for (std::size_t first = 0; first < tokens.size(); ++first) {
            for (std::size_t second = first + 1;
                 second < tokens.size();
                 ++second) {
                const int a = tokens[first];
                const int b = tokens[second];
                const int file_a = (a % Layout::PawnSquares) % 8;
                const int file_b = (b % Layout::PawnSquares) % 8;
                if (std::abs(file_a - file_b) <= 1) {
                    lists.pawn_pair[static_cast<std::size_t>(
                        perspective_index
                    )].push_back(pawn_pair_index(a, b));
                }
            }
        }
    }

    sort_unique_feature_lists(lists);
    return lists;
}

[[nodiscard]] FeatureLists collect_features(const Position& pos) {
    FeatureLists lists{};
    const LookupTables& tables = lookup_tables();
    const std::array<bool, COLOR_NB> mirrors = perspective_mirrors(pos);
    const std::array<Square, COLOR_NB> kings{{
        safe_king_square(pos, WHITE),
        safe_king_square(pos, BLACK)
    }};
    std::array<std::vector<int>, COLOR_NB> pawn_tokens{};

    Bitboard occupied = pos.occupied;
    while (occupied != 0ULL) {
        const Square square = static_cast<Square>(
            std::countr_zero(static_cast<std::uint64_t>(occupied))
        );
        occupied &= occupied - 1;

        const Piece piece = piece_on(pos, square);
        if (piece == PIECE_NONE)
            continue;
        const PieceType piece_type = type_of(piece);
        const Color piece_color = color_of(piece);

        for (int perspective_index = WHITE;
             perspective_index <= BLACK;
             ++perspective_index) {
            const Color perspective =
                static_cast<Color>(perspective_index);
            const int relative_color =
                piece_color == perspective ? 0 : 1;
            const int row = tables.piece_row_by_king
                [static_cast<std::size_t>(perspective_index)]
                [static_cast<std::size_t>(kings[static_cast<std::size_t>(
                    perspective_index
                )])]
                [static_cast<std::size_t>(relative_color)]
                [static_cast<std::size_t>(piece_type)]
                [static_cast<std::size_t>(square)];
            lists.piece[static_cast<std::size_t>(perspective_index)]
                .push_back(row);

            if (piece_type == PAWN) {
                const i16 token = tables.pawn_token
                    [static_cast<std::size_t>(perspective_index)]
                    [static_cast<std::size_t>(square)]
                    [static_cast<std::size_t>(relative_color)];
                if (token >= 0) {
                    pawn_tokens[static_cast<std::size_t>(perspective_index)]
                        .push_back(token);
                }
            }
        }

        Bitboard targets = occupied_attacks(piece, square, pos.occupied);
        while (targets != 0ULL) {
            const Square target = static_cast<Square>(
                std::countr_zero(static_cast<std::uint64_t>(targets))
            );
            targets &= targets - 1;

            const Piece victim = piece_on(pos, target);
            if (victim == PIECE_NONE)
                continue;
            const PieceType victim_type = type_of(victim);
            const Color victim_color = color_of(victim);

            for (int perspective_index = WHITE;
                 perspective_index <= BLACK;
                 ++perspective_index) {
                const Color perspective =
                    static_cast<Color>(perspective_index);
                const int attacker_relative =
                    piece_color == perspective ? 0 : 1;
                const int victim_relative =
                    victim_color == perspective ? 0 : 1;
                const int orientation = orientation_index(
                    perspective,
                    mirrors[static_cast<std::size_t>(perspective_index)]
                );
                const i32 base = tables.edge_base
                    [static_cast<std::size_t>(orientation)]
                    [static_cast<std::size_t>(attacker_relative)]
                    [static_cast<std::size_t>(piece_type)]
                    [static_cast<std::size_t>(square)]
                    [static_cast<std::size_t>(target)];
                if (base >= 0) {
                    const int row = static_cast<int>(base)
                        + victim_relative * Layout::PieceTypes
                        + static_cast<int>(victim_type);
                    lists.attack[static_cast<std::size_t>(
                        perspective_index
                    )].push_back(row);
                }
            }
        }
    }

    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        std::vector<int>& tokens =
            pawn_tokens[static_cast<std::size_t>(perspective_index)];
        for (std::size_t first = 0; first < tokens.size(); ++first) {
            for (std::size_t second = first + 1;
                 second < tokens.size();
                 ++second) {
                const i16 row = tables.pawn_pair_row
                    [static_cast<std::size_t>(tokens[first])]
                    [static_cast<std::size_t>(tokens[second])];
                if (row >= 0) {
                    lists.pawn_pair[static_cast<std::size_t>(
                        perspective_index
                    )].push_back(row);
                }
            }
        }
    }

    sort_unique_feature_lists(lists);
    return lists;
}

void clear_piece_accumulators(
    std::array<PieceAccumulator, COLOR_NB>& piece_accumulators
) noexcept {
    for (auto& accumulator : piece_accumulators)
        accumulator.fill(0);
}

void clear_attack_accumulators(
    std::array<AttackAccumulator, COLOR_NB>& attack_accumulators
) noexcept {
    for (auto& accumulator : attack_accumulators)
        accumulator.fill(0);
}

void clear_pawn_accumulators(
    std::array<PawnPairAccumulator, COLOR_NB>& pawn_accumulators
) noexcept {
    for (auto& accumulator : pawn_accumulators)
        accumulator.fill(0);
}

void rebuild_piece_accumulators(
    const Position& pos,
    std::array<PieceAccumulator, COLOR_NB>& piece_accumulators,
    RowAddBackend backend
) noexcept {
    clear_piece_accumulators(piece_accumulators);
    const FeatureLists lists = collect_features(pos);
    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const std::size_t p = static_cast<std::size_t>(perspective_index);
        for (const int row : lists.piece[p])
            add_piece_row(piece_accumulators[p], row, backend);
    }
}

void rebuild_attack_accumulators(
    const Position& pos,
    std::array<AttackAccumulator, COLOR_NB>& attack_accumulators,
    RowAddBackend backend
) noexcept {
    clear_attack_accumulators(attack_accumulators);

    const LookupTables& tables = lookup_tables();
    const std::array<bool, COLOR_NB> mirrors = perspective_mirrors(pos);

    Bitboard occupied = pos.occupied;
    while (occupied != 0ULL) {
        const Square square = static_cast<Square>(
            std::countr_zero(static_cast<std::uint64_t>(occupied))
        );
        occupied &= occupied - 1;

        const Piece piece = piece_on(pos, square);
        if (piece == PIECE_NONE)
            continue;
        const PieceType piece_type = type_of(piece);
        const Color piece_color = color_of(piece);

        Bitboard targets = occupied_attacks(piece, square, pos.occupied);
        while (targets != 0ULL) {
            const Square target = static_cast<Square>(
                std::countr_zero(static_cast<std::uint64_t>(targets))
            );
            targets &= targets - 1;

            const Piece victim = piece_on(pos, target);
            if (victim == PIECE_NONE)
                continue;
            const PieceType victim_type = type_of(victim);
            const Color victim_color = color_of(victim);

            for (int perspective_index = WHITE;
                 perspective_index <= BLACK;
                 ++perspective_index) {
                const Color perspective =
                    static_cast<Color>(perspective_index);
                const std::size_t p =
                    static_cast<std::size_t>(perspective_index);
                const int attacker_relative =
                    piece_color == perspective ? 0 : 1;
                const int victim_relative =
                    victim_color == perspective ? 0 : 1;
                const int orientation =
                    orientation_index(perspective, mirrors[p]);
                const i32 base = tables.edge_base
                    [static_cast<std::size_t>(orientation)]
                    [static_cast<std::size_t>(attacker_relative)]
                    [static_cast<std::size_t>(piece_type)]
                    [static_cast<std::size_t>(square)]
                    [static_cast<std::size_t>(target)];
                if (base >= 0) {
                    const int row = static_cast<int>(base)
                        + victim_relative * Layout::PieceTypes
                        + static_cast<int>(victim_type);
                    add_attack_row(attack_accumulators[p], row, backend);
                }
            }
        }
    }
}

void rebuild_accumulators_formula(
    const Position& pos,
    std::array<PieceAccumulator, COLOR_NB>& piece_accumulators,
    std::array<AttackAccumulator, COLOR_NB>& attack_accumulators,
    std::array<PawnPairAccumulator, COLOR_NB>& pawn_pair_accumulators
) noexcept {
    clear_accumulator_arrays(
        piece_accumulators,
        attack_accumulators,
        pawn_pair_accumulators
    );
    const FeatureLists lists = collect_features_formula(pos);
    add_feature_lists_to_accumulators(
        lists,
        piece_accumulators,
        attack_accumulators,
        pawn_pair_accumulators
    );
}

void rebuild_accumulators(
    const Position& pos,
    std::array<PieceAccumulator, COLOR_NB>& piece_accumulators,
    std::array<AttackAccumulator, COLOR_NB>& attack_accumulators,
    std::array<PawnPairAccumulator, COLOR_NB>& pawn_pair_accumulators,
    RowAddBackend backend = RowAddBackend::Scalar
) noexcept {
    clear_accumulator_arrays(
        piece_accumulators,
        attack_accumulators,
        pawn_pair_accumulators
    );

    const LookupTables& tables = lookup_tables();
    const std::array<bool, COLOR_NB> mirrors = perspective_mirrors(pos);
    const std::array<Square, COLOR_NB> kings{{
        safe_king_square(pos, WHITE),
        safe_king_square(pos, BLACK)
    }};
    std::array<std::array<i16, Layout::Squares>, COLOR_NB> pawn_tokens{};
    std::array<int, COLOR_NB> pawn_token_counts{};

    Bitboard occupied = pos.occupied;
    while (occupied != 0ULL) {
        const Square square = static_cast<Square>(
            std::countr_zero(static_cast<std::uint64_t>(occupied))
        );
        occupied &= occupied - 1;

        const Piece piece = piece_on(pos, square);
        if (piece == PIECE_NONE)
            continue;
        const PieceType piece_type = type_of(piece);
        const Color piece_color = color_of(piece);

        for (int perspective_index = WHITE;
             perspective_index <= BLACK;
             ++perspective_index) {
            const Color perspective =
                static_cast<Color>(perspective_index);
            const std::size_t p =
                static_cast<std::size_t>(perspective_index);
            const int relative_color =
                piece_color == perspective ? 0 : 1;
            const int row = tables.piece_row_by_king[p]
                [static_cast<std::size_t>(kings[p])]
                [static_cast<std::size_t>(relative_color)]
                [static_cast<std::size_t>(piece_type)]
                [static_cast<std::size_t>(square)];
            add_piece_row(piece_accumulators[p], row, backend);

            if (piece_type == PAWN) {
                const i16 token = tables.pawn_token[p]
                    [static_cast<std::size_t>(square)]
                    [static_cast<std::size_t>(relative_color)];
                if (token >= 0) {
                    const int index = pawn_token_counts[p]++;
                    if (index < Layout::Squares)
                        pawn_tokens[p][static_cast<std::size_t>(index)] =
                            token;
                }
            }
        }

        Bitboard targets = occupied_attacks(piece, square, pos.occupied);
        while (targets != 0ULL) {
            const Square target = static_cast<Square>(
                std::countr_zero(static_cast<std::uint64_t>(targets))
            );
            targets &= targets - 1;

            const Piece victim = piece_on(pos, target);
            if (victim == PIECE_NONE)
                continue;
            const PieceType victim_type = type_of(victim);
            const Color victim_color = color_of(victim);

            for (int perspective_index = WHITE;
                 perspective_index <= BLACK;
                 ++perspective_index) {
                const Color perspective =
                    static_cast<Color>(perspective_index);
                const std::size_t p =
                    static_cast<std::size_t>(perspective_index);
                const int attacker_relative =
                    piece_color == perspective ? 0 : 1;
                const int victim_relative =
                    victim_color == perspective ? 0 : 1;
                const int orientation =
                    orientation_index(perspective, mirrors[p]);
                const i32 base = tables.edge_base
                    [static_cast<std::size_t>(orientation)]
                    [static_cast<std::size_t>(attacker_relative)]
                    [static_cast<std::size_t>(piece_type)]
                    [static_cast<std::size_t>(square)]
                    [static_cast<std::size_t>(target)];
                if (base >= 0) {
                    const int row = static_cast<int>(base)
                        + victim_relative * Layout::PieceTypes
                        + static_cast<int>(victim_type);
                    add_attack_row(attack_accumulators[p], row, backend);
                }
            }
        }
    }

    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const std::size_t p = static_cast<std::size_t>(perspective_index);
        const int token_count = std::min(
            pawn_token_counts[p],
            Layout::Squares
        );
        for (int first = 0; first < token_count; ++first) {
            for (int second = first + 1;
                 second < token_count;
                 ++second) {
                const i16 row = tables.pawn_pair_row
                    [static_cast<std::size_t>(pawn_tokens[p][
                        static_cast<std::size_t>(first)
                    ])]
                    [static_cast<std::size_t>(pawn_tokens[p][
                        static_cast<std::size_t>(second)
                    ])];
                if (row >= 0)
                    add_pawn_pair_row(pawn_pair_accumulators[p], row, backend);
            }
        }
    }
}

void reset_accumulator_lut(
    X2K16PawnAccumulator& accumulator,
    const Position& pos,
    RowAddBackend backend
) noexcept {
    rebuild_accumulators(
        pos,
        accumulator.piece,
        accumulator.attack,
        accumulator.pawn,
        backend
    );
    accumulator.piece_valid = true;
    accumulator.pawn_valid = true;
    accumulator.attack_valid = true;
    accumulator.position_hash = pos.key;
}

void merge_accumulator(
    const PieceAccumulator& piece,
    const AttackAccumulator& attack,
    const PawnPairAccumulator& pawn_pair,
    MergedAccumulator& merged
) noexcept {
    for (int column = 0; column < Layout::MergedHiddenSize; ++column) {
        i32 value = static_cast<i32>(
            g_network.l0b[static_cast<std::size_t>(column)]
        );
        value += piece[static_cast<std::size_t>(column)];
        value += pawn_pair[static_cast<std::size_t>(column)];
        if (column < Layout::AttackHiddenSize)
            value += attack[static_cast<std::size_t>(column)];
        merged[static_cast<std::size_t>(column)] = value;
    }
}

struct ForwardMask {
    bool l0b = true;
    bool piece = true;
    bool attack = true;
    bool pawn_pair = true;
};

void merge_accumulator_masked(
    const PieceAccumulator& piece,
    const AttackAccumulator& attack,
    const PawnPairAccumulator& pawn_pair,
    const ForwardMask mask,
    MergedAccumulator& merged
) noexcept {
    for (int column = 0; column < Layout::MergedHiddenSize; ++column) {
        i32 value = 0;
        if (mask.l0b) {
            value += static_cast<i32>(
                g_network.l0b[static_cast<std::size_t>(column)]
            );
        }
        if (mask.piece)
            value += piece[static_cast<std::size_t>(column)];
        if (mask.pawn_pair)
            value += pawn_pair[static_cast<std::size_t>(column)];
        if (mask.attack && column < Layout::AttackHiddenSize)
            value += attack[static_cast<std::size_t>(column)];
        merged[static_cast<std::size_t>(column)] = value;
    }
}

[[nodiscard]] constexpr u8 pairwise_value(
    i32 left,
    i32 right
) noexcept {
    left = std::clamp<i32>(left, 0, static_cast<i32>(kQa));
    right = std::clamp<i32>(right, 0, static_cast<i32>(kQa));
    return static_cast<u8>(left * right / static_cast<i32>(kQa));
}

void pairwise_crelu(
    const MergedAccumulator& accumulator,
    Pairwise& output
) noexcept {
    constexpr int Half = Layout::MergedHiddenSize / 2;
    for (int index = 0; index < Half; ++index) {
        output[static_cast<std::size_t>(index)] = pairwise_value(
            accumulator[static_cast<std::size_t>(index)],
            accumulator[static_cast<std::size_t>(index + Half)]
        );
    }
}

template<std::size_t N>
[[nodiscard]] i32 max_abs_value(const std::array<i32, N>& values) noexcept {
    i32 result = 0;
    for (const i32 value : values) {
        const i32 absolute = value < 0 ? -value : value;
        if (absolute > result)
            result = absolute;
    }
    return result;
}

struct PairwiseDebugStats {
    i32 merged_max_abs = 0;
    int nonzero = 0;
    int clipped = 0;
};

[[nodiscard]] PairwiseDebugStats pairwise_debug_stats(
    const MergedAccumulator& accumulator
) noexcept {
    PairwiseDebugStats stats{};
    constexpr int Half = Layout::MergedHiddenSize / 2;
    for (int index = 0; index < Half; ++index) {
        const i32 left = accumulator[static_cast<std::size_t>(index)];
        const i32 right = accumulator[static_cast<std::size_t>(index + Half)];
        stats.merged_max_abs = std::max(
            stats.merged_max_abs,
            std::max(left < 0 ? -left : left, right < 0 ? -right : right)
        );

        const i32 clipped_left =
            std::clamp<i32>(left, 0, static_cast<i32>(kQa));
        const i32 clipped_right =
            std::clamp<i32>(right, 0, static_cast<i32>(kQa));
        if (left != clipped_left || right != clipped_right)
            ++stats.clipped;
        if ((clipped_left * clipped_right / static_cast<i32>(kQa)) != 0)
            ++stats.nonzero;
    }
    return stats;
}

[[nodiscard]] constexpr float crelu(float value) noexcept {
    return std::clamp(value, 0.0F, 1.0F);
}

[[nodiscard]] int output_bucket(const Position& pos) noexcept {
    const int count = static_cast<int>(
        std::popcount(static_cast<std::uint64_t>(pos.occupied))
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

[[nodiscard]] double forward(
    const Position& pos,
    const std::array<PieceAccumulator, COLOR_NB>& piece_accumulators,
    const std::array<AttackAccumulator, COLOR_NB>& attack_accumulators,
    const std::array<PawnPairAccumulator, COLOR_NB>& pawn_pair_accumulators
) noexcept {
    std::array<MergedAccumulator, COLOR_NB> merged{};
    std::array<Pairwise, COLOR_NB> pairwise{};
    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const std::size_t p = static_cast<std::size_t>(perspective_index);
        merge_accumulator(
            piece_accumulators[p],
            attack_accumulators[p],
            pawn_pair_accumulators[p],
            merged[p]
        );
        pairwise_crelu(merged[p], pairwise[p]);
    }

    const Color stm = static_cast<Color>(pos.side_to_move);
    const Pairwise& stm_pairwise =
        stm == WHITE ? pairwise[WHITE] : pairwise[BLACK];
    const Pairwise& ntm_pairwise =
        stm == WHITE ? pairwise[BLACK] : pairwise[WHITE];

    BackendInput backend_input{};
    std::copy(
        stm_pairwise.begin(),
        stm_pairwise.end(),
        backend_input.begin()
    );
    std::copy(
        ntm_pairwise.begin(),
        ntm_pairwise.end(),
        backend_input.begin() + Layout::PairwiseSize
    );

    const std::size_t bucket =
        static_cast<std::size_t>(output_bucket(pos));

    Hidden1 hidden1{};
    const std::int8_t* l1_weights =
        g_network.l1w.data()
        + bucket * Layout::L1Size * Layout::HeadInputSize;
    const float* l1_bias =
        g_network.l1b.data() + bucket * Layout::L1Size;
    constexpr float L1Scale =
        1.0F / (static_cast<float>(kQa) * static_cast<float>(kL1Quant));
    for (int row = 0; row < Layout::L1Size; ++row) {
        float sum = l1_bias[row];
        const std::int8_t* weights =
            l1_weights + static_cast<std::size_t>(row)
                * Layout::HeadInputSize;
        for (int column = 0; column < Layout::HeadInputSize; ++column) {
            const u8 input =
                backend_input[static_cast<std::size_t>(column)];
            if (input == 0)
                continue;
            sum += static_cast<float>(input)
                * static_cast<float>(weights[column])
                * L1Scale;
        }
        hidden1[static_cast<std::size_t>(row)] = crelu(sum);
    }

    Hidden2 hidden2{};
    const float* l2_weights =
        g_network.l2w.data()
        + bucket * Layout::L2Size * Layout::L1Size;
    const float* l2_bias =
        g_network.l2b.data() + bucket * Layout::L2Size;
    for (int row = 0; row < Layout::L2Size; ++row) {
        float sum = l2_bias[row];
        const float* weights =
            l2_weights + static_cast<std::size_t>(row) * Layout::L1Size;
        for (int column = 0; column < Layout::L1Size; ++column) {
            sum += hidden1[static_cast<std::size_t>(column)]
                * weights[column];
        }
        hidden2[static_cast<std::size_t>(row)] = crelu(sum);
    }

    float output = g_network.l3b[bucket];
    const float* l3_weights =
        g_network.l3w.data() + bucket * Layout::L2Size;
    for (int column = 0; column < Layout::L2Size; ++column) {
        output += hidden2[static_cast<std::size_t>(column)]
            * l3_weights[column];
    }
    return static_cast<double>(output);
}

[[nodiscard]] double forward_profiled(
    const Position& pos,
    const std::array<PieceAccumulator, COLOR_NB>& piece_accumulators,
    const std::array<AttackAccumulator, COLOR_NB>& attack_accumulators,
    const std::array<PawnPairAccumulator, COLOR_NB>& pawn_pair_accumulators,
    EvaluationProfile& profile
) noexcept {
    const auto merge_begin = Clock::now();
    std::array<MergedAccumulator, COLOR_NB> merged{};
    std::array<Pairwise, COLOR_NB> pairwise{};
    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const std::size_t p = static_cast<std::size_t>(perspective_index);
        merge_accumulator(
            piece_accumulators[p],
            attack_accumulators[p],
            pawn_pair_accumulators[p],
            merged[p]
        );
        pairwise_crelu(merged[p], pairwise[p]);
    }
    const auto merge_end = Clock::now();
    profile.merge_activation_time_us += elapsed_us(merge_begin, merge_end);

    const auto head_begin = Clock::now();
    const Color stm = static_cast<Color>(pos.side_to_move);
    const Pairwise& stm_pairwise =
        stm == WHITE ? pairwise[WHITE] : pairwise[BLACK];
    const Pairwise& ntm_pairwise =
        stm == WHITE ? pairwise[BLACK] : pairwise[WHITE];

    BackendInput backend_input{};
    std::copy(
        stm_pairwise.begin(),
        stm_pairwise.end(),
        backend_input.begin()
    );
    std::copy(
        ntm_pairwise.begin(),
        ntm_pairwise.end(),
        backend_input.begin() + Layout::PairwiseSize
    );

    const std::size_t bucket =
        static_cast<std::size_t>(output_bucket(pos));

    Hidden1 hidden1{};
    const std::int8_t* l1_weights =
        g_network.l1w.data()
        + bucket * Layout::L1Size * Layout::HeadInputSize;
    const float* l1_bias =
        g_network.l1b.data() + bucket * Layout::L1Size;
    constexpr float L1Scale =
        1.0F / (static_cast<float>(kQa) * static_cast<float>(kL1Quant));
    for (int row = 0; row < Layout::L1Size; ++row) {
        float sum = l1_bias[row];
        const std::int8_t* weights =
            l1_weights + static_cast<std::size_t>(row)
                * Layout::HeadInputSize;
        for (int column = 0; column < Layout::HeadInputSize; ++column) {
            const u8 input =
                backend_input[static_cast<std::size_t>(column)];
            if (input == 0)
                continue;
            sum += static_cast<float>(input)
                * static_cast<float>(weights[column])
                * L1Scale;
        }
        hidden1[static_cast<std::size_t>(row)] = crelu(sum);
    }

    Hidden2 hidden2{};
    const float* l2_weights =
        g_network.l2w.data()
        + bucket * Layout::L2Size * Layout::L1Size;
    const float* l2_bias =
        g_network.l2b.data() + bucket * Layout::L2Size;
    for (int row = 0; row < Layout::L2Size; ++row) {
        float sum = l2_bias[row];
        const float* weights =
            l2_weights + static_cast<std::size_t>(row) * Layout::L1Size;
        for (int column = 0; column < Layout::L1Size; ++column) {
            sum += hidden1[static_cast<std::size_t>(column)]
                * weights[column];
        }
        hidden2[static_cast<std::size_t>(row)] = crelu(sum);
    }

    float output = g_network.l3b[bucket];
    const float* l3_weights =
        g_network.l3w.data() + bucket * Layout::L2Size;
    for (int column = 0; column < Layout::L2Size; ++column) {
        output += hidden2[static_cast<std::size_t>(column)]
            * l3_weights[column];
    }
    const auto head_end = Clock::now();
    profile.head_time_us += elapsed_us(head_begin, head_end);
    return static_cast<double>(output);
}

[[nodiscard]] double forward_masked(
    const Position& pos,
    const std::array<PieceAccumulator, COLOR_NB>& piece_accumulators,
    const std::array<AttackAccumulator, COLOR_NB>& attack_accumulators,
    const std::array<PawnPairAccumulator, COLOR_NB>& pawn_pair_accumulators,
    const ForwardMask mask
) noexcept {
    std::array<MergedAccumulator, COLOR_NB> merged{};
    std::array<Pairwise, COLOR_NB> pairwise{};
    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const std::size_t p = static_cast<std::size_t>(perspective_index);
        merge_accumulator_masked(
            piece_accumulators[p],
            attack_accumulators[p],
            pawn_pair_accumulators[p],
            mask,
            merged[p]
        );
        pairwise_crelu(merged[p], pairwise[p]);
    }

    const Color stm = static_cast<Color>(pos.side_to_move);
    const Pairwise& stm_pairwise =
        stm == WHITE ? pairwise[WHITE] : pairwise[BLACK];
    const Pairwise& ntm_pairwise =
        stm == WHITE ? pairwise[BLACK] : pairwise[WHITE];

    BackendInput backend_input{};
    std::copy(
        stm_pairwise.begin(),
        stm_pairwise.end(),
        backend_input.begin()
    );
    std::copy(
        ntm_pairwise.begin(),
        ntm_pairwise.end(),
        backend_input.begin() + Layout::PairwiseSize
    );

    const std::size_t bucket =
        static_cast<std::size_t>(output_bucket(pos));

    Hidden1 hidden1{};
    const std::int8_t* l1_weights =
        g_network.l1w.data()
        + bucket * Layout::L1Size * Layout::HeadInputSize;
    const float* l1_bias =
        g_network.l1b.data() + bucket * Layout::L1Size;
    constexpr float L1Scale =
        1.0F / (static_cast<float>(kQa) * static_cast<float>(kL1Quant));
    for (int row = 0; row < Layout::L1Size; ++row) {
        float sum = l1_bias[row];
        const std::int8_t* weights =
            l1_weights + static_cast<std::size_t>(row)
                * Layout::HeadInputSize;
        for (int column = 0; column < Layout::HeadInputSize; ++column) {
            const u8 input =
                backend_input[static_cast<std::size_t>(column)];
            if (input == 0)
                continue;
            sum += static_cast<float>(input)
                * static_cast<float>(weights[column])
                * L1Scale;
        }
        hidden1[static_cast<std::size_t>(row)] = crelu(sum);
    }

    Hidden2 hidden2{};
    const float* l2_weights =
        g_network.l2w.data()
        + bucket * Layout::L2Size * Layout::L1Size;
    const float* l2_bias =
        g_network.l2b.data() + bucket * Layout::L2Size;
    for (int row = 0; row < Layout::L2Size; ++row) {
        float sum = l2_bias[row];
        const float* weights =
            l2_weights + static_cast<std::size_t>(row) * Layout::L1Size;
        for (int column = 0; column < Layout::L1Size; ++column) {
            sum += hidden1[static_cast<std::size_t>(column)]
                * weights[column];
        }
        hidden2[static_cast<std::size_t>(row)] = crelu(sum);
    }

    float output = g_network.l3b[bucket];
    const float* l3_weights =
        g_network.l3w.data() + bucket * Layout::L2Size;
    for (int column = 0; column < Layout::L2Size; ++column) {
        output += hidden2[static_cast<std::size_t>(column)]
            * l3_weights[column];
    }
    return static_cast<double>(output);
}

[[nodiscard]] int score_from_output(double output) noexcept {
    const double scaled = output * static_cast<double>(kScale);
    if (scaled >= static_cast<double>(std::numeric_limits<int>::max()))
        return std::numeric_limits<int>::max();
    if (scaled <= static_cast<double>(std::numeric_limits<int>::min()))
        return std::numeric_limits<int>::min();
    return static_cast<int>(std::llround(scaled));
}

void dump_vector(
    std::ostream& output,
    const char* color,
    const char* family,
    const std::vector<int>& values
) {
    output << "info string " << color << ' ' << family;
    for (const int value : values)
        output << ' ' << value;
    output << '\n';
}

[[nodiscard]] char piece_to_fen_char(Piece piece) noexcept {
    switch (piece) {
    case W_PAWN: return 'P';
    case W_KNIGHT: return 'N';
    case W_BISHOP: return 'B';
    case W_ROOK: return 'R';
    case W_QUEEN: return 'Q';
    case W_KING: return 'K';
    case B_PAWN: return 'p';
    case B_KNIGHT: return 'n';
    case B_BISHOP: return 'b';
    case B_ROOK: return 'r';
    case B_QUEEN: return 'q';
    case B_KING: return 'k';
    default: return '?';
    }
}

[[nodiscard]] std::string square_to_fen(Square square) {
    if (square == NO_SQ)
        return "-";
    std::string result;
    result.push_back(static_cast<char>('a' + file_of_sq(square)));
    result.push_back(static_cast<char>('1' + rank_of_sq(square)));
    return result;
}

[[nodiscard]] std::string castling_to_fen(int rights) {
    std::string result;
    if ((rights & WHITE_OO) != 0)
        result.push_back('K');
    if ((rights & WHITE_OOO) != 0)
        result.push_back('Q');
    if ((rights & BLACK_OO) != 0)
        result.push_back('k');
    if ((rights & BLACK_OOO) != 0)
        result.push_back('q');
    return result.empty() ? "-" : result;
}

[[nodiscard]] std::string position_to_fen(const Position& pos) {
    std::string fen;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const Square square = rank * 8 + file;
            const Piece piece = piece_on(pos, square);
            if (piece == PIECE_NONE) {
                ++empty;
                continue;
            }
            if (empty != 0) {
                fen.push_back(static_cast<char>('0' + empty));
                empty = 0;
            }
            fen.push_back(piece_to_fen_char(piece));
        }
        if (empty != 0)
            fen.push_back(static_cast<char>('0' + empty));
        if (rank != 0)
            fen.push_back('/');
    }

    fen += pos.side_to_move == WHITE ? " w " : " b ";
    fen += castling_to_fen(pos.castling_rights);
    fen.push_back(' ');
    fen += square_to_fen(pos.ep_sq);
    fen.push_back(' ');
    fen += std::to_string(pos.halfmove_clock);
    fen.push_back(' ');
    fen += std::to_string(pos.fullmove_number);
    return fen;
}

[[nodiscard]] u32 next_stress_random(u32& state) noexcept {
    state = state * 1664525u + 1013904223u;
    return state;
}

[[nodiscard]] char promotion_char(Move move) noexcept {
    switch (promo_piece(move)) {
    case KNIGHT: return 'n';
    case BISHOP: return 'b';
    case ROOK: return 'r';
    case QUEEN: return 'q';
    default: return 'q';
    }
}

[[nodiscard]] std::string move_to_simple_uci(Move move) {
    std::string text;
    text += square_to_fen(from_sq(move));
    text += square_to_fen(to_sq(move));
    if (move_is_promotion(move))
        text.push_back(promotion_char(move));
    return text;
}

[[nodiscard]] bool equal_vectors(
    const std::vector<int>& left,
    const std::vector<int>& right
) noexcept {
    return left.size() == right.size()
        && std::equal(left.begin(), left.end(), right.begin());
}

[[nodiscard]] bool feature_lists_equal(
    const FeatureLists& left,
    const FeatureLists& right
) noexcept {
    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const std::size_t p = static_cast<std::size_t>(perspective_index);
        if (!equal_vectors(left.piece[p], right.piece[p])
            || !equal_vectors(left.attack[p], right.attack[p])
            || !equal_vectors(left.pawn_pair[p], right.pawn_pair[p])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] int first_vector_diff(
    const std::vector<int>& left,
    const std::vector<int>& right
) noexcept {
    const std::size_t size = std::min(left.size(), right.size());
    for (std::size_t index = 0; index < size; ++index) {
        if (left[index] != right[index])
            return static_cast<int>(index);
    }
    if (left.size() != right.size())
        return static_cast<int>(size);
    return -1;
}

[[nodiscard]] EvaluationProfile profile_lut_evaluation(
    const Position& pos,
    RowAddBackend backend
) noexcept {
    EvaluationProfile profile{};
    const auto total_begin = Clock::now();

    const auto feature_begin = Clock::now();
    const FeatureLists lists = collect_features(pos);
    const auto feature_end = Clock::now();
    profile.feature_gen_time_us = elapsed_us(feature_begin, feature_end);

    X2K16PawnAccumulator accumulator{};
    clear_accumulator(accumulator);
    add_feature_lists_to_accumulators(
        lists,
        accumulator.piece,
        accumulator.attack,
        accumulator.pawn,
        &profile,
        backend
    );
    accumulator.piece_valid = true;
    accumulator.pawn_valid = true;
    accumulator.attack_valid = true;
    accumulator.position_hash = pos.key;

    profile.raw = score_from_output(forward_profiled(
        pos,
        accumulator.piece,
        accumulator.attack,
        accumulator.pawn,
        profile
    ));
    profile.searchcp = magnus::mnue::search_score_to_cp(profile.raw, pos);
    const auto total_end = Clock::now();
    profile.total_eval_time_us = elapsed_us(total_begin, total_end);
    return profile;
}

void rebuild_piece_pawn_incremental(
    const Position& pos,
    X2K16PawnAccumulator& accumulator,
    RowAddBackend backend
) noexcept {
    const FeatureLists lists = collect_features(pos);
    clear_piece_accumulators(accumulator.piece);
    clear_pawn_accumulators(accumulator.pawn);

    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const std::size_t p = static_cast<std::size_t>(perspective_index);
        for (const int row : lists.piece[p])
            add_piece_row(accumulator.piece[p], row, backend);
        for (const int row : lists.pawn_pair[p])
            add_pawn_pair_row(accumulator.pawn[p], row, backend);
    }

    accumulator.piece_valid = true;
    accumulator.pawn_valid = true;
    accumulator.attack_valid = false;
    accumulator.position_hash = pos.key;
}

void apply_piece_feature_delta(
    X2K16PawnAccumulator& accumulator,
    const FeatureLists& before,
    const FeatureLists& after,
    RowAddBackend backend
) noexcept {
    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const std::size_t p = static_cast<std::size_t>(perspective_index);
        apply_sorted_row_delta(
            before.piece[p],
            after.piece[p],
            [&] (int row) noexcept {
                add_piece_row(accumulator.piece[p], row, backend);
            },
            [&] (int row) noexcept {
                sub_piece_row(accumulator.piece[p], row, backend);
            }
        );
    }
}

void apply_pawn_pair_feature_delta(
    X2K16PawnAccumulator& accumulator,
    const FeatureLists& before,
    const FeatureLists& after,
    RowAddBackend backend
) noexcept {
    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const std::size_t p = static_cast<std::size_t>(perspective_index);
        apply_sorted_row_delta(
            before.pawn_pair[p],
            after.pawn_pair[p],
            [&] (int row) noexcept {
                add_pawn_pair_row(accumulator.pawn[p], row, backend);
            },
            [&] (int row) noexcept {
                sub_pawn_pair_row(accumulator.pawn[p], row, backend);
            }
        );
    }
}

void update_piece_pawn_incremental(
    X2K16PawnAccumulator& accumulator,
    const Position& before,
    const Position& after,
    Move move,
    const StateInfo& state,
    RowAddBackend backend
) noexcept {
    if (!accumulator.piece_valid || !accumulator.pawn_valid
        || accumulator.position_hash != before.key) {
        rebuild_piece_pawn_incremental(after, accumulator, backend);
        return;
    }

    const Piece moving = piece_on(before, from_sq(move));
    const PieceType moving_type = type_of(moving);
    const bool king_moved =
        moving != PIECE_NONE && moving_type == KING;
    const bool pawn_affected =
        (moving != PIECE_NONE && moving_type == PAWN)
        || (state.captured != PIECE_NONE
            && type_of(state.captured) == PAWN);

    FeatureLists before_lists{};
    FeatureLists after_lists{};
    const bool need_lists = !king_moved || pawn_affected;
    if (need_lists) {
        before_lists = collect_features(before);
        after_lists = collect_features(after);
    }

    if (king_moved) {
        rebuild_piece_accumulators(
            after,
            accumulator.piece,
            backend
        );
        accumulator.piece_valid = true;
    } else {
        apply_piece_feature_delta(
            accumulator,
            before_lists,
            after_lists,
            backend
        );
    }

    if (pawn_affected) {
        apply_pawn_pair_feature_delta(
            accumulator,
            before_lists,
            after_lists,
            backend
        );
    }

    accumulator.pawn_valid = true;
    accumulator.attack_valid = false;
    accumulator.position_hash = after.key;
}

[[nodiscard]] int evaluate_piece_pawn_incremental(
    const Position& pos,
    X2K16PawnAccumulator& accumulator,
    RowAddBackend backend
) noexcept {
    if (!accumulator.piece_valid || !accumulator.pawn_valid
        || accumulator.position_hash != pos.key) {
        rebuild_piece_pawn_incremental(pos, accumulator, backend);
    }

    rebuild_attack_accumulators(pos, accumulator.attack, backend);
    accumulator.attack_valid = true;
    return score_from_output(forward(
        pos,
        accumulator.piece,
        accumulator.attack,
        accumulator.pawn
    ));
}

struct BranchChecksums {
    i64 piece = 0;
    i64 attack = 0;
    i64 pawn = 0;
};

[[nodiscard]] BranchChecksums branch_checksums(
    const X2K16PawnAccumulator& accumulator
) noexcept {
    return {
        checksum_branch(accumulator.piece),
        checksum_branch(accumulator.attack),
        checksum_branch(accumulator.pawn)
    };
}

} // namespace

bool load(const std::string& file_path) {
    clear(g_network);

    std::error_code error_code;
    const std::uintmax_t bytes =
        std::filesystem::file_size(file_path, error_code);
    if (error_code) {
        g_network.error =
            "could not determine MNUE-X2-K16-pawn-Q8-A384 file size";
        return false;
    }
    if (bytes != file_bytes()) {
        g_network.error =
            "MNUE-X2-K16-pawn-Q8-A384 file size mismatch: got "
            + std::to_string(bytes)
            + ", expected " + std::to_string(file_bytes());
        return false;
    }

    std::ifstream input(file_path, std::ios::binary);
    if (!input) {
        g_network.error = "could not open MNUE-X2-K16-pawn-Q8-A384 file";
        return false;
    }

    FileHeader header{};
    if (!read_exact(input, &header, 1)) {
        g_network.error =
            "could not read MNUE-X2-K16-pawn-Q8-A384 header";
        return false;
    }
    const std::string mismatch = header_error(header);
    if (!mismatch.empty()) {
        g_network.error = mismatch;
        return false;
    }

    resize_tensors(g_network);
    if (!read_payload(input, g_network)) {
        clear(g_network);
        g_network.error = "truncated MNUE-X2-K16-pawn-Q8-A384 payload";
        return false;
    }
    if (!g_network.valid()) {
        clear(g_network);
        g_network.error =
            "invalid MNUE-X2-K16-pawn-Q8-A384 tensor dimensions";
        return false;
    }
    if (attack_index_table().size != Layout::AttackInputSize) {
        clear(g_network);
        g_network.error =
            "MNUE-X2-K16-pawn-Q8-A384 attack index table size mismatch";
        return false;
    }

    g_network.file_bytes = static_cast<std::size_t>(bytes);
    g_network.file_path = file_path;
    g_network.is_loaded = true;
    (void)lookup_tables();
    return true;
}

void unload() noexcept {
    clear(g_network);
}

bool loaded() noexcept {
    return g_network.is_loaded && g_network.valid();
}

const std::string& path() noexcept {
    return g_network.file_path;
}

const std::string& last_error() noexcept {
    return g_network.error;
}

std::size_t network_bytes() noexcept {
    return g_network.file_bytes;
}

std::uintmax_t expected_payload_bytes() noexcept {
    return payload_bytes();
}

std::uintmax_t expected_file_bytes() noexcept {
    return file_bytes();
}

int evaluate_reference(
    const Position& pos,
    const memory::Memory& mem
) noexcept {
    (void)mem;
    if (!loaded())
        return 0;

    std::array<PieceAccumulator, COLOR_NB> piece_accumulators{};
    std::array<AttackAccumulator, COLOR_NB> attack_accumulators{};
    std::array<PawnPairAccumulator, COLOR_NB> pawn_pair_accumulators{};
    rebuild_accumulators_formula(
        pos,
        piece_accumulators,
        attack_accumulators,
        pawn_pair_accumulators
    );
    return score_from_output(forward(
        pos,
        piece_accumulators,
        attack_accumulators,
        pawn_pair_accumulators
    ));
}

[[nodiscard]] int evaluate_lut_with_backend(
    const Position& pos,
    RowAddBackend backend
) noexcept {
    if (!loaded())
        return 0;

    X2K16PawnAccumulator accumulator{};
    reset_accumulator_lut(accumulator, pos, backend);
    return score_from_output(forward(
        pos,
        accumulator.piece,
        accumulator.attack,
        accumulator.pawn
    ));
}

[[nodiscard]] int evaluate_lut_scalar(
    const Position& pos
) noexcept {
    return evaluate_lut_with_backend(pos, RowAddBackend::Scalar);
}

[[nodiscard]] int evaluate_lut_avx2(
    const Position& pos
) noexcept {
    return evaluate_lut_with_backend(pos, RowAddBackend::Avx2);
}

int evaluate_lut(
    const Position& pos,
    const memory::Memory& mem
) noexcept {
    (void)mem;
    return evaluate_lut_with_backend(pos, active_rowadd_backend());
}

const char* backend_name() noexcept {
    return rowadd_backend_name(active_rowadd_backend());
}

void debug_dump_evaluation(
    const Position& pos,
    const memory::Memory& mem,
    std::ostream& output
) {
    (void)mem;
    if (!loaded()) {
        output << "info string MNUE-X2-K16-pawn-Q8-A384 unavailable: "
            << last_error() << '\n';
        return;
    }

    std::array<PieceAccumulator, COLOR_NB> piece_accumulators{};
    std::array<AttackAccumulator, COLOR_NB> attack_accumulators{};
    std::array<PawnPairAccumulator, COLOR_NB> pawn_pair_accumulators{};
    rebuild_accumulators(
        pos,
        piece_accumulators,
        attack_accumulators,
        pawn_pair_accumulators
    );

    const auto raw_for = [&] (ForwardMask mask) noexcept {
        return score_from_output(forward_masked(
            pos,
            piece_accumulators,
            attack_accumulators,
            pawn_pair_accumulators,
            mask
        ));
    };

    const int total_raw = raw_for(ForwardMask{});
    const int head_bias_only_raw = raw_for(ForwardMask{
        .l0b = false,
        .piece = false,
        .attack = false,
        .pawn_pair = false
    });
    const int l0b_only_raw = raw_for(ForwardMask{
        .l0b = true,
        .piece = false,
        .attack = false,
        .pawn_pair = false
    });
    const int piece_only_raw = raw_for(ForwardMask{
        .l0b = true,
        .piece = true,
        .attack = false,
        .pawn_pair = false
    });
    const int attack_only_raw = raw_for(ForwardMask{
        .l0b = true,
        .piece = false,
        .attack = true,
        .pawn_pair = false
    });
    const int pawn_only_raw = raw_for(ForwardMask{
        .l0b = true,
        .piece = false,
        .attack = false,
        .pawn_pair = true
    });

    std::array<MergedAccumulator, COLOR_NB> merged{};
    std::array<PairwiseDebugStats, COLOR_NB> pairwise_stats{};
    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const std::size_t p = static_cast<std::size_t>(perspective_index);
        merge_accumulator(
            piece_accumulators[p],
            attack_accumulators[p],
            pawn_pair_accumulators[p],
            merged[p]
        );
        pairwise_stats[p] = pairwise_debug_stats(merged[p]);
    }

    const std::size_t stm =
        pos.side_to_move == WHITE ? static_cast<std::size_t>(WHITE)
                                  : static_cast<std::size_t>(BLACK);
    const std::size_t ntm =
        pos.side_to_move == WHITE ? static_cast<std::size_t>(BLACK)
                                  : static_cast<std::size_t>(WHITE);
    const FeatureLists lists = collect_features(pos);
    const int searchcp =
        magnus::mnue::search_score_to_cp(total_raw, pos);

    output << "info string mnue x2k16 debug total_raw "
        << total_raw << '\n';
    output << "info string mnue x2k16 debug searchcp "
        << searchcp << '\n';
    output << "info string mnue x2k16 debug output_bucket "
        << output_bucket(pos) << '\n';
    output << "info string mnue x2k16 debug head_bias_only_raw "
        << head_bias_only_raw << '\n';
    output << "info string mnue x2k16 debug l0b_only_raw "
        << l0b_only_raw << '\n';
    output << "info string mnue x2k16 debug piece_only_raw "
        << piece_only_raw << '\n';
    output << "info string mnue x2k16 debug attack_only_raw "
        << attack_only_raw << '\n';
    output << "info string mnue x2k16 debug pawn_only_raw "
        << pawn_only_raw << '\n';
    output << "info string mnue x2k16 debug piece_delta_raw "
        << piece_only_raw - l0b_only_raw << '\n';
    output << "info string mnue x2k16 debug attack_delta_raw "
        << attack_only_raw - l0b_only_raw << '\n';
    output << "info string mnue x2k16 debug pawn_delta_raw "
        << pawn_only_raw - l0b_only_raw << '\n';

    output << "info string mnue x2k16 debug piece_acc_stm_max_abs "
        << max_abs_value(piece_accumulators[stm]) << '\n';
    output << "info string mnue x2k16 debug piece_acc_ntm_max_abs "
        << max_abs_value(piece_accumulators[ntm]) << '\n';
    output << "info string mnue x2k16 debug attack_acc_stm_max_abs "
        << max_abs_value(attack_accumulators[stm]) << '\n';
    output << "info string mnue x2k16 debug attack_acc_ntm_max_abs "
        << max_abs_value(attack_accumulators[ntm]) << '\n';
    output << "info string mnue x2k16 debug pawn_acc_stm_max_abs "
        << max_abs_value(pawn_pair_accumulators[stm]) << '\n';
    output << "info string mnue x2k16 debug pawn_acc_ntm_max_abs "
        << max_abs_value(pawn_pair_accumulators[ntm]) << '\n';

    output << "info string mnue x2k16 debug merged_stm_max_abs "
        << pairwise_stats[stm].merged_max_abs << '\n';
    output << "info string mnue x2k16 debug merged_ntm_max_abs "
        << pairwise_stats[ntm].merged_max_abs << '\n';
    output << "info string mnue x2k16 debug crelu_stm_nonzero "
        << pairwise_stats[stm].nonzero << '\n';
    output << "info string mnue x2k16 debug crelu_ntm_nonzero "
        << pairwise_stats[ntm].nonzero << '\n';
    output << "info string mnue x2k16 debug crelu_stm_clipped "
        << pairwise_stats[stm].clipped << '\n';
    output << "info string mnue x2k16 debug crelu_ntm_clipped "
        << pairwise_stats[ntm].clipped << '\n';

    output << "info string mnue x2k16 debug active_piece_features_stm "
        << lists.piece[stm].size() << '\n';
    output << "info string mnue x2k16 debug active_piece_features_ntm "
        << lists.piece[ntm].size() << '\n';
    output << "info string mnue x2k16 debug active_attack_features_stm "
        << lists.attack[stm].size() << '\n';
    output << "info string mnue x2k16 debug active_attack_features_ntm "
        << lists.attack[ntm].size() << '\n';
    output << "info string mnue x2k16 debug active_pawn_pair_features_stm "
        << lists.pawn_pair[stm].size() << '\n';
    output << "info string mnue x2k16 debug active_pawn_pair_features_ntm "
        << lists.pawn_pair[ntm].size() << '\n';
}

void debug_compare_evaluation(
    const Position& pos,
    const memory::Memory& mem,
    std::ostream& output
) {
    (void)mem;
    if (!loaded()) {
        output << "info string MNUE-X2-K16-pawn-Q8-A384 unavailable: "
            << last_error() << '\n';
        return;
    }

    const FeatureLists formula_features = collect_features_formula(pos);
    const FeatureLists lut_features = collect_features(pos);

    std::array<PieceAccumulator, COLOR_NB> formula_piece{};
    std::array<AttackAccumulator, COLOR_NB> formula_attack{};
    std::array<PawnPairAccumulator, COLOR_NB> formula_pawn{};
    rebuild_accumulators_formula(
        pos,
        formula_piece,
        formula_attack,
        formula_pawn
    );

    X2K16PawnAccumulator scalar_lut_accumulator{};
    reset_accumulator_lut(
        scalar_lut_accumulator,
        pos,
        RowAddBackend::Scalar
    );
    X2K16PawnAccumulator avx2_lut_accumulator{};
    reset_accumulator_lut(
        avx2_lut_accumulator,
        pos,
        RowAddBackend::Avx2
    );

    const int reference_raw = score_from_output(forward(
        pos,
        formula_piece,
        formula_attack,
        formula_pawn
    ));
    const int scalar_lut_raw = score_from_output(forward(
        pos,
        scalar_lut_accumulator.piece,
        scalar_lut_accumulator.attack,
        scalar_lut_accumulator.pawn
    ));
    const int avx2_lut_raw = score_from_output(forward(
        pos,
        avx2_lut_accumulator.piece,
        avx2_lut_accumulator.attack,
        avx2_lut_accumulator.pawn
    ));
    const int reference_cp =
        magnus::mnue::search_score_to_cp(reference_raw, pos);
    const int scalar_lut_cp =
        magnus::mnue::search_score_to_cp(scalar_lut_raw, pos);
    const int avx2_lut_cp =
        magnus::mnue::search_score_to_cp(avx2_lut_raw, pos);

    output << "info string x2 compare reference_raw "
        << reference_raw
        << " scalar_lut_raw " << scalar_lut_raw
        << " avx2_lut_raw " << avx2_lut_raw
        << " diff_ref_lut " << (scalar_lut_raw - reference_raw)
        << " diff_lut_avx2 " << (avx2_lut_raw - scalar_lut_raw)
        << " feature_rows_equal "
        << (feature_lists_equal(formula_features, lut_features) ? 1 : 0)
        << '\n';
    output << "info string x2 compare reference_cp "
        << reference_cp
        << " scalar_lut_cp " << scalar_lut_cp
        << " avx2_lut_cp " << avx2_lut_cp
        << " cp_diff_ref_lut " << (scalar_lut_cp - reference_cp)
        << " cp_diff_lut_avx2 " << (avx2_lut_cp - scalar_lut_cp)
        << '\n';
    output << "info string x2 compare backend_reference scalar-full-rebuild"
        << " backend_scalar scalar-lut-full-rebuild"
        << " backend_active " << backend_name()
        << " avx2_compiled " << (avx2_rowadd_compiled() ? 1 : 0)
        << '\n';
    output << "info string x2 compare fen current old_raw "
        << reference_raw << " new_raw " << scalar_lut_raw
        << " diff " << (scalar_lut_raw - reference_raw) << '\n';
    output << "info string x2 compare fen current old_cp "
        << reference_cp << " new_cp " << scalar_lut_cp
        << " cp_diff " << (scalar_lut_cp - reference_cp) << '\n';
    output << "info string x2 compare feature_rows_equal "
        << (feature_lists_equal(formula_features, lut_features) ? 1 : 0)
        << '\n';

    for (int perspective_index = WHITE;
         perspective_index <= BLACK;
         ++perspective_index) {
        const std::size_t p = static_cast<std::size_t>(perspective_index);
        const char* color = perspective_index == WHITE ? "white" : "black";
        output << "info string x2 compare " << color
            << " piece_formula " << formula_features.piece[p].size()
            << " piece_lut " << lut_features.piece[p].size()
            << " piece_first_diff "
            << first_vector_diff(
                formula_features.piece[p],
                lut_features.piece[p]
            )
            << '\n';
        output << "info string x2 compare " << color
            << " attack_formula " << formula_features.attack[p].size()
            << " attack_lut " << lut_features.attack[p].size()
            << " attack_first_diff "
            << first_vector_diff(
                formula_features.attack[p],
                lut_features.attack[p]
            )
            << '\n';
        output << "info string x2 compare " << color
            << " pawn_formula " << formula_features.pawn_pair[p].size()
            << " pawn_lut " << lut_features.pawn_pair[p].size()
            << " pawn_first_diff "
            << first_vector_diff(
                formula_features.pawn_pair[p],
                lut_features.pawn_pair[p]
            )
            << '\n';
    }
}

void debug_profile_evaluation(
    const Position& pos,
    const memory::Memory& mem,
    std::ostream& output
) {
    if (!loaded()) {
        output << "info string MNUE-X2-K16-pawn-Q8-A384 unavailable: "
            << last_error() << '\n';
        return;
    }

    const EvaluationProfile profile =
        profile_lut_evaluation(pos, active_rowadd_backend());
    output << "info string mnue x2k16 profile backend "
        << backend_name() << '\n';
    output << "info string mnue x2k16 profile raw "
        << profile.raw << '\n';
    output << "info string mnue x2k16 profile searchcp "
        << profile.searchcp << '\n';
    output << "info string mnue x2k16 profile evals 1\n";
    output << "info string mnue x2k16 profile total_eval_time_us "
        << profile.total_eval_time_us << '\n';
    output << "info string mnue x2k16 profile feature_gen_time_us "
        << profile.feature_gen_time_us << '\n';
    output << "info string mnue x2k16 profile piece_l0_time_us "
        << profile.piece_l0_time_us << '\n';
    output << "info string mnue x2k16 profile attack_l0_time_us "
        << profile.attack_l0_time_us << '\n';
    output << "info string mnue x2k16 profile pawn_l0_time_us "
        << profile.pawn_l0_time_us << '\n';
    output << "info string mnue x2k16 profile merge_activation_time_us "
        << profile.merge_activation_time_us << '\n';
    output << "info string mnue x2k16 profile head_time_us "
        << profile.head_time_us << '\n';
    output << "info string mnue x2k16 profile active_piece_rows "
        << profile.active_piece_rows << '\n';
    output << "info string mnue x2k16 profile active_attack_rows "
        << profile.active_attack_rows << '\n';
    output << "info string mnue x2k16 profile active_pawn_rows "
        << profile.active_pawn_rows << '\n';

    constexpr int WarmupIterations = 16;
    constexpr int BenchIterations = 128;
    int reference_checksum = 0;
    int scalar_lut_checksum = 0;
    int avx2_lut_checksum = 0;
    int incremental_checksum = 0;
    X2K16PawnAccumulator incremental_accumulator{};
    rebuild_piece_pawn_incremental(
        pos,
        incremental_accumulator,
        active_rowadd_backend()
    );

    for (int iteration = 0; iteration < WarmupIterations; ++iteration) {
        reference_checksum += evaluate_reference(pos, mem);
        scalar_lut_checksum += evaluate_lut_scalar(pos);
        avx2_lut_checksum += evaluate_lut_avx2(pos);
        incremental_checksum += evaluate_piece_pawn_incremental(
            pos,
            incremental_accumulator,
            active_rowadd_backend()
        );
    }

    const auto reference_begin = Clock::now();
    for (int iteration = 0; iteration < BenchIterations; ++iteration)
        reference_checksum += evaluate_reference(pos, mem);
    const auto reference_end = Clock::now();

    const auto scalar_lut_begin = Clock::now();
    for (int iteration = 0; iteration < BenchIterations; ++iteration)
        scalar_lut_checksum += evaluate_lut_scalar(pos);
    const auto scalar_lut_end = Clock::now();

    const auto avx2_lut_begin = Clock::now();
    for (int iteration = 0; iteration < BenchIterations; ++iteration)
        avx2_lut_checksum += evaluate_lut_avx2(pos);
    const auto avx2_lut_end = Clock::now();

    const auto incremental_begin = Clock::now();
    for (int iteration = 0; iteration < BenchIterations; ++iteration) {
        incremental_checksum += evaluate_piece_pawn_incremental(
            pos,
            incremental_accumulator,
            active_rowadd_backend()
        );
    }
    const auto incremental_end = Clock::now();

    const i64 reference_us = std::max<i64>(
        1,
        elapsed_us(reference_begin, reference_end)
    );
    const i64 scalar_lut_us = std::max<i64>(
        1,
        elapsed_us(scalar_lut_begin, scalar_lut_end)
    );
    const i64 avx2_lut_us = std::max<i64>(
        1,
        elapsed_us(avx2_lut_begin, avx2_lut_end)
    );
    const i64 incremental_us = std::max<i64>(
        1,
        elapsed_us(incremental_begin, incremental_end)
    );
    const double reference_eps =
        static_cast<double>(BenchIterations) * 1000000.0
        / static_cast<double>(reference_us);
    const double scalar_lut_eps =
        static_cast<double>(BenchIterations) * 1000000.0
        / static_cast<double>(scalar_lut_us);
    const double avx2_lut_eps =
        static_cast<double>(BenchIterations) * 1000000.0
        / static_cast<double>(avx2_lut_us);
    const double incremental_eps =
        static_cast<double>(BenchIterations) * 1000000.0
        / static_cast<double>(incremental_us);

    output << "info string mnue x2k16 profile warmup_iterations "
        << WarmupIterations << '\n';
    output << "info string mnue x2k16 profile bench_iterations "
        << BenchIterations << '\n';
    output << "info string mnue x2k16 profile avx2_compiled "
        << (avx2_rowadd_compiled() ? 1 : 0) << '\n';
    output << "info string mnue x2k16 profile oracle_total_time_us "
        << reference_us << '\n';
    output << "info string mnue x2k16 profile scalar_lut_total_time_us "
        << scalar_lut_us << '\n';
    output << "info string mnue x2k16 profile avx2_lut_total_time_us "
        << avx2_lut_us << '\n';
    output << "info string mnue x2k16 profile incremental_total_time_us "
        << incremental_us << '\n';
    output << "info string mnue x2k16 profile lut_total_time_us "
        << (active_rowadd_backend() == RowAddBackend::Avx2
                ? avx2_lut_us
                : scalar_lut_us)
        << '\n';
    output << "info string mnue x2k16 profile oracle_evals_per_sec "
        << static_cast<i64>(reference_eps) << '\n';
    output << "info string mnue x2k16 profile scalar_lut_evals_per_sec "
        << static_cast<i64>(scalar_lut_eps) << '\n';
    output << "info string mnue x2k16 profile avx2_lut_evals_per_sec "
        << static_cast<i64>(avx2_lut_eps) << '\n';
    output << "info string mnue x2k16 profile incremental_evals_per_sec "
        << static_cast<i64>(incremental_eps) << '\n';
    output << "info string mnue x2k16 profile lut_evals_per_sec "
        << static_cast<i64>(
            active_rowadd_backend() == RowAddBackend::Avx2
                ? avx2_lut_eps
                : scalar_lut_eps
        ) << '\n';
    output << "info string mnue x2k16 profile bench_raw_diff "
        << (scalar_lut_checksum - reference_checksum) << '\n';
    output << "info string mnue x2k16 profile bench_raw_diff_lut_avx2 "
        << (avx2_lut_checksum - scalar_lut_checksum) << '\n';
    output << "info string mnue x2k16 profile bench_raw_diff_full_inc "
        << (incremental_checksum - avx2_lut_checksum) << '\n';
}

void debug_stress_evaluation(
    const Position& pos,
    const memory::Memory& mem,
    int positions,
    std::ostream& output
) {
    if (!loaded()) {
        output << "info string MNUE-X2-K16-pawn-Q8-A384 unavailable: "
            << last_error() << '\n';
        return;
    }

    positions = std::clamp(positions, 1, 100000);
    Position work = pos;
    u32 random_state = 0x9E3779B9u
        ^ static_cast<u32>(pos.key)
        ^ static_cast<u32>(positions);

    int checked = 0;
    for (; checked < positions; ++checked) {
        const int scalar_raw = evaluate_lut_scalar(work);
        const int avx2_raw = evaluate_lut_avx2(work);
        if (scalar_raw != avx2_raw) {
            const FeatureLists lists = collect_features(work);
            output << "info string mnue x2k16 stress mismatch index "
                << checked << '\n';
            output << "info string mnue x2k16 stress fen "
                << position_to_fen(work) << '\n';
            output << "info string mnue x2k16 stress scalar_lut_raw "
                << scalar_raw << " avx2_lut_raw " << avx2_raw
                << " diff " << (avx2_raw - scalar_raw) << '\n';
            output << "info string mnue x2k16 stress active_piece_rows "
                << active_rows(lists.piece)
                << " active_attack_rows " << active_rows(lists.attack)
                << " active_pawn_rows " << active_rows(lists.pawn_pair)
                << '\n';
            return;
        }

        MoveList legal_moves{};
        generate_legal(work, mem, legal_moves);
        if (legal_moves.size <= 0) {
            work = pos;
            continue;
        }

        const int index = static_cast<int>(
            next_stress_random(random_state)
            % static_cast<u32>(legal_moves.size)
        );
        StateInfo state{};
        make_move(work, legal_moves.moves[index], mem.tables, state);
    }

    output << "info string mnue x2k16 stress positions "
        << checked
        << " scalar_lut_equals_avx2 1"
        << " avx2_compiled " << (avx2_rowadd_compiled() ? 1 : 0)
        << '\n';
    output << "info string mnue x2k16 stress final_fen "
        << position_to_fen(work) << '\n';
}

void debug_incremental_compare(
    const Position& pos,
    const memory::Memory& mem,
    std::ostream& output
) {
    (void)mem;
    if (!loaded()) {
        output << "info string MNUE-X2-K16-pawn-Q8-A384 unavailable: "
            << last_error() << '\n';
        return;
    }

    const RowAddBackend backend = active_rowadd_backend();
    const int ref_raw = evaluate_reference(pos, mem);
    const int scalar_lut_raw = evaluate_lut_scalar(pos);
    const int avx2_full_raw = evaluate_lut_avx2(pos);

    X2K16PawnAccumulator full_accumulator{};
    reset_accumulator_lut(full_accumulator, pos, backend);

    X2K16PawnAccumulator incremental_accumulator{};
    rebuild_piece_pawn_incremental(pos, incremental_accumulator, backend);
    const int inc_raw = evaluate_piece_pawn_incremental(
        pos,
        incremental_accumulator,
        backend
    );

    const BranchChecksums full_checksums =
        branch_checksums(full_accumulator);
    const BranchChecksums inc_checksums =
        branch_checksums(incremental_accumulator);

    output << "info string x2 inc compare ref_raw " << ref_raw
        << " scalar_lut_raw " << scalar_lut_raw
        << " avx2_full_raw " << avx2_full_raw
        << " inc_raw " << inc_raw
        << " diff_full_inc " << (inc_raw - avx2_full_raw)
        << '\n';
    output << "info string x2 inc compare backend_full "
        << backend_name()
        << " backend_inc avx2-piece-pawn-inc"
        << " avx2_compiled " << (avx2_rowadd_compiled() ? 1 : 0)
        << '\n';
    output << "info string x2 inc compare checksum_piece_full "
        << full_checksums.piece
        << " checksum_piece_inc " << inc_checksums.piece
        << " equal " << (full_checksums.piece == inc_checksums.piece ? 1 : 0)
        << '\n';
    output << "info string x2 inc compare checksum_pawn_full "
        << full_checksums.pawn
        << " checksum_pawn_inc " << inc_checksums.pawn
        << " equal " << (full_checksums.pawn == inc_checksums.pawn ? 1 : 0)
        << '\n';
    output << "info string x2 inc compare checksum_attack_full "
        << full_checksums.attack
        << " checksum_attack_inc " << inc_checksums.attack
        << " equal " << (full_checksums.attack == inc_checksums.attack ? 1 : 0)
        << '\n';
}

void debug_incremental_stress(
    const Position& pos,
    const memory::Memory& mem,
    int positions,
    std::ostream& output
) {
    if (!loaded()) {
        output << "info string MNUE-X2-K16-pawn-Q8-A384 unavailable: "
            << last_error() << '\n';
        return;
    }

    positions = std::clamp(positions, 1, 100000);
    const RowAddBackend backend = active_rowadd_backend();
    Position work = pos;
    X2K16PawnAccumulator incremental_accumulator{};
    rebuild_piece_pawn_incremental(work, incremental_accumulator, backend);
    u32 random_state = 0xC2B2AE35u
        ^ static_cast<u32>(pos.key >> 32)
        ^ static_cast<u32>(positions);
    std::string last_move = "root";

    int checked = 0;
    for (; checked < positions; ++checked) {
        const int full_raw = evaluate_lut_avx2(work);
        const int inc_raw = evaluate_piece_pawn_incremental(
            work,
            incremental_accumulator,
            backend
        );

        X2K16PawnAccumulator full_accumulator{};
        reset_accumulator_lut(full_accumulator, work, backend);
        const BranchChecksums full_checksums =
            branch_checksums(full_accumulator);
        const BranchChecksums inc_checksums =
            branch_checksums(incremental_accumulator);
        const bool checksums_equal =
            full_checksums.piece == inc_checksums.piece
            && full_checksums.pawn == inc_checksums.pawn
            && full_checksums.attack == inc_checksums.attack;

        if (full_raw != inc_raw || !checksums_equal) {
            const FeatureLists lists = collect_features(work);
            output << "info string x2 incstress mismatch index "
                << checked << '\n';
            output << "info string x2 incstress fen "
                << position_to_fen(work) << '\n';
            output << "info string x2 incstress last_move "
                << last_move << '\n';
            output << "info string x2 incstress full_raw "
                << full_raw << " incremental_raw " << inc_raw
                << " diff " << (inc_raw - full_raw) << '\n';
            output << "info string x2 incstress checksum_piece_full "
                << full_checksums.piece
                << " checksum_piece_inc " << inc_checksums.piece
                << " equal "
                << (full_checksums.piece == inc_checksums.piece ? 1 : 0)
                << '\n';
            output << "info string x2 incstress checksum_pawn_full "
                << full_checksums.pawn
                << " checksum_pawn_inc " << inc_checksums.pawn
                << " equal "
                << (full_checksums.pawn == inc_checksums.pawn ? 1 : 0)
                << '\n';
            output << "info string x2 incstress checksum_attack_full "
                << full_checksums.attack
                << " checksum_attack_inc " << inc_checksums.attack
                << " equal "
                << (full_checksums.attack == inc_checksums.attack ? 1 : 0)
                << '\n';
            output << "info string x2 incstress active_piece_rows "
                << active_rows(lists.piece)
                << " active_attack_rows " << active_rows(lists.attack)
                << " active_pawn_rows " << active_rows(lists.pawn_pair)
                << '\n';
            return;
        }

        MoveList legal_moves{};
        generate_legal(work, mem, legal_moves);
        if (legal_moves.size <= 0) {
            work = pos;
            rebuild_piece_pawn_incremental(
                work,
                incremental_accumulator,
                backend
            );
            last_move = "reset";
            continue;
        }

        const int index = static_cast<int>(
            next_stress_random(random_state)
            % static_cast<u32>(legal_moves.size)
        );
        const Move move = legal_moves.moves[index];
        const Position before = work;
        StateInfo state{};
        make_move(work, move, mem.tables, state);
        update_piece_pawn_incremental(
            incremental_accumulator,
            before,
            work,
            move,
            state,
            backend
        );
        last_move = move_to_simple_uci(move);
    }

    output << "info string x2 incstress positions "
        << checked
        << " full_equals_incremental 1"
        << " avx2_compiled " << (avx2_rowadd_compiled() ? 1 : 0)
        << '\n';
    output << "info string x2 incstress backend_full "
        << backend_name()
        << " backend_inc avx2-piece-pawn-inc"
        << '\n';
    output << "info string x2 incstress final_fen "
        << position_to_fen(work) << '\n';
}

void debug_dump_network(std::ostream& output) {
    output << "info string MNUE-X2-K16-pawn-Q8-A384 loaded "
        << (loaded() ? 1 : 0) << '\n';
    if (!loaded()) {
        output << "info string MNUE-X2-K16-pawn-Q8-A384 error "
            << last_error() << '\n';
        return;
    }

    output << "info string loaded MNUE-X2-K16-pawn-Q8-A384 "
        << path() << '\n';
    output << "info string MNUE-X2-K16-pawn-Q8-A384 backend: "
        << backend_name() << '\n';
    output << "info string MNUE-X2-K16-pawn-Q8-A384 oracle: scalar-full-rebuild\n";
    output << "info string architecture X2-K16-pawn-Q8-A384 version "
        << kVersion << " arch " << Layout::ArchId
        << " feature_version " << Layout::FeatureVersion << '\n';
    output << "info string inputs Piece " << Layout::PieceInputSize
        << " AttackEdge " << Layout::AttackInputSize
        << " PawnPair " << Layout::PawnPairInputSize << '\n';
    output << "info string hidden Piece " << Layout::PieceHiddenSize
        << " Attack " << Layout::AttackHiddenSize
        << " PawnPair " << Layout::PawnPairHiddenSize
        << " Merged " << Layout::MergedHiddenSize << '\n';
    output << "info string head " << Layout::HeadInputSize << 'x'
        << Layout::L1Size << 'x' << Layout::L2Size
        << "x1 buckets " << Layout::OutputBuckets << '\n';
    output << "info string scales Eval " << kScale
        << " QA " << kQa
        << " PieceQ " << kPieceQuant
        << " PieceRescale " << kPieceRescale
        << " AttackQ " << kAttackQuant
        << " AttackRescale " << kAttackRescale
        << " PawnPairQ " << kPawnPairQuant
        << " PawnPairRescale " << kPawnPairRescale
        << " L1Q " << kL1Quant << '\n';
    output << "info string payload bytes " << expected_payload_bytes()
        << " file bytes " << expected_file_bytes() << '\n';
    output << "info string network size "
        << static_cast<double>(network_bytes()) / 1048576.0
        << " MiB bytes " << network_bytes() << '\n';
    output << "info string k16_bucket white_e1 "
        << king_bucket16(relative_square(WHITE, 4, false))
        << " white_a1 "
        << king_bucket16(relative_square(WHITE, 0, false))
        << " white_h1 "
        << king_bucket16(relative_square(WHITE, 7, false))
        << " black_e8 "
        << king_bucket16(relative_square(BLACK, 60, false))
        << " black_a8 "
        << king_bucket16(relative_square(BLACK, 56, false))
        << " black_h8 "
        << king_bucket16(relative_square(BLACK, 63, false))
        << '\n';
}

void debug_dump_features(const Position& pos, std::ostream& output) {
    const FeatureLists lists = collect_features(pos);
    output << "info string mnue_x2_k16_pawn_q8_a384_features\n";
    output << "info string output_bucket " << output_bucket(pos) << '\n';
    dump_vector(output, "white", "piece", lists.piece[WHITE]);
    dump_vector(output, "black", "piece", lists.piece[BLACK]);
    dump_vector(output, "white", "attack", lists.attack[WHITE]);
    dump_vector(output, "black", "attack", lists.attack[BLACK]);
    dump_vector(output, "white", "pawn_pair", lists.pawn_pair[WHITE]);
    dump_vector(output, "black", "pawn_pair", lists.pawn_pair[BLACK]);
}

} // namespace magnus::mnue::x2k16
