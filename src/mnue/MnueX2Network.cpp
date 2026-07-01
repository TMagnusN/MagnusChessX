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

#include "mnue/MnueX2Network.h"

#include "Memory.h"
#include "Nnue.h"
#include "board/Position.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace magnus::mnue::x2 {
namespace {

using i16 = std::int16_t;
using i32 = std::int32_t;
using u32 = std::uint32_t;

constexpr u32 kMagic = 0x45554E4Du;
constexpr u32 kVersion = 3;
constexpr u32 kHeaderBytes = 80;
constexpr u32 kScale = 400;
constexpr u32 kQa = 255;
constexpr u32 kAttackQuant = 64;
constexpr u32 kAttackRescale = 4;
constexpr u32 kL1Quant = 64;
constexpr double kUciWdlAs[] = {
    4.44037236, -27.44028449, 69.36512228, 175.98749706
};
constexpr double kUciWdlBs[] = {
    -2.09838237, 15.76765588, -39.56299152, 90.47624591
};

struct FileHeader {
    u32 magic;
    u32 version;
    u32 arch;
    u32 header_bytes;
    u32 piece_input_size;
    u32 attack_input_size;
    u32 hidden_size;
    u32 input_buckets;
    u32 output_buckets;
    u32 l1_size;
    u32 l2_size;
    u32 scale;
    u32 qa;
    u32 attack_quant;
    u32 attack_rescale;
    u32 l1_quant;
    u32 feature_version;
    u32 flags;
    u32 reserved0;
    u32 reserved1;
};

static_assert(sizeof(FileHeader) == kHeaderBytes);

struct Network {
    bool is_loaded = false;
    bool headered = false;
    int scale = static_cast<int>(kScale);
    std::size_t file_bytes = 0;
    std::string file_path{};
    std::string error{};

    std::vector<i16> piece_l0w{};
    std::vector<std::int8_t> attack_l0w{};
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
                    * Layout::HiddenSize
            && attack_l0w.size()
                == static_cast<std::size_t>(Layout::AttackInputSize)
                    * Layout::HiddenSize
            && l0b.size() == static_cast<std::size_t>(Layout::HiddenSize)
            && l1w.size()
                == static_cast<std::size_t>(Layout::OutputBuckets)
                    * Layout::L1Size * Layout::HiddenSize
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

[[nodiscard]] constexpr std::uintmax_t expected_payload_bytes() noexcept {
    return static_cast<std::uintmax_t>(Layout::PieceInputSize)
            * Layout::HiddenSize * sizeof(i16)
        + static_cast<std::uintmax_t>(Layout::AttackInputSize)
            * Layout::HiddenSize * sizeof(std::int8_t)
        + static_cast<std::uintmax_t>(Layout::HiddenSize) * sizeof(i16)
        + static_cast<std::uintmax_t>(Layout::OutputBuckets)
            * Layout::L1Size * Layout::HiddenSize * sizeof(std::int8_t)
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

[[nodiscard]] constexpr std::uintmax_t expected_file_bytes() noexcept {
    return kHeaderBytes + expected_payload_bytes();
}

static_assert(expected_file_bytes() == 81072240);

void clear(Network& network) noexcept {
    network = {};
}

[[nodiscard]] int game_ply(const Position& pos) noexcept {
    const int fullmove = std::max(1, pos.fullmove_number);
    return (fullmove - 1) * 2 + (pos.side_to_move == BLACK ? 1 : 0);
}

[[nodiscard]] nnue::WdlTriplet uci_wdl_from_cp(
    int cp,
    const Position& pos
) noexcept {
    const double m =
        static_cast<double>(std::min(240, game_ply(pos))) / 64.0;
    const double a =
        (((kUciWdlAs[0] * m + kUciWdlAs[1]) * m + kUciWdlAs[2]) * m)
        + kUciWdlAs[3];
    const double b =
        (((kUciWdlBs[0] * m + kUciWdlBs[1]) * m + kUciWdlBs[2]) * m)
        + kUciWdlBs[3];

    const double x = std::clamp(static_cast<double>(cp), -2000.0, 2000.0);
    const double win = 1.0 / (1.0 + std::exp((a - x) / b));
    const double loss = 1.0 / (1.0 + std::exp((a + x) / b));
    const double draw = 1.0 - win - loss;

    return {
        .win = static_cast<int>(std::round(1000.0 * win)),
        .draw = static_cast<int>(std::round(1000.0 * draw)),
        .loss = static_cast<int>(std::round(1000.0 * loss))
    };
}

[[nodiscard]] bool header_matches(const FileHeader& header) noexcept {
    return header.magic == kMagic
        && header.version == kVersion
        && header.arch == Layout::ArchId
        && header.header_bytes == kHeaderBytes
        && header.piece_input_size == Layout::PieceInputSize
        && header.attack_input_size == Layout::AttackInputSize
        && header.hidden_size == Layout::HiddenSize
        && header.input_buckets == Layout::InputBuckets
        && header.output_buckets == Layout::OutputBuckets
        && header.l1_size == Layout::L1Size
        && header.l2_size == Layout::L2Size
        && header.scale == kScale
        && header.qa == kQa
        && header.attack_quant == kAttackQuant
        && header.attack_rescale == kAttackRescale
        && header.l1_quant == kL1Quant
        && header.feature_version == Layout::FeatureVersion
        && header.flags == 0
        && header.reserved0 == 0
        && header.reserved1 == 0;
}

void resize_tensors(Network& network) {
    network.piece_l0w.resize(
        static_cast<std::size_t>(Layout::PieceInputSize) * Layout::HiddenSize
    );
    network.attack_l0w.resize(
        static_cast<std::size_t>(Layout::AttackInputSize) * Layout::HiddenSize
    );
    network.l0b.resize(Layout::HiddenSize);
    network.l1w.resize(
        static_cast<std::size_t>(Layout::OutputBuckets)
            * Layout::L1Size * Layout::HiddenSize
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
        && read_exact(input, network.l0b.data(), network.l0b.size())
        && read_exact(input, network.l1w.data(), network.l1w.size())
        && read_exact(input, network.l1b.data(), network.l1b.size())
        && read_exact(input, network.l2w.data(), network.l2w.size())
        && read_exact(input, network.l2b.data(), network.l2b.size())
        && read_exact(input, network.l3w.data(), network.l3w.size())
        && read_exact(input, network.l3b.data(), network.l3b.size());
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

[[nodiscard]] constexpr int king_bucket10(Square relative_king) noexcept {
    const int file = std::min(
        file_of_sq(relative_king),
        7 - file_of_sq(relative_king)
    );
    switch (rank_of_sq(relative_king)) {
    case 0: return file;
    case 1: return 4 + file;
    case 2:
    case 3: return 8;
    default: return 9;
    }
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

[[nodiscard]] std::size_t attack_feature_index(
    int attacker_color,
    PieceType attacker_type,
    Square attacker_square,
    int victim_color,
    PieceType victim_type,
    Square victim_square
) noexcept {
    const Bitboard empty_attacks = empty_board_attacks(
        attacker_color,
        attacker_type,
        attacker_square
    );
    const Bitboard before =
        victim_square == 0 ? 0ULL : ((1ULL << victim_square) - 1ULL);
    const std::size_t target_slot = static_cast<std::size_t>(
        std::popcount(static_cast<std::uint64_t>(empty_attacks & before))
    );
    const std::size_t victim_class =
        static_cast<std::size_t>(victim_color) * Layout::PieceTypes
        + static_cast<std::size_t>(victim_type);

    return attack_index_table()
            .base[static_cast<std::size_t>(attacker_color)]
                 [static_cast<std::size_t>(attacker_type)]
                 [static_cast<std::size_t>(attacker_square)]
        + target_slot * Layout::VictimClasses
        + victim_class;
}

using Accumulator = std::array<i32, Layout::HiddenSize>;
using Pairwise = std::array<std::uint8_t, Layout::HiddenSize / 2>;
using BackendInput = std::array<std::uint8_t, Layout::HiddenSize>;
using Hidden1 = std::array<float, Layout::L1Size>;
using Hidden2 = std::array<float, Layout::L2Size>;

void add_piece_row(Accumulator& accumulator, std::size_t feature) noexcept {
    const i16* row =
        g_network.piece_l0w.data() + feature * Layout::HiddenSize;
    for (int column = 0; column < Layout::HiddenSize; ++column)
        accumulator[static_cast<std::size_t>(column)] += row[column];
}

void add_attack_row(Accumulator& accumulator, std::size_t feature) noexcept {
    const std::int8_t* row =
        g_network.attack_l0w.data() + feature * Layout::HiddenSize;
    for (int column = 0; column < Layout::HiddenSize; ++column) {
        accumulator[static_cast<std::size_t>(column)] +=
            static_cast<i32>(row[column])
            * static_cast<i32>(kAttackRescale);
    }
}

void rebuild_accumulator(
    const Position& pos,
    Color perspective,
    Accumulator& accumulator
) noexcept {
    std::copy(g_network.l0b.begin(), g_network.l0b.end(), accumulator.begin());

    const Square own_king = relative_square(
        perspective,
        safe_king_square(pos, perspective),
        false
    );
    const bool mirror = mirror_for_king(own_king);
    const int bucket = king_bucket10(own_king);

    Bitboard occupied = pos.occupied;
    while (occupied != 0ULL) {
        const Square square = static_cast<Square>(
            std::countr_zero(static_cast<std::uint64_t>(occupied))
        );
        occupied &= occupied - 1;

        const Piece piece = piece_on(pos, square);
        const PieceType piece_type = type_of(piece);
        const int relative_color = color_of(piece) == perspective ? 0 : 1;
        const Square relative_sq =
            relative_square(perspective, square, mirror);
        add_piece_row(
            accumulator,
            static_cast<std::size_t>(piece_feature_index(
                bucket,
                relative_color,
                piece_type,
                relative_sq
            ))
        );
    }

    occupied = pos.occupied;
    while (occupied != 0ULL) {
        const Square from = static_cast<Square>(
            std::countr_zero(static_cast<std::uint64_t>(occupied))
        );
        occupied &= occupied - 1;

        const Piece attacker = piece_on(pos, from);
        const PieceType attacker_type = type_of(attacker);
        const int attacker_color =
            color_of(attacker) == perspective ? 0 : 1;
        const Square relative_from =
            relative_square(perspective, from, mirror);

        Bitboard targets = occupied_attacks(attacker, from, pos.occupied);
        while (targets != 0ULL) {
            const Square to = static_cast<Square>(
                std::countr_zero(static_cast<std::uint64_t>(targets))
            );
            targets &= targets - 1;

            const Piece victim = piece_on(pos, to);
            const int victim_color =
                color_of(victim) == perspective ? 0 : 1;
            const Square relative_to =
                relative_square(perspective, to, mirror);
            add_attack_row(
                accumulator,
                attack_feature_index(
                    attacker_color,
                    attacker_type,
                    relative_from,
                    victim_color,
                    type_of(victim),
                    relative_to
                )
            );
        }
    }
}

[[nodiscard]] constexpr std::uint8_t pairwise_value(
    i32 left,
    i32 right
) noexcept {
    left = std::clamp<i32>(left, 0, static_cast<i32>(kQa));
    right = std::clamp<i32>(right, 0, static_cast<i32>(kQa));
    return static_cast<std::uint8_t>(left * right / static_cast<i32>(kQa));
}

void pairwise_crelu(
    const Accumulator& accumulator,
    Pairwise& output
) noexcept {
    constexpr std::size_t Half = Layout::HiddenSize / 2;
    for (std::size_t index = 0; index < Half; ++index) {
        output[index] = pairwise_value(
            accumulator[index],
            accumulator[index + Half]
        );
    }
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
    const Accumulator& white_accumulator,
    const Accumulator& black_accumulator
) noexcept {
    Pairwise white_pairwise{};
    Pairwise black_pairwise{};
    pairwise_crelu(white_accumulator, white_pairwise);
    pairwise_crelu(black_accumulator, black_pairwise);

    const Color stm = static_cast<Color>(pos.side_to_move);
    const Pairwise& stm_pairwise =
        stm == WHITE ? white_pairwise : black_pairwise;
    const Pairwise& ntm_pairwise =
        stm == WHITE ? black_pairwise : white_pairwise;

    BackendInput backend_input{};
    constexpr std::size_t Half = Layout::HiddenSize / 2;
    std::copy(
        stm_pairwise.begin(),
        stm_pairwise.end(),
        backend_input.begin()
    );
    std::copy(
        ntm_pairwise.begin(),
        ntm_pairwise.end(),
        backend_input.begin() + Half
    );

    const std::size_t bucket =
        static_cast<std::size_t>(output_bucket(pos));

    Hidden1 hidden1{};
    const std::int8_t* l1_weights =
        g_network.l1w.data()
        + bucket * Layout::L1Size * Layout::HiddenSize;
    const float* l1_bias =
        g_network.l1b.data() + bucket * Layout::L1Size;
    constexpr float L1Scale =
        1.0F / (static_cast<float>(kQa) * static_cast<float>(kL1Quant));
    for (int row = 0; row < Layout::L1Size; ++row) {
        float sum = l1_bias[row];
        const std::int8_t* weights =
            l1_weights + static_cast<std::size_t>(row) * Layout::HiddenSize;
        for (int column = 0; column < Layout::HiddenSize; ++column) {
            const std::uint8_t input =
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
    const double scaled =
        output * static_cast<double>(g_network.scale);
    if (scaled >= static_cast<double>(std::numeric_limits<int>::max()))
        return std::numeric_limits<int>::max();
    if (scaled <= static_cast<double>(std::numeric_limits<int>::min()))
        return std::numeric_limits<int>::min();
    return static_cast<int>(std::llround(scaled));
}

} // namespace

bool load(const std::string& file_path) {
    clear(g_network);

    std::error_code error_code;
    const std::uintmax_t bytes =
        std::filesystem::file_size(file_path, error_code);
    if (error_code) {
        g_network.error = "could not determine MNUE-X2 file size";
        return false;
    }
    const bool headered = bytes == expected_file_bytes();
    const bool raw_payload =
        bytes >= expected_payload_bytes()
        && bytes - expected_payload_bytes() < 64;
    if (!headered && !raw_payload) {
        g_network.error =
            "MNUE-X2 file size mismatch: got " + std::to_string(bytes)
            + ", expected " + std::to_string(expected_file_bytes())
            + " headered or " + std::to_string(expected_payload_bytes())
            + " raw payload plus less than 64 padding bytes";
        return false;
    }

    std::ifstream input(file_path, std::ios::binary);
    if (!input) {
        g_network.error = "could not open MNUE-X2 file";
        return false;
    }

    if (headered) {
        FileHeader header{};
        if (!read_exact(input, &header, 1) || !header_matches(header)) {
            g_network.error = "MNUE-X2 header mismatch";
            return false;
        }
    }

    resize_tensors(g_network);

    if (!read_payload(input, g_network)) {
        clear(g_network);
        g_network.error = "truncated MNUE-X2 payload";
        return false;
    }

    if (!g_network.valid()) {
        clear(g_network);
        g_network.error = "invalid MNUE-X2 tensor dimensions";
        return false;
    }
    if (attack_index_table().size != Layout::AttackInputSize) {
        clear(g_network);
        g_network.error = "MNUE-X2 attack index table size mismatch";
        return false;
    }

    g_network.headered = headered;
    g_network.scale = static_cast<int>(kScale);
    g_network.file_bytes = static_cast<std::size_t>(bytes);
    g_network.file_path = file_path;
    g_network.is_loaded = true;
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

int evaluate_reference(
    const Position& pos,
    const memory::Memory& mem
) noexcept {
    (void)mem;
    if (!loaded())
        return 0;

    Accumulator white_accumulator{};
    Accumulator black_accumulator{};
    rebuild_accumulator(pos, WHITE, white_accumulator);
    rebuild_accumulator(pos, BLACK, black_accumulator);
    return score_from_output(
        forward(pos, white_accumulator, black_accumulator)
    );
}

int search_score(int v, const Position& pos) noexcept {
    (void)pos;
    return v;
}

int to_cp(int v, const Position& pos) noexcept {
    (void)pos;
    return v;
}

int search_score_to_cp(int score, const Position& pos) noexcept {
    return to_cp(score, pos);
}

nnue::WdlTriplet search_score_to_wdl(
    int score,
    const Position& pos
) noexcept {
    return uci_wdl_from_cp(search_score_to_cp(score, pos), pos);
}

void debug_dump_network(std::ostream& output) {
    output << "info string MNUE-X2 loaded " << (loaded() ? 1 : 0) << '\n';
    if (!loaded()) {
        output << "info string MNUE-X2 error " << last_error() << '\n';
        return;
    }

    output << "info string MNUE-X2 evaluation full-rebuild scalar\n";
    output << "info string MNUE-X2 network using " << path() << '\n';
    output << "info string MNUE-X2 source "
        << (g_network.headered ? "headered" : "raw-quantised-payload")
        << '\n';
    output << "info string architecture X2 version " << kVersion
        << " arch " << Layout::ArchId
        << " feature_version " << Layout::FeatureVersion << '\n';
    output << "info string inputs Piece " << Layout::PieceInputSize
        << " AttackEdge " << Layout::AttackInputSize
        << " total " << Layout::InputSize
        << " max_active " << Layout::MaxActive << '\n';
    output << "info string transformer Piece "
        << Layout::PieceInputSize << 'x' << Layout::HiddenSize
        << " int16 AttackEdge "
        << Layout::AttackInputSize << 'x' << Layout::HiddenSize
        << " int8 bias " << Layout::HiddenSize << " int16\n";
    output << "info string head " << Layout::HiddenSize << 'x'
        << Layout::L1Size << 'x' << Layout::L2Size
        << "x1 buckets " << Layout::OutputBuckets
        << " l1 int8 tail float\n";
    output << "info string scales Score " << g_network.scale
        << " PieceQA " << kQa
        << " Attack " << kAttackQuant
        << " AttackRescale " << kAttackRescale
        << " L1 " << kL1Quant << '\n';
    output << "info string network size "
        << static_cast<double>(network_bytes()) / 1048576.0
        << " MiB bytes " << network_bytes() << '\n';
}

} // namespace magnus::mnue::x2
