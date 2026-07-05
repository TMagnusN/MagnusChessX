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

/* ===== ANNOTATED: 繁體中文註釋已添加 =====
 * 本檔案是 MagnusChessX Thinking 西洋棋引擎的一部分。
 * 詳細說明請參閱對應的 .cpp 實作檔案。
 */


#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "Memory.h"

namespace magnus {

/*
Attack generation is split into leaper lookups and slider backends. The public
API hides whether bishops and rooks are served by classical scans, dense tables,
magic multiplication, or BMI2/PEXT indexing.
*/

using AttackBitboard = memory::Bitboard;
using AttackColor    = memory::Color;
using AttackKey      = memory::Key;

constexpr int ATTACK_SQ_NB    = memory::SQ_NB;
constexpr int ATTACK_FILE_NB  = memory::FILE_NB;
constexpr int ATTACK_RANK_NB  = memory::RANK_NB;
constexpr int ATTACK_COLOR_NB = memory::COLOR_NB;
constexpr int ATTACK_PIECE_NB = memory::PIECE_NB;

// Small wrappers keep the attack layer independent from the main Types names.
constexpr AttackBitboard attack_bb_of(int sq) noexcept {
    return 1ULL << sq;
}

constexpr int attack_file_of(int sq) noexcept {
    return sq & 7;
}

constexpr int attack_rank_of(int sq) noexcept {
    return sq >> 3;
}

constexpr bool attack_on_board(int sq) noexcept {
    return sq >= 0 && sq < 64;
}

inline AttackBitboard pawn_attacks(
    const memory::Memory& mem,
    AttackColor c,
    int sq
) noexcept {
    return mem.tables.pawn_attacks[c][sq];
}

inline AttackBitboard knight_attacks(
    const memory::Memory& mem,
    int sq
) noexcept {
    return mem.tables.knight_attacks[sq];
}

inline AttackBitboard king_attacks(
    const memory::Memory& mem,
    int sq
) noexcept {
    return mem.tables.king_attacks[sq];
}

inline AttackBitboard between_bb(
    const memory::Memory& mem,
    int a,
    int b
) noexcept {
    return mem.tables.between[a][b];
}

inline AttackBitboard line_bb(
    const memory::Memory& mem,
    int a,
    int b
) noexcept {
    return mem.tables.line[a][b];
}

inline std::uint8_t chebyshev_distance(
    const memory::Memory& mem,
    int a,
    int b
) noexcept {
    return mem.tables.chebyshev[a][b];
}

inline std::uint8_t manhattan_distance(
    const memory::Memory& mem,
    int a,
    int b
) noexcept {
    return mem.tables.manhattan[a][b];
}

/*
 * 攻擊生成後端 — AttackBackendKind：CLASSICAL(掃描) / TABLE(稠密查表) / MAGIC / PEXT(BMI2)
 * SliderAttackEntry：滑子攻擊表的元數據（遮罩/偏移/相關位元/移位）
 * 跳子/騎士/國王攻擊直接從 Tables 查表（O(1)）
 * 主教/城堡攻擊根據後端選擇：經典掃描 / 稠密索引表 / PEXT 硬體加速
 */
enum class AttackBackendKind : int {
    CLASSICAL = 0,
    TABLE     = 1,
    MAGIC     = 2,
    PEXT      = 3
};

struct SliderAttackEntry {
    // Lookup metadata for one bishop or rook source square.
    AttackBitboard mask = 0ULL;
    AttackBitboard magic = 0ULL;
    std::uint32_t offset = 0;
    std::uint8_t relevant_bits = 0;
    std::uint8_t shift = 0;
};

void attack_init_backend(memory::Memory& mem) noexcept;
void attack_set_backend(AttackBackendKind kind) noexcept;
void attack_auto_select_backend() noexcept;
bool attack_select_backend(std::string_view name) noexcept;
AttackBackendKind attack_backend_kind() noexcept;
const char* attack_backend_name() noexcept;
bool attack_backend_uses_slider_tables() noexcept;
bool attack_backend_pext_supported() noexcept;

const SliderAttackEntry& bishop_slider_entry(int sq) noexcept;
const SliderAttackEntry& rook_slider_entry(int sq) noexcept;
std::size_t bishop_slider_table_size() noexcept;
std::size_t rook_slider_table_size() noexcept;
const AttackBitboard* bishop_slider_table_data() noexcept;
const AttackBitboard* rook_slider_table_data() noexcept;

AttackBitboard bishop_attacks(
    const memory::Memory& mem,
    int sq,
    AttackBitboard occupied
) noexcept;

AttackBitboard rook_attacks(
    const memory::Memory& mem,
    int sq,
    AttackBitboard occupied
) noexcept;

AttackBitboard bishop_attacks_fast(
    const memory::Memory& mem,
    int sq,
    AttackBitboard occupied
) noexcept;

AttackBitboard rook_attacks_fast(
    const memory::Memory& mem,
    int sq,
    AttackBitboard occupied
) noexcept;

inline AttackBitboard queen_attacks_fast(
    const memory::Memory& mem,
    int sq,
    AttackBitboard occupied
) noexcept {
    return bishop_attacks_fast(mem, sq, occupied)
         | rook_attacks_fast(mem, sq, occupied);
}

inline AttackBitboard queen_attacks(
    const memory::Memory& mem,
    int sq,
    AttackBitboard occupied
) noexcept {
    return bishop_attacks(mem, sq, occupied) | rook_attacks(mem, sq, occupied);
}

AttackBitboard bishop_rays(int sq) noexcept;
AttackBitboard rook_rays(int sq) noexcept;

inline AttackBitboard queen_rays(int sq) noexcept {
    return bishop_rays(sq) | rook_rays(sq);
}

AttackBitboard bishop_xray_attacks(
    const memory::Memory& mem,
    int sq,
    AttackBitboard occupied,
    AttackBitboard blockers
) noexcept;

AttackBitboard rook_xray_attacks(
    const memory::Memory& mem,
    int sq,
    AttackBitboard occupied,
    AttackBitboard blockers
) noexcept;

inline AttackBitboard queen_xray_attacks(
    const memory::Memory& mem,
    int sq,
    AttackBitboard occupied,
    AttackBitboard blockers
) noexcept {
    return bishop_xray_attacks(mem, sq, occupied, blockers)
         | rook_xray_attacks(mem, sq, occupied, blockers);
}

bool same_file(int a, int b) noexcept;
bool same_rank(int a, int b) noexcept;
bool same_diagonal(int a, int b) noexcept;
bool aligned(int a, int b) noexcept;

inline bool square_in_mask(int sq, AttackBitboard mask) noexcept {
    return (mask & attack_bb_of(sq)) != 0;
}

} // namespace magnus
