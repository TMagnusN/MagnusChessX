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

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

#include "Types.h"
#include "Tables.h"
#include "board/Position.h"

namespace magnus::memory {

/*
The transposition table is organized as fixed-size 64-byte clusters with four
lanes each. That layout keeps probe/store traffic cache friendly while still
allowing a simple depth-and-age replacement policy.
*/

using ::magnus::u8;
using ::magnus::u16;
using ::magnus::u32;
using ::magnus::u64;
using ::magnus::i16;
using ::magnus::Move;
using ::magnus::Key;
using ::magnus::mix64;

enum Bound : u8 {
    BOUND_NONE  = 0,
    BOUND_UPPER = 1,
    BOUND_LOWER = 2,
    BOUND_EXACT = 3
};

struct TTData {
    // Compact logical view of one TT entry.
    u32 tag32 = 0;
    u16 move  = 0;
    i16 score = 0;
    i16 eval  = 0;
    i16 depth = 0;
    u8  age   = 0;
    u8  flags = 0;
    u16 spare = 0;
};

struct alignas(64) TTCluster {
    // Physical storage for four packed TT entries inside one cache line.
    u32 tag32[4]{};
    u16 move[4]{};
    i16 score[4]{};
    i16 eval[4]{};
    i16 depth[4]{};
    u8  age[4]{};
    u8  flags[4]{};
    u16 spare[4]{};
};

static_assert(sizeof(TTCluster) == 64, "TTCluster must be 64 bytes.");

struct TT {
    // Global TT allocation plus search-generation bookkeeping.
    TTCluster* clusters = nullptr;
    std::size_t cluster_count = 0;
    std::size_t cluster_mask = 0;

    std::atomic_flag* locks = nullptr;
    std::size_t lock_count = 0;
    std::size_t lock_mask = 0;

    u8 generation = 1;
};

struct TTSlotRef {
    TTCluster* cluster = nullptr;
    int lane = 0;
};

struct TTProbe {
    bool hit = false;
    TTSlotRef slot{};
    TTData data{};
};

// Low-level cluster helpers and public probe/save API.
[[nodiscard]] TTData tt_cluster_load(const TTCluster& c, int lane) noexcept;
void tt_cluster_store(TTCluster& c, int lane, const TTData& d) noexcept;
void tt_cluster_clear(TTCluster& c) noexcept;

[[nodiscard]] int tt_replacement_score(const TTCluster& c, int lane, u8 current_age) noexcept;

void tt_free(TT& tt) noexcept;
void tt_clear(TT& tt) noexcept;
void tt_resize_mb(TT& tt, std::size_t mb);
void tt_new_search(TT& tt) noexcept;

[[nodiscard]] std::size_t tt_index(const TT& tt, Key key) noexcept;
[[nodiscard]] constexpr int rule50_bucket(int halfmove_clock) noexcept {
    return halfmove_clock < 50
        ? 0
        : std::min(15, (halfmove_clock - 8) / 8);
}

static_assert(rule50_bucket(0) == 0);
static_assert(rule50_bucket(49) == 0);
static_assert(rule50_bucket(50) == 5);
static_assert(rule50_bucket(58) == 6);
static_assert(rule50_bucket(66) == 7);
static_assert(rule50_bucket(200) == 15);

[[nodiscard]] inline Key tt_key(const Position& pos, const Tables& tables) noexcept {
    return pos.halfmove_clock < 50
        ? pos.key
        : pos.key ^ tables.zobrist.rule50[rule50_bucket(pos.halfmove_clock)];
}

void tt_prefetch(const TT& tt, Key key) noexcept;
[[nodiscard]] TTProbe tt_probe(TT& tt, Key key) noexcept;

void tt_save(
    TT& tt,
    Key key,
    Move move,
    i16 score,
    i16 eval,
    i16 depth,
    Bound bound,
    bool pv
) noexcept;

[[nodiscard]] int tt_hashfull(const TT& tt, int max_age = 0) noexcept;

} // namespace magnus::memory
