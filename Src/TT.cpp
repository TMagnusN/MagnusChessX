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

#include "TT.h"

#include <bit>
#include <cstring>

/*
The TT implementation focuses on a cheap probe path: compare four 32-bit tags
at once, prefer empty lanes when possible, and otherwise replace the weakest
entry based on depth, age, and bound quality.
*/

/* ===== 繁體中文註釋 =====
 * 本檔案是 MagnusChessX Thinking 西洋棋引擎的一部分。
 * 實作詳情請參閱對應的 .h 標頭檔案。
 */

namespace magnus::memory {

namespace {

[[nodiscard]] inline u32 tt_tag32_from_key(Key key) noexcept {
    return static_cast<u32>(key >> 32);
}

template <typename T>
[[nodiscard]] inline T tt_atomic_load(
    const T& value,
    std::memory_order order = std::memory_order_relaxed
) noexcept {
    return std::atomic_ref<T>(const_cast<T&>(value)).load(order);
}

template <typename T>
inline void tt_atomic_store(
    T& value,
    T desired,
    std::memory_order order = std::memory_order_relaxed
) noexcept {
    std::atomic_ref<T>(value).store(desired, order);
}

[[nodiscard]] inline int lane_mask4_eq_u32_sse(const u32* ptr, u32 tag32) noexcept {
    int mask = 0;
    for (int lane = 0; lane < 4; ++lane) {
        if (tt_atomic_load(ptr[lane], std::memory_order_acquire) == tag32)
            mask |= (1 << lane);
    }
    return mask;
}

[[nodiscard]] inline int first_lane_from_mask4(int mask) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(mask);
#else
    int lane = 0;
    while (((mask >> lane) & 1) == 0) ++lane;
    return lane;
#endif
}

[[nodiscard]] inline int empty_lane_mask4_u8(const u8* ages) noexcept {
    int mask = 0;
    for (int lane = 0; lane < 4; ++lane) {
        if (tt_atomic_load(ages[lane], std::memory_order_relaxed) == 0)
            mask |= (1 << lane);
    }
    return mask;
}

[[nodiscard]] inline int first_live_match_lane4(
    const TTCluster& c,
    int tag_mask
) noexcept {
    if ((tag_mask & 0x1) && tt_atomic_load(c.age[0], std::memory_order_relaxed) != 0) return 0;
    if ((tag_mask & 0x2) && tt_atomic_load(c.age[1], std::memory_order_relaxed) != 0) return 1;
    if ((tag_mask & 0x4) && tt_atomic_load(c.age[2], std::memory_order_relaxed) != 0) return 2;
    if ((tag_mask & 0x8) && tt_atomic_load(c.age[3], std::memory_order_relaxed) != 0) return 3;
    return -1;
}

[[nodiscard]] inline int replacement_score_lane(
    const TTCluster& c,
    int lane,
    u8 current_age
) noexcept {
    // Deeper, newer, exact, and PV entries are more expensive to overwrite.
    const u8 lane_age = tt_atomic_load(c.age[lane], std::memory_order_relaxed);
    const u8 lane_flags = tt_atomic_load(c.flags[lane], std::memory_order_relaxed);
    const i16 lane_depth = tt_atomic_load(c.depth[lane], std::memory_order_relaxed);
    const int age_penalty = static_cast<int>(static_cast<u8>(current_age - lane_age));
    const int exact_bonus = ((lane_flags & 0x3U) == BOUND_EXACT) ? 8 : 0;
    const int pv_bonus    = (lane_flags & 0x4U) ? 4 : 0;

    return static_cast<int>(lane_depth) - age_penalty * 2 + exact_bonus + pv_bonus;
}

[[nodiscard]] inline int best_replacement_lane4(
    const TTCluster& c,
    u8 current_age
) noexcept {
    int best_lane = 0;
    int best_score = replacement_score_lane(c, 0, current_age);

    const int s1 = replacement_score_lane(c, 1, current_age);
    if (s1 < best_score) {
        best_score = s1;
        best_lane = 1;
    }

    const int s2 = replacement_score_lane(c, 2, current_age);
    if (s2 < best_score) {
        best_score = s2;
        best_lane = 2;
    }

    const int s3 = replacement_score_lane(c, 3, current_age);
    if (s3 < best_score) {
        best_score = s3;
        best_lane = 3;
    }

    return best_lane;
}

[[nodiscard]] inline int tt_relative_age(u8 current_age, u8 entry_age) noexcept {
    if (entry_age == 0)
        return 256;

    return current_age >= entry_age
        ? static_cast<int>(current_age - entry_age)
        : static_cast<int>(current_age) + 254 - static_cast<int>(entry_age);
}

} // namespace

TTData tt_cluster_load(const TTCluster& c, int lane) noexcept {
    TTData d;
    d.tag32 = tt_atomic_load(c.tag32[lane], std::memory_order_acquire);
    d.move  = tt_atomic_load(c.move[lane], std::memory_order_relaxed);
    d.score = tt_atomic_load(c.score[lane], std::memory_order_relaxed);
    d.eval  = tt_atomic_load(c.eval[lane], std::memory_order_relaxed);
    d.depth = tt_atomic_load(c.depth[lane], std::memory_order_relaxed);
    d.age   = tt_atomic_load(c.age[lane], std::memory_order_relaxed);
    d.flags = tt_atomic_load(c.flags[lane], std::memory_order_relaxed);
    d.spare = tt_atomic_load(c.spare[lane], std::memory_order_relaxed);
    return d;
}

void tt_cluster_store(TTCluster& c, int lane, const TTData& d) noexcept {
    // Publish by tag: invalidate the slot, write payload, then publish tag.
    tt_atomic_store(c.tag32[lane], 0U, std::memory_order_relaxed);
    tt_atomic_store(c.move[lane], d.move, std::memory_order_relaxed);
    tt_atomic_store(c.score[lane], d.score, std::memory_order_relaxed);
    tt_atomic_store(c.eval[lane], d.eval, std::memory_order_relaxed);
    tt_atomic_store(c.depth[lane], d.depth, std::memory_order_relaxed);
    tt_atomic_store(c.age[lane], d.age, std::memory_order_relaxed);
    tt_atomic_store(c.flags[lane], d.flags, std::memory_order_relaxed);
    tt_atomic_store(c.spare[lane], d.spare, std::memory_order_relaxed);
    tt_atomic_store(c.tag32[lane], d.tag32, std::memory_order_release);
}

/*
 * 置換表實作
 * tt_cluster_clear() — 清空單個快取行（所有 4 個條目歸零）
 * tt_resize_mb() — 調整 TT 大小（分配新記憶體、釋放舊的）
 * tt_new_search() — 遞增世代編號（使舊條目 age 過期）
 * tt_save() — 儲存條目（選擇最佳替換位置，寫入所有欄位）
 * tt_probe() — 探測 TT（比較 4 個 tag32，回傳命中條目）
 */
void tt_cluster_clear(TTCluster& c) noexcept {
    for (int lane = 0; lane < 4; ++lane) {
        tt_atomic_store(c.tag32[lane], 0U, std::memory_order_relaxed);
        tt_atomic_store(c.move[lane], static_cast<u16>(0), std::memory_order_relaxed);
        tt_atomic_store(c.score[lane], static_cast<i16>(0), std::memory_order_relaxed);
        tt_atomic_store(c.eval[lane], static_cast<i16>(0), std::memory_order_relaxed);
        tt_atomic_store(c.depth[lane], static_cast<i16>(0), std::memory_order_relaxed);
        tt_atomic_store(c.age[lane], static_cast<u8>(0), std::memory_order_relaxed);
        tt_atomic_store(c.flags[lane], static_cast<u8>(0), std::memory_order_relaxed);
        tt_atomic_store(c.spare[lane], static_cast<u16>(0), std::memory_order_relaxed);
    }
}

int tt_replacement_score(const TTCluster& c, int lane, u8 current_age) noexcept {
    if (tt_atomic_load(c.age[lane], std::memory_order_relaxed) == 0)
        return -1000000000;

    return replacement_score_lane(c, lane, current_age);
}

void tt_free(TT& tt) noexcept {
    delete[] tt.clusters;
    delete[] tt.locks;
    tt.clusters = nullptr;
    tt.cluster_count = 0;
    tt.cluster_mask = 0;
    tt.locks = nullptr;
    tt.lock_count = 0;
    tt.lock_mask = 0;
    tt.generation = 1;
}

void tt_clear(TT& tt) noexcept {
    if (!tt.clusters) return;
    for (std::size_t i = 0; i < tt.cluster_count; ++i)
        tt_cluster_clear(tt.clusters[i]);
    tt.generation = 1;
}

void tt_resize_mb(TT& tt, std::size_t mb) {
    std::size_t bytes = mb * 1024ULL * 1024ULL;
    std::size_t count = bytes / sizeof(TTCluster);
    if (count == 0) count = 1;
    count = std::bit_ceil(count);

    TTCluster* new_clusters = new TTCluster[count]{};
    TTCluster* old_clusters = tt.clusters;

    tt.clusters = new_clusters;
    tt.cluster_count = count;
    tt.cluster_mask = count - 1;
    tt.generation = 1;

    delete[] old_clusters;
}

void tt_new_search(TT& tt) noexcept {
    if (tt.generation >= 254)
        tt.generation = 1;
    else
        ++tt.generation;
}

std::size_t tt_index(const TT& tt, Key key) noexcept {
    return mix64(key) & tt.cluster_mask;
}

void tt_prefetch(const TT& tt, Key key) noexcept {
    if (!tt.clusters) return;
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(&tt.clusters[tt_index(tt, key)], 0, 3);
#elif defined(_MSC_VER)
    _mm_prefetch(reinterpret_cast<const char*>(&tt.clusters[tt_index(tt, key)]), _MM_HINT_T0);
#endif
}

TTProbe tt_probe(TT& tt, Key key) noexcept {
    // Probe returns both the hit data (if any) and the slot that should be
    // updated on save, so the caller only hashes once.
    TTProbe res{};
    if (!tt.clusters) return res;

    TTCluster& c = tt.clusters[tt_index(tt, key)];
    const u32 tag32 = tt_tag32_from_key(key);

    const int tag_mask = lane_mask4_eq_u32_sse(c.tag32, tag32);
    if (tag_mask) {
        for (int lane = 0; lane < 4; ++lane) {
            if (((tag_mask >> lane) & 1) == 0)
                continue;

            if (tt_atomic_load(c.tag32[lane], std::memory_order_acquire) != tag32)
                continue;

            TTData data = tt_cluster_load(c, lane);
            if (data.age == 0)
                continue;

            if (tt_atomic_load(c.tag32[lane], std::memory_order_acquire) != tag32)
                continue;

            res.hit = true;
            res.slot.cluster = &c;
            res.slot.lane = lane;
            res.data = data;
            return res;
        }
    }

    const int empty_mask = empty_lane_mask4_u8(c.age);
    if (empty_mask) {
        const int lane = first_lane_from_mask4(empty_mask);
        res.slot.cluster = &c;
        res.slot.lane = lane;
        return res;
    }

    const int lane = best_replacement_lane4(c, tt.generation);
    res.slot.cluster = &c;
    res.slot.lane = lane;
    return res;
}

void tt_save(
    TT& tt,
    Key key,
    Move move,
    i16 score,
    i16 eval,
    i16 depth,
    Bound bound,
    bool pv
) noexcept {
    // Replacement is conservative when the existing entry is deeper and more exact.
    if (!tt.clusters) return;

    TTProbe pr = tt_probe(tt, key);
    TTData old = pr.hit ? pr.data : TTData{};

    if (move == 0 && pr.hit)
        move = old.move;

    if (pr.hit) {
        const Bound old_bound = static_cast<Bound>(old.flags & 0x3U);
        const bool old_pv = (old.flags & 0x4U) != 0;

        const bool weaker =
            depth < old.depth &&
            bound != BOUND_EXACT &&
            old_bound == BOUND_EXACT &&
            !pv &&
            old_pv;

        if (weaker) return;
    }

    TTData nw;
    nw.tag32 = tt_tag32_from_key(key);
    nw.move  = move;
    nw.score = score;
    nw.eval  = eval;
    nw.depth = depth;
    nw.age   = tt.generation;
    nw.flags = static_cast<u8>(bound) | (pv ? 0x4U : 0U);
    nw.spare = 0;

    tt_cluster_store(*pr.slot.cluster, pr.slot.lane, nw);
}

int tt_hashfull(const TT& tt, int max_age) noexcept {
    if (!tt.clusters || tt.cluster_count == 0) return 0;

    constexpr int STOCKFISH_SAMPLE_CLUSTERS = 1000;
    const int n = static_cast<int>(std::min<std::size_t>(
        tt.cluster_count,
        static_cast<std::size_t>(STOCKFISH_SAMPLE_CLUSTERS)
    ));

    if (n == 0)
        return 0;

    int used = 0;

    for (int i = 0; i < n; ++i) {
        const TTCluster& c = tt.clusters[i];
        for (int lane = 0; lane < 4; ++lane) {
            const u8 age = tt_atomic_load(c.age[lane], std::memory_order_relaxed);
            used += age != 0
                && tt_relative_age(tt.generation, age) <= max_age;
        }
    }

    return (used * 1000) / (n * 4);
}

} // namespace magnus::memory
