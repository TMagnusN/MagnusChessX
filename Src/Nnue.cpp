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

#include "Nnue.h"
#include "board/Position.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <immintrin.h>
#include <string>

/*
 * NNUE 評估器實作 — Efficiently Updatable Neural Network
 *
 * 架構：Chess768 雙視角 → 128 隱藏層神經元 → 1 輸出
 *
 * 核心組件：
 *   1. 增量累加器更新
 *      - on_piece_added/removed/moved 鉤子
 *      - 每次 make/unmake 僅更新 O(隱藏層) 而非 O(輸入×隱藏層)
 *      - AVX2 SIMD 加速（每次處理 16 個神經元）
 *
 *   2. 前向傳播 (forward pass)
 *      - Clipped ReLU: clamp(x, 0, 255)
 *      - 平方激活: screlu(x) = clamp(x, 0, 255)^2
 *      - AVX2 點積 (forward_dot_avx2)
 *      - 標量後備路徑（無 AVX2 時）
 *
 *   3. 分數轉換
 *      - CP 查表 (build_cp_lookup_table): 基於材料的多項式模型
 *      - 勝率模型 (win_rate_model): Sigmoid 函數
 *      - WDL 三元組 (uci_wdl_from_cp): 勝/和/負千分比
 *
 *   4. 檔案載入
 *      - 原生 .nnue 格式（含標頭 magic + version + dimensions + scale）
 *      - Bullet quantised .bin 格式（Rust simple.rs 輸出，無標頭）
 *
 * 網路狀態儲存在 NativeNetwork (g_net) 中，為全域單例。
 * 累加器儲存在 Position 結構體中（每個視角一個）。
 */
namespace magnus::nnue {

namespace {

using i16 = std::int16_t;
using i32 = std::int32_t;
using u32 = std::uint32_t;

constexpr u32 kMagic   = 0x554E4E56u; // "VNNU" in little-endian file bytes
constexpr u32 kVersion = 1;

constexpr int kInputs = kInputSize;
constexpr int kHidden = kHiddenSize;
constexpr int kClip   = kActivationClip;
constexpr int kDefaultScale = 400;

struct FileHeader {
    u32 magic;
    u32 version;
    u32 input_size;
    u32 hidden_size;
    i32 scale;
};

struct NativeNetwork {
    bool is_loaded = false;
    bool is_bullet_simple = false;
    std::string loaded_path;
    std::string desc;
    std::string error;

    i32 scale = kDefaultScale;

    std::array<i16, kInputs * kHidden> w0{};
    std::array<i16, kHidden> b0{};

    std::array<i16, 2 * kHidden> w1{};
    i16 b1 = 0;
};

NativeNetwork g_net{};
std::atomic<u32> g_generation{1};
std::atomic<bool> g_loading{false};

constexpr int kWinRateMaterialMin = 17;
constexpr int kWinRateMaterialMax = 78;
constexpr int kCpLookupMaxRaw = 32768;
constexpr int kMaterialBucketCount = kWinRateMaterialMax - kWinRateMaterialMin + 1;
constexpr double kWinRateAs[] = {
    -72.32565836, 185.93832038, -144.58862193, 416.44950446
};
constexpr double kWinRateBs[] = {
    83.86794042, -136.06112997, 69.98820887, 47.62901433
};
constexpr double kUciWdlAs[] = {
    4.44037236, -27.44028449, 69.36512228, 175.98749706
};
constexpr double kUciWdlBs[] = {
    -2.09838237, 15.76765588, -39.56299152, 90.47624591
};
using CpLookupRow = std::array<i16, kCpLookupMaxRaw + 1>;
using CpLookupTable = std::array<CpLookupRow, kMaterialBucketCount>;

[[nodiscard]] CpLookupTable build_cp_lookup_table() {
    CpLookupTable lut{};
    for (int mat_idx = 0; mat_idx < kMaterialBucketCount; ++mat_idx) {
        const int material = kWinRateMaterialMin + mat_idx;
        const double m = static_cast<double>(material) / 58.0;
        const double a =
            (((kWinRateAs[0] * m + kWinRateAs[1]) * m + kWinRateAs[2]) * m)
            + kWinRateAs[3];
        // Precompute reciprocal scale to move the division out of the inner loop.
        const float scale = 100.0f / static_cast<float>(a);
        auto& row = lut[static_cast<std::size_t>(mat_idx)];
        for (int raw = 0; raw <= kCpLookupMaxRaw; ++raw) {
            row[static_cast<std::size_t>(raw)] = static_cast<i16>(
                static_cast<float>(raw) * scale + 0.5f
            );
        }
    }
    return lut;
}

// Meyer's Singleton avoids the Static Initialization Order Fiasco.
[[nodiscard]] const CpLookupTable& cp_lookup_table() noexcept {
    static const CpLookupTable table = build_cp_lookup_table();
    return table;
}

template<typename T>
[[nodiscard]] T read_le(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

[[nodiscard]] inline Color opposite(Color c) noexcept {
    return c == WHITE ? BLACK : WHITE;
}

[[nodiscard]] inline int game_ply(const Position& pos) noexcept {
    const int fullmove = std::max(1, pos.fullmove_number);
    return (fullmove - 1) * 2 + (pos.side_to_move == BLACK ? 1 : 0);
}

[[nodiscard]] inline WdlTriplet uci_wdl_from_cp(
    int cp,
    const Position& pos
) noexcept {
    // The WDL model expects displayed UCI cp units on its normalized x-axis.
    const double m = static_cast<double>(std::min(240, game_ply(pos))) / 64.0;
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

[[nodiscard]] inline Square flip_vertical_sq(Square sq) noexcept {
    return static_cast<Square>(static_cast<int>(sq) ^ 56);
}

[[nodiscard]] inline Square orient_square(Square sq, Color persp) noexcept {
    return persp == WHITE ? sq : flip_vertical_sq(sq);
}

// Maps PAWN..KING -> 0..5.
// Adjust this switch if your engine uses different enum values.
[[nodiscard]] inline int piece_plane(PieceType pt) noexcept {
    switch (pt) {
        case PAWN:   return 0;
        case KNIGHT: return 1;
        case BISHOP: return 2;
        case ROOK:   return 3;
        case QUEEN:  return 4;
        case KING:   return 5;
        default:     return -1;
    }
}

[[nodiscard]] inline int chess768_index(Color persp, Piece pc, Square sq) noexcept {
    const PieceType pt = type_of(pc);
    const int plane = piece_plane(pt);
    if (plane < 0)
        return -1;

    const Color pc_color = color_of(pc);
    const int color_plane = (pc_color == persp) ? 0 : 1;
    const Square osq = orient_square(sq, persp);

    return (color_plane * 6 + plane) * 64 + static_cast<int>(osq);
}

[[nodiscard]] inline bool accumulator_matches(const Position& pos) noexcept {
    return pos.nnue_acc_valid && pos.nnue_generation == g_generation.load(std::memory_order_relaxed);
}

[[nodiscard]] inline int win_rate_material(const Position& pos) noexcept {
    return std::clamp(
        magnus::non_king_material(pos),
        kWinRateMaterialMin,
        kWinRateMaterialMax
    );
}

[[nodiscard]] inline int lookup_cp(int raw, int material) noexcept {
    const auto& row =
        cp_lookup_table()[static_cast<std::size_t>(material - kWinRateMaterialMin)];
    const i64 abs_raw = raw >= 0 ? static_cast<i64>(raw) : -static_cast<i64>(raw);
    const int index = static_cast<int>(std::min<i64>(abs_raw, kCpLookupMaxRaw));
    const int cp = static_cast<int>(row[static_cast<std::size_t>(index)]);
    return raw >= 0 ? cp : -cp;
}

inline void invalidate_accumulator(Position& pos) noexcept {
    pos.nnue_acc_valid = false;
    pos.nnue_generation = g_generation.load(std::memory_order_relaxed);
}

#if defined(__AVX2__)
inline void apply_feature_delta_avx2(
    std::array<i16, kHidden>& acc,
    const i16* weights,
    bool add
) noexcept {
    i16* acc_ptr = acc.data();
    for (int i = 0; i < kHidden; i += 16) {
        __m256i acc_vec = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc_ptr + i));
        __m256i weight_vec = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights + i));

        if (add) {
            acc_vec = _mm256_add_epi16(acc_vec, weight_vec);
        } else {
            acc_vec = _mm256_sub_epi16(acc_vec, weight_vec);
        }

        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(acc_ptr + i), acc_vec);
    }
}

inline void apply_feature_move_delta_avx2(
    std::array<i16, kHidden>& acc,
    const i16* add_weights,
    const i16* sub_weights
) noexcept {
    i16* acc_ptr = acc.data();
    for (int i = 0; i < kHidden; i += 16) {
        __m256i acc_vec = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc_ptr + i));
        __m256i add_vec = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(add_weights + i));
        __m256i sub_vec = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(sub_weights + i));

        acc_vec = _mm256_add_epi16(acc_vec, _mm256_sub_epi16(add_vec, sub_vec));

        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(acc_ptr + i), acc_vec);
    }
}

#endif

inline void apply_feature_delta(
    std::array<i16, kHidden>& acc,
    int feature_index,
    int delta
) noexcept {
    const i16* w = &g_net.w0[static_cast<std::size_t>(feature_index) * kHidden];
#if defined(__AVX2__)
    apply_feature_delta_avx2(acc, w, delta > 0);
    return;
#endif
    if (delta > 0) {
        for (int i = 0; i < kHidden; ++i)
            acc[i] = static_cast<i16>(static_cast<i32>(acc[i]) + static_cast<i32>(w[i]));
        return;
    }

    for (int i = 0; i < kHidden; ++i)
        acc[i] = static_cast<i16>(static_cast<i32>(acc[i]) - static_cast<i32>(w[i]));
}

inline void apply_feature_move_delta(
    std::array<i16, kHidden>& acc,
    int add_feature_index,
    int sub_feature_index
) noexcept {
    const i16* add = &g_net.w0[static_cast<std::size_t>(add_feature_index) * kHidden];
    const i16* sub = &g_net.w0[static_cast<std::size_t>(sub_feature_index) * kHidden];
#if defined(__AVX2__)
    apply_feature_move_delta_avx2(acc, add, sub);
    return;
#endif
    for (int i = 0; i < kHidden; ++i)
        acc[i] = static_cast<i16>(
            static_cast<i32>(acc[i]) +
            static_cast<i32>(add[i]) -
            static_cast<i32>(sub[i])
        );
}

inline void apply_piece_delta(
    Position& pos,
    Color piece_color,
    PieceType piece_type,
    Square sq,
    int delta
) noexcept {
    const Piece pc = make_piece(piece_color, piece_type);
    for (int persp = WHITE; persp <= BLACK; ++persp) {
        const int idx = chess768_index(static_cast<Color>(persp), pc, sq);
        if (idx >= 0)
            apply_feature_delta(pos.nnue_acc[persp], idx, delta);
    }
}

inline void apply_piece_move_delta(
    Position& pos,
    Color piece_color,
    PieceType piece_type,
    Square from,
    Square to
) noexcept {
    const Piece pc = make_piece(piece_color, piece_type);
    for (int persp = WHITE; persp <= BLACK; ++persp) {
        const Color perspective = static_cast<Color>(persp);
        const int sub_idx = chess768_index(perspective, pc, from);
        const int add_idx = chess768_index(perspective, pc, to);
        if (sub_idx >= 0 && add_idx >= 0) {
            apply_feature_move_delta(pos.nnue_acc[persp], add_idx, sub_idx);
            continue;
        }

        if (sub_idx >= 0)
            apply_feature_delta(pos.nnue_acc[persp], sub_idx, -1);
        if (add_idx >= 0)
            apply_feature_delta(pos.nnue_acc[persp], add_idx, 1);
    }
}

void rebuild_accumulator(const Position& pos) noexcept {
    for (int persp = WHITE; persp <= BLACK; ++persp)
        pos.nnue_acc[persp] = g_net.b0;

    Bitboard bb = pieces(pos);
    while (bb) {
        const Square sq =
            static_cast<Square>(std::countr_zero(static_cast<std::uint64_t>(bb)));
        bb &= (bb - 1);

        const Piece pc = piece_on(pos, sq);
        if (pc == PIECE_NONE)
            continue;

        for (int persp = WHITE; persp <= BLACK; ++persp) {
            const int idx = chess768_index(static_cast<Color>(persp), pc, sq);
            if (idx < 0)
                continue;

            apply_feature_delta(pos.nnue_acc[persp], idx, 1);
        }
    }

    pos.nnue_generation = g_generation.load(std::memory_order_relaxed);
    pos.nnue_acc_valid = true;
}

inline void ensure_accumulator(const Position& pos) noexcept {
    if (!accumulator_matches(pos))
        rebuild_accumulator(pos);
}

void clear_network() noexcept {
    u32 next_gen = g_generation.load(std::memory_order_relaxed) + 1;
    if (next_gen == 0) next_gen = 1;
    g_generation.store(next_gen, std::memory_order_release);

    g_net.is_loaded = false;
    g_net.loaded_path.clear();
    g_net.desc.clear();
    g_net.error.clear();
    g_net.scale = kDefaultScale;
    g_net.is_bullet_simple = false;
    g_net.w0.fill(0);
    g_net.b0.fill(0);
    g_net.w1.fill(0);
    g_net.b1 = 0;
}

[[nodiscard]] inline i32 screlu(i32 x) noexcept {
    const i32 y = std::clamp(x, 0, kClip);
    return y * y;
}

#if defined(__AVX2__)
[[nodiscard]] inline i32 horizontal_sum_epi32_avx2(__m256i v) noexcept {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return _mm_cvtsi128_si32(sum128);
}

[[nodiscard]] inline __m256i clipped_square_epi32_avx2(
    __m128i v16
) noexcept {
    const __m256i v32 = _mm256_cvtepi16_epi32(v16);
    return _mm256_mullo_epi32(v32, v32);
}

[[nodiscard]] inline __m256i load_weights_epi32_avx2(
    const i16* weights
) noexcept {
    __m128i w16;
    std::memcpy(&w16, weights, sizeof(__m128i));
    return _mm256_cvtepi16_epi32(w16);
}

[[nodiscard]] inline i32 forward_dot_avx2(
    const std::array<i16, kHidden>& us,
    const std::array<i16, kHidden>& them
) noexcept {
    const i16* us_ptr = us.data();
    const i16* them_ptr = them.data();
    const i16* w_us_ptr = g_net.w1.data();
    const i16* w_them_ptr = g_net.w1.data() + kHidden;

    const __m256i vzero = _mm256_setzero_si256();
    const __m256i vclip = _mm256_set1_epi16(static_cast<i16>(kClip));
    __m256i sum_us = _mm256_setzero_si256();
    __m256i sum_them = _mm256_setzero_si256();

    int i = 0;
    for (; i + 15 < kHidden; i += 16) {
        __m256i us16;
        std::memcpy(&us16, us_ptr + i, sizeof(__m256i));
        __m256i them16;
        std::memcpy(&them16, them_ptr + i, sizeof(__m256i));

        const __m256i clipped_us =
            _mm256_min_epi16(_mm256_max_epi16(us16, vzero), vclip);
        const __m256i clipped_them =
            _mm256_min_epi16(_mm256_max_epi16(them16, vzero), vclip);

        const __m128i us_lo = _mm256_castsi256_si128(clipped_us);
        const __m128i us_hi = _mm256_extracti128_si256(clipped_us, 1);
        const __m128i them_lo = _mm256_castsi256_si128(clipped_them);
        const __m128i them_hi = _mm256_extracti128_si256(clipped_them, 1);

        const __m256i sq_us_lo = clipped_square_epi32_avx2(us_lo);
        const __m256i sq_us_hi = clipped_square_epi32_avx2(us_hi);
        const __m256i sq_them_lo = clipped_square_epi32_avx2(them_lo);
        const __m256i sq_them_hi = clipped_square_epi32_avx2(them_hi);

        const __m256i w_us_lo = load_weights_epi32_avx2(w_us_ptr + i);
        const __m256i w_us_hi = load_weights_epi32_avx2(w_us_ptr + i + 8);
        const __m256i w_them_lo = load_weights_epi32_avx2(w_them_ptr + i);
        const __m256i w_them_hi = load_weights_epi32_avx2(w_them_ptr + i + 8);

        sum_us = _mm256_add_epi32(sum_us, _mm256_mullo_epi32(w_us_lo, sq_us_lo));
        sum_us = _mm256_add_epi32(sum_us, _mm256_mullo_epi32(w_us_hi, sq_us_hi));
        sum_them = _mm256_add_epi32(sum_them, _mm256_mullo_epi32(w_them_lo, sq_them_lo));
        sum_them = _mm256_add_epi32(sum_them, _mm256_mullo_epi32(w_them_hi, sq_them_hi));
    }

    i32 sum = horizontal_sum_epi32_avx2(_mm256_add_epi32(sum_us, sum_them));
    // kHidden (=128) is an exact multiple of 16, so the AVX2 loop covers every element.
    static_assert(kHidden % 16 == 0, "kHidden must be a multiple of 16 for the AVX2 path");

    return sum;
}
#endif

[[nodiscard]] int forward(const Position& pos) noexcept {
    const Color stm = static_cast<Color>(pos.side_to_move);
    const Color nstm = opposite(stm);
    ensure_accumulator(pos);
    const auto& us = pos.nnue_acc[stm];
    const auto& them = pos.nnue_acc[nstm];
    i32 sum = 0;

#if defined(__AVX2__)
    sum = forward_dot_avx2(us, them);
#else
    {
        u32 acc = 0;
        for (int i = 0; i < kHidden; ++i) {
            const u32 wu = static_cast<u32>(static_cast<i32>(g_net.w1[i]));
            const u32 wt = static_cast<u32>(static_cast<i32>(g_net.w1[kHidden + i]));
            acc += wu * static_cast<u32>(screlu(static_cast<i32>(us[i])));
            acc += wt * static_cast<u32>(screlu(static_cast<i32>(them[i])));
        }
        sum = static_cast<i32>(acc);
    }
#endif

    sum /= kClip;                     // divide by QA
    sum += static_cast<i32>(g_net.b1);
    sum *= g_net.scale;              // multiply by SCALE
    sum /= (kClip * 64);             // divide by QA * QB

    return static_cast<int>(sum);
}

bool load_bullet_simple_quantised(const std::string& path) {
    unload();

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        g_net.error = "cannot open NNUE file: " + path;
        return false;
    }

    // Rust layout for simple.rs quantised.bin:
    // feature_weights: [Accumulator; 768]  where Accumulator = [i16; 128], align 64
    // feature_bias:    [i16; 128]
    // output_weights:  [i16; 256]
    // output_bias:     i16
    // trailing padding to struct alignment (64 bytes)

    in.read(reinterpret_cast<char*>(g_net.w0.data()), sizeof(g_net.w0));
    in.read(reinterpret_cast<char*>(g_net.b0.data()), sizeof(g_net.b0));
    in.read(reinterpret_cast<char*>(g_net.w1.data()), sizeof(g_net.w1));
    in.read(reinterpret_cast<char*>(&g_net.b1), sizeof(g_net.b1));

    if (!in) {
        clear_network();
        g_net.error = "truncated Bullet quantised.bin";
        return false;
    }

    // ignore any trailing alignment padding
    g_net.scale = kDefaultScale;
    g_net.is_loaded = true;
    g_net.is_bullet_simple = true;
    g_net.loaded_path = path;
    g_net.desc = "Bullet simple quantised NNUE (Chess768 dual-perspective 128x1)";
    return true;
}

} // namespace

bool load(const std::string& path) {
    g_loading.store(true, std::memory_order_release);
    unload();

    if (path.size() >= 4 && path.substr(path.size() - 4) == ".bin") {
        bool ok = load_bullet_simple_quantised(path);
        g_loading.store(false, std::memory_order_release);
        return ok;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        g_net.error = "cannot open NNUE file: " + path;
        g_loading.store(false, std::memory_order_release);
        return false;
    }

    const FileHeader h{
        .magic = read_le<u32>(in),
        .version = read_le<u32>(in),
        .input_size = read_le<u32>(in),
        .hidden_size = read_le<u32>(in),
        .scale = read_le<i32>(in)
    };

    if (!in) {
        g_net.error = "failed to read NNUE header";
        g_loading.store(false, std::memory_order_release);
        return false;
    }

    if (h.magic != kMagic) {
        g_net.error = "bad NNUE magic";
        g_loading.store(false, std::memory_order_release);
        return false;
    }

    if (h.version != kVersion) {
        g_net.error = "unsupported NNUE version";
        g_loading.store(false, std::memory_order_release);
        return false;
    }

    if (h.input_size != kInputs || h.hidden_size != kHidden) {
        g_net.error = "network dimensions mismatch";
        g_loading.store(false, std::memory_order_release);
        return false;
    }

    g_net.scale = h.scale > 0 ? h.scale : kDefaultScale;

    in.read(reinterpret_cast<char*>(g_net.w0.data()), sizeof(g_net.w0));
    in.read(reinterpret_cast<char*>(g_net.b0.data()), sizeof(g_net.b0));
    in.read(reinterpret_cast<char*>(g_net.w1.data()), sizeof(g_net.w1));
    in.read(reinterpret_cast<char*>(&g_net.b1), sizeof(g_net.b1));

    if (!in) {
        clear_network();
        g_net.error = "truncated NNUE file";
        g_loading.store(false, std::memory_order_release);
        return false;
    }

    g_net.is_loaded = true;
    g_net.is_bullet_simple = false;
    g_net.loaded_path = path;
    g_net.desc = "Magnus native NNUE v1 (Chess768 dual-perspective 128x1)";
    g_loading.store(false, std::memory_order_release);
    return true;
}

void unload() noexcept {
    clear_network();
}

bool loaded() noexcept {
    return g_net.is_loaded;
}

const std::string& path() noexcept {
    return g_net.loaded_path;
}

const std::string& description() noexcept {
    return g_net.desc;
}

const std::string& last_error() noexcept {
    return g_net.error;
}

int eval(const Position& pos) noexcept {
    if (!g_net.is_loaded || g_loading.load(std::memory_order_acquire))
        return 0; // sentinel: caller must check loaded() before calling eval()
    return forward(pos);
}

void on_position_cleared(Position& pos) noexcept {
    invalidate_accumulator(pos);
}

void on_piece_added(
    Position& pos,
    Color color,
    PieceType piece_type,
    Square sq
) noexcept {
    if (!g_net.is_loaded || !accumulator_matches(pos))
        return;

    apply_piece_delta(pos, color, piece_type, sq, 1);
}

void on_piece_removed(
    Position& pos,
    Color color,
    PieceType piece_type,
    Square sq
) noexcept {
    if (!g_net.is_loaded || !accumulator_matches(pos))
        return;

    apply_piece_delta(pos, color, piece_type, sq, -1);
}

void on_piece_moved(
    Position& pos,
    Color color,
    PieceType piece_type,
    Square from,
    Square to
) noexcept {
    if (!g_net.is_loaded || !accumulator_matches(pos))
        return;

    apply_piece_move_delta(pos, color, piece_type, from, to);
}

WinRateParams win_rate_params(const Position& pos) noexcept {
    const double m = static_cast<double>(win_rate_material(pos)) / 58.0;
    const double a =
        (((kWinRateAs[0] * m + kWinRateAs[1]) * m + kWinRateAs[2]) * m)
        + kWinRateAs[3];
    const double b =
        (((kWinRateBs[0] * m + kWinRateBs[1]) * m + kWinRateBs[2]) * m)
        + kWinRateBs[3];

    return {a, b};
}

int to_cp(int v, const Position& pos) noexcept {
    return lookup_cp(v, win_rate_material(pos));
}

int win_rate_model(int v, const Position& pos) noexcept {
    auto [a, b] = win_rate_params(pos);

    if (std::abs(b) < 1e-9)
        return v >= static_cast<int>(a) ? 1000 : 0;

    return static_cast<int>(
        0.5 + 1000.0 / (1.0 + std::exp((a - static_cast<double>(v)) / b))
    );
}

int search_score(int v, const Position& pos) noexcept {
    (void)pos;
    return v;
}

int search_score_to_winrate(int score, const Position& pos) noexcept {
    return search_score_to_wdl(score, pos).win;
}

WdlTriplet search_score_to_wdl(int score, const Position& pos) noexcept {
    return uci_wdl_from_cp(search_score_to_cp(score, pos), pos);
}

int search_score_to_cp(int score, const Position& pos) noexcept {
    return to_cp(score, pos);
}

} // namespace magnus::nnue
