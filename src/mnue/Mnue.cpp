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

#include "Mnue.h"
#include "board/Position.h"
#include "board/MoveGen.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <limits>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

// Symbols provided by mnue/MnueEmbedded.S — the embedded MNUE P2/P2Pro blob in .rodata.
// Weak: resolve to nullptr when the object is not linked (e.g. non-GCC builds).
extern "C" const char mnue_p2_embedded_data[] __attribute__((weak));
extern "C" const char mnue_p2_embedded_end[]   __attribute__((weak));

namespace magnus::mnue {
namespace {

using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using u32 = std::uint32_t;

constexpr u32 kMnueMagic = 0x45554E4Du; // "MNUE" in little-endian file bytes
constexpr u32 kMnueVersion = 1;
constexpr int kQa = 255;
constexpr int kQb = 64;
constexpr int kScale = 400;
constexpr int kClip = 255;
constexpr int kMnueMaterialMax = 78;
constexpr int kMnueCpMaxRaw = 32768;
// Keep opening scores conservative: typical start-position search values of
// 35-45 MNUE raw units should display near 20-30 cp.
constexpr double kMnueCpBase = 144.0;
constexpr double kMnueCpMaterial = 16.0;
constexpr double kMnueCpEndgameDiscount = 8.0;
constexpr double kMnueCpMinDenominator = 128.0;
constexpr double kMnueCpMaxDenominator = 168.0;

template<class Layout>
struct Network {
    bool loaded = false;
    int scale = kScale;
    std::string path;
    std::string error;

    // Row-major by feature: w0[feature][hidden]. This matches the existing
    // Chess768 loader/inference convention and keeps feature delta contiguous.
    std::vector<i16> w0;
    std::vector<i16> b0;

    // w1[bucket][perspective][hidden], perspective 0 = stm/us, 1 = nstm/them.
    std::vector<i16> w1;
    std::vector<i16> b1;
    int w1_max_abs = 0;
    bool forward_madd_safe = false;
    bool forward_i32_safe = false;

    [[nodiscard]] bool valid() const noexcept {
        return w0.size() == static_cast<std::size_t>(Layout::InputSize) * Layout::HiddenSize
            && b0.size() == static_cast<std::size_t>(Layout::HiddenSize)
            && w1.size() == static_cast<std::size_t>(Layout::OutputBuckets) * 2 * Layout::HiddenSize
            && b1.size() == static_cast<std::size_t>(Layout::OutputBuckets);
    }
};

struct FileHeader {
    u32 magic;
    u32 version;
    u32 arch;
    u32 input_size;
    u32 hidden_size;
    u32 input_buckets;
    u32 output_buckets;
    i32 scale;
    i32 qa;
    i32 qb;
};

static_assert(sizeof(FileHeader) == 40);

Network<P2Layout> g_p2;
Network<P2ProLayout> g_p2pro;
Network<P4Layout> g_p4;
std::string g_error;
std::atomic<u32> g_p2_generation{1};

enum class P2ActiveKind : std::uint8_t {
    None,
    P2,
    P2Pro
};

P2ActiveKind g_p2_active = P2ActiveKind::None;
const std::string kEmptyPath;

[[nodiscard]] const char* p2_kind_eval_name(P2ActiveKind kind) noexcept {
    switch (kind) {
        case P2ActiveKind::P2:    return "mnue-p2";
        case P2ActiveKind::P2Pro: return "mnue-p2pro";
        case P2ActiveKind::None:  return "mnue-p2";
    }
    return "mnue-p2";
}

[[nodiscard]] const char* p2_kind_arch_name(P2ActiveKind kind) noexcept {
    switch (kind) {
        case P2ActiveKind::P2:    return "P2";
        case P2ActiveKind::P2Pro: return "P2Pro";
        case P2ActiveKind::None:  return "P2";
    }
    return "P2";
}

[[nodiscard]] const char* p2_kind_short_name(P2ActiveKind kind) noexcept {
    switch (kind) {
        case P2ActiveKind::P2:    return "p2";
        case P2ActiveKind::P2Pro: return "p2pro";
        case P2ActiveKind::None:  return "p2";
    }
    return "p2";
}

template<typename T>
[[nodiscard]] bool read_exact(std::istream& in, T* data, std::size_t count) {
    if (count == 0)
        return true;
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(sizeof(T) * count));
    return static_cast<bool>(in);
}

template<class Layout>
[[nodiscard]] constexpr std::uintmax_t payload_bytes() noexcept {
    return static_cast<std::uintmax_t>(
        static_cast<std::size_t>(Layout::InputSize) * Layout::HiddenSize +
        static_cast<std::size_t>(Layout::HiddenSize) +
        static_cast<std::size_t>(Layout::OutputBuckets) * 2 * Layout::HiddenSize +
        static_cast<std::size_t>(Layout::OutputBuckets)
    ) * sizeof(i16);
}

template<class Layout>
[[nodiscard]] constexpr std::uintmax_t file_bytes() noexcept {
    return sizeof(FileHeader) + payload_bytes<Layout>();
}

template<class Layout>
void clear_network(Network<Layout>& net) noexcept {
    net.loaded = false;
    net.scale = kScale;
    net.path.clear();
    net.error.clear();
    net.w0.clear();
    net.b0.clear();
    net.w1.clear();
    net.b1.clear();
    net.w1_max_abs = 0;
    net.forward_madd_safe = false;
    net.forward_i32_safe = false;
}

template<class Layout>
void resize_network(Network<Layout>& net) {
    net.w0.resize(static_cast<std::size_t>(Layout::InputSize) * Layout::HiddenSize);
    net.b0.resize(static_cast<std::size_t>(Layout::HiddenSize));
    net.w1.resize(static_cast<std::size_t>(Layout::OutputBuckets) * 2 * Layout::HiddenSize);
    net.b1.resize(static_cast<std::size_t>(Layout::OutputBuckets));
}

template<class Layout>
[[nodiscard]] constexpr int max_safe_i32_w1_abs() noexcept {
    constexpr i64 lane_terms = (Layout::HiddenSize + 7) / 8;
    constexpr i64 max_lane_sum_per_weight =
        static_cast<i64>(kClip) * static_cast<i64>(kClip) * lane_terms;
    return static_cast<int>(
        static_cast<i64>(std::numeric_limits<i32>::max()) / max_lane_sum_per_weight
    );
}

template<class Layout>
void refresh_network_traits(Network<Layout>& net) noexcept {
    int max_abs = 0;
    for (const i16 w : net.w1) {
        const int abs_w = w < 0 ? -static_cast<int>(w) : static_cast<int>(w);
        max_abs = std::max(max_abs, abs_w);
    }

    net.w1_max_abs = max_abs;
    net.forward_madd_safe = max_abs <= 128;
    net.forward_i32_safe = max_abs <= max_safe_i32_w1_abs<Layout>();
}

template<class Layout>
[[nodiscard]] bool read_network_payload(std::istream& in, Network<Layout>& net) {
    resize_network(net);

    if (!read_exact(in, net.w0.data(), net.w0.size()) ||
        !read_exact(in, net.b0.data(), net.b0.size()) ||
        !read_exact(in, net.w1.data(), net.w1.size()) ||
        !read_exact(in, net.b1.data(), net.b1.size())) {
        clear_network(net);
        return false;
    }

    refresh_network_traits(net);
    return net.valid();
}


[[nodiscard]] u32 next_p2_generation() noexcept {
    u32 next = g_p2_generation.load(std::memory_order_relaxed) + 1;
    if (next == 0)
        next = 1;
    g_p2_generation.store(next, std::memory_order_release);
    return next;
}

[[nodiscard]] inline std::uint8_t p2_perspective_mask(Color perspective) noexcept {
    return static_cast<std::uint8_t>(1u << static_cast<unsigned>(perspective));
}


// Core parsing: reads MNUE header + payload from any std::istream.
// `known_size` is the total byte size when available (0 = unknown).
// Used by both file-based load_network() and embedded load_p2_embedded().
template<class Layout>
[[nodiscard]] bool load_network_from_stream(
    std::istream& in, Network<Layout>& net, const std::string& path,
    std::uintmax_t known_size = 0) {
    clear_network(net);

    FileHeader header{};
    if (!read_exact(in, &header, 1)) {
        net.error = "could not read MNUE header";
        g_error = net.error;
        return false;
    }

    if (header.magic != kMnueMagic || header.version != kMnueVersion) {
        // Headerless fallback: only when size matches raw payload exactly.
        // A file that happens to have enough bytes but is the wrong format
        // must NOT be silently accepted.
        if (known_size == 0) {
            in.clear();
            in.seekg(0, std::ios::end);
            const auto endpos = in.tellg();
            if (endpos >= 0)
                known_size = static_cast<std::uintmax_t>(endpos);
            in.seekg(0, std::ios::beg);
        }
        if (known_size == payload_bytes<Layout>()) {
            in.clear();
            in.seekg(0, std::ios::beg);
            if (read_network_payload(in, net)) {
                net.loaded = true;
                net.path = path;
                g_error.clear();
                return true;
            }
        }

        net.error = "bad MNUE magic/version";
        g_error = net.error;
        return false;
    }

    if (header.arch != Layout::ArchId ||
        header.input_size != Layout::InputSize ||
        header.hidden_size != Layout::HiddenSize ||
        header.input_buckets != Layout::InputBuckets ||
        header.output_buckets != Layout::OutputBuckets ||
        header.qa != kQa ||
        header.qb != kQb) {
        net.error = "MNUE header dimensions do not match requested layout";
        g_error = net.error;
        return false;
    }

    net.scale = header.scale;
    if (!read_network_payload(in, net)) {
        net.error = "truncated MNUE weight file";
        g_error = net.error;
        return false;
    }

    if (!net.valid()) {
        clear_network(net);
        net.error = "internal MNUE size check failed";
        g_error = net.error;
        return false;
    }

    net.loaded = true;
    net.path = path;
    g_error.clear();
    return true;
}

template<class Layout>
[[nodiscard]] bool load_network(Network<Layout>& net, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        net.error = "could not open MNUE file: " + path;
        g_error = net.error;
        return false;
    }
    std::error_code ec;
    const auto file_sz = std::filesystem::file_size(path, ec);
    return load_network_from_stream(
        in, net, path,
        ec ? static_cast<std::uintmax_t>(0) : file_sz
    );
}

[[nodiscard]] inline Square flip_vertical_sq(Square sq) noexcept {
    return static_cast<Square>(static_cast<int>(sq) ^ 56);
}

[[nodiscard]] inline Square relative_square(Color perspective, Square sq) noexcept {
    return perspective == WHITE ? sq : flip_vertical_sq(sq);
}

[[nodiscard]] inline int file_of(Square sq) noexcept {
    return static_cast<int>(sq) & 7;
}

[[nodiscard]] inline int rank_of(Square sq) noexcept {
    return static_cast<int>(sq) >> 3;
}

[[nodiscard]] inline Square king_square_of(const Position& pos, Color color) noexcept {
    const Square sq = king_square(pos, color);
    return sq == NO_SQ ? static_cast<Square>(0) : sq;
}

[[nodiscard]] inline int nonking_piece_index(PieceType pt) noexcept {
    switch (pt) {
        case PAWN:   return 0;
        case KNIGHT: return 1;
        case BISHOP: return 2;
        case ROOK:   return 3;
        case QUEEN:  return 4;
        default:     return -1;
    }
}

[[nodiscard]] inline int king_zone16(Square relative_king_sq) noexcept {
    const int file_group = file_of(relative_king_sq) / 2; // 0..3
    const int rank_group = rank_of(relative_king_sq) / 2; // 0..3
    return rank_group * 4 + file_group;                  // 0..15
}

[[nodiscard]] inline int king_zone32(Square relative_king_sq) noexcept {
    const int file_group = file_of(relative_king_sq);      // 0..7
    const int rank_group = rank_of(relative_king_sq) / 2;  // 0..3
    return rank_group * 8 + file_group;                   // 0..31
}

[[nodiscard]] inline int phase_bucket2(const Position& pos) noexcept {
    // Must match the Bullet trainer exactly:
    //   knight/bishop = 1, rook = 2, queen = 4, pawns ignored.
    //
    // Trainer:
    //   let mut npm = 0;
    //   N/B += 1, R += 2, Q += 4;
    //   phase = usize::from(npm < 10);
    //
    // Therefore:
    //   phase 0 = opening/middlegame
    //   phase 1 = simplified/endgame
    return mnue_phase_units(pos) < 10 ? 1 : 0;
}

[[nodiscard]] inline int game_ply(const Position& pos) noexcept {
    const int fullmove = std::max(1, pos.fullmove_number);
    return (fullmove - 1) * 2 + (pos.side_to_move == BLACK ? 1 : 0);
}

[[nodiscard]] inline int mnue_material_units(const Position& pos) noexcept {
    return std::clamp(non_king_material(pos), 0, kMnueMaterialMax);
}

[[nodiscard]] inline double mnue_material_ratio(const Position& pos) noexcept {
    return static_cast<double>(mnue_material_units(pos))
        / static_cast<double>(kMnueMaterialMax);
}

[[nodiscard]] inline double mnue_cp_denominator(const Position& pos) noexcept {
    const double material = mnue_material_ratio(pos);
    double denom = kMnueCpBase + kMnueCpMaterial * material;
    if (phase_bucket2(pos) != 0)
        denom -= kMnueCpEndgameDiscount;
    return std::clamp(
        denom,
        kMnueCpMinDenominator,
        kMnueCpMaxDenominator
    );
}

[[nodiscard]] inline int mnue_to_cp(int raw, const Position& pos) noexcept {
    const i64 abs_raw = raw >= 0 ? static_cast<i64>(raw) : -static_cast<i64>(raw);
    const double clipped = static_cast<double>(
        std::min<i64>(abs_raw, kMnueCpMaxRaw)
    );
    const int cp = static_cast<int>(
        std::round(clipped * 100.0 / mnue_cp_denominator(pos))
    );
    return raw >= 0 ? cp : -cp;
}

[[nodiscard]] inline WinRateParams mnue_win_rate_params(const Position& pos) noexcept {
    const double material = mnue_material_ratio(pos);
    const double ply = static_cast<double>(std::min(240, game_ply(pos))) / 240.0;
    const double low_material = 1.0 - material;

    double a = 138.0 + 36.0 * material + 8.0 * ply;
    double b = 54.0 + 32.0 * material + 6.0 * ply;

    if (mnue_material_units(pos) <= 4) {
        a += 130.0 * low_material;
        b += 22.0 * low_material;
    }

    return {a, b};
}

[[nodiscard]] inline WdlTriplet mnue_wdl_from_cp(
    int cp,
    const Position& pos
) noexcept {
    const auto [a, b] = mnue_win_rate_params(pos);
    const double x = std::clamp(static_cast<double>(cp), -2400.0, 2400.0);
    const double slope = std::max(1.0, b);

    const double win = 1.0 / (1.0 + std::exp((a - x) / slope));
    const double loss = 1.0 / (1.0 + std::exp((a + x) / slope));

    int win_i = std::clamp(static_cast<int>(std::round(1000.0 * win)), 0, 1000);
    int loss_i = std::clamp(static_cast<int>(std::round(1000.0 * loss)), 0, 1000);
    if (win_i + loss_i > 1000) {
        const int total = win_i + loss_i;
        win_i = (win_i * 1000) / total;
        loss_i = 1000 - win_i;
    }

    return {
        .win = win_i,
        .draw = 1000 - win_i - loss_i,
        .loss = loss_i
    };
}

template<class Layout>
[[nodiscard]] inline int input_bucket(const Position& pos, Color perspective) noexcept {
    const Square rel_ksq = relative_square(perspective, king_square_of(pos, perspective));
    if constexpr (Layout::InputBuckets == 16) {
        return king_zone16(rel_ksq);
    } else {
        return king_zone32(rel_ksq);
    }
}

template<class Layout>
[[nodiscard]] inline int output_bucket(const Position& pos, Color stm) noexcept {
    const Square rel_ksq = relative_square(stm, king_square_of(pos, stm));
    return phase_bucket2(pos) * 16 + king_zone16(rel_ksq); // 0..31
}

template<class Layout>
[[nodiscard]] inline int feature_index(
    Color perspective,
    int bucket,
    Piece pc,
    Square sq
) noexcept {
    const PieceType pt = type_of(pc);
    const int pt_idx = nonking_piece_index(pt);
    if (pt_idx < 0)
        return -1;

    const int rel_color = color_of(pc) == perspective ? 0 : 1;
    const int rel_sq = static_cast<int>(relative_square(perspective, sq));

    return (((bucket * 2 + rel_color) * 5 + pt_idx) * 64 + rel_sq);
}

template<class Layout>
[[nodiscard]] inline int feature_index(
    const Position& pos,
    Color perspective,
    Piece pc,
    Square sq
) noexcept {
    return feature_index<Layout>(
        perspective,
        input_bucket<Layout>(pos, perspective),
        pc,
        sq
    );
}

template<class Layout>
using Accumulator = std::array<i16, Layout::HiddenSize>;

// Stockfish refreshes stale NNUE accumulators from a king-indexed cache.
// P2-family nets use the same idea with MNUE's coarser input bucket as the key.
template<class Layout>
struct P2RefreshEntry {
    Accumulator<Layout> accumulation{};
    std::array<Piece, SQ_NB> board{};
    bool initialized = false;

    void reset_to_biases(const Network<Layout>& net) noexcept {
        std::copy(net.b0.begin(), net.b0.end(), accumulation.begin());
        board.fill(PIECE_NONE);
        initialized = true;
    }
};

template<class Layout>
struct P2RefreshCache {
    u32 generation = 0;
    std::array<std::array<P2RefreshEntry<Layout>, Layout::InputBuckets>, COLOR_NB> entries{};

    void sync(u32 current_generation) noexcept {
        if (generation == current_generation)
            return;

        for (auto& side_entries : entries)
            for (auto& entry : side_entries)
                entry.initialized = false;

        generation = current_generation;
    }
};

struct P2PerspectiveDiff {
    std::array<std::uint16_t, 2> removed{};
    std::array<std::uint16_t, 2> added{};
    std::uint8_t removed_count = 0;
    std::uint8_t added_count = 0;
    bool refresh = false;
};

#if defined(__AVX2__)
template<class Layout>
inline void add_feature_vector_avx2(
    std::array<i16, Layout::HiddenSize>& acc,
    const i16* w
) noexcept {
    i16* acc_ptr = acc.data();
    for (int i = 0; i < Layout::HiddenSize; i += 16) {
        const __m256i a = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc_ptr + i)
        );
        const __m256i b = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(w + i)
        );
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(acc_ptr + i),
            _mm256_add_epi16(a, b)
        );
    }
}

template<class Layout>
inline void sub_feature_vector_avx2(
    std::array<i16, Layout::HiddenSize>& acc,
    const i16* w
) noexcept {
    i16* acc_ptr = acc.data();
    for (int i = 0; i < Layout::HiddenSize; i += 16) {
        const __m256i a = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc_ptr + i)
        );
        const __m256i b = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(w + i)
        );
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(acc_ptr + i),
            _mm256_sub_epi16(a, b)
        );
    }
}

#endif

template<class Layout>
void add_feature(
    Accumulator<Layout>& acc,
    const Network<Layout>& net,
    int idx
) noexcept {
    const i16* w = net.w0.data() + static_cast<std::size_t>(idx) * Layout::HiddenSize;
#if defined(__AVX2__)
    add_feature_vector_avx2<Layout>(acc, w);
#else
    for (int i = 0; i < Layout::HiddenSize; ++i)
        acc[static_cast<std::size_t>(i)] = static_cast<i16>(
            static_cast<i32>(acc[static_cast<std::size_t>(i)]) + static_cast<i32>(w[i])
        );
#endif
}


template<class Layout>
void sub_feature(
    Accumulator<Layout>& acc,
    const Network<Layout>& net,
    int idx
) noexcept {
    const i16* w = net.w0.data() + static_cast<std::size_t>(idx) * Layout::HiddenSize;
#if defined(__AVX2__)
    sub_feature_vector_avx2<Layout>(acc, w);
#else
    for (int i = 0; i < Layout::HiddenSize; ++i)
        acc[static_cast<std::size_t>(i)] = static_cast<i16>(
            static_cast<i32>(acc[static_cast<std::size_t>(i)]) - static_cast<i32>(w[i])
        );
#endif
}

template<class Layout>
void apply_p2_diff(
    Accumulator<Layout>& dst,
    const Accumulator<Layout>& src,
    const Network<Layout>& net,
    const P2PerspectiveDiff& diff,
    bool forward
) noexcept {
    const auto& added = forward ? diff.added : diff.removed;
    const auto& removed = forward ? diff.removed : diff.added;
    const std::uint8_t added_count = forward ? diff.added_count : diff.removed_count;
    const std::uint8_t removed_count = forward ? diff.removed_count : diff.added_count;

    const i16* add0 = added_count > 0
        ? net.w0.data() + static_cast<std::size_t>(added[0]) * Layout::HiddenSize
        : nullptr;
    const i16* add1 = added_count > 1
        ? net.w0.data() + static_cast<std::size_t>(added[1]) * Layout::HiddenSize
        : nullptr;
    const i16* sub0 = removed_count > 0
        ? net.w0.data() + static_cast<std::size_t>(removed[0]) * Layout::HiddenSize
        : nullptr;
    const i16* sub1 = removed_count > 1
        ? net.w0.data() + static_cast<std::size_t>(removed[1]) * Layout::HiddenSize
        : nullptr;

#if defined(__AVX2__)
    // P2 stack states explicitly align both perspective accumulators to a
    // cache line, so the source/destination traffic can use aligned moves.
    const auto apply = [&]<int AddCount, int SubCount>() noexcept {
        for (int i = 0; i < Layout::HiddenSize; i += 16) {
            __m256i value = _mm256_load_si256(
                reinterpret_cast<const __m256i*>(src.data() + i)
            );
            if constexpr (AddCount >= 1)
                value = _mm256_add_epi16(
                    value,
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(add0 + i))
                );
            if constexpr (AddCount >= 2)
                value = _mm256_add_epi16(
                    value,
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(add1 + i))
                );
            if constexpr (SubCount >= 1)
                value = _mm256_sub_epi16(
                    value,
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sub0 + i))
                );
            if constexpr (SubCount >= 2)
                value = _mm256_sub_epi16(
                    value,
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sub1 + i))
                );
            _mm256_store_si256(reinterpret_cast<__m256i*>(dst.data() + i), value);
        }
    };

    switch ((static_cast<unsigned>(added_count) << 2) | removed_count) {
        case 0x0: apply.template operator()<0, 0>(); break;
        case 0x1: apply.template operator()<0, 1>(); break;
        case 0x2: apply.template operator()<0, 2>(); break;
        case 0x4: apply.template operator()<1, 0>(); break;
        case 0x5: apply.template operator()<1, 1>(); break;
        case 0x6: apply.template operator()<1, 2>(); break;
        case 0x8: apply.template operator()<2, 0>(); break;
        case 0x9: apply.template operator()<2, 1>(); break;
        case 0xA: apply.template operator()<2, 2>(); break;
        default:  assert(false); break;
    }
#else
    for (int i = 0; i < Layout::HiddenSize; ++i) {
        i32 value = src[static_cast<std::size_t>(i)];
        if (add0 != nullptr) value += add0[i];
        if (add1 != nullptr) value += add1[i];
        if (sub0 != nullptr) value -= sub0[i];
        if (sub1 != nullptr) value -= sub1[i];
        dst[static_cast<std::size_t>(i)] = static_cast<i16>(value);
    }
#endif
}

using P2MoveDiff = std::array<P2PerspectiveDiff, COLOR_NB>;

void append_p2_feature(
    P2MoveDiff& move_diff,
    const std::array<int, COLOR_NB>& buckets,
    Piece pc,
    Square sq,
    bool added
) noexcept {
    if (pc == PIECE_NONE || type_of(pc) == KING)
        return;

    for (int persp = WHITE; persp <= BLACK; ++persp) {
        P2PerspectiveDiff& diff = move_diff[static_cast<std::size_t>(persp)];
        if (diff.refresh)
            continue;

        const int idx = feature_index<P2Layout>(
            static_cast<Color>(persp),
            buckets[static_cast<std::size_t>(persp)],
            pc,
            sq
        );
        if (idx < 0)
            continue;

        if (added) {
            assert(diff.added_count < diff.added.size());
            diff.added[diff.added_count++] = static_cast<std::uint16_t>(idx);
        } else {
            assert(diff.removed_count < diff.removed.size());
            diff.removed[diff.removed_count++] = static_cast<std::uint16_t>(idx);
        }
    }
}

[[nodiscard]] P2MoveDiff make_p2_move_diff(
    const Position& pos,
    Move move
) noexcept {
    P2MoveDiff move_diff{};
    const std::array<int, COLOR_NB> buckets{{
        input_bucket<P2Layout>(pos, WHITE),
        input_bucket<P2Layout>(pos, BLACK)
    }};
    const Square from = from_sq(move);
    const Square to = to_sq(move);
    const Piece moving = piece_on(pos, from);
    assert(moving != PIECE_NONE);

    const Color moving_color = color_of(moving);
    const PieceType moving_type = type_of(moving);
    if (moving_type == KING) {
        const int from_bucket = king_zone16(relative_square(moving_color, from));
        const int to_bucket = king_zone16(relative_square(moving_color, to));
        move_diff[static_cast<std::size_t>(moving_color)].refresh =
            from_bucket != to_bucket;
    }

    if (move_is_capture(move)) {
        const Square captured_sq = move_is_ep(move)
            ? static_cast<Square>(
                static_cast<int>(to) + (moving_color == WHITE ? -8 : 8)
            )
            : to;
        append_p2_feature(
            move_diff,
            buckets,
            piece_on(pos, captured_sq),
            captured_sq,
            false
        );
    }

    append_p2_feature(move_diff, buckets, moving, from, false);
    if (move_is_promotion(move)) {
        append_p2_feature(
            move_diff,
            buckets,
            make_piece(moving_color, promo_piece(move)),
            to,
            true
        );
    } else {
        append_p2_feature(move_diff, buckets, moving, to, true);
    }

    if (move_is_castle(move)) {
        const bool king_side = move_flag(move) == MOVE_OO;
        const Square rook_from = moving_color == WHITE
            ? static_cast<Square>(king_side ? 7 : 0)
            : static_cast<Square>(king_side ? 63 : 56);
        const Square rook_to = moving_color == WHITE
            ? static_cast<Square>(king_side ? 5 : 3)
            : static_cast<Square>(king_side ? 61 : 59);
        const Piece rook = make_piece(moving_color, ROOK);
        append_p2_feature(move_diff, buckets, rook, rook_from, false);
        append_p2_feature(move_diff, buckets, rook, rook_to, true);
    }

    return move_diff;
}

template<class Layout>
void rebuild_accumulator(
    const Position& pos,
    Color perspective,
    Accumulator<Layout>& acc,
    const Network<Layout>& net
) noexcept {
    std::copy(net.b0.begin(), net.b0.end(), acc.begin());
    const int bucket = input_bucket<Layout>(pos, perspective);

    Bitboard bb = pieces(pos);
    while (bb) {
        const Square sq = static_cast<Square>(
            std::countr_zero(static_cast<std::uint64_t>(bb))
        );
        bb &= (bb - 1);

        const Piece pc = piece_on(pos, sq);
        if (pc == PIECE_NONE)
            continue;

        const int idx = feature_index<Layout>(perspective, bucket, pc, sq);
        if (idx >= 0)
            add_feature(acc, net, idx);
    }
}

[[nodiscard]] inline i32 screlu(i32 x) noexcept {
    const i32 y = std::clamp(x, 0, kClip);
    return y * y;
}


#if defined(__AVX512F__) && defined(__AVX512BW__)
[[nodiscard]] inline i64 horizontal_sum_epi64_avx512(__m512i v) noexcept {
    return static_cast<i64>(_mm512_reduce_add_epi64(v));
}

[[nodiscard]] inline i64 horizontal_sum_epi32_to_i64_avx512(__m512i v) noexcept {
    const __m256i lo = _mm512_castsi512_si256(v);
    const __m256i hi = _mm512_extracti64x4_epi64(v, 1);
    return horizontal_sum_epi64_avx512(
        _mm512_add_epi64(
            _mm512_cvtepi32_epi64(lo),
            _mm512_cvtepi32_epi64(hi)
        )
    );
}

inline __m512i add_i32x16_to_i64_avx512(__m512i sum, __m512i v) noexcept {
    const __m256i lo = _mm512_castsi512_si256(v);
    const __m256i hi = _mm512_extracti64x4_epi64(v, 1);
    sum = _mm512_add_epi64(sum, _mm512_cvtepi32_epi64(lo));
    return _mm512_add_epi64(sum, _mm512_cvtepi32_epi64(hi));
}

[[nodiscard]] i64 dot_pair_screlu_i16_madd_avx512(
    const i16* acc0,
    const i16* weights0,
    const i16* acc1,
    const i16* weights1,
    int count
) noexcept {
    const __m512i zero16 = _mm512_setzero_si512();
    const __m512i clip16 = _mm512_set1_epi16(static_cast<i16>(kClip));
    __m512i sum0_32 = _mm512_setzero_si512();
    __m512i sum1_32 = _mm512_setzero_si512();

    int i = 0;
    for (; i + 31 < count; i += 32) {
        const __m512i acc0_16_raw = _mm512_loadu_si512(
            reinterpret_cast<const void*>(acc0 + i)
        );
        const __m512i w0_16 = _mm512_loadu_si512(
            reinterpret_cast<const void*>(weights0 + i)
        );
        const __m512i acc1_16_raw = _mm512_loadu_si512(
            reinterpret_cast<const void*>(acc1 + i)
        );
        const __m512i w1_16 = _mm512_loadu_si512(
            reinterpret_cast<const void*>(weights1 + i)
        );

        const __m512i clipped0_16 = _mm512_min_epi16(
            _mm512_max_epi16(acc0_16_raw, zero16),
            clip16
        );
        const __m512i clipped1_16 = _mm512_min_epi16(
            _mm512_max_epi16(acc1_16_raw, zero16),
            clip16
        );

        const __m512i aw0_16 = _mm512_mullo_epi16(clipped0_16, w0_16);
        const __m512i aw1_16 = _mm512_mullo_epi16(clipped1_16, w1_16);
        sum0_32 = _mm512_add_epi32(
            sum0_32,
            _mm512_madd_epi16(clipped0_16, aw0_16)
        );
        sum1_32 = _mm512_add_epi32(
            sum1_32,
            _mm512_madd_epi16(clipped1_16, aw1_16)
        );
    }

    i64 total = horizontal_sum_epi32_to_i64_avx512(sum0_32)
        + horizontal_sum_epi32_to_i64_avx512(sum1_32);
    for (; i < count; ++i) {
        total += static_cast<i64>(screlu(acc0[i])) * static_cast<i64>(weights0[i]);
        total += static_cast<i64>(screlu(acc1[i])) * static_cast<i64>(weights1[i]);
    }

    return total;
}

[[nodiscard]] i64 dot_pair_screlu_i16_i32_avx512(
    const i16* acc0,
    const i16* weights0,
    const i16* acc1,
    const i16* weights1,
    int count
) noexcept {
    const __m512i zero16 = _mm512_setzero_si512();
    const __m512i clip16 = _mm512_set1_epi16(static_cast<i16>(kClip));
    __m512i sum0_32 = _mm512_setzero_si512();
    __m512i sum1_32 = _mm512_setzero_si512();

    int i = 0;
    for (; i + 31 < count; i += 32) {
        const __m512i acc0_16_raw = _mm512_loadu_si512(
            reinterpret_cast<const void*>(acc0 + i)
        );
        const __m512i w0_16_raw = _mm512_loadu_si512(
            reinterpret_cast<const void*>(weights0 + i)
        );
        const __m512i acc1_16_raw = _mm512_loadu_si512(
            reinterpret_cast<const void*>(acc1 + i)
        );
        const __m512i w1_16_raw = _mm512_loadu_si512(
            reinterpret_cast<const void*>(weights1 + i)
        );

        const __m512i clipped0_16 = _mm512_min_epi16(
            _mm512_max_epi16(acc0_16_raw, zero16),
            clip16
        );
        const __m512i clipped1_16 = _mm512_min_epi16(
            _mm512_max_epi16(acc1_16_raw, zero16),
            clip16
        );

        const __m256i acc0_lo16 = _mm512_castsi512_si256(clipped0_16);
        const __m256i acc0_hi16 = _mm512_extracti64x4_epi64(clipped0_16, 1);
        const __m256i w0_lo16 = _mm512_castsi512_si256(w0_16_raw);
        const __m256i w0_hi16 = _mm512_extracti64x4_epi64(w0_16_raw, 1);
        const __m256i acc1_lo16 = _mm512_castsi512_si256(clipped1_16);
        const __m256i acc1_hi16 = _mm512_extracti64x4_epi64(clipped1_16, 1);
        const __m256i w1_lo16 = _mm512_castsi512_si256(w1_16_raw);
        const __m256i w1_hi16 = _mm512_extracti64x4_epi64(w1_16_raw, 1);

        const __m512i acc0_lo32 = _mm512_cvtepi16_epi32(acc0_lo16);
        const __m512i acc0_hi32 = _mm512_cvtepi16_epi32(acc0_hi16);
        const __m512i w0_lo32 = _mm512_cvtepi16_epi32(w0_lo16);
        const __m512i w0_hi32 = _mm512_cvtepi16_epi32(w0_hi16);
        const __m512i acc1_lo32 = _mm512_cvtepi16_epi32(acc1_lo16);
        const __m512i acc1_hi32 = _mm512_cvtepi16_epi32(acc1_hi16);
        const __m512i w1_lo32 = _mm512_cvtepi16_epi32(w1_lo16);
        const __m512i w1_hi32 = _mm512_cvtepi16_epi32(w1_hi16);

        const __m512i sq0_lo32 = _mm512_mullo_epi32(acc0_lo32, acc0_lo32);
        const __m512i sq0_hi32 = _mm512_mullo_epi32(acc0_hi32, acc0_hi32);
        const __m512i sq1_lo32 = _mm512_mullo_epi32(acc1_lo32, acc1_lo32);
        const __m512i sq1_hi32 = _mm512_mullo_epi32(acc1_hi32, acc1_hi32);

        sum0_32 = _mm512_add_epi32(
            sum0_32,
            _mm512_mullo_epi32(sq0_lo32, w0_lo32)
        );
        sum0_32 = _mm512_add_epi32(
            sum0_32,
            _mm512_mullo_epi32(sq0_hi32, w0_hi32)
        );
        sum1_32 = _mm512_add_epi32(
            sum1_32,
            _mm512_mullo_epi32(sq1_lo32, w1_lo32)
        );
        sum1_32 = _mm512_add_epi32(
            sum1_32,
            _mm512_mullo_epi32(sq1_hi32, w1_hi32)
        );
    }

    i64 total = horizontal_sum_epi32_to_i64_avx512(sum0_32)
        + horizontal_sum_epi32_to_i64_avx512(sum1_32);
    for (; i < count; ++i) {
        total += static_cast<i64>(screlu(acc0[i])) * static_cast<i64>(weights0[i]);
        total += static_cast<i64>(screlu(acc1[i])) * static_cast<i64>(weights1[i]);
    }

    return total;
}

[[nodiscard]] i64 dot_screlu_i16_avx512(
    const i16* acc,
    const i16* weights,
    int count
) noexcept {
    const __m512i zero16 = _mm512_setzero_si512();
    const __m512i clip16 = _mm512_set1_epi16(static_cast<i16>(kClip));
    __m512i sum64 = _mm512_setzero_si512();

    int i = 0;
    for (; i + 31 < count; i += 32) {
        const __m512i acc16_raw = _mm512_loadu_si512(
            reinterpret_cast<const void*>(acc + i)
        );
        const __m512i w16_raw = _mm512_loadu_si512(
            reinterpret_cast<const void*>(weights + i)
        );
        const __m512i clipped16 = _mm512_min_epi16(
            _mm512_max_epi16(acc16_raw, zero16),
            clip16
        );

        const __m256i acc_lo16 = _mm512_castsi512_si256(clipped16);
        const __m256i acc_hi16 = _mm512_extracti64x4_epi64(clipped16, 1);
        const __m256i w_lo16 = _mm512_castsi512_si256(w16_raw);
        const __m256i w_hi16 = _mm512_extracti64x4_epi64(w16_raw, 1);

        const __m512i acc_lo32 = _mm512_cvtepi16_epi32(acc_lo16);
        const __m512i acc_hi32 = _mm512_cvtepi16_epi32(acc_hi16);
        const __m512i w_lo32 = _mm512_cvtepi16_epi32(w_lo16);
        const __m512i w_hi32 = _mm512_cvtepi16_epi32(w_hi16);
        const __m512i prod_lo32 = _mm512_mullo_epi32(
            _mm512_mullo_epi32(acc_lo32, acc_lo32),
            w_lo32
        );
        const __m512i prod_hi32 = _mm512_mullo_epi32(
            _mm512_mullo_epi32(acc_hi32, acc_hi32),
            w_hi32
        );

        sum64 = add_i32x16_to_i64_avx512(sum64, prod_lo32);
        sum64 = add_i32x16_to_i64_avx512(sum64, prod_hi32);
    }

    i64 total = horizontal_sum_epi64_avx512(sum64);
    for (; i < count; ++i)
        total += static_cast<i64>(screlu(acc[i])) * static_cast<i64>(weights[i]);

    return total;
}
#endif

#if defined(__AVX2__) \
    && !(defined(__AVX512F__) && defined(__AVX512BW__))
[[nodiscard]] inline i64 horizontal_sum_epi64_avx2(__m256i v) noexcept {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    const __m128i sum = _mm_add_epi64(lo, hi);
    return static_cast<i64>(_mm_cvtsi128_si64(sum))
        + static_cast<i64>(_mm_extract_epi64(sum, 1));
}

[[nodiscard]] inline i64 horizontal_sum_epi32_to_i64_avx2(__m256i v) noexcept {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    const __m256i lo64 = _mm256_cvtepi32_epi64(lo);
    const __m256i hi64 = _mm256_cvtepi32_epi64(hi);
    return horizontal_sum_epi64_avx2(_mm256_add_epi64(lo64, hi64));
}

[[nodiscard]] inline __m256i cvt_i32x4_to_i64x4_avx2(__m128i v) noexcept {
    return _mm256_cvtepi32_epi64(v);
}

[[nodiscard]] i64 dot_pair_screlu_i16_madd_avx2(
    const i16* acc0,
    const i16* weights0,
    const i16* acc1,
    const i16* weights1,
    int count
) noexcept {
    const __m256i zero16 = _mm256_setzero_si256();
    const __m256i clip16 = _mm256_set1_epi16(static_cast<i16>(kClip));
    __m256i sum0_32 = _mm256_setzero_si256();
    __m256i sum1_32 = _mm256_setzero_si256();

    int i = 0;
    for (; i + 15 < count; i += 16) {
        const __m256i acc0_16_raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc0 + i)
        );
        const __m256i w0_16 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights0 + i)
        );
        const __m256i acc1_16_raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc1 + i)
        );
        const __m256i w1_16 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights1 + i)
        );

        const __m256i clipped0_16 = _mm256_min_epi16(
            _mm256_max_epi16(acc0_16_raw, zero16),
            clip16
        );
        const __m256i clipped1_16 = _mm256_min_epi16(
            _mm256_max_epi16(acc1_16_raw, zero16),
            clip16
        );

        const __m256i aw0_16 = _mm256_mullo_epi16(clipped0_16, w0_16);
        const __m256i aw1_16 = _mm256_mullo_epi16(clipped1_16, w1_16);

        sum0_32 = _mm256_add_epi32(
            sum0_32,
            _mm256_madd_epi16(clipped0_16, aw0_16)
        );
        sum1_32 = _mm256_add_epi32(
            sum1_32,
            _mm256_madd_epi16(clipped1_16, aw1_16)
        );
    }

    i64 total = horizontal_sum_epi32_to_i64_avx2(sum0_32)
        + horizontal_sum_epi32_to_i64_avx2(sum1_32);
    for (; i < count; ++i) {
        total += static_cast<i64>(screlu(acc0[i])) * static_cast<i64>(weights0[i]);
        total += static_cast<i64>(screlu(acc1[i])) * static_cast<i64>(weights1[i]);
    }

    return total;
}

[[nodiscard]] i64 dot_pair_screlu_i16_i32_avx2(
    const i16* acc0,
    const i16* weights0,
    const i16* acc1,
    const i16* weights1,
    int count
) noexcept {
    const __m256i zero16 = _mm256_setzero_si256();
    const __m256i clip16 = _mm256_set1_epi16(static_cast<i16>(kClip));
    __m256i sum0_32 = _mm256_setzero_si256();
    __m256i sum1_32 = _mm256_setzero_si256();

    int i = 0;
    for (; i + 15 < count; i += 16) {
        const __m256i acc0_16_raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc0 + i)
        );
        const __m256i w0_16_raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights0 + i)
        );
        const __m256i acc1_16_raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc1 + i)
        );
        const __m256i w1_16_raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights1 + i)
        );

        const __m256i clipped0_16 = _mm256_min_epi16(
            _mm256_max_epi16(acc0_16_raw, zero16),
            clip16
        );
        const __m256i clipped1_16 = _mm256_min_epi16(
            _mm256_max_epi16(acc1_16_raw, zero16),
            clip16
        );

        const __m128i acc0_lo16 = _mm256_castsi256_si128(clipped0_16);
        const __m128i acc0_hi16 = _mm256_extracti128_si256(clipped0_16, 1);
        const __m128i w0_lo16 = _mm256_castsi256_si128(w0_16_raw);
        const __m128i w0_hi16 = _mm256_extracti128_si256(w0_16_raw, 1);
        const __m128i acc1_lo16 = _mm256_castsi256_si128(clipped1_16);
        const __m128i acc1_hi16 = _mm256_extracti128_si256(clipped1_16, 1);
        const __m128i w1_lo16 = _mm256_castsi256_si128(w1_16_raw);
        const __m128i w1_hi16 = _mm256_extracti128_si256(w1_16_raw, 1);

        const __m256i acc0_lo32 = _mm256_cvtepi16_epi32(acc0_lo16);
        const __m256i acc0_hi32 = _mm256_cvtepi16_epi32(acc0_hi16);
        const __m256i w0_lo32 = _mm256_cvtepi16_epi32(w0_lo16);
        const __m256i w0_hi32 = _mm256_cvtepi16_epi32(w0_hi16);
        const __m256i acc1_lo32 = _mm256_cvtepi16_epi32(acc1_lo16);
        const __m256i acc1_hi32 = _mm256_cvtepi16_epi32(acc1_hi16);
        const __m256i w1_lo32 = _mm256_cvtepi16_epi32(w1_lo16);
        const __m256i w1_hi32 = _mm256_cvtepi16_epi32(w1_hi16);

        const __m256i sq0_lo32 = _mm256_mullo_epi32(acc0_lo32, acc0_lo32);
        const __m256i sq0_hi32 = _mm256_mullo_epi32(acc0_hi32, acc0_hi32);
        const __m256i sq1_lo32 = _mm256_mullo_epi32(acc1_lo32, acc1_lo32);
        const __m256i sq1_hi32 = _mm256_mullo_epi32(acc1_hi32, acc1_hi32);

        sum0_32 = _mm256_add_epi32(
            sum0_32,
            _mm256_mullo_epi32(sq0_lo32, w0_lo32)
        );
        sum0_32 = _mm256_add_epi32(
            sum0_32,
            _mm256_mullo_epi32(sq0_hi32, w0_hi32)
        );
        sum1_32 = _mm256_add_epi32(
            sum1_32,
            _mm256_mullo_epi32(sq1_lo32, w1_lo32)
        );
        sum1_32 = _mm256_add_epi32(
            sum1_32,
            _mm256_mullo_epi32(sq1_hi32, w1_hi32)
        );
    }

    i64 total = horizontal_sum_epi32_to_i64_avx2(sum0_32)
        + horizontal_sum_epi32_to_i64_avx2(sum1_32);
    for (; i < count; ++i) {
        total += static_cast<i64>(screlu(acc0[i])) * static_cast<i64>(weights0[i]);
        total += static_cast<i64>(screlu(acc1[i])) * static_cast<i64>(weights1[i]);
    }

    return total;
}

[[nodiscard]] i64 dot_screlu_i16_avx2(
    const i16* acc,
    const i16* weights,
    int count
) noexcept {
    const __m256i zero16 = _mm256_setzero_si256();
    const __m256i clip16 = _mm256_set1_epi16(static_cast<i16>(kClip));
    __m256i sum64 = _mm256_setzero_si256();

    int i = 0;
    for (; i + 15 < count; i += 16) {
        const __m256i acc16_raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc + i)
        );
        const __m256i w16_raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights + i)
        );

        const __m256i clipped16 = _mm256_min_epi16(
            _mm256_max_epi16(acc16_raw, zero16),
            clip16
        );

        const __m128i acc_lo16 = _mm256_castsi256_si128(clipped16);
        const __m128i acc_hi16 = _mm256_extracti128_si256(clipped16, 1);
        const __m128i w_lo16 = _mm256_castsi256_si128(w16_raw);
        const __m128i w_hi16 = _mm256_extracti128_si256(w16_raw, 1);

        const __m256i acc_lo32 = _mm256_cvtepi16_epi32(acc_lo16);
        const __m256i acc_hi32 = _mm256_cvtepi16_epi32(acc_hi16);
        const __m256i w_lo32 = _mm256_cvtepi16_epi32(w_lo16);
        const __m256i w_hi32 = _mm256_cvtepi16_epi32(w_hi16);

        const __m256i sq_lo32 = _mm256_mullo_epi32(acc_lo32, acc_lo32);
        const __m256i sq_hi32 = _mm256_mullo_epi32(acc_hi32, acc_hi32);
        const __m256i prod_lo32 = _mm256_mullo_epi32(sq_lo32, w_lo32);
        const __m256i prod_hi32 = _mm256_mullo_epi32(sq_hi32, w_hi32);

        sum64 = _mm256_add_epi64(
            sum64,
            cvt_i32x4_to_i64x4_avx2(_mm256_castsi256_si128(prod_lo32))
        );
        sum64 = _mm256_add_epi64(
            sum64,
            cvt_i32x4_to_i64x4_avx2(_mm256_extracti128_si256(prod_lo32, 1))
        );
        sum64 = _mm256_add_epi64(
            sum64,
            cvt_i32x4_to_i64x4_avx2(_mm256_castsi256_si128(prod_hi32))
        );
        sum64 = _mm256_add_epi64(
            sum64,
            cvt_i32x4_to_i64x4_avx2(_mm256_extracti128_si256(prod_hi32, 1))
        );
    }

    i64 total = horizontal_sum_epi64_avx2(sum64);
    for (; i < count; ++i)
        total += static_cast<i64>(screlu(acc[i])) * static_cast<i64>(weights[i]);

    return total;
}
#endif

template<class Layout>
[[nodiscard]] int forward(
    const Position& pos,
    const Network<Layout>& net,
    const Accumulator<Layout>& stm_acc,
    const Accumulator<Layout>& nstm_acc
) noexcept {
    const Color stm = static_cast<Color>(pos.side_to_move);
    const int bucket = output_bucket<Layout>(pos, stm);
    const std::size_t base = static_cast<std::size_t>(bucket) * 2 * Layout::HiddenSize;
    const i16* w_stm = net.w1.data() + base;
    const i16* w_nstm = w_stm + Layout::HiddenSize;

    i64 output = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (net.forward_madd_safe) {
        output += dot_pair_screlu_i16_madd_avx512(
            stm_acc.data(),
            w_stm,
            nstm_acc.data(),
            w_nstm,
            Layout::HiddenSize
        );
    } else if (net.forward_i32_safe) {
        output += dot_pair_screlu_i16_i32_avx512(
            stm_acc.data(),
            w_stm,
            nstm_acc.data(),
            w_nstm,
            Layout::HiddenSize
        );
    } else {
        output += dot_screlu_i16_avx512(stm_acc.data(), w_stm, Layout::HiddenSize);
        output += dot_screlu_i16_avx512(nstm_acc.data(), w_nstm, Layout::HiddenSize);
    }
#elif defined(__AVX2__)
    if (net.forward_madd_safe) {
        output += dot_pair_screlu_i16_madd_avx2(
            stm_acc.data(),
            w_stm,
            nstm_acc.data(),
            w_nstm,
            Layout::HiddenSize
        );
    } else if (net.forward_i32_safe) {
        output += dot_pair_screlu_i16_i32_avx2(
            stm_acc.data(),
            w_stm,
            nstm_acc.data(),
            w_nstm,
            Layout::HiddenSize
        );
    } else {
        output += dot_screlu_i16_avx2(stm_acc.data(), w_stm, Layout::HiddenSize);
        output += dot_screlu_i16_avx2(nstm_acc.data(), w_nstm, Layout::HiddenSize);
    }
#else
    for (int i = 0; i < Layout::HiddenSize; ++i)
        output += static_cast<i64>(screlu(stm_acc[static_cast<std::size_t>(i)]))
            * static_cast<i64>(w_stm[i]);

    for (int i = 0; i < Layout::HiddenSize; ++i)
        output += static_cast<i64>(screlu(nstm_acc[static_cast<std::size_t>(i)]))
            * static_cast<i64>(w_nstm[i]);
#endif

    output /= kQa;
    output += static_cast<i64>(net.b1[static_cast<std::size_t>(bucket)]);
    output *= net.scale;
    output /= static_cast<i64>(kQa) * kQb;

    return static_cast<int>(std::clamp<i64>(output, -32000, 32000));
}

template<class Layout>
[[nodiscard]] int forward_reference(
    const Position& pos,
    const Network<Layout>& net,
    const Accumulator<Layout>& stm_acc,
    const Accumulator<Layout>& nstm_acc
) noexcept {
    const Color stm = static_cast<Color>(pos.side_to_move);
    const int bucket = output_bucket<Layout>(pos, stm);
    const std::size_t base = static_cast<std::size_t>(bucket) * 2 * Layout::HiddenSize;
    const i16* w_stm = net.w1.data() + base;
    const i16* w_nstm = w_stm + Layout::HiddenSize;

    i64 output = 0;
    for (int i = 0; i < Layout::HiddenSize; ++i)
        output += static_cast<i64>(screlu(stm_acc[static_cast<std::size_t>(i)]))
            * static_cast<i64>(w_stm[i]);

    for (int i = 0; i < Layout::HiddenSize; ++i)
        output += static_cast<i64>(screlu(nstm_acc[static_cast<std::size_t>(i)]))
            * static_cast<i64>(w_nstm[i]);

    output /= kQa;
    output += static_cast<i64>(net.b1[static_cast<std::size_t>(bucket)]);
    output *= net.scale;
    output /= static_cast<i64>(kQa) * kQb;

    return static_cast<int>(std::clamp<i64>(output, -32000, 32000));
}

template<class Layout>
[[nodiscard]] int eval_lazy(
    const Position& pos,
    const Network<Layout>& net
) noexcept {
    if (!net.loaded || !net.valid())
        return 0;

    thread_local Accumulator<Layout> white_acc{};
    thread_local Accumulator<Layout> black_acc{};

    rebuild_accumulator<Layout>(pos, WHITE, white_acc, net);
    rebuild_accumulator<Layout>(pos, BLACK, black_acc, net);

    const Color stm = static_cast<Color>(pos.side_to_move);
    const auto& stm_acc = stm == WHITE ? white_acc : black_acc;
    const auto& nstm_acc = stm == WHITE ? black_acc : white_acc;
    return forward<Layout>(pos, net, stm_acc, nstm_acc);
}

template<class Layout>
void refresh_p2_accumulator_from_cache(
    P2RefreshCache<Layout>& cache,
    const Network<Layout>& net,
    const Position& pos,
    Color perspective,
    u32 current_generation,
    Accumulator<Layout>& out
) noexcept {
    if (!net.loaded || !net.valid())
        return;

    cache.sync(current_generation);
    const int bucket = input_bucket<Layout>(pos, perspective);
    P2RefreshEntry<Layout>& entry =
        cache.entries[static_cast<std::size_t>(perspective)]
                     [static_cast<std::size_t>(bucket)];
    if (!entry.initialized)
        entry.reset_to_biases(net);

    for (int sq_idx = 0; sq_idx < SQ_NB; ++sq_idx) {
        const Square sq = static_cast<Square>(sq_idx);
        const Piece old_pc = entry.board[static_cast<std::size_t>(sq_idx)];
        const Piece new_pc = piece_on(pos, sq);
        if (old_pc == new_pc)
            continue;

        const int old_idx = feature_index<Layout>(
            perspective,
            bucket,
            old_pc,
            sq
        );
        if (old_idx >= 0)
            sub_feature<Layout>(entry.accumulation, net, old_idx);

        const int new_idx = feature_index<Layout>(
            perspective,
            bucket,
            new_pc,
            sq
        );
        if (new_idx >= 0)
            add_feature<Layout>(entry.accumulation, net, new_idx);

        entry.board[static_cast<std::size_t>(sq_idx)] = new_pc;
    }

    out = entry.accumulation;
}

} // namespace

template<class Layout>
struct P2StackStateSet {
    static constexpr std::size_t Capacity = 132;

    struct alignas(64) State {
        alignas(64) std::array<Accumulator<Layout>, COLOR_NB> accumulation{};
        P2MoveDiff diff{};
        std::uint8_t computed_mask = 0;
    };

    std::array<State, Capacity> states{};
    P2RefreshCache<Layout> refresh_cache{};
    std::size_t state_count = 1;
    u32 generation = 0;

    void sync_generation(u32 current) noexcept {
        if (generation == current)
            return;

        for (std::size_t i = 0; i < state_count; ++i)
            states[i].computed_mask = 0;
        refresh_cache.sync(current);
        generation = current;
    }

    void reset(u32 current_generation) noexcept {
        sync_generation(current_generation);
        state_count = 1;
        states[0].diff = {};
        states[0].computed_mask = 0;
    }

    void push(const Position& pos, Move move, u32 current_generation) noexcept {
        sync_generation(current_generation);
        assert(state_count < states.size());
        if (state_count >= states.size())
            return;

        State& next = states[state_count++];
        next.diff = make_p2_move_diff(pos, move);
        next.computed_mask = 0;
    }

    void pop() noexcept {
        assert(state_count > 1);
        if (state_count > 1)
            --state_count;
    }

    [[nodiscard]] const Accumulator<Layout>& ensure(
        const Position& pos,
        Color perspective,
        const Network<Layout>& net,
        u32 current_generation
    ) noexcept {
        sync_generation(current_generation);

        const std::uint8_t mask = p2_perspective_mask(perspective);
        const std::size_t current = state_count - 1;
        if ((states[current].computed_mask & mask) != 0)
            return states[current].accumulation[static_cast<std::size_t>(perspective)];

        std::size_t boundary = current;
        bool found_computed = false;
        for (std::size_t i = current;; --i) {
            if ((states[i].computed_mask & mask) != 0) {
                boundary = i;
                found_computed = true;
                break;
            }
            if (i > 0
                && states[i].diff[static_cast<std::size_t>(perspective)].refresh) {
                boundary = i;
                break;
            }
            if (i == 0) {
                boundary = 0;
                break;
            }
        }

        if (found_computed) {
            for (std::size_t i = boundary + 1; i <= current; ++i) {
                apply_p2_diff<Layout>(
                    states[i].accumulation[static_cast<std::size_t>(perspective)],
                    states[i - 1].accumulation[static_cast<std::size_t>(perspective)],
                    net,
                    states[i].diff[static_cast<std::size_t>(perspective)],
                    true
                );
                states[i].computed_mask = static_cast<std::uint8_t>(
                    states[i].computed_mask | mask
                );
            }
        } else {
            refresh_p2_accumulator_from_cache(
                refresh_cache,
                net,
                pos,
                perspective,
                generation,
                states[current].accumulation[static_cast<std::size_t>(perspective)]
            );
            states[current].computed_mask = static_cast<std::uint8_t>(
                states[current].computed_mask | mask
            );

            for (std::size_t i = current; i > boundary; --i) {
                apply_p2_diff<Layout>(
                    states[i - 1].accumulation[static_cast<std::size_t>(perspective)],
                    states[i].accumulation[static_cast<std::size_t>(perspective)],
                    net,
                    states[i].diff[static_cast<std::size_t>(perspective)],
                    false
                );
                states[i - 1].computed_mask = static_cast<std::uint8_t>(
                    states[i - 1].computed_mask | mask
                );
            }
        }

        return states[current].accumulation[static_cast<std::size_t>(perspective)];
    }
};

struct P2AccumulatorStack::Impl {
    std::unique_ptr<P2StackStateSet<P2Layout>> p2;
    std::unique_ptr<P2StackStateSet<P2ProLayout>> p2pro;

    [[nodiscard]] P2StackStateSet<P2Layout>& p2_state() {
        if (!p2)
            p2 = std::make_unique<P2StackStateSet<P2Layout>>();
        return *p2;
    }

    [[nodiscard]] P2StackStateSet<P2ProLayout>& p2pro_state() {
        if (!p2pro)
            p2pro = std::make_unique<P2StackStateSet<P2ProLayout>>();
        return *p2pro;
    }

    [[nodiscard]] static u32 current_generation() noexcept {
        return g_p2_generation.load(std::memory_order_acquire);
    }

    void reset() noexcept {
        const u32 current = current_generation();
        switch (g_p2_active) {
            case P2ActiveKind::P2:
                p2_state().reset(current);
                break;
            case P2ActiveKind::P2Pro:
                p2pro_state().reset(current);
                break;
            case P2ActiveKind::None:
                if (p2)
                    p2->reset(current);
                if (p2pro)
                    p2pro->reset(current);
                break;
        }
    }

    void push(const Position& pos, Move move) noexcept {
        const u32 current = current_generation();
        switch (g_p2_active) {
            case P2ActiveKind::P2:
                p2_state().push(pos, move, current);
                break;
            case P2ActiveKind::P2Pro:
                p2pro_state().push(pos, move, current);
                break;
            case P2ActiveKind::None:
                break;
        }
    }

    void pop() noexcept {
        switch (g_p2_active) {
            case P2ActiveKind::P2:
                if (p2)
                    p2->pop();
                break;
            case P2ActiveKind::P2Pro:
                if (p2pro)
                    p2pro->pop();
                break;
            case P2ActiveKind::None:
                break;
        }
    }

    [[nodiscard]] std::size_t size() const noexcept {
        switch (g_p2_active) {
            case P2ActiveKind::P2:
                return p2 ? p2->state_count : 1;
            case P2ActiveKind::P2Pro:
                return p2pro ? p2pro->state_count : 1;
            case P2ActiveKind::None:
                return 1;
        }
        return 1;
    }

    [[nodiscard]] const Accumulator<P2Layout>& ensure_p2(
        const Position& pos,
        Color perspective
    ) noexcept {
        return p2_state().ensure(
            pos,
            perspective,
            g_p2,
            current_generation()
        );
    }

    [[nodiscard]] const Accumulator<P2ProLayout>& ensure_p2pro(
        const Position& pos,
        Color perspective
    ) noexcept {
        return p2pro_state().ensure(
            pos,
            perspective,
            g_p2pro,
            current_generation()
        );
    }
};

P2AccumulatorStack::P2AccumulatorStack() noexcept
    : impl_(std::make_unique<Impl>()) {
    impl_->reset();
}

P2AccumulatorStack::~P2AccumulatorStack() = default;
P2AccumulatorStack::P2AccumulatorStack(P2AccumulatorStack&&) noexcept = default;
P2AccumulatorStack& P2AccumulatorStack::operator=(P2AccumulatorStack&&) noexcept = default;

void P2AccumulatorStack::reset() noexcept {
    impl_->reset();
}

void P2AccumulatorStack::push(const Position& pos, Move move) noexcept {
    impl_->push(pos, move);
}

void P2AccumulatorStack::pop() noexcept {
    impl_->pop();
}

std::size_t P2AccumulatorStack::size() const noexcept {
    return impl_->size();
}

bool load_p2(const std::string& path) {
    Network<P2Layout> p2_candidate{};
    if (load_network(p2_candidate, path)) {
        g_p2 = std::move(p2_candidate);
        clear_network(g_p2pro);
        g_p2_active = P2ActiveKind::P2;
        (void)next_p2_generation();
        return true;
    }

    const std::string p2_error = g_error;
    Network<P2ProLayout> p2pro_candidate{};
    if (load_network(p2pro_candidate, path)) {
        clear_network(g_p2);
        g_p2pro = std::move(p2pro_candidate);
        g_p2_active = P2ActiveKind::P2Pro;
        (void)next_p2_generation();
        return true;
    }

    g_p2_active = P2ActiveKind::None;
    clear_network(g_p2);
    clear_network(g_p2pro);
    g_error = "failed to load MNUE P2/P2Pro: P2: " + p2_error
        + "; P2Pro: " + g_error;
    return false;
}

bool load_p4(const std::string& path) {
    return load_network(g_p4, path);
}

template<class Layout>
[[nodiscard]] bool load_p2_family_from_memory(
    Network<Layout>& target,
    const char* data,
    std::size_t size,
    const std::string& path
) {
    struct MemBuf : std::streambuf {
        MemBuf(const char* b, const char* e) {
            char* gb = const_cast<char*>(b);
            char* ge = const_cast<char*>(e);
            setg(gb, gb, ge);
        }
    } buf(data, data + size);

    std::istream stream(&buf);
    Network<Layout> candidate{};
    if (!load_network_from_stream(
            stream,
            candidate,
            path,
            static_cast<std::uintmax_t>(size)
        )) {
        return false;
    }

    target = std::move(candidate);
    return true;
}

// Load the compile-time embedded P2/P2Pro network from .rodata.
// Uses a zero-copy memory streambuf to avoid an extra 20-28 MiB string allocation.
bool load_p2_embedded() {
    if (!mnue_p2_embedded_data || !mnue_p2_embedded_end) {
        g_error = "embedded MNUE P2/P2Pro not linked (missing MnueEmbedded.o)";
        return false;
    }

    const auto begin =
        reinterpret_cast<std::uintptr_t>(mnue_p2_embedded_data);
    const auto end =
        reinterpret_cast<std::uintptr_t>(mnue_p2_embedded_end);

    if (end <= begin) {
        g_error = "invalid embedded MNUE range";
        return false;
    }

    const std::size_t size = static_cast<std::size_t>(end - begin);

    if (load_p2_family_from_memory(
            g_p2,
            mnue_p2_embedded_data,
            size,
            kEmbeddedP2Filename
        )) {
        clear_network(g_p2pro);
        g_p2_active = P2ActiveKind::P2;
        (void)next_p2_generation();
        return true;
    }

    const std::string p2_error = g_error;
    if (load_p2_family_from_memory(
            g_p2pro,
            mnue_p2_embedded_data,
            size,
            kEmbeddedP2Filename
        )) {
        clear_network(g_p2);
        g_p2_active = P2ActiveKind::P2Pro;
        (void)next_p2_generation();
        return true;
    }

    g_p2_active = P2ActiveKind::None;
    clear_network(g_p2);
    clear_network(g_p2pro);
    g_error = "failed to load embedded MNUE P2/P2Pro: P2: " + p2_error
        + "; P2Pro: " + g_error;
    return false;
}

bool p2_embedded_available() noexcept {
    if (!mnue_p2_embedded_data || !mnue_p2_embedded_end)
        return false;
    const auto begin =
        reinterpret_cast<std::uintptr_t>(mnue_p2_embedded_data);
    const auto end =
        reinterpret_cast<std::uintptr_t>(mnue_p2_embedded_end);
    return end > begin;
}

void unload_p2() noexcept {
    clear_network(g_p2);
    clear_network(g_p2pro);
    g_p2_active = P2ActiveKind::None;
    (void)next_p2_generation();
}

void unload_p4() noexcept {
    clear_network(g_p4);
}

void unload_all() noexcept {
    unload_p2();
    unload_p4();
}

bool p2_loaded() noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:    return g_p2.loaded && g_p2.valid();
        case P2ActiveKind::P2Pro: return g_p2pro.loaded && g_p2pro.valid();
        case P2ActiveKind::None:  return false;
    }
    return false;
}

bool p4_loaded() noexcept {
    return g_p4.loaded;
}

const std::string& p2_path() noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:    return g_p2.path;
        case P2ActiveKind::P2Pro: return g_p2pro.path;
        case P2ActiveKind::None:  return kEmptyPath;
    }
    return kEmptyPath;
}

const std::string& p4_path() noexcept {
    return g_p4.path;
}

const std::string& last_error() noexcept {
    return g_error;
}

const char* eval_simd_name() noexcept {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    return "AVX-512BW";
#elif defined(__AVX2__)
    return "AVX2";
#else
    return "scalar";
#endif
}

const char* p2_eval_name() noexcept {
    return p2_kind_eval_name(g_p2_active);
}

const char* p2_arch_name() noexcept {
    return p2_kind_arch_name(g_p2_active);
}

const char* p2_short_name() noexcept {
    return p2_kind_short_name(g_p2_active);
}

int p2_input_size() noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:    return P2Layout::InputSize;
        case P2ActiveKind::P2Pro: return P2ProLayout::InputSize;
        case P2ActiveKind::None:  return 0;
    }
    return 0;
}

int p2_hidden_size() noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:    return P2Layout::HiddenSize;
        case P2ActiveKind::P2Pro: return P2ProLayout::HiddenSize;
        case P2ActiveKind::None:  return 0;
    }
    return 0;
}

int p2_input_buckets() noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:    return P2Layout::InputBuckets;
        case P2ActiveKind::P2Pro: return P2ProLayout::InputBuckets;
        case P2ActiveKind::None:  return 0;
    }
    return 0;
}

int p2_output_buckets() noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:    return P2Layout::OutputBuckets;
        case P2ActiveKind::P2Pro: return P2ProLayout::OutputBuckets;
        case P2ActiveKind::None:  return 0;
    }
    return 0;
}

std::size_t p2_file_bytes() noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:
            return static_cast<std::size_t>(file_bytes<P2Layout>());
        case P2ActiveKind::P2Pro:
            return static_cast<std::size_t>(file_bytes<P2ProLayout>());
        case P2ActiveKind::None:
            return 0;
    }
    return 0;
}

int eval_p2(const Position& pos) noexcept {
    struct StandaloneContext {
        P2AccumulatorStack stack{};
        Key key = 0;
        bool initialized = false;
    };

    thread_local StandaloneContext context{};
    if (!context.initialized || context.key != pos.key) {
        context.stack.reset();
        context.key = pos.key;
        context.initialized = true;
    }

    return eval_p2(pos, context.stack);
}

int eval_p2(const Position& pos, P2AccumulatorStack& stack) noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2: {
            if (!g_p2.loaded || !g_p2.valid())
                return 0;

            const auto& white_acc = stack.impl_->ensure_p2(pos, WHITE);
            const auto& black_acc = stack.impl_->ensure_p2(pos, BLACK);
            const Color stm = static_cast<Color>(pos.side_to_move);
            const auto& stm_acc = stm == WHITE ? white_acc : black_acc;
            const auto& nstm_acc = stm == WHITE ? black_acc : white_acc;
            return forward<P2Layout>(pos, g_p2, stm_acc, nstm_acc);
        }
        case P2ActiveKind::P2Pro: {
            if (!g_p2pro.loaded || !g_p2pro.valid())
                return 0;

            const auto& white_acc = stack.impl_->ensure_p2pro(pos, WHITE);
            const auto& black_acc = stack.impl_->ensure_p2pro(pos, BLACK);
            const Color stm = static_cast<Color>(pos.side_to_move);
            const auto& stm_acc = stm == WHITE ? white_acc : black_acc;
            const auto& nstm_acc = stm == WHITE ? black_acc : white_acc;
            return forward<P2ProLayout>(pos, g_p2pro, stm_acc, nstm_acc);
        }
        case P2ActiveKind::None:
            return 0;
    }
    return 0;
}

int debug_eval_p2_reference(const Position& pos) noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2: {
            if (!g_p2.loaded || !g_p2.valid())
                return 0;

            Accumulator<P2Layout> white_acc{};
            Accumulator<P2Layout> black_acc{};
            rebuild_accumulator<P2Layout>(pos, WHITE, white_acc, g_p2);
            rebuild_accumulator<P2Layout>(pos, BLACK, black_acc, g_p2);
            const Color stm = static_cast<Color>(pos.side_to_move);
            const auto& stm_acc = stm == WHITE ? white_acc : black_acc;
            const auto& nstm_acc = stm == WHITE ? black_acc : white_acc;
            return forward_reference<P2Layout>(pos, g_p2, stm_acc, nstm_acc);
        }
        case P2ActiveKind::P2Pro: {
            if (!g_p2pro.loaded || !g_p2pro.valid())
                return 0;

            Accumulator<P2ProLayout> white_acc{};
            Accumulator<P2ProLayout> black_acc{};
            rebuild_accumulator<P2ProLayout>(pos, WHITE, white_acc, g_p2pro);
            rebuild_accumulator<P2ProLayout>(pos, BLACK, black_acc, g_p2pro);
            const Color stm = static_cast<Color>(pos.side_to_move);
            const auto& stm_acc = stm == WHITE ? white_acc : black_acc;
            const auto& nstm_acc = stm == WHITE ? black_acc : white_acc;
            return forward_reference<P2ProLayout>(pos, g_p2pro, stm_acc, nstm_acc);
        }
        case P2ActiveKind::None:
            return 0;
    }
    return 0;
}

bool p2_i32_forward_enabled() noexcept {
#if (defined(__AVX512F__) && defined(__AVX512BW__)) || defined(__AVX2__)
    switch (g_p2_active) {
        case P2ActiveKind::P2:
            return g_p2.loaded && g_p2.valid() && g_p2.forward_i32_safe;
        case P2ActiveKind::P2Pro:
            return g_p2pro.loaded && g_p2pro.valid() && g_p2pro.forward_i32_safe;
        case P2ActiveKind::None:
            return false;
    }
    return false;
#else
    return false;
#endif
}

int p2_w1_max_abs() noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:
            return g_p2.loaded && g_p2.valid() ? g_p2.w1_max_abs : 0;
        case P2ActiveKind::P2Pro:
            return g_p2pro.loaded && g_p2pro.valid() ? g_p2pro.w1_max_abs : 0;
        case P2ActiveKind::None:
            return 0;
    }
    return 0;
}

int eval_p4_lazy(const Position& pos) noexcept {
    return eval_lazy(pos, g_p4);
}

template<class Layout>
void debug_dump_p2_features_impl(
    const Position& pos,
    const Network<Layout>& net,
    const char* label,
    std::ostream& out
) {
    out << "mnue debug " << label << '\n';
    out << "side_to_move " << pos.side_to_move << '\n';
    out << "hidden_size " << Layout::HiddenSize << '\n';

    const int out_bucket = output_bucket<Layout>(
        pos,
        static_cast<Color>(pos.side_to_move)
    );
    out << "output_bucket " << out_bucket << '\n';

    if (net.loaded && net.valid()) {
        out << "p2_loaded 1\n";
        out << "p2_eval_raw " << eval_p2(pos) << '\n';
    } else {
        out << "p2_loaded 0\n";
    }

    for (int persp = WHITE; persp <= BLACK; ++persp) {
        const Color perspective = static_cast<Color>(persp);
        const Square ksq = king_square_of(pos, perspective);
        const Square rksq = relative_square(perspective, ksq);
        const int bucket = input_bucket<Layout>(pos, perspective);

        out << "perspective " << persp
            << " king_sq " << static_cast<int>(ksq)
            << " rel_king_sq " << static_cast<int>(rksq)
            << " input_bucket " << bucket
            << '\n';

        Bitboard bb = pieces(pos);
        while (bb) {
            const Square sq = static_cast<Square>(
                std::countr_zero(static_cast<std::uint64_t>(bb))
            );
            bb &= (bb - 1);

            const Piece pc = piece_on(pos, sq);
            if (pc == PIECE_NONE)
                continue;

            const int idx = feature_index<Layout>(perspective, bucket, pc, sq);
            if (idx < 0)
                continue;

            out << "  feature"
                << " sq " << static_cast<int>(sq)
                << " rel_sq " << static_cast<int>(relative_square(perspective, sq))
                << " piece " << static_cast<int>(pc)
                << " color " << static_cast<int>(color_of(pc))
                << " type " << static_cast<int>(type_of(pc))
                << " idx " << idx
                << '\n';
        }
    }
}

void debug_dump_p2_features(const Position& pos, std::ostream& out) {
    switch (g_p2_active) {
        case P2ActiveKind::P2:
            debug_dump_p2_features_impl<P2Layout>(pos, g_p2, "p2", out);
            return;
        case P2ActiveKind::P2Pro:
            debug_dump_p2_features_impl<P2ProLayout>(pos, g_p2pro, "p2pro", out);
            return;
        case P2ActiveKind::None:
            out << "mnue debug p2\n";
            out << "p2_loaded 0\n";
            return;
    }
}

template<class Layout>
bool debug_check_p2_incremental_impl(
    const Position& pos,
    P2StackStateSet<Layout>& set,
    const Network<Layout>& net,
    const char* label,
    std::ostream& out
) noexcept {
    if (!net.loaded || !net.valid()) {
        out << "mnuecheck " << label << " loaded 0\n";
        return false;
    }

    const u32 current_generation = g_p2_generation.load(std::memory_order_acquire);
    set.sync_generation(current_generation);
    const std::size_t current = set.state_count - 1;
    const std::uint8_t valid_mask_before = set.states[current].computed_mask;
    const bool valid_before =
        valid_mask_before
        == static_cast<std::uint8_t>(
            p2_perspective_mask(WHITE) | p2_perspective_mask(BLACK)
        );
    const u32 generation_before = set.generation;

    const auto& white_acc = set.ensure(pos, WHITE, net, current_generation);
    const auto& black_acc = set.ensure(pos, BLACK, net, current_generation);
    const Color stm = static_cast<Color>(pos.side_to_move);
    const auto& stm_acc = stm == WHITE ? white_acc : black_acc;
    const auto& nstm_acc = stm == WHITE ? black_acc : white_acc;
    const int incremental = forward<Layout>(pos, net, stm_acc, nstm_acc);

    Accumulator<Layout> white_rebuild{};
    Accumulator<Layout> black_rebuild{};
    rebuild_accumulator<Layout>(pos, WHITE, white_rebuild, net);
    rebuild_accumulator<Layout>(pos, BLACK, black_rebuild, net);

    const auto& rebuilt_stm = stm == WHITE ? white_rebuild : black_rebuild;
    const auto& rebuilt_nstm = stm == WHITE ? black_rebuild : white_rebuild;
    const int rebuilt = forward<Layout>(pos, net, rebuilt_stm, rebuilt_nstm);

    const bool white_acc_equal = std::equal(
        set.states[current].accumulation[WHITE].begin(),
        set.states[current].accumulation[WHITE].end(),
        white_rebuild.begin()
    );
    const bool black_acc_equal = std::equal(
        set.states[current].accumulation[BLACK].begin(),
        set.states[current].accumulation[BLACK].end(),
        black_rebuild.begin()
    );
    const bool score_equal = incremental == rebuilt;
    const bool ok = score_equal && white_acc_equal && black_acc_equal;

    out << "mnuecheck " << label << " loaded 1"
        << " valid_before " << (valid_before ? 1 : 0)
        << " valid_mask_before " << static_cast<int>(valid_mask_before)
        << " generation_before " << generation_before
        << " generation_current " << g_p2_generation.load(std::memory_order_relaxed)
        << " stack_size " << set.state_count
        << " incremental " << incremental
        << " rebuild " << rebuilt
        << " score_equal " << (score_equal ? 1 : 0)
        << " white_acc_equal " << (white_acc_equal ? 1 : 0)
        << " black_acc_equal " << (black_acc_equal ? 1 : 0)
        << " ok " << (ok ? 1 : 0)
        << '\n';

    return ok;
}

bool debug_check_p2_incremental(
    const Position& pos,
    P2AccumulatorStack& stack,
    std::ostream& out
) noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:
            return debug_check_p2_incremental_impl<P2Layout>(
                pos,
                stack.impl_->p2_state(),
                g_p2,
                "p2",
                out
            );
        case P2ActiveKind::P2Pro:
            return debug_check_p2_incremental_impl<P2ProLayout>(
                pos,
                stack.impl_->p2pro_state(),
                g_p2pro,
                "p2pro",
                out
            );
        case P2ActiveKind::None:
            out << "mnuecheck p2 loaded 0\n";
            return false;
    }
    return false;
}

bool debug_check_p2_incremental(
    const Position& pos,
    std::ostream& out
) noexcept {
    P2AccumulatorStack stack{};
    stack.reset();
    return debug_check_p2_incremental(pos, stack, out);
}

int search_score(int v, const Position& pos) noexcept {
    (void)pos;
    return v;
}

int material_units(const Position& pos) noexcept {
    return mnue_material_units(pos);
}

WinRateParams win_rate_params(const Position& pos) noexcept {
    return mnue_win_rate_params(pos);
}

int to_cp(int v, const Position& pos) noexcept {
    return mnue_to_cp(v, pos);
}

int win_rate_model(int v, const Position& pos) noexcept {
    return mnue_wdl_from_cp(to_cp(v, pos), pos).win;
}

int search_score_to_cp(int score, const Position& pos) noexcept {
    return to_cp(score, pos);
}

int search_score_to_winrate(int score, const Position& pos) noexcept {
    return search_score_to_wdl(score, pos).win;
}

WdlTriplet search_score_to_wdl(int score, const Position& pos) noexcept {
    return mnue_wdl_from_cp(search_score_to_cp(score, pos), pos);
}

} // namespace magnus::mnue
