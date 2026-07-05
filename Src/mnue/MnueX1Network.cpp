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

#include "mnue/MnueX1Network.h"

#include "Memory.h"
#include "board/Position.h"
#include "mnue/MnueX1Features.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <limits>
#include <string>
#include <vector>

namespace magnus::mnue::x1 {
namespace {

using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using u8 = std::uint8_t;
using u32 = std::uint32_t;

constexpr u32 kMagic = 0x45554E4Du;
constexpr u32 kVersion = 2;
constexpr u32 kHeaderBytes = 64;
constexpr int kQa = 255;
constexpr int kQb = 64;

struct FileHeader {
    u32 magic;
    u32 version;
    u32 arch;
    u32 header_bytes;
    u32 input_size;
    u32 hidden_size;
    u32 input_buckets;
    u32 output_buckets;
    u32 l1_size;
    u32 l2_size;
    i32 scale;
    i32 qa;
    i32 qb;
    u32 feature_version;
    u32 flags;
    u32 reserved;
};

static_assert(sizeof(FileHeader) == kHeaderBytes);

struct Network {
    bool is_loaded = false;
    int scale = 400;
    std::string file_path{};
    std::string error{};

    // l0 is feature-major. Backend weights are output-major after the
    // trainer's transpose transform.
    std::vector<i16> l0w{};
    std::vector<i16> l0b{};
    std::vector<i16> l1w{};
    std::vector<std::int8_t> l1w_chunked{};
    std::vector<i16> l1b{};
    std::vector<i16> l2w{};
    std::vector<i16> l2b{};
    std::vector<i16> l3w{};
    std::vector<i32> l3b{};

    [[nodiscard]] bool valid() const noexcept {
        return l0w.size()
                == static_cast<std::size_t>(Layout::InputSize)
                    * Layout::HiddenSize
            && l0b.size() == static_cast<std::size_t>(Layout::HiddenSize)
            && l1w.size()
                == static_cast<std::size_t>(Layout::OutputBuckets)
                    * Layout::L1Size * Layout::HiddenSize
            && l1w_chunked.size() == l1w.size()
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
std::atomic<u32> g_generation{1};

[[nodiscard]] u32 next_generation() noexcept {
    u32 next = g_generation.load(std::memory_order_relaxed) + 1;
    if (next == 0)
        next = 1;
    g_generation.store(next, std::memory_order_release);
    return next;
}

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

[[nodiscard]] constexpr std::uintmax_t expected_file_bytes() noexcept {
    return kHeaderBytes
        + static_cast<std::uintmax_t>(Layout::InputSize)
            * Layout::HiddenSize * sizeof(i16)
        + static_cast<std::uintmax_t>(Layout::HiddenSize) * sizeof(i16)
        + static_cast<std::uintmax_t>(Layout::OutputBuckets)
            * Layout::L1Size * Layout::HiddenSize * sizeof(i16)
        + static_cast<std::uintmax_t>(Layout::OutputBuckets)
            * Layout::L1Size * sizeof(i16)
        + static_cast<std::uintmax_t>(Layout::OutputBuckets)
            * Layout::L2Size * Layout::L1Size * sizeof(i16)
        + static_cast<std::uintmax_t>(Layout::OutputBuckets)
            * Layout::L2Size * sizeof(i16)
        + static_cast<std::uintmax_t>(Layout::OutputBuckets)
            * Layout::L2Size * sizeof(i16)
        + static_cast<std::uintmax_t>(Layout::OutputBuckets) * sizeof(i32);
}

void clear(Network& network) noexcept {
    network = {};
}

[[nodiscard]] bool header_matches(const FileHeader& header) noexcept {
    return header.magic == kMagic
        && header.version == kVersion
        && header.arch == Layout::ArchId
        && header.header_bytes == kHeaderBytes
        && header.input_size == Layout::InputSize
        && header.hidden_size == Layout::HiddenSize
        && header.input_buckets == Layout::InputBuckets
        && header.output_buckets == Layout::OutputBuckets
        && header.l1_size == Layout::L1Size
        && header.l2_size == Layout::L2Size
        && header.qa == kQa
        && header.qb == kQb
        && header.feature_version == Layout::FeatureVersion
        && header.flags == 0;
}

template<std::size_t InputSize, std::size_t OutputSize>
void affine_qa_qb(
    const std::array<i32, InputSize>& input,
    const i16* weights,
    const i16* biases,
    std::array<i32, OutputSize>& output
) noexcept {
    for (std::size_t row = 0; row < OutputSize; ++row) {
        i64 sum = 0;
        const i16* row_weights = weights + row * InputSize;
        for (std::size_t column = 0; column < InputSize; ++column) {
            sum += static_cast<i64>(input[column])
                * static_cast<i64>(row_weights[column]);
        }
        output[row] = static_cast<i32>(sum / kQb)
            + static_cast<i32>(biases[row]);
    }
}

[[nodiscard]] constexpr i32 crelu(i32 value) noexcept {
    return std::clamp(value, 0, kQa);
}

[[nodiscard]] constexpr i32 screlu_qa(i32 value) noexcept {
    const i32 clipped = crelu(value);
    return clipped * clipped / kQa;
}

using Accumulator = std::array<i32, Layout::HiddenSize>;
using Pairwise = std::array<i32, Layout::HiddenSize / 2>;
using BackendInput = std::array<i32, Layout::HiddenSize>;
using FastAccumulator = std::array<i16, Layout::HiddenSize>;
using FastPairwise = std::array<u8, Layout::HiddenSize / 2>;
using FastBackendInput = std::array<u8, Layout::HiddenSize>;
using L1Output = std::array<i32, Layout::L1Size>;
using L2Output = std::array<i32, Layout::L2Size>;

void rebuild_accumulator(
    const Position& pos,
    const memory::Memory& mem,
    Color perspective,
    Accumulator& accumulator
) noexcept {
    for (int i = 0; i < Layout::HiddenSize; ++i)
        accumulator[static_cast<std::size_t>(i)] = g_network.l0b[
            static_cast<std::size_t>(i)
        ];

    FeatureList features{};
    const std::size_t count =
        collect_features(pos, mem, perspective, features);
    for (std::size_t active = 0; active < count; ++active) {
        const std::size_t feature = features[active];
        const i16* row = g_network.l0w.data()
            + feature * Layout::HiddenSize;
        for (int i = 0; i < Layout::HiddenSize; ++i) {
            accumulator[static_cast<std::size_t>(i)]
                += static_cast<i32>(row[i]);
        }
    }
}

void pairwise_crelu(
    const Accumulator& accumulator,
    Pairwise& output
) noexcept {
    constexpr std::size_t Half = Layout::HiddenSize / 2;
    for (std::size_t i = 0; i < Half; ++i) {
        output[i] = crelu(accumulator[i])
            * crelu(accumulator[i + Half])
            / kQa;
    }
}

void rebuild_accumulator_fast(
    const Position& pos,
    const memory::Memory& mem,
    Color perspective,
    FastAccumulator& accumulator
) noexcept {
    std::copy(
        g_network.l0b.begin(),
        g_network.l0b.end(),
        accumulator.begin()
    );

    FeatureList features{};
    const std::size_t count =
        collect_features(pos, mem, perspective, features);
    for (std::size_t active = 0; active < count; ++active) {
        const std::size_t feature = features[active];
        const i16* row = g_network.l0w.data()
            + feature * Layout::HiddenSize;

#if defined(__AVX2__)
        for (int i = 0; i < Layout::HiddenSize; i += 16) {
            const __m256i value = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(accumulator.data() + i)
            );
            const __m256i weight = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row + i)
            );
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(accumulator.data() + i),
                _mm256_add_epi16(value, weight)
            );
        }
#else
        for (int i = 0; i < Layout::HiddenSize; ++i) {
            accumulator[static_cast<std::size_t>(i)] = static_cast<i16>(
                static_cast<i32>(accumulator[static_cast<std::size_t>(i)])
                + static_cast<i32>(row[i])
            );
        }
#endif
    }
}

void add_feature_row(
    FastAccumulator& accumulator,
    std::size_t feature
) noexcept {
    const i16* row = g_network.l0w.data()
        + feature * Layout::HiddenSize;
#if defined(__AVX2__)
    for (int i = 0; i < Layout::HiddenSize; i += 16) {
        const __m256i value = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(accumulator.data() + i)
        );
        const __m256i weight = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(row + i)
        );
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(accumulator.data() + i),
            _mm256_add_epi16(value, weight)
        );
    }
#else
    for (int i = 0; i < Layout::HiddenSize; ++i) {
        accumulator[static_cast<std::size_t>(i)] = static_cast<i16>(
            static_cast<i32>(accumulator[static_cast<std::size_t>(i)])
            + static_cast<i32>(row[i])
        );
    }
#endif
}

void sub_feature_row(
    FastAccumulator& accumulator,
    std::size_t feature
) noexcept {
    const i16* row = g_network.l0w.data()
        + feature * Layout::HiddenSize;
#if defined(__AVX2__)
    for (int i = 0; i < Layout::HiddenSize; i += 16) {
        const __m256i value = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(accumulator.data() + i)
        );
        const __m256i weight = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(row + i)
        );
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(accumulator.data() + i),
            _mm256_sub_epi16(value, weight)
        );
    }
#else
    for (int i = 0; i < Layout::HiddenSize; ++i) {
        accumulator[static_cast<std::size_t>(i)] = static_cast<i16>(
            static_cast<i32>(accumulator[static_cast<std::size_t>(i)])
            - static_cast<i32>(row[i])
        );
    }
#endif
}

void copy_accumulator(
    FastAccumulator& destination,
    const FastAccumulator& source
) noexcept {
#if defined(__AVX2__)
    for (int i = 0; i < Layout::HiddenSize; i += 16) {
        const __m256i value = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(source.data() + i)
        );
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(destination.data() + i),
            value
        );
    }
#else
    destination = source;
#endif
}

void rebuild_from_features(
    const FeatureList& features,
    std::size_t count,
    FastAccumulator& accumulator
) noexcept {
    std::copy(
        g_network.l0b.begin(),
        g_network.l0b.end(),
        accumulator.begin()
    );
    for (std::size_t i = 0; i < count; ++i)
        add_feature_row(accumulator, features[i]);
}

void apply_feature_diff(
    FastAccumulator& destination,
    const FastAccumulator& source,
    const FeatureList& old_features,
    std::size_t old_count,
    const FeatureList& new_features,
    std::size_t new_count,
    AccumulatorStack::Stats& stats
) noexcept {
    copy_accumulator(destination, source);

    std::size_t old_index = 0;
    std::size_t new_index = 0;
    while (old_index < old_count || new_index < new_count) {
        if (old_index == old_count) {
            add_feature_row(destination, new_features[new_index++]);
            ++stats.added_rows;
            continue;
        }
        if (new_index == new_count) {
            sub_feature_row(destination, old_features[old_index++]);
            ++stats.removed_rows;
            continue;
        }

        const u16 old_feature = old_features[old_index];
        const u16 new_feature = new_features[new_index];
        if (old_feature == new_feature) {
            ++old_index;
            ++new_index;
        } else if (old_feature < new_feature) {
            sub_feature_row(destination, old_feature);
            ++stats.removed_rows;
            ++old_index;
        } else {
            add_feature_row(destination, new_feature);
            ++stats.added_rows;
            ++new_index;
        }
    }
}

void pairwise_crelu_fast(
    const FastAccumulator& accumulator,
    FastPairwise& output
) noexcept {
    constexpr std::size_t Half = Layout::HiddenSize / 2;
#if defined(__AVX2__)
    const __m256i zero16 = _mm256_setzero_si256();
    const __m256i clip16 = _mm256_set1_epi16(kQa);
    const __m256i one32 = _mm256_set1_epi32(1);

    for (std::size_t i = 0; i < Half; i += 16) {
        const __m256i left16 = _mm256_min_epi16(
            _mm256_max_epi16(
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        accumulator.data() + i
                    )
                ),
                zero16
            ),
            clip16
        );
        const __m256i right16 = _mm256_min_epi16(
            _mm256_max_epi16(
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        accumulator.data() + Half + i
                    )
                ),
                zero16
            ),
            clip16
        );

        const __m256i product_lo = _mm256_mullo_epi32(
            _mm256_cvtepu16_epi32(
                _mm256_castsi256_si128(left16)
            ),
            _mm256_cvtepu16_epi32(
                _mm256_castsi256_si128(right16)
            )
        );
        const __m256i product_hi = _mm256_mullo_epi32(
            _mm256_cvtepu16_epi32(
                _mm256_extracti128_si256(left16, 1)
            ),
            _mm256_cvtepu16_epi32(
                _mm256_extracti128_si256(right16, 1)
            )
        );

        const auto divide_255 = [&](const __m256i product) noexcept {
            __m256i quotient = _mm256_add_epi32(product, one32);
            quotient = _mm256_add_epi32(
                quotient,
                _mm256_srli_epi32(quotient, 8)
            );
            return _mm256_srli_epi32(quotient, 8);
        };

        __m256i packed16 = _mm256_packus_epi32(
            divide_255(product_lo),
            divide_255(product_hi)
        );
        packed16 = _mm256_permute4x64_epi64(packed16, 0xD8);
        const __m128i packed8 = _mm_packus_epi16(
            _mm256_castsi256_si128(packed16),
            _mm256_extracti128_si256(packed16, 1)
        );
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(output.data() + i),
            packed8
        );
    }
#else
    for (std::size_t i = 0; i < Half; ++i) {
        const i32 left = std::clamp<i32>(accumulator[i], 0, kQa);
        const i32 right = std::clamp<i32>(
            accumulator[i + Half],
            0,
            kQa
        );
        output[i] = static_cast<u8>(left * right / kQa);
    }
#endif
}

void affine_l1_sparse(
    const FastBackendInput& input,
    std::size_t bucket,
    const i16* biases,
    L1Output& output
) noexcept {
#if defined(__AVX2__)
    constexpr int ChunkSize = 4;
    constexpr int Chunks = Layout::HiddenSize / ChunkSize;
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256i sums0 = _mm256_setzero_si256();
    __m256i sums1 = _mm256_setzero_si256();

    const std::int8_t* bucket_weights = g_network.l1w_chunked.data()
        + bucket * Chunks * Layout::L1Size * ChunkSize;
    for (int chunk = 0; chunk < Chunks; ++chunk) {
        u32 packed_input = 0;
        std::memcpy(
            &packed_input,
            input.data() + chunk * ChunkSize,
            sizeof(packed_input)
        );
        if (packed_input == 0)
            continue;

        const __m256i values =
            _mm256_set1_epi32(static_cast<i32>(packed_input));
        const std::int8_t* chunk_weights = bucket_weights
            + chunk * Layout::L1Size * ChunkSize;
        const __m256i weights0 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(chunk_weights)
        );
        const __m256i weights1 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(chunk_weights + 32)
        );

        sums0 = _mm256_add_epi32(
            sums0,
            _mm256_madd_epi16(
                _mm256_maddubs_epi16(values, weights0),
                ones16
            )
        );
        sums1 = _mm256_add_epi32(
            sums1,
            _mm256_madd_epi16(
                _mm256_maddubs_epi16(values, weights1),
                ones16
            )
        );
    }

    alignas(32) std::array<i32, Layout::L1Size> sums{};
    _mm256_store_si256(
        reinterpret_cast<__m256i*>(sums.data()),
        sums0
    );
    _mm256_store_si256(
        reinterpret_cast<__m256i*>(sums.data() + 8),
        sums1
    );
    for (int row = 0; row < Layout::L1Size; ++row) {
        output[static_cast<std::size_t>(row)] =
            sums[static_cast<std::size_t>(row)] / kQb
            + static_cast<i32>(biases[row]);
    }
#else
    const i16* weights = g_network.l1w.data()
        + bucket * Layout::L1Size * Layout::HiddenSize;
    for (int row = 0; row < Layout::L1Size; ++row) {
        const i16* row_weights =
            weights + static_cast<std::size_t>(row) * Layout::HiddenSize;
        i64 sum = 0;
        for (int column = 0; column < Layout::HiddenSize; column += 4) {
            if (input[static_cast<std::size_t>(column)] == 0
                && input[static_cast<std::size_t>(column + 1)] == 0
                && input[static_cast<std::size_t>(column + 2)] == 0
                && input[static_cast<std::size_t>(column + 3)] == 0) {
                continue;
            }

            for (int offset = 0; offset < 4; ++offset) {
                sum += static_cast<i64>(
                    input[static_cast<std::size_t>(column + offset)]
                ) * static_cast<i64>(row_weights[column + offset]);
            }
        }

        output[static_cast<std::size_t>(row)] =
            static_cast<i32>(sum / kQb)
            + static_cast<i32>(biases[row]);
    }
#endif
}

[[nodiscard]] int forward_fast(
    const Position& pos,
    const FastAccumulator& white_accumulator,
    const FastAccumulator& black_accumulator
) noexcept {
    FastPairwise white_pairwise{};
    FastPairwise black_pairwise{};
    pairwise_crelu_fast(white_accumulator, white_pairwise);
    pairwise_crelu_fast(black_accumulator, black_pairwise);

    const Color stm = static_cast<Color>(pos.side_to_move);
    const FastPairwise& stm_pairwise =
        stm == WHITE ? white_pairwise : black_pairwise;
    const FastPairwise& ntm_pairwise =
        stm == WHITE ? black_pairwise : white_pairwise;

    FastBackendInput backend_input{};
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

    L1Output l1{};
    affine_l1_sparse(
        backend_input,
        bucket,
        g_network.l1b.data() + bucket * Layout::L1Size,
        l1
    );
    for (i32& value : l1)
        value = screlu_qa(value);

    L2Output l2{};
    affine_qa_qb(
        l1,
        g_network.l2w.data()
            + bucket * Layout::L2Size * Layout::L1Size,
        g_network.l2b.data() + bucket * Layout::L2Size,
        l2
    );
    for (i32& value : l2)
        value = screlu_qa(value);

    i64 output = g_network.l3b[bucket];
    const i16* output_weights =
        g_network.l3w.data() + bucket * Layout::L2Size;
    for (int i = 0; i < Layout::L2Size; ++i) {
        output += static_cast<i64>(l2[static_cast<std::size_t>(i)])
            * static_cast<i64>(output_weights[i]);
    }

    output *= g_network.scale;
    output /= static_cast<i64>(kQa) * kQb;
    return static_cast<int>(std::clamp<i64>(
        output,
        std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max()
    ));
}

} // namespace

struct AccumulatorStack::Impl {
    static constexpr std::size_t Capacity = 132;

    struct alignas(64) State {
        alignas(64) std::array<FastAccumulator, COLOR_NB> accumulation{};
        std::array<FeatureList, COLOR_NB> features{};
        std::array<std::uint8_t, COLOR_NB> feature_count{};
        std::uint8_t feature_mask = 0;
        std::uint8_t computed_mask = 0;
    };

    std::array<State, Capacity> states{};
    std::size_t state_count = 1;
    u32 generation = 0;
    AccumulatorStack::Stats statistics{};

    [[nodiscard]] static constexpr std::uint8_t mask(
        Color perspective
    ) noexcept {
        return static_cast<std::uint8_t>(
            1u << static_cast<unsigned>(perspective)
        );
    }

    void sync_generation() noexcept {
        const u32 current = g_generation.load(std::memory_order_acquire);
        if (generation == current)
            return;

        state_count = 1;
        states[0].feature_mask = 0;
        states[0].computed_mask = 0;
        generation = current;
        statistics = {};
    }

    void reset() noexcept {
        sync_generation();
        state_count = 1;
        states[0].feature_mask = 0;
        states[0].computed_mask = 0;
        statistics = {};
    }

    void push() noexcept {
        sync_generation();
        assert(state_count < states.size());
        if (state_count >= states.size())
            return;

        State& state = states[state_count++];
        state.feature_mask = 0;
        state.computed_mask = 0;
    }

    void pop() noexcept {
        assert(state_count > 1);
        if (state_count > 1)
            --state_count;
    }

    void ensure_features(
        State& state,
        const Position& pos,
        const memory::Memory& mem,
        Color perspective
    ) noexcept {
        (void)perspective;
        const std::uint8_t all_masks = static_cast<std::uint8_t>(
            mask(WHITE) | mask(BLACK)
        );
        if (state.feature_mask == all_masks)
            return;

        PerspectiveFeatureCounts counts{};
        collect_feature_pairs(pos, mem, state.features, counts);
        ++statistics.feature_builds;
        for (int index = WHITE; index <= BLACK; ++index) {
            FeatureList& features =
                state.features[static_cast<std::size_t>(index)];
            const std::size_t count =
                counts[static_cast<std::size_t>(index)];
            std::sort(features.begin(), features.begin() + count);
            state.feature_count[static_cast<std::size_t>(index)] =
                static_cast<std::uint8_t>(count);
        }
        state.feature_mask = all_masks;
    }

    [[nodiscard]] const FastAccumulator& ensure(
        const Position& pos,
        const memory::Memory& mem,
        Color perspective
    ) noexcept {
        sync_generation();

        const std::uint8_t perspective_mask = mask(perspective);
        const std::size_t current = state_count - 1;
        State& current_state = states[current];
        if ((current_state.computed_mask & perspective_mask) != 0) {
            return current_state.accumulation[
                static_cast<std::size_t>(perspective)
            ];
        }

        ensure_features(current_state, pos, mem, perspective);

        std::size_t source_index = current;
        bool found_source = false;
        while (source_index > 0) {
            --source_index;
            if ((states[source_index].computed_mask & perspective_mask) != 0) {
                found_source = true;
                break;
            }
        }

        FastAccumulator& destination = current_state.accumulation[
            static_cast<std::size_t>(perspective)
        ];
        const std::size_t perspective_index =
            static_cast<std::size_t>(perspective);
        const std::size_t current_count =
            current_state.feature_count[perspective_index];

        if (found_source) {
            State& source_state = states[source_index];
            assert(
                (source_state.feature_mask & perspective_mask) != 0
            );
            apply_feature_diff(
                destination,
                source_state.accumulation[perspective_index],
                source_state.features[perspective_index],
                source_state.feature_count[perspective_index],
                current_state.features[perspective_index],
                current_count,
                statistics
            );
            ++statistics.diff_updates;
        } else {
            rebuild_from_features(
                current_state.features[perspective_index],
                current_count,
                destination
            );
            ++statistics.rebuilds;
        }

        current_state.computed_mask = static_cast<std::uint8_t>(
            current_state.computed_mask | perspective_mask
        );
        return destination;
    }
};

AccumulatorStack::AccumulatorStack() noexcept
    : impl_(std::make_unique<Impl>()) {
    impl_->reset();
}

AccumulatorStack::~AccumulatorStack() = default;
AccumulatorStack::AccumulatorStack(AccumulatorStack&&) noexcept = default;
AccumulatorStack& AccumulatorStack::operator=(AccumulatorStack&&) noexcept =
    default;

void AccumulatorStack::reset() noexcept {
    impl_->reset();
}

void AccumulatorStack::push() noexcept {
    impl_->push();
}

void AccumulatorStack::pop() noexcept {
    impl_->pop();
}

std::size_t AccumulatorStack::size() const noexcept {
    return impl_->state_count;
}

AccumulatorStack::Stats AccumulatorStack::stats() const noexcept {
    return impl_->statistics;
}

bool load(const std::string& file_path) {
    clear(g_network);

    std::error_code error_code;
    const std::uintmax_t bytes =
        std::filesystem::file_size(file_path, error_code);
    if (error_code || bytes != expected_file_bytes()) {
        g_network.error = "MNUE-X1 file size mismatch";
        return false;
    }

    std::ifstream input(file_path, std::ios::binary);
    if (!input) {
        g_network.error = "could not open MNUE-X1 file";
        return false;
    }

    FileHeader header{};
    if (!read_exact(input, &header, 1) || !header_matches(header)) {
        g_network.error = "MNUE-X1 header mismatch";
        return false;
    }

    g_network.l0w.resize(
        static_cast<std::size_t>(Layout::InputSize) * Layout::HiddenSize
    );
    g_network.l0b.resize(Layout::HiddenSize);
    g_network.l1w.resize(
        static_cast<std::size_t>(Layout::OutputBuckets)
            * Layout::L1Size * Layout::HiddenSize
    );
    g_network.l1b.resize(
        static_cast<std::size_t>(Layout::OutputBuckets) * Layout::L1Size
    );
    g_network.l2w.resize(
        static_cast<std::size_t>(Layout::OutputBuckets)
            * Layout::L2Size * Layout::L1Size
    );
    g_network.l2b.resize(
        static_cast<std::size_t>(Layout::OutputBuckets) * Layout::L2Size
    );
    g_network.l3w.resize(
        static_cast<std::size_t>(Layout::OutputBuckets) * Layout::L2Size
    );
    g_network.l3b.resize(Layout::OutputBuckets);

    if (!read_exact(input, g_network.l0w.data(), g_network.l0w.size())
        || !read_exact(input, g_network.l0b.data(), g_network.l0b.size())
        || !read_exact(input, g_network.l1w.data(), g_network.l1w.size())
        || !read_exact(input, g_network.l1b.data(), g_network.l1b.size())
        || !read_exact(input, g_network.l2w.data(), g_network.l2w.size())
        || !read_exact(input, g_network.l2b.data(), g_network.l2b.size())
        || !read_exact(input, g_network.l3w.data(), g_network.l3w.size())
        || !read_exact(input, g_network.l3b.data(), g_network.l3b.size())) {
        clear(g_network);
        g_network.error = "truncated MNUE-X1 payload";
        return false;
    }

    constexpr std::size_t ChunkSize = 4;
    constexpr std::size_t Chunks = Layout::HiddenSize / ChunkSize;
    g_network.l1w_chunked.resize(g_network.l1w.size());
    for (std::size_t bucket = 0; bucket < Layout::OutputBuckets; ++bucket) {
        for (std::size_t chunk = 0; chunk < Chunks; ++chunk) {
            for (std::size_t row = 0; row < Layout::L1Size; ++row) {
                for (std::size_t offset = 0; offset < ChunkSize; ++offset) {
                    const i16 weight = g_network.l1w[
                        (bucket * Layout::L1Size + row)
                            * Layout::HiddenSize
                        + chunk * ChunkSize + offset
                    ];
                    if (weight < std::numeric_limits<std::int8_t>::min()
                        || weight > std::numeric_limits<std::int8_t>::max()) {
                        clear(g_network);
                        g_network.error =
                            "MNUE-X1 L1 weight does not fit i8";
                        return false;
                    }

                    g_network.l1w_chunked[
                        ((bucket * Chunks + chunk) * Layout::L1Size + row)
                            * ChunkSize
                        + offset
                    ] = static_cast<std::int8_t>(weight);
                }
            }
        }
    }

    if (!g_network.valid()) {
        clear(g_network);
        g_network.error = "invalid MNUE-X1 tensor dimensions";
        return false;
    }

    g_network.scale = header.scale;
    g_network.file_path = file_path;
    g_network.is_loaded = true;
    (void)next_generation();
    return true;
}

void unload() noexcept {
    clear(g_network);
    (void)next_generation();
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

int evaluate_reference(
    const Position& pos,
    const memory::Memory& mem
) noexcept {
    if (!loaded())
        return 0;

    Accumulator white_accumulator{};
    Accumulator black_accumulator{};
    rebuild_accumulator(pos, mem, WHITE, white_accumulator);
    rebuild_accumulator(pos, mem, BLACK, black_accumulator);

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

    L1Output l1{};
    affine_qa_qb(
        backend_input,
        g_network.l1w.data()
            + bucket * Layout::L1Size * Layout::HiddenSize,
        g_network.l1b.data() + bucket * Layout::L1Size,
        l1
    );
    for (i32& value : l1)
        value = screlu_qa(value);

    L2Output l2{};
    affine_qa_qb(
        l1,
        g_network.l2w.data()
            + bucket * Layout::L2Size * Layout::L1Size,
        g_network.l2b.data() + bucket * Layout::L2Size,
        l2
    );
    for (i32& value : l2)
        value = screlu_qa(value);

    i64 output = g_network.l3b[bucket];
    const i16* output_weights =
        g_network.l3w.data() + bucket * Layout::L2Size;
    for (int i = 0; i < Layout::L2Size; ++i) {
        output += static_cast<i64>(l2[static_cast<std::size_t>(i)])
            * static_cast<i64>(output_weights[i]);
    }

    output *= g_network.scale;
    output /= static_cast<i64>(kQa) * kQb;
    return static_cast<int>(std::clamp<i64>(
        output,
        std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max()
    ));
}

int evaluate_fast(
    const Position& pos,
    const memory::Memory& mem
) noexcept {
    if (!loaded())
        return 0;

    FastAccumulator white_accumulator{};
    FastAccumulator black_accumulator{};
    rebuild_accumulator_fast(
        pos,
        mem,
        WHITE,
        white_accumulator
    );
    rebuild_accumulator_fast(
        pos,
        mem,
        BLACK,
        black_accumulator
    );
    return forward_fast(pos, white_accumulator, black_accumulator);
}

int evaluate_incremental(
    const Position& pos,
    const memory::Memory& mem,
    AccumulatorStack& stack
) noexcept {
    if (!loaded())
        return 0;

    const FastAccumulator& white =
        stack.impl_->ensure(pos, mem, WHITE);
    const FastAccumulator& black =
        stack.impl_->ensure(pos, mem, BLACK);
    return forward_fast(pos, white, black);
}

} // namespace magnus::mnue::x1
