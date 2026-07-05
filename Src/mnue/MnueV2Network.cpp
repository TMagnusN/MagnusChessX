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

#include "mnue/MnueV2Network.h"
#include "mnue/MnueV2Telemetry.h"

#include "Memory.h"
#include "board/Attack.h"
#include "board/MoveGen.h"
#include "board/Position.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>

#ifndef MAGNUS_MNUEV2_TELEMETRY
#define MAGNUS_MNUEV2_TELEMETRY 0
#endif

#if (defined(__x86_64__) || defined(_M_X64)) \
    && (defined(__GNUC__) || defined(__clang__))
#define MAGNUS_MNUEV2_HAS_AVX2_KERNEL 1
#define MAGNUS_MNUEV2_TARGET_AVX2 __attribute__((target("avx2"), noinline))
#define MAGNUS_MNUEV2_INLINE_AVX2 \
    __attribute__((target("avx2"), always_inline)) inline
#elif defined(__AVX2__)
#define MAGNUS_MNUEV2_HAS_AVX2_KERNEL 1
#define MAGNUS_MNUEV2_TARGET_AVX2
#define MAGNUS_MNUEV2_INLINE_AVX2 inline
#else
#define MAGNUS_MNUEV2_HAS_AVX2_KERNEL 0
#define MAGNUS_MNUEV2_TARGET_AVX2
#define MAGNUS_MNUEV2_INLINE_AVX2 inline
#endif

namespace magnus::mnue::v2 {
namespace {

constexpr std::array<u8, 8> kMagic{
    'M', 'N', 'U', 'E', 'V', '2', 0, 0
};
constexpr u32 kSectionCount = 12;
constexpr u32 kHeaderBytes = 668;
constexpr u32 kSectionDescriptorBytes = 48;
constexpr u32 kDtypeI8 = 1;
constexpr u32 kDtypeI16 = 2;
constexpr u32 kDtypeI32 = 3;

constexpr std::array<std::string_view, kSectionCount> kSectionNames{{
    "position.weights",
    "position.bias",
    "attack.weights",
    "attack.bias",
    "structure.weights",
    "structure.bias",
    "head1.weights",
    "head1.bias",
    "head2.weights",
    "head2.bias",
    "output.weights",
    "output.bias"
}};

constexpr std::array<u32, kSectionCount> kFixedDtypes{{
    kDtypeI16,
    kDtypeI32,
    0, // Attack weights may be int8 or int16.
    kDtypeI32,
    kDtypeI16,
    kDtypeI32,
    kDtypeI16,
    kDtypeI32,
    kDtypeI16,
    kDtypeI32,
    kDtypeI16,
    kDtypeI32
}};

struct SectionView {
    std::string name{};
    u32 dtype = 0;
    u64 offset = 0;
    u64 bytes = 0;
};

struct Network {
    bool is_loaded = false;
    std::string file_path{};
    std::string error{};
    std::size_t file_bytes = 0;

    int score_scale = 400;
    float position_scale = 1.0F / 255.0F;
    float attack_scale = 1.0F / 255.0F;
    float head_scale = 1.0F / 64.0F;
    float output_scale = 1.0F / 64.0F;
    AttackWeightType attack_type = AttackWeightType::Int16;

    std::vector<i16> position_weights{};
    std::vector<i32> position_bias{};
    std::vector<i16> attack_weights_i16{};
    std::vector<i8> attack_weights_i8{};
    std::vector<i32> attack_bias{};
    std::vector<i16> structure_weights{};
    std::vector<i32> structure_bias{};

    std::vector<i16> head1_weights{};
    std::vector<i16> head1_weights_transposed{};
    std::vector<i32> head1_bias{};
    std::vector<i16> head2_weights{};
    std::vector<i16> head2_weights_transposed{};
    std::vector<i32> head2_bias{};
    std::vector<i16> output_weights{};
    std::vector<i32> output_bias{};

    [[nodiscard]] bool valid() const noexcept {
        const bool attack_valid =
            attack_type == AttackWeightType::Int8
                ? attack_weights_i8.size()
                    == static_cast<std::size_t>(
                        Layout::AttackFeatureCount
                    ) * Layout::AttackAccumulatorWidth
                : attack_weights_i16.size()
                    == static_cast<std::size_t>(
                        Layout::AttackFeatureCount
                    ) * Layout::AttackAccumulatorWidth;
        return is_loaded
            && position_weights.size()
                == static_cast<std::size_t>(
                    Layout::PositionFeatureCount
                ) * Layout::PositionAccumulatorWidth
            && position_bias.size()
                == static_cast<std::size_t>(
                    Layout::PositionAccumulatorWidth
                )
            && attack_valid
            && attack_bias.size()
                == static_cast<std::size_t>(
                    Layout::AttackAccumulatorWidth
                )
            && structure_weights.size()
                == static_cast<std::size_t>(
                    Layout::StructureFeatureCount
                ) * Layout::StructureAccumulatorWidth
            && structure_bias.size()
                == static_cast<std::size_t>(
                    Layout::StructureAccumulatorWidth
                )
            && head1_weights.size()
                == static_cast<std::size_t>(Layout::OutputBuckets)
                    * Layout::HeadHidden1 * Layout::ConcatWidth
            && head1_weights_transposed.size()
                == head1_weights.size()
            && head1_bias.size()
                == static_cast<std::size_t>(Layout::OutputBuckets)
                    * Layout::HeadHidden1
            && head2_weights.size()
                == static_cast<std::size_t>(Layout::OutputBuckets)
                    * Layout::HeadHidden2 * Layout::HeadHidden1
            && head2_weights_transposed.size()
                == head2_weights.size()
            && head2_bias.size()
                == static_cast<std::size_t>(Layout::OutputBuckets)
                    * Layout::HeadHidden2
            && output_weights.size()
                == static_cast<std::size_t>(Layout::OutputBuckets)
                    * Layout::HeadHidden2
            && output_bias.size()
                == static_cast<std::size_t>(Layout::OutputBuckets);
    }
};

Network g_network{};
std::atomic<u32> g_generation{1};
std::atomic<bool> g_force_scalar{false};

using PositionAccumulator =
    std::array<i32, Layout::PositionAccumulatorWidth>;
using AttackAccumulator =
    std::array<i32, Layout::AttackAccumulatorWidth>;
using StructureAccumulator =
    std::array<i32, Layout::StructureAccumulatorWidth>;
using HeadInput = std::array<float, Layout::ConcatWidth>;
using Head1 = std::array<float, Layout::HeadHidden1>;
using Head2 = std::array<float, Layout::HeadHidden2>;

struct ForwardTrace {
    double input_sum = 0.0;
    double hidden1_sum = 0.0;
    double hidden2_sum = 0.0;
};

[[nodiscard]] u32 next_generation() noexcept {
    u32 next = g_generation.load(std::memory_order_relaxed) + 1;
    if (next == 0)
        next = 1;
    g_generation.store(next, std::memory_order_release);
    return next;
}

void clear_network() noexcept {
    g_network = {};
}

[[nodiscard]] bool checked_range(
    std::size_t offset,
    std::size_t length,
    std::size_t size
) noexcept {
    return offset <= size && length <= size - offset;
}

[[nodiscard]] bool read_u32_le(
    const std::vector<u8>& bytes,
    std::size_t& cursor,
    u32& value
) noexcept {
    if (!checked_range(cursor, 4, bytes.size()))
        return false;
    value = static_cast<u32>(bytes[cursor])
        | (static_cast<u32>(bytes[cursor + 1]) << 8)
        | (static_cast<u32>(bytes[cursor + 2]) << 16)
        | (static_cast<u32>(bytes[cursor + 3]) << 24);
    cursor += 4;
    return true;
}

[[nodiscard]] bool read_u64_le(
    const std::vector<u8>& bytes,
    std::size_t& cursor,
    u64& value
) noexcept {
    if (!checked_range(cursor, 8, bytes.size()))
        return false;
    value = 0;
    for (int shift = 0; shift < 64; shift += 8)
        value |= static_cast<u64>(bytes[cursor++]) << shift;
    return true;
}

[[nodiscard]] bool read_f32_le(
    const std::vector<u8>& bytes,
    std::size_t& cursor,
    float& value
) noexcept {
    u32 bits = 0;
    if (!read_u32_le(bytes, cursor, bits))
        return false;
    value = std::bit_cast<float>(bits);
    return true;
}

[[nodiscard]] u64 fnv1a64(
    const u8* data,
    std::size_t size
) noexcept {
    u64 hash = 0xcbf29ce484222325ULL;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

[[nodiscard]] std::size_t dtype_bytes(u32 dtype) noexcept {
    switch (dtype) {
        case kDtypeI8: return 1;
        case kDtypeI16: return 2;
        case kDtypeI32: return 4;
        default: return 0;
    }
}

[[nodiscard]] constexpr std::array<std::size_t, kSectionCount>
section_elements() noexcept {
    return {{
        static_cast<std::size_t>(Layout::PositionFeatureCount)
            * Layout::PositionAccumulatorWidth,
        Layout::PositionAccumulatorWidth,
        static_cast<std::size_t>(Layout::AttackFeatureCount)
            * Layout::AttackAccumulatorWidth,
        Layout::AttackAccumulatorWidth,
        static_cast<std::size_t>(Layout::StructureFeatureCount)
            * Layout::StructureAccumulatorWidth,
        Layout::StructureAccumulatorWidth,
        static_cast<std::size_t>(Layout::OutputBuckets)
            * Layout::HeadHidden1 * Layout::ConcatWidth,
        static_cast<std::size_t>(Layout::OutputBuckets)
            * Layout::HeadHidden1,
        static_cast<std::size_t>(Layout::OutputBuckets)
            * Layout::HeadHidden2 * Layout::HeadHidden1,
        static_cast<std::size_t>(Layout::OutputBuckets)
            * Layout::HeadHidden2,
        static_cast<std::size_t>(Layout::OutputBuckets)
            * Layout::HeadHidden2,
        Layout::OutputBuckets
    }};
}

template<typename T>
[[nodiscard]] bool decode_signed_section(
    const std::vector<u8>& bytes,
    const SectionView& section,
    std::vector<T>& output
) {
    static_assert(
        std::is_same_v<T, i8>
        || std::is_same_v<T, i16>
        || std::is_same_v<T, i32>
    );
    if (section.bytes % sizeof(T) != 0)
        return false;
    output.resize(static_cast<std::size_t>(section.bytes / sizeof(T)));
    std::size_t cursor = static_cast<std::size_t>(section.offset);
    for (T& value : output) {
        if constexpr (std::is_same_v<T, i8>) {
            if (!checked_range(cursor, 1, bytes.size()))
                return false;
            value = static_cast<i8>(bytes[cursor++]);
        } else if constexpr (std::is_same_v<T, i16>) {
            if (!checked_range(cursor, 2, bytes.size()))
                return false;
            const u16 bits = static_cast<u16>(bytes[cursor])
                | static_cast<u16>(bytes[cursor + 1] << 8);
            value = std::bit_cast<i16>(bits);
            cursor += 2;
        } else {
            u32 bits = 0;
            if (!read_u32_le(bytes, cursor, bits))
                return false;
            value = std::bit_cast<i32>(bits);
        }
    }
    return true;
}

[[nodiscard]] bool load_file_bytes(
    const std::string& path,
    std::vector<u8>& bytes
) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        g_network.error = "could not open MNUEv2 file: " + path;
        return false;
    }
    input.seekg(0, std::ios::end);
    const std::streamoff length = input.tellg();
    if (length < 0) {
        g_network.error = "could not determine MNUEv2 file size";
        return false;
    }
    input.seekg(0, std::ios::beg);
    bytes.resize(static_cast<std::size_t>(length));
    if (!bytes.empty()) {
        input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())
        );
    }
    if (!input) {
        g_network.error = "truncated MNUEv2 file";
        return false;
    }
    return true;
}

[[nodiscard]] bool parse_network(
    const std::string& path,
    const std::vector<u8>& bytes
) {
    if (bytes.size() < kHeaderBytes) {
        g_network.error = "MNUEv2 file is smaller than the fixed header";
        return false;
    }

    if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
        g_network.error = "bad MNUEv2 magic";
        return false;
    }

    std::size_t cursor = kMagic.size();
    std::array<u32, 15> fields{};
    for (u32& field : fields) {
        if (!read_u32_le(bytes, cursor, field)) {
            g_network.error = "truncated MNUEv2 fixed header";
            return false;
        }
    }

    const std::array<u32, 15> expected{{
        Layout::FormatVersion,
        Layout::ArchitectureId,
        kHeaderBytes,
        kSectionCount,
        Layout::PositionFeatureCount,
        Layout::AttackFeatureCount,
        Layout::StructureFeatureCount,
        Layout::PositionAccumulatorWidth,
        Layout::AttackAccumulatorWidth,
        Layout::StructureAccumulatorWidth,
        Layout::ConcatWidth,
        Layout::HeadHidden1,
        Layout::HeadHidden2,
        Layout::OutputBuckets,
        400
    }};
    if (fields != expected) {
        g_network.error =
            "MNUEv2 format/version/architecture/dimensions mismatch";
        return false;
    }

    float position_scale = 0.0F;
    float attack_scale = 0.0F;
    float head_scale = 0.0F;
    float output_scale = 0.0F;
    if (!read_f32_le(bytes, cursor, position_scale)
        || !read_f32_le(bytes, cursor, attack_scale)
        || !read_f32_le(bytes, cursor, head_scale)
        || !read_f32_le(bytes, cursor, output_scale)) {
        g_network.error = "truncated MNUEv2 scale fields";
        return false;
    }
    if (!std::isfinite(position_scale) || position_scale <= 0.0F
        || !std::isfinite(attack_scale) || attack_scale <= 0.0F
        || !std::isfinite(head_scale) || head_scale <= 0.0F
        || !std::isfinite(output_scale) || output_scale <= 0.0F) {
        g_network.error = "MNUEv2 contains invalid quantisation scales";
        return false;
    }

    u64 expected_checksum = 0;
    if (!read_u64_le(bytes, cursor, expected_checksum)) {
        g_network.error = "truncated MNUEv2 checksum";
        return false;
    }

    std::array<SectionView, kSectionCount> sections{};
    for (std::size_t index = 0; index < sections.size(); ++index) {
        if (!checked_range(cursor, 24, bytes.size())) {
            g_network.error = "truncated MNUEv2 section name";
            return false;
        }
        const char* name_data =
            reinterpret_cast<const char*>(bytes.data() + cursor);
        std::size_t name_length = 0;
        while (name_length < 24 && name_data[name_length] != '\0')
            ++name_length;
        if (name_length == 24) {
            g_network.error = "unterminated MNUEv2 section name";
            return false;
        }
        sections[index].name.assign(name_data, name_length);
        cursor += 24;

        u32 reserved = 0;
        if (!read_u32_le(bytes, cursor, sections[index].dtype)
            || !read_u32_le(bytes, cursor, reserved)
            || !read_u64_le(bytes, cursor, sections[index].offset)
            || !read_u64_le(bytes, cursor, sections[index].bytes)) {
            g_network.error = "truncated MNUEv2 section descriptor";
            return false;
        }
        if (reserved != 0) {
            g_network.error = "non-zero MNUEv2 section reserved field";
            return false;
        }
    }

    if (cursor != kHeaderBytes) {
        g_network.error = "MNUEv2 header byte count mismatch";
        return false;
    }

    const auto elements = section_elements();
    u64 expected_offset = kHeaderBytes;
    for (std::size_t index = 0; index < sections.size(); ++index) {
        const SectionView& section = sections[index];
        if (section.name != kSectionNames[index]) {
            g_network.error = "MNUEv2 section order/name mismatch at "
                + std::to_string(index);
            return false;
        }
        const bool attack_weights = index == 2;
        if (attack_weights) {
            if (section.dtype != kDtypeI8
                && section.dtype != kDtypeI16) {
                g_network.error =
                    "MNUEv2 Attack weights must be int8 or int16";
                return false;
            }
        } else if (section.dtype != kFixedDtypes[index]) {
            g_network.error = "MNUEv2 section dtype mismatch for "
                + section.name;
            return false;
        }
        const std::size_t element_bytes = dtype_bytes(section.dtype);
        if (element_bytes == 0
            || section.bytes
                != elements[index] * element_bytes) {
            g_network.error = "MNUEv2 section byte size mismatch for "
                + section.name;
            return false;
        }
        if (section.offset != expected_offset) {
            g_network.error =
                "MNUEv2 sections overlap, contain gaps, or are reordered";
            return false;
        }
        if (section.offset > bytes.size()
            || section.bytes > bytes.size() - section.offset) {
            g_network.error = "MNUEv2 section exceeds file bounds";
            return false;
        }
        expected_offset += section.bytes;
    }
    if (expected_offset != bytes.size()) {
        g_network.error =
            "MNUEv2 file has truncated sections or trailing payload";
        return false;
    }
    const float expected_attack_scale =
        sections[2].dtype == kDtypeI8
            ? 1.0F / 127.0F
            : 1.0F / 255.0F;
    if (std::bit_cast<u32>(position_scale)
            != std::bit_cast<u32>(1.0F / 255.0F)
        || std::bit_cast<u32>(attack_scale)
            != std::bit_cast<u32>(expected_attack_scale)
        || std::bit_cast<u32>(head_scale)
            != std::bit_cast<u32>(1.0F / 64.0F)
        || std::bit_cast<u32>(output_scale)
            != std::bit_cast<u32>(1.0F / 64.0F)) {
        g_network.error =
            "MNUEv2 quantisation scales do not match format v4";
        return false;
    }

    const u64 checksum = fnv1a64(
        bytes.data() + kHeaderBytes,
        bytes.size() - kHeaderBytes
    );
    if (checksum != expected_checksum) {
        g_network.error = "MNUEv2 payload checksum mismatch";
        return false;
    }

    if (!decode_signed_section(
            bytes,
            sections[0],
            g_network.position_weights
        )
        || !decode_signed_section(
            bytes,
            sections[1],
            g_network.position_bias
        )
        || !decode_signed_section(
            bytes,
            sections[3],
            g_network.attack_bias
        )
        || !decode_signed_section(
            bytes,
            sections[4],
            g_network.structure_weights
        )
        || !decode_signed_section(
            bytes,
            sections[5],
            g_network.structure_bias
        )
        || !decode_signed_section(
            bytes,
            sections[6],
            g_network.head1_weights
        )
        || !decode_signed_section(
            bytes,
            sections[7],
            g_network.head1_bias
        )
        || !decode_signed_section(
            bytes,
            sections[8],
            g_network.head2_weights
        )
        || !decode_signed_section(
            bytes,
            sections[9],
            g_network.head2_bias
        )
        || !decode_signed_section(
            bytes,
            sections[10],
            g_network.output_weights
        )
        || !decode_signed_section(
            bytes,
            sections[11],
            g_network.output_bias
        )) {
        g_network.error = "failed to decode MNUEv2 tensor payload";
        return false;
    }

    if (sections[2].dtype == kDtypeI8) {
        g_network.attack_type = AttackWeightType::Int8;
        if (!decode_signed_section(
                bytes,
                sections[2],
                g_network.attack_weights_i8
            )) {
            g_network.error = "failed to decode int8 Attack table";
            return false;
        }
        g_network.attack_weights_i16.clear();
    } else {
        g_network.attack_type = AttackWeightType::Int16;
        if (!decode_signed_section(
                bytes,
                sections[2],
                g_network.attack_weights_i16
            )) {
            g_network.error = "failed to decode int16 Attack table";
            return false;
        }
        g_network.attack_weights_i8.clear();
    }

    g_network.score_scale = static_cast<int>(fields[14]);
    g_network.position_scale = position_scale;
    g_network.attack_scale = attack_scale;
    g_network.head_scale = head_scale;
    g_network.output_scale = output_scale;
    g_network.head1_weights_transposed.resize(
        g_network.head1_weights.size()
    );
    for (int bucket = 0;
         bucket < Layout::OutputBuckets;
         ++bucket) {
        for (int column = 0;
             column < Layout::ConcatWidth;
             ++column) {
            for (int row = 0;
                 row < Layout::HeadHidden1;
                 ++row) {
                const std::size_t source =
                    (
                        static_cast<std::size_t>(bucket)
                            * Layout::HeadHidden1
                        + static_cast<std::size_t>(row)
                    ) * Layout::ConcatWidth
                    + static_cast<std::size_t>(column);
                const std::size_t destination =
                    (
                        static_cast<std::size_t>(bucket)
                            * Layout::ConcatWidth
                        + static_cast<std::size_t>(column)
                    ) * Layout::HeadHidden1
                    + static_cast<std::size_t>(row);
                g_network.head1_weights_transposed[destination] =
                    g_network.head1_weights[source];
            }
        }
    }
    g_network.head2_weights_transposed.resize(
        g_network.head2_weights.size()
    );
    for (int bucket = 0;
         bucket < Layout::OutputBuckets;
         ++bucket) {
        for (int column = 0;
             column < Layout::HeadHidden1;
             ++column) {
            for (int row = 0;
                 row < Layout::HeadHidden2;
                 ++row) {
                const std::size_t source =
                    (
                        static_cast<std::size_t>(bucket)
                            * Layout::HeadHidden2
                        + static_cast<std::size_t>(row)
                    ) * Layout::HeadHidden1
                    + static_cast<std::size_t>(column);
                const std::size_t destination =
                    (
                        static_cast<std::size_t>(bucket)
                            * Layout::HeadHidden1
                        + static_cast<std::size_t>(column)
                    ) * Layout::HeadHidden2
                    + static_cast<std::size_t>(row);
                g_network.head2_weights_transposed[destination] =
                    g_network.head2_weights[source];
            }
        }
    }
    g_network.file_path = path;
    g_network.file_bytes = bytes.size();
    g_network.is_loaded = true;
    if (!g_network.valid()) {
        g_network.error = "MNUEv2 decoded tensor dimensions are invalid";
        g_network.is_loaded = false;
        return false;
    }
    return true;
}

template<std::size_t Width, std::size_t Capacity>
void rebuild_i16(
    const std::vector<i16>& weights,
    const std::vector<i32>& bias,
    const FeatureList<Capacity>& features,
    std::array<i32, Width>& accumulator
) noexcept {
    std::copy(bias.begin(), bias.end(), accumulator.begin());
    for (std::size_t active = 0; active < features.count; ++active) {
        const std::size_t feature = features.indices[active];
        const i16* row = weights.data() + feature * Width;
        for (std::size_t column = 0; column < Width; ++column)
            accumulator[column] += static_cast<i32>(row[column]);
    }
}

void rebuild_attack(
    const AttackFeatureList& features,
    AttackAccumulator& accumulator
) noexcept {
    std::copy(
        g_network.attack_bias.begin(),
        g_network.attack_bias.end(),
        accumulator.begin()
    );
    if (g_network.attack_type == AttackWeightType::Int8) {
        for (std::size_t active = 0; active < features.count; ++active) {
            const std::size_t feature = features.indices[active];
            const i8* row = g_network.attack_weights_i8.data()
                + feature * Layout::AttackAccumulatorWidth;
            for (int column = 0;
                 column < Layout::AttackAccumulatorWidth;
                 ++column) {
                accumulator[column] += static_cast<i32>(row[column]);
            }
        }
    } else {
        rebuild_i16(
            g_network.attack_weights_i16,
            g_network.attack_bias,
            features,
            accumulator
        );
    }
}

[[nodiscard]] bool mnuev2_avx2_available() noexcept {
#if MAGNUS_MNUEV2_HAS_AVX2_KERNEL \
    && (defined(__GNUC__) || defined(__clang__))
    static const bool available = []() noexcept {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2");
    }();
    return available;
#elif MAGNUS_MNUEV2_HAS_AVX2_KERNEL
    return true;
#else
    return false;
#endif
}

#if MAGNUS_MNUEV2_HAS_AVX2_KERNEL
MAGNUS_MNUEV2_TARGET_AVX2 void add_i16_row_avx2(
    const i16* row,
    i32* accumulator,
    std::size_t width,
    int sign
) noexcept {
    assert((width % 8) == 0);
    for (std::size_t column = 0; column < width; column += 8) {
        const __m128i packed = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(row + column)
        );
        const __m256i values = _mm256_cvtepi16_epi32(packed);
        const __m256i current = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                accumulator + column
            )
        );
        const __m256i result = sign > 0
            ? _mm256_add_epi32(current, values)
            : _mm256_sub_epi32(current, values);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(accumulator + column),
            result
        );
    }
}
#endif

template<std::size_t Width>
void add_i16_row(
    const std::vector<i16>& weights,
    std::array<i32, Width>& accumulator,
    u32 feature,
    int sign
) noexcept {
    CycleScope cycles(CycleKind::RowApplication);
    const i16* row =
        weights.data() + static_cast<std::size_t>(feature) * Width;
#if MAGNUS_MNUEV2_HAS_AVX2_KERNEL
    if (!g_force_scalar.load(std::memory_order_relaxed)
        && mnuev2_avx2_available()) {
        add_i16_row_avx2(
            row,
            accumulator.data(),
            Width,
            sign
        );
        return;
    }
#endif
    for (std::size_t column = 0; column < Width; ++column) {
        accumulator[column] += sign * static_cast<i32>(row[column]);
    }
}

void add_attack_row_scalar(
    AttackAccumulator& accumulator,
    u32 feature,
    int sign
) noexcept {
    if (g_network.attack_type == AttackWeightType::Int8) {
        const i8* row = g_network.attack_weights_i8.data()
            + static_cast<std::size_t>(feature)
                * Layout::AttackAccumulatorWidth;
        for (int column = 0;
             column < Layout::AttackAccumulatorWidth;
             ++column) {
            accumulator[column] += sign * static_cast<i32>(row[column]);
        }
    } else {
        add_i16_row(
            g_network.attack_weights_i16,
            accumulator,
            feature,
            sign
        );
    }
}

[[nodiscard]] bool attack_avx2_available() noexcept {
    return mnuev2_avx2_available();
}

#if MAGNUS_MNUEV2_HAS_AVX2_KERNEL
MAGNUS_MNUEV2_TARGET_AVX2 void add_attack_row_avx2(
    AttackAccumulator& accumulator,
    u32 feature,
    int sign
) noexcept {
    if (g_network.attack_type != AttackWeightType::Int8) {
        add_attack_row_scalar(accumulator, feature, sign);
        return;
    }

    const i8* row = g_network.attack_weights_i8.data()
        + static_cast<std::size_t>(feature)
            * Layout::AttackAccumulatorWidth;
    for (int column = 0;
         column < Layout::AttackAccumulatorWidth;
         column += 16) {
        const __m128i packed = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(row + column)
        );
        const __m256i low =
            _mm256_cvtepi8_epi32(packed);
        const __m128i packed_high = _mm_srli_si128(packed, 8);
        const __m256i high =
            _mm256_cvtepi8_epi32(packed_high);
        const __m256i accumulator_low = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                accumulator.data() + column
            )
        );
        const __m256i accumulator_high = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                accumulator.data() + column + 8
            )
        );
        const __m256i result_low = sign > 0
            ? _mm256_add_epi32(accumulator_low, low)
            : _mm256_sub_epi32(accumulator_low, low);
        const __m256i result_high = sign > 0
            ? _mm256_add_epi32(accumulator_high, high)
            : _mm256_sub_epi32(accumulator_high, high);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(
                accumulator.data() + column
            ),
            result_low
        );
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(
                accumulator.data() + column + 8
            ),
            result_high
        );
    }
}
#endif

void add_attack_row(
    AttackAccumulator& accumulator,
    u32 feature,
    int sign
) noexcept {
    CycleScope cycles(CycleKind::RowApplication);
#if MAGNUS_MNUEV2_HAS_AVX2_KERNEL
    if (g_network.attack_type == AttackWeightType::Int8
        && !g_force_scalar.load(std::memory_order_relaxed)
        && attack_avx2_available()) {
        add_attack_row_avx2(accumulator, feature, sign);
        return;
    }
#endif
    add_attack_row_scalar(accumulator, feature, sign);
}

template<std::size_t Capacity, std::size_t Width, typename Update>
void apply_diff(
    const FeatureList<Capacity>& old_features,
    const FeatureList<Capacity>& new_features,
    const std::array<i32, Width>& source,
    std::array<i32, Width>& destination,
    BranchTelemetry& telemetry,
    MoveClass move_class,
    Update update
) noexcept {
    CycleScope cycles(CycleKind::SetDiff);
    destination = source;
    std::size_t old_index = 0;
    std::size_t new_index = 0;
    u64 changed = 0;
    while (old_index < old_features.count
           || new_index < new_features.count) {
        if (old_index == old_features.count) {
            update(destination, new_features.indices[new_index++], 1);
#if MAGNUS_MNUEV2_TELEMETRY
            ++telemetry.rows_added;
#endif
            ++changed;
            continue;
        }
        if (new_index == new_features.count) {
            update(destination, old_features.indices[old_index++], -1);
#if MAGNUS_MNUEV2_TELEMETRY
            ++telemetry.rows_removed;
#endif
            ++changed;
            continue;
        }
        const u32 old_feature = old_features.indices[old_index];
        const u32 new_feature = new_features.indices[new_index];
        if (old_feature == new_feature) {
            ++old_index;
            ++new_index;
#if MAGNUS_MNUEV2_TELEMETRY
            ++telemetry.cancelled_deltas;
#endif
        } else if (old_feature < new_feature) {
            update(destination, old_feature, -1);
            ++old_index;
#if MAGNUS_MNUEV2_TELEMETRY
            ++telemetry.rows_removed;
#endif
            ++changed;
        } else {
            update(destination, new_feature, 1);
            ++new_index;
#if MAGNUS_MNUEV2_TELEMETRY
            ++telemetry.rows_added;
#endif
            ++changed;
        }
    }
#if MAGNUS_MNUEV2_TELEMETRY
    telemetry.unique_rows_changed += changed;
    telemetry.max_rows_changed =
        std::max(telemetry.max_rows_changed, changed);
    const std::size_t histogram_index = std::min<std::size_t>(
        static_cast<std::size_t>(changed),
        telemetry.rows_changed_histogram.size() - 1
    );
    ++telemetry.rows_changed_histogram[histogram_index];
    const std::size_t move_index = static_cast<std::size_t>(move_class);
    if (move_index < MoveClassCount) {
        ++telemetry.updates_by_move_class[move_index];
        telemetry.rows_by_move_class[move_index] += changed;
    }
    ++telemetry.moves;
#else
    (void)telemetry;
    (void)move_class;
#endif
}

template<std::size_t Width, std::size_t PerspectiveWidth>
void append_branch(
    const std::array<i32, Width>& stm,
    const std::array<i32, Width>& ntm,
    float scale,
    HeadInput& output,
    std::size_t& offset,
    u64* clipping_count = nullptr
) noexcept {
    static_assert(PerspectiveWidth * 2 == Width);
    const auto append_perspective = [&](const auto& accumulator) {
        for (std::size_t index = 0; index < PerspectiveWidth; ++index) {
            const float left_raw =
                static_cast<float>(accumulator[index]) * scale;
            const float right_raw =
                static_cast<float>(
                    accumulator[index + PerspectiveWidth]
                ) * scale;
            const float left = std::clamp(left_raw, 0.0F, 1.0F);
            const float right = std::clamp(right_raw, 0.0F, 1.0F);
            if (clipping_count != nullptr) {
                *clipping_count += static_cast<u64>(left_raw != left);
                *clipping_count += static_cast<u64>(right_raw != right);
            }
            output[offset++] = left * right;
        }
    };
    append_perspective(stm);
    append_perspective(ntm);
}

[[nodiscard]] float screlu(float value) noexcept {
    const float clipped = std::clamp(value, 0.0F, 1.0F);
    return clipped * clipped;
}

#if MAGNUS_MNUEV2_HAS_AVX2_KERNEL
[[nodiscard]] MAGNUS_MNUEV2_INLINE_AVX2 float dot_i16_f32_avx2(
    const i16* weights,
    const float* input,
    int count
) noexcept {
    assert((count % 8) == 0);
    __m256 sum = _mm256_setzero_ps();
    for (int column = 0; column < count; column += 8) {
        const __m128i packed = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(weights + column)
        );
        const __m256 values = _mm256_cvtepi32_ps(
            _mm256_cvtepi16_epi32(packed)
        );
        sum = _mm256_add_ps(
            sum,
            _mm256_mul_ps(
                values,
                _mm256_loadu_ps(input + column)
            )
        );
    }
    const __m128 low = _mm256_castps256_ps128(sum);
    const __m128 high = _mm256_extractf128_ps(sum, 1);
    __m128 reduced = _mm_add_ps(low, high);
    reduced = _mm_hadd_ps(reduced, reduced);
    reduced = _mm_hadd_ps(reduced, reduced);
    return _mm_cvtss_f32(reduced);
}

MAGNUS_MNUEV2_TARGET_AVX2 void head_partial_avx2(
    const float* input,
    int input_offset,
    int width,
    std::size_t bucket,
    std::array<float, Layout::HeadHidden1>& output
) noexcept;

MAGNUS_MNUEV2_TARGET_AVX2 void hidden2_transposed_avx2(
    const Head1& hidden1,
    std::size_t bucket,
    Head2& hidden2
) noexcept;

MAGNUS_MNUEV2_TARGET_AVX2 [[nodiscard]] float dense_head_avx2(
    const HeadInput& input,
    std::size_t bucket
) noexcept {
    Head1 position_partial{};
    Head1 attack_partial{};
    Head1 structure_partial{};
    head_partial_avx2(
        input.data(),
        0,
        Layout::PositionAccumulatorWidth,
        bucket,
        position_partial
    );
    head_partial_avx2(
        input.data(),
        Layout::PositionAccumulatorWidth,
        Layout::AttackAccumulatorWidth,
        bucket,
        attack_partial
    );
    head_partial_avx2(
        input.data(),
        Layout::PositionAccumulatorWidth
            + Layout::AttackAccumulatorWidth,
        Layout::StructureAccumulatorWidth,
        bucket,
        structure_partial
    );
    Head1 hidden1{};
    const std::size_t head1_row_base =
        bucket * Layout::HeadHidden1;
    for (int row = 0; row < Layout::HeadHidden1; ++row) {
        const std::size_t global_row = head1_row_base + row;
        const float sum =
            static_cast<float>(g_network.head1_bias[global_row])
                * g_network.head_scale
            + position_partial[row]
            + attack_partial[row]
            + structure_partial[row];
        hidden1[row] = screlu(sum);
    }

    Head2 hidden2{};
    hidden2_transposed_avx2(hidden1, bucket, hidden2);

    const i16* output_weights = g_network.output_weights.data()
        + bucket * Layout::HeadHidden2;
    return static_cast<float>(g_network.output_bias[bucket])
            * g_network.output_scale
        + dot_i16_f32_avx2(
            output_weights,
            hidden2.data(),
            Layout::HeadHidden2
        ) * g_network.output_scale;
}

MAGNUS_MNUEV2_TARGET_AVX2 void head_partial_avx2(
    const float* input,
    int input_offset,
    int width,
    std::size_t bucket,
    std::array<float, Layout::HeadHidden1>& output
) noexcept {
    __m256 sums[4]{
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps()
    };
    const i16* weights =
        g_network.head1_weights_transposed.data()
        + (
            bucket * Layout::ConcatWidth
            + static_cast<std::size_t>(input_offset)
        ) * Layout::HeadHidden1;
    for (int column = 0; column < width; ++column) {
        const __m256 activation =
            _mm256_set1_ps(input[input_offset + column]);
        const i16* column_weights =
            weights
            + static_cast<std::size_t>(column)
                * Layout::HeadHidden1;
        for (int group = 0; group < 4; ++group) {
            const __m128i packed = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(
                    column_weights + group * 8
                )
            );
            const __m256 values = _mm256_cvtepi32_ps(
                _mm256_cvtepi16_epi32(packed)
            );
            sums[group] = _mm256_add_ps(
                sums[group],
                _mm256_mul_ps(values, activation)
            );
        }
    }
    const __m256 scale = _mm256_set1_ps(g_network.head_scale);
    for (int group = 0; group < 4; ++group) {
        _mm256_storeu_ps(
            output.data() + group * 8,
            _mm256_mul_ps(sums[group], scale)
        );
    }
}

MAGNUS_MNUEV2_TARGET_AVX2 void hidden2_transposed_avx2(
    const Head1& hidden1,
    std::size_t bucket,
    Head2& hidden2
) noexcept {
    __m256 sums[4]{
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps()
    };
    const i16* weights =
        g_network.head2_weights_transposed.data()
        + bucket * Layout::HeadHidden1 * Layout::HeadHidden2;
    for (int column = 0;
         column < Layout::HeadHidden1;
         ++column) {
        const __m256 activation =
            _mm256_set1_ps(hidden1[column]);
        const i16* column_weights =
            weights + column * Layout::HeadHidden2;
        for (int group = 0; group < 4; ++group) {
            const __m128i packed = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(
                    column_weights + group * 8
                )
            );
            const __m256 values = _mm256_cvtepi32_ps(
                _mm256_cvtepi16_epi32(packed)
            );
            sums[group] = _mm256_add_ps(
                sums[group],
                _mm256_mul_ps(values, activation)
            );
        }
    }
    const __m256 scale = _mm256_set1_ps(g_network.head_scale);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0F);
    const i32* bias = g_network.head2_bias.data()
        + bucket * Layout::HeadHidden2;
    for (int group = 0; group < 4; ++group) {
        __m256 values = _mm256_add_ps(
            _mm256_mul_ps(sums[group], scale),
            _mm256_mul_ps(
                _mm256_cvtepi32_ps(
                    _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(
                            bias + group * 8
                        )
                    )
                ),
                scale
            )
        );
        values = _mm256_min_ps(
            one,
            _mm256_max_ps(zero, values)
        );
        _mm256_storeu_ps(
            hidden2.data() + group * 8,
            _mm256_mul_ps(values, values)
        );
    }
}

MAGNUS_MNUEV2_TARGET_AVX2 [[nodiscard]] float dense_tail_avx2(
    const Head1& hidden1,
    std::size_t bucket
) noexcept {
    Head2 hidden2{};
    hidden2_transposed_avx2(hidden1, bucket, hidden2);
    const i16* output_weights = g_network.output_weights.data()
        + bucket * Layout::HeadHidden2;
    return static_cast<float>(g_network.output_bias[bucket])
            * g_network.output_scale
        + dot_i16_f32_avx2(
            output_weights,
            hidden2.data(),
            Layout::HeadHidden2
        ) * g_network.output_scale;
}

MAGNUS_MNUEV2_TARGET_AVX2 void build_head_input_avx2(
    const std::array<PositionAccumulator, COLOR_NB>& position,
    const std::array<AttackAccumulator, COLOR_NB>& attack,
    const std::array<StructureAccumulator, COLOR_NB>& structure,
    Color stm,
    HeadInput& output
) noexcept {
    const Color ntm = ~stm;
    std::size_t offset = 0;
    const auto append = [&offset, &output](
        const auto& first,
        const auto& second,
        int perspective_width,
        float scale
    ) noexcept {
        const __m256 scale_vector = _mm256_set1_ps(scale);
        const __m256 zero = _mm256_setzero_ps();
        const __m256 one = _mm256_set1_ps(1.0F);
        const auto append_perspective = [&](
            const auto& accumulator
        ) noexcept {
            for (int column = 0;
                 column < perspective_width;
                 column += 8) {
                __m256 left = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(
                        _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(
                                accumulator.data() + column
                            )
                        )
                    ),
                    scale_vector
                );
                __m256 right = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(
                        _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(
                                accumulator.data()
                                    + perspective_width
                                    + column
                            )
                        )
                    ),
                    scale_vector
                );
                left = _mm256_min_ps(
                    one,
                    _mm256_max_ps(zero, left)
                );
                right = _mm256_min_ps(
                    one,
                    _mm256_max_ps(zero, right)
                );
                _mm256_storeu_ps(
                    output.data() + offset,
                    _mm256_mul_ps(left, right)
                );
                offset += 8;
            }
        };
        append_perspective(first);
        append_perspective(second);
    };
    append(
        position[stm],
        position[ntm],
        Layout::PositionPerspectiveWidth,
        g_network.position_scale
    );
    append(
        attack[stm],
        attack[ntm],
        Layout::AttackPerspectiveWidth,
        g_network.attack_scale
    );
    append(
        structure[stm],
        structure[ntm],
        Layout::StructurePerspectiveWidth,
        g_network.position_scale
    );
    assert(offset == output.size());
}
#endif

[[nodiscard]] double forward(
    const Position& pos,
    const std::array<PositionAccumulator, COLOR_NB>& position,
    const std::array<AttackAccumulator, COLOR_NB>& attack,
    const std::array<StructureAccumulator, COLOR_NB>& structure,
    u64* attack_clips = nullptr,
    ForwardTrace* trace = nullptr
) noexcept {
    CycleScope cycles(CycleKind::SelectedHead);
    const Color stm = static_cast<Color>(pos.side_to_move);
    const Color ntm = ~stm;
    HeadInput input{};
#if MAGNUS_MNUEV2_HAS_AVX2_KERNEL
    if (attack_clips == nullptr
        && !g_force_scalar.load(std::memory_order_relaxed)
        && mnuev2_avx2_available()) {
        build_head_input_avx2(
            position,
            attack,
            structure,
            stm,
            input
        );
    } else
#endif
    {
        std::size_t offset = 0;
        append_branch<
            Layout::PositionAccumulatorWidth,
            Layout::PositionPerspectiveWidth
        >(
            position[stm],
            position[ntm],
            g_network.position_scale,
            input,
            offset
        );
        append_branch<
            Layout::AttackAccumulatorWidth,
            Layout::AttackPerspectiveWidth
        >(
            attack[stm],
            attack[ntm],
            g_network.attack_scale,
            input,
            offset,
            attack_clips
        );
        append_branch<
            Layout::StructureAccumulatorWidth,
            Layout::StructurePerspectiveWidth
        >(
            structure[stm],
            structure[ntm],
            g_network.position_scale,
            input,
            offset
        );
        assert(offset == input.size());
    }
    if (trace != nullptr) {
        for (const float value : input)
            trace->input_sum += value;
    }

    const std::size_t bucket =
        static_cast<std::size_t>(material_bucket(pos));
#if MAGNUS_MNUEV2_HAS_AVX2_KERNEL
    if (trace == nullptr
        && !g_force_scalar.load(std::memory_order_relaxed)
        && mnuev2_avx2_available()) {
        return static_cast<double>(dense_head_avx2(input, bucket));
    }
#endif
    Head1 hidden1{};
    const std::size_t head1_row_base =
        bucket * Layout::HeadHidden1;
    for (int row = 0; row < Layout::HeadHidden1; ++row) {
        const std::size_t global_row = head1_row_base + row;
        float sum =
            static_cast<float>(g_network.head1_bias[global_row])
            * g_network.head_scale;
        const i16* weights = g_network.head1_weights.data()
            + global_row * Layout::ConcatWidth;
        for (int column = 0;
             column < Layout::ConcatWidth;
             ++column) {
            sum += input[column]
                * static_cast<float>(weights[column])
                * g_network.head_scale;
        }
        hidden1[row] = screlu(sum);
        if (trace != nullptr)
            trace->hidden1_sum += hidden1[row];
    }

    Head2 hidden2{};
    const std::size_t head2_row_base =
        bucket * Layout::HeadHidden2;
    for (int row = 0; row < Layout::HeadHidden2; ++row) {
        const std::size_t global_row = head2_row_base + row;
        float sum =
            static_cast<float>(g_network.head2_bias[global_row])
            * g_network.head_scale;
        const i16* weights = g_network.head2_weights.data()
            + global_row * Layout::HeadHidden1;
        for (int column = 0;
             column < Layout::HeadHidden1;
             ++column) {
            sum += hidden1[column]
                * static_cast<float>(weights[column])
                * g_network.head_scale;
        }
        hidden2[row] = screlu(sum);
        if (trace != nullptr)
            trace->hidden2_sum += hidden2[row];
    }

    float output =
        static_cast<float>(g_network.output_bias[bucket])
        * g_network.output_scale;
    const i16* output_weights = g_network.output_weights.data()
        + bucket * Layout::HeadHidden2;
    for (int column = 0;
         column < Layout::HeadHidden2;
         ++column) {
        output += hidden2[column]
            * static_cast<float>(output_weights[column])
            * g_network.output_scale;
    }
    return static_cast<double>(output);
}

[[nodiscard]] int score_from_output(double output) noexcept {
    const double scaled =
        output * static_cast<double>(g_network.score_scale);
    if (scaled >= static_cast<double>(std::numeric_limits<int>::max()))
        return std::numeric_limits<int>::max();
    if (scaled <= static_cast<double>(std::numeric_limits<int>::min()))
        return std::numeric_limits<int>::min();
    return static_cast<int>(std::llround(scaled));
}

template<std::size_t Width>
[[nodiscard]] u64 accumulator_hash(
    const std::array<i32, Width>& accumulator
) noexcept {
    u64 hash = 0xcbf29ce484222325ULL;
    for (const i32 value : accumulator) {
        const u32 bits = std::bit_cast<u32>(value);
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<u8>(bits >> shift);
            hash *= 0x100000001b3ULL;
        }
    }
    return hash;
}

void rebuild_all(
    const EncodedFeatures& features,
    std::array<PositionAccumulator, COLOR_NB>& position,
    std::array<AttackAccumulator, COLOR_NB>& attack,
    std::array<StructureAccumulator, COLOR_NB>& structure
) noexcept {
    for (int perspective = WHITE; perspective <= BLACK; ++perspective) {
        rebuild_i16(
            g_network.position_weights,
            g_network.position_bias,
            features.position[perspective],
            position[perspective]
        );
        rebuild_attack(
            features.attack[perspective],
            attack[perspective]
        );
        rebuild_i16(
            g_network.structure_weights,
            g_network.structure_bias,
            features.structure[perspective],
            structure[perspective]
        );
    }
}

} // namespace

namespace {

enum class SemanticBranch : u8 {
    Position = 0,
    Attack = 1,
    Structure = 2
};

struct SlotUndo {
    u32 old_feature = Layout::NoFeature;
    u16 slot = 0;
    u8 branch = 0;
    u8 perspective = 0;
};

struct SemanticFrame {
    std::size_t undo_begin = 0;
    std::size_t row_begin = 0;
    int old_bucket = 0;
    int new_bucket = 0;
    Move move = Move(0);
    Piece moved_piece = PIECE_NONE;
    Piece captured_piece = PIECE_NONE;
    std::array<Square, 4> changed_squares{
        NO_SQ, NO_SQ, NO_SQ, NO_SQ
    };
    u8 changed_count = 0;
    std::array<u8, COLOR_NB> old_position_bucket{};
    Bitboard old_king_ray_targets = 0ULL;
    Bitboard affected_attack_sources = 0ULL;
    u8 dirty_branch_mask = 0;
    bool updated = false;
};

struct HeadPartialCache {
    std::array<
        std::array<
            std::array<float, Layout::HeadHidden1>,
            COLOR_NB
        >,
        3
    > partial{};
    std::array<std::array<u32, COLOR_NB>, 3> cached_epoch{};
    std::array<u32, 3> branch_epoch{{1, 1, 1}};
    int bucket = -1;

    void invalidate() noexcept {
        for (auto& branch : cached_epoch)
            branch.fill(0);
        branch_epoch = {{1, 1, 1}};
        bucket = -1;
    }
};

struct SemanticState {
    static constexpr std::size_t MaxPly = 132;
    static constexpr std::size_t UndoCapacity = 131072;
    static constexpr std::size_t RowCapacity = 131072;
    static constexpr std::size_t EvalCacheCapacity = 16384;
    static constexpr u32 FeatureMask = (1u << 18) - 1u;

    struct EvalCacheEntry {
        Key key = 0ULL;
        int score = 0;
        u32 epoch = 0;
    };

    bool initialized = false;
    u32 generation = 0;
    int current_ply = 0;
    int materialized_ply = 0;

    std::array<
        std::array<u32, Layout::PositionSlotCount>,
        COLOR_NB
    > position_slots{};
    std::array<
        std::array<u32, Layout::AttackSlotCount>,
        COLOR_NB
    > attack_slots{};
    std::array<
        std::array<u32, Layout::StructureSlotCount>,
        COLOR_NB
    > structure_slots{};

    std::array<
        std::array<u8, Layout::PositionFeatureCount>,
        COLOR_NB
    > position_refs{};
    std::array<
        std::array<u8, Layout::AttackFeatureCount>,
        COLOR_NB
    > attack_refs{};
    std::array<
        std::array<u8, Layout::StructureFeatureCount>,
        COLOR_NB
    > structure_refs{};

    AttackGraph attack_graph{};
    std::array<PositionAccumulator, COLOR_NB> position{};
    std::array<AttackAccumulator, COLOR_NB> attack{};
    std::array<StructureAccumulator, COLOR_NB> structure{};
    HeadPartialCache head_cache{};

    std::array<SemanticFrame, MaxPly> frames{};
    std::array<SlotUndo, UndoCapacity> undo_arena{};
    std::array<u32, RowCapacity> row_arena{};
    std::size_t undo_top = 0;
    std::size_t row_top = 0;
    std::array<EvalCacheEntry, EvalCacheCapacity> eval_cache{};
    u32 eval_cache_epoch = 1;
    Telemetry statistics{};

    static_assert(
        (EvalCacheCapacity & (EvalCacheCapacity - 1)) == 0
    );

    [[nodiscard]] static constexpr u32 pack_delta(
        SemanticBranch branch,
        Color perspective,
        u32 feature,
        bool add
    ) noexcept {
        return (feature & FeatureMask)
            | (static_cast<u32>(branch) << 18)
            | (static_cast<u32>(perspective) << 20)
            | (static_cast<u32>(add) << 21);
    }

    [[nodiscard]] static constexpr u32 delta_feature(u32 packed) noexcept {
        return packed & FeatureMask;
    }

    [[nodiscard]] static constexpr SemanticBranch delta_branch(
        u32 packed
    ) noexcept {
        return static_cast<SemanticBranch>((packed >> 18) & 3u);
    }

    [[nodiscard]] static constexpr Color delta_perspective(
        u32 packed
    ) noexcept {
        return static_cast<Color>((packed >> 20) & 1u);
    }

    [[nodiscard]] static constexpr bool delta_add(u32 packed) noexcept {
        return ((packed >> 21) & 1u) != 0u;
    }

    void clear_slots() noexcept {
        for (auto& perspective : position_slots)
            perspective.fill(Layout::NoFeature);
        for (auto& perspective : attack_slots)
            perspective.fill(Layout::NoFeature);
        for (auto& perspective : structure_slots)
            perspective.fill(Layout::NoFeature);
    }

    void reset() noexcept {
        initialized = false;
        current_ply = 0;
        materialized_ply = 0;
        undo_top = 0;
        row_top = 0;
        head_cache.invalidate();
        ++eval_cache_epoch;
        if (eval_cache_epoch == 0) {
            for (EvalCacheEntry& entry : eval_cache)
                entry.epoch = 0;
            eval_cache_epoch = 1;
        }
        statistics = {};
    }

    [[nodiscard]] bool probe_eval_cache(
        Key key,
        int& score
    ) const noexcept {
        const std::size_t index = static_cast<std::size_t>(
            (key ^ (key >> 32)) & (EvalCacheCapacity - 1)
        );
        const EvalCacheEntry& entry = eval_cache[index];
        if (entry.epoch != eval_cache_epoch || entry.key != key)
            return false;
        score = entry.score;
        return true;
    }

    void store_eval_cache(Key key, int score) noexcept {
        const std::size_t index = static_cast<std::size_t>(
            (key ^ (key >> 32)) & (EvalCacheCapacity - 1)
        );
        eval_cache[index] = {key, score, eval_cache_epoch};
    }

    [[nodiscard]] u8& ref(
        SemanticBranch branch,
        Color perspective,
        u32 feature
    ) noexcept {
        switch (branch) {
            case SemanticBranch::Position:
                return position_refs[perspective][feature];
            case SemanticBranch::Attack:
                return attack_refs[perspective][feature];
            case SemanticBranch::Structure:
                return structure_refs[perspective][feature];
        }
        assert(false);
        return position_refs[0][0];
    }

    [[nodiscard]] u32& slot(
        SemanticBranch branch,
        Color perspective,
        std::size_t index
    ) noexcept {
        switch (branch) {
            case SemanticBranch::Position:
                return position_slots[perspective][index];
            case SemanticBranch::Attack:
                return attack_slots[perspective][index];
            case SemanticBranch::Structure:
                return structure_slots[perspective][index];
        }
        assert(false);
        return position_slots[0][0];
    }

    void apply_row(u32 packed, bool inverse = false) noexcept {
        const u32 feature = delta_feature(packed);
        const Color perspective = delta_perspective(packed);
        int sign = delta_add(packed) ? 1 : -1;
        if (inverse)
            sign = -sign;
        switch (delta_branch(packed)) {
            case SemanticBranch::Position:
                add_i16_row(
                    g_network.position_weights,
                    position[perspective],
                    feature,
                    sign
                );
                break;
            case SemanticBranch::Attack:
                add_attack_row(
                    attack[perspective],
                    feature,
                    sign
                );
                break;
            case SemanticBranch::Structure:
                add_i16_row(
                    g_network.structure_weights,
                    structure[perspective],
                    feature,
                    sign
                );
                break;
        }
    }

    void prefetch_row(u32 packed) const noexcept {
        const u32 feature = delta_feature(packed);
        const char* row = nullptr;
        switch (delta_branch(packed)) {
            case SemanticBranch::Position:
                row = reinterpret_cast<const char*>(
                    g_network.position_weights.data()
                    + static_cast<std::size_t>(feature)
                        * Layout::PositionAccumulatorWidth
                );
                break;
            case SemanticBranch::Attack:
                if (g_network.attack_type == AttackWeightType::Int8) {
                    row = reinterpret_cast<const char*>(
                        g_network.attack_weights_i8.data()
                        + static_cast<std::size_t>(feature)
                            * Layout::AttackAccumulatorWidth
                    );
                } else {
                    row = reinterpret_cast<const char*>(
                        g_network.attack_weights_i16.data()
                        + static_cast<std::size_t>(feature)
                            * Layout::AttackAccumulatorWidth
                    );
                }
                break;
            case SemanticBranch::Structure:
                row = reinterpret_cast<const char*>(
                    g_network.structure_weights.data()
                    + static_cast<std::size_t>(feature)
                        * Layout::StructureAccumulatorWidth
                );
                break;
        }
        if (row == nullptr)
            return;
        _mm_prefetch(row, _MM_HINT_T0);
        _mm_prefetch(row + 64, _MM_HINT_T0);
        _mm_prefetch(row + 128, _MM_HINT_T0);
    }

#if MAGNUS_MNUEV2_HAS_AVX2_KERNEL
    MAGNUS_MNUEV2_TARGET_AVX2 void apply_rows_avx2(
        std::size_t begin,
        std::size_t end,
        bool inverse
    ) noexcept {
        const auto apply_i16 = [](
            const i16* row,
            i32* accumulator,
            int width,
            int sign
        ) noexcept {
            for (int column = 0; column < width; column += 8) {
                const __m128i packed = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(row + column)
                );
                const __m256i values =
                    _mm256_cvtepi16_epi32(packed);
                const __m256i current = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        accumulator + column
                    )
                );
                const __m256i result = sign > 0
                    ? _mm256_add_epi32(current, values)
                    : _mm256_sub_epi32(current, values);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(
                        accumulator + column
                    ),
                    result
                );
            }
        };
        const auto apply_i8 = [](
            const i8* row,
            i32* accumulator,
            int sign
        ) noexcept {
            for (int column = 0;
                 column < Layout::AttackAccumulatorWidth;
                 column += 16) {
                const __m128i packed = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(row + column)
                );
                const __m256i low =
                    _mm256_cvtepi8_epi32(packed);
                const __m256i high = _mm256_cvtepi8_epi32(
                    _mm_srli_si128(packed, 8)
                );
                const __m256i current_low =
                    _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(
                            accumulator + column
                        )
                    );
                const __m256i current_high =
                    _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(
                            accumulator + column + 8
                        )
                    );
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(
                        accumulator + column
                    ),
                    sign > 0
                        ? _mm256_add_epi32(current_low, low)
                        : _mm256_sub_epi32(current_low, low)
                );
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(
                        accumulator + column + 8
                    ),
                    sign > 0
                        ? _mm256_add_epi32(current_high, high)
                        : _mm256_sub_epi32(current_high, high)
                );
            }
        };
        const auto apply_i16_pair = [](
            const i16* first,
            const i16* second,
            i32* accumulator,
            int width,
            int first_sign
        ) noexcept {
            for (int column = 0; column < width; column += 8) {
                const __m256i first_values =
                    _mm256_cvtepi16_epi32(
                        _mm_loadu_si128(
                            reinterpret_cast<const __m128i*>(
                                first + column
                            )
                        )
                    );
                const __m256i second_values =
                    _mm256_cvtepi16_epi32(
                        _mm_loadu_si128(
                            reinterpret_cast<const __m128i*>(
                                second + column
                            )
                        )
                    );
                const __m256i delta = first_sign > 0
                    ? _mm256_sub_epi32(
                        first_values,
                        second_values
                    )
                    : _mm256_sub_epi32(
                        second_values,
                        first_values
                    );
                const __m256i current = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        accumulator + column
                    )
                );
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(
                        accumulator + column
                    ),
                    _mm256_add_epi32(current, delta)
                );
            }
        };
        const auto apply_i8_pair = [](
            const i8* first,
            const i8* second,
            i32* accumulator,
            int first_sign
        ) noexcept {
            for (int column = 0;
                 column < Layout::AttackAccumulatorWidth;
                 column += 8) {
                const __m256i first_values =
                    _mm256_cvtepi8_epi32(
                        _mm_loadl_epi64(
                            reinterpret_cast<const __m128i*>(
                                first + column
                            )
                        )
                    );
                const __m256i second_values =
                    _mm256_cvtepi8_epi32(
                        _mm_loadl_epi64(
                            reinterpret_cast<const __m128i*>(
                                second + column
                            )
                        )
                    );
                const __m256i delta = first_sign > 0
                    ? _mm256_sub_epi32(
                        first_values,
                        second_values
                    )
                    : _mm256_sub_epi32(
                        second_values,
                        first_values
                    );
                const __m256i current = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        accumulator + column
                    )
                );
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(
                        accumulator + column
                    ),
                    _mm256_add_epi32(current, delta)
                );
            }
        };

        for (std::size_t index = begin; index < end;) {
            constexpr std::size_t PrefetchDistance = 4;
            if (index + PrefetchDistance < end) {
                prefetch_row(
                    row_arena[index + PrefetchDistance]
                );
            }
            const u32 packed = row_arena[index];
            const u32 feature = delta_feature(packed);
            const Color perspective = delta_perspective(packed);
            int sign = delta_add(packed) ? 1 : -1;
            if (inverse)
                sign = -sign;
            const bool pair =
                index + 1 < end
                && delta_branch(row_arena[index + 1])
                    == delta_branch(packed)
                && delta_perspective(row_arena[index + 1])
                    == perspective
                && delta_add(row_arena[index + 1])
                    != delta_add(packed);
            if (pair) {
                const u32 second_feature =
                    delta_feature(row_arena[index + 1]);
                switch (delta_branch(packed)) {
                    case SemanticBranch::Position: {
                        const i16* first =
                            g_network.position_weights.data()
                            + static_cast<std::size_t>(feature)
                                * Layout::PositionAccumulatorWidth;
                        const i16* second =
                            g_network.position_weights.data()
                            + static_cast<std::size_t>(second_feature)
                                * Layout::PositionAccumulatorWidth;
                        apply_i16_pair(
                            first,
                            second,
                            position[perspective].data(),
                            Layout::PositionAccumulatorWidth,
                            sign
                        );
                        break;
                    }
                    case SemanticBranch::Attack:
                        if (g_network.attack_type
                            == AttackWeightType::Int8) {
                            const i8* first =
                                g_network.attack_weights_i8.data()
                                + static_cast<std::size_t>(feature)
                                    * Layout::AttackAccumulatorWidth;
                            const i8* second =
                                g_network.attack_weights_i8.data()
                                + static_cast<std::size_t>(second_feature)
                                    * Layout::AttackAccumulatorWidth;
                            apply_i8_pair(
                                first,
                                second,
                                attack[perspective].data(),
                                sign
                            );
                        } else {
                            const i16* first =
                                g_network.attack_weights_i16.data()
                                + static_cast<std::size_t>(feature)
                                    * Layout::AttackAccumulatorWidth;
                            const i16* second =
                                g_network.attack_weights_i16.data()
                                + static_cast<std::size_t>(second_feature)
                                    * Layout::AttackAccumulatorWidth;
                            apply_i16_pair(
                                first,
                                second,
                                attack[perspective].data(),
                                Layout::AttackAccumulatorWidth,
                                sign
                            );
                        }
                        break;
                    case SemanticBranch::Structure: {
                        const i16* first =
                            g_network.structure_weights.data()
                            + static_cast<std::size_t>(feature)
                                * Layout::StructureAccumulatorWidth;
                        const i16* second =
                            g_network.structure_weights.data()
                            + static_cast<std::size_t>(second_feature)
                                * Layout::StructureAccumulatorWidth;
                        apply_i16_pair(
                            first,
                            second,
                            structure[perspective].data(),
                            Layout::StructureAccumulatorWidth,
                            sign
                        );
                        break;
                    }
                }
                index += 2;
                continue;
            }
            switch (delta_branch(packed)) {
                case SemanticBranch::Position: {
                    const i16* row =
                        g_network.position_weights.data()
                        + static_cast<std::size_t>(feature)
                            * Layout::PositionAccumulatorWidth;
                    apply_i16(
                        row,
                        position[perspective].data(),
                        Layout::PositionAccumulatorWidth,
                        sign
                    );
                    break;
                }
                case SemanticBranch::Attack:
                    if (g_network.attack_type
                        == AttackWeightType::Int8) {
                        const i8* row =
                            g_network.attack_weights_i8.data()
                            + static_cast<std::size_t>(feature)
                                * Layout::AttackAccumulatorWidth;
                        apply_i8(
                            row,
                            attack[perspective].data(),
                            sign
                        );
                    } else {
                        const i16* row =
                            g_network.attack_weights_i16.data()
                            + static_cast<std::size_t>(feature)
                                * Layout::AttackAccumulatorWidth;
                        apply_i16(
                            row,
                            attack[perspective].data(),
                            Layout::AttackAccumulatorWidth,
                            sign
                        );
                    }
                    break;
                case SemanticBranch::Structure: {
                    const i16* row =
                        g_network.structure_weights.data()
                        + static_cast<std::size_t>(feature)
                            * Layout::StructureAccumulatorWidth;
                    apply_i16(
                        row,
                        structure[perspective].data(),
                        Layout::StructureAccumulatorWidth,
                        sign
                    );
                    break;
                }
            }
            ++index;
        }
    }
#endif

    void apply_rows(
        std::size_t begin,
        std::size_t end,
        bool inverse = false
    ) noexcept {
#if MAGNUS_MNUEV2_HAS_AVX2_KERNEL && !MAGNUS_MNUEV2_TELEMETRY
        if (!g_force_scalar.load(std::memory_order_relaxed)
            && mnuev2_avx2_available()) {
            apply_rows_avx2(begin, end, inverse);
            return;
        }
#endif
        if (!inverse) {
            for (std::size_t index = begin; index < end; ++index) {
                constexpr std::size_t PrefetchDistance = 4;
                if (index + PrefetchDistance < end) {
                    prefetch_row(
                        row_arena[index + PrefetchDistance]
                    );
                }
                apply_row(row_arena[index]);
            }
            return;
        }
        for (std::size_t index = end; index > begin; --index) {
            constexpr std::size_t PrefetchDistance = 4;
            if (index >= begin + PrefetchDistance + 1) {
                prefetch_row(
                    row_arena[index - PrefetchDistance - 1]
                );
            }
            apply_row(row_arena[index - 1], true);
        }
    }

    void initialize_slot(
        SemanticBranch branch,
        Color perspective,
        std::size_t index,
        u32 feature
    ) noexcept {
        slot(branch, perspective, index) = feature;
        if (feature == Layout::NoFeature)
            return;
        u8& count = ref(branch, perspective, feature);
        assert(count != std::numeric_limits<u8>::max());
        if (count++ == 0)
            apply_row(pack_delta(branch, perspective, feature, true));
    }

    void initialize(
        const Position& pos,
        const memory::Memory& mem
    ) noexcept {
        clear_slots();
        for (auto& perspective : position_refs)
            perspective.fill(0);
        for (auto& perspective : attack_refs)
            perspective.fill(0);
        for (auto& perspective : structure_refs)
            perspective.fill(0);
        for (int perspective = WHITE;
             perspective <= BLACK;
             ++perspective) {
            std::copy(
                g_network.position_bias.begin(),
                g_network.position_bias.end(),
                position[perspective].begin()
            );
            std::copy(
                g_network.attack_bias.begin(),
                g_network.attack_bias.end(),
                attack[perspective].begin()
            );
            std::copy(
                g_network.structure_bias.begin(),
                g_network.structure_bias.end(),
                structure[perspective].begin()
            );
        }
        (void)build_attack_graph(pos, mem, attack_graph);
        for (int perspective = WHITE;
             perspective <= BLACK;
             ++perspective) {
            const Color color = static_cast<Color>(perspective);
            for (std::size_t index = 0;
                 index < Layout::PositionSlotCount;
                 ++index) {
                initialize_slot(
                    SemanticBranch::Position,
                    color,
                    index,
                    compute_position_slot(
                        pos,
                        color,
                        static_cast<Square>(index)
                    )
                );
            }
            for (std::size_t index = 0;
                 index < Layout::AttackSlotCount;
                 ++index) {
                initialize_slot(
                    SemanticBranch::Attack,
                    color,
                    index,
                    compute_attack_slot(
                        pos,
                        mem,
                        attack_graph,
                        color,
                        index
                    )
                );
            }
            for (std::size_t index = 0;
                 index < Layout::StructureSlotCount;
                 ++index) {
                initialize_slot(
                    SemanticBranch::Structure,
                    color,
                    index,
                    compute_structure_slot(pos, color, index)
                );
            }
        }
        generation = g_generation.load(std::memory_order_acquire);
        initialized = true;
        current_ply = 0;
        materialized_ply = 0;
        undo_top = 0;
        row_top = 0;
        head_cache.invalidate();
        ++eval_cache_epoch;
        if (eval_cache_epoch == 0) {
            for (EvalCacheEntry& entry : eval_cache)
                entry.epoch = 0;
            eval_cache_epoch = 1;
        }
    }

    void ensure_initialized(
        const Position& pos,
        const memory::Memory& mem
    ) noexcept {
        const u32 current_generation =
            g_generation.load(std::memory_order_acquire);
        if (!initialized || generation != current_generation)
            initialize(pos, mem);
    }

    void emit_delta(
        SemanticBranch branch,
        Color perspective,
        u32 feature,
        bool add
    ) noexcept {
        assert(row_top < row_arena.size());
        if (row_top >= row_arena.size())
            return;
        row_arena[row_top++] =
            pack_delta(branch, perspective, feature, add);
        BranchTelemetry* branch_stats = nullptr;
        switch (branch) {
            case SemanticBranch::Position:
                branch_stats = &statistics.position;
                break;
            case SemanticBranch::Attack:
                branch_stats = &statistics.attack;
                break;
            case SemanticBranch::Structure:
                branch_stats = &statistics.structure;
                break;
        }
        if (add)
            ++branch_stats->rows_added;
        else
            ++branch_stats->rows_removed;
        frames[current_ply].dirty_branch_mask |= static_cast<u8>(
            1u << static_cast<unsigned>(branch)
        );
    }

    void update_slot(
        SemanticBranch branch,
        Color perspective,
        std::size_t index,
        u32 new_feature
    ) noexcept {
        u32& current = slot(branch, perspective, index);
        const u32 old_feature = current;
        if (old_feature == new_feature)
            return;
        assert(undo_top < undo_arena.size());
        if (undo_top >= undo_arena.size())
            return;
        undo_arena[undo_top++] = {
            old_feature,
            static_cast<u16>(index),
            static_cast<u8>(branch),
            static_cast<u8>(perspective)
        };
        if (old_feature != Layout::NoFeature) {
            u8& count = ref(branch, perspective, old_feature);
            assert(count > 0);
            if (--count == 0)
                emit_delta(branch, perspective, old_feature, false);
        }
        if (new_feature != Layout::NoFeature) {
            u8& count = ref(branch, perspective, new_feature);
            assert(count != std::numeric_limits<u8>::max());
            if (count++ == 0)
                emit_delta(branch, perspective, new_feature, true);
        }
        current = new_feature;
    }

    void refresh_all_slots(
        const Position& pos,
        const memory::Memory& mem,
        const SemanticFrame& frame
    ) noexcept {
        refresh_position_dirty(pos, frame);
        refresh_structure_dirty(pos, frame);
        refresh_attack_dirty(pos, mem, frame);
    }

    [[nodiscard]] static Bitboard ray_first_sliders(
        const Position& pos,
        const memory::Memory& mem,
        Square changed
    ) noexcept {
        const Bitboard diagonal_sliders =
            pieces(pos, WHITE, BISHOP)
            | pieces(pos, BLACK, BISHOP)
            | pieces(pos, WHITE, QUEEN)
            | pieces(pos, BLACK, QUEEN);
        const Bitboard orthogonal_sliders =
            pieces(pos, WHITE, ROOK)
            | pieces(pos, BLACK, ROOK)
            | pieces(pos, WHITE, QUEEN)
            | pieces(pos, BLACK, QUEEN);
        return (
            bishop_attacks_fast(mem, changed, pos.occupied)
            & diagonal_sliders
        ) | (
            rook_attacks_fast(mem, changed, pos.occupied)
            & orthogonal_sliders
        );
    }

    [[nodiscard]] static Bitboard occupied_slider_sources(
        const Position& pos,
        Bitboard sources
    ) noexcept {
        Bitboard result = 0ULL;
        while (sources != 0ULL) {
            const Square source = static_cast<Square>(
                std::countr_zero(static_cast<u64>(sources))
            );
            sources &= sources - 1;
            const Piece piece = piece_on(pos, source);
            if (piece == PIECE_NONE)
                continue;
            const PieceType type = type_of(piece);
            if (type == BISHOP || type == ROOK || type == QUEEN)
                result |= bb_of(source);
        }
        return result;
    }

    [[nodiscard]] static Bitboard king_ray_target_for_changed(
        const Position& pos,
        Color color,
        Square changed
    ) noexcept {
        const Square king = king_square(pos, color);
        const int df = file_of(changed) - file_of(king);
        const int dr = rank_of(changed) - rank_of(king);
        if (df != 0 && dr != 0 && std::abs(df) != std::abs(dr))
            return 0ULL;
        const int sf = (df > 0) - (df < 0);
        const int sr = (dr > 0) - (dr < 0);
        if (sf == 0 && sr == 0)
            return 0ULL;
        int file = file_of(king) + sf;
        int rank = rank_of(king) + sr;
        while (file >= 0 && file < 8 && rank >= 0 && rank < 8) {
            const Square square = rank * 8 + file;
            const Piece piece = piece_on(pos, square);
            if (piece != PIECE_NONE) {
                return color_of(piece) == color
                        && type_of(piece) != KING
                    ? bb_of(square)
                    : 0ULL;
            }
            file += sf;
            rank += sr;
        }
        return 0ULL;
    }

    void refresh_attack_slot_both(
        const Position& pos,
        const memory::Memory& mem,
        std::size_t index
    ) noexcept {
        std::array<u32, COLOR_NB> features{};
        compute_attack_slot_pair(
            pos,
            mem,
            attack_graph,
            index,
            features
        );
        for (int perspective = WHITE;
             perspective <= BLACK;
             ++perspective) {
            const Color color = static_cast<Color>(perspective);
            update_slot(
                SemanticBranch::Attack,
                color,
                index,
                features[perspective]
            );
        }
    }

    void refresh_attack_dirty(
        const Position& pos,
        const memory::Memory& mem,
        const SemanticFrame& frame
    ) noexcept {
        CycleScope cycles(CycleKind::AttackSemantic);
        constexpr std::size_t StatusBegin = 0;
        constexpr std::size_t RelationBegin =
            StatusBegin + Layout::AttackStatusSlotCount;
        constexpr std::size_t EdgeBegin =
            RelationBegin + Layout::AttackRelationSlotCount;
        constexpr std::size_t PressureBegin =
            EdgeBegin + Layout::AttackEdgeSlotCount;

        Bitboard affected_sources = 0ULL;
        std::array<Bitboard, 4> old_incoming{};
        for (u8 changed = 0;
             changed < frame.changed_count;
             ++changed) {
            const Square square = frame.changed_squares[changed];
            old_incoming[changed] = attack_graph.attackers_to[square];
            affected_sources |= bb_of(square);
            affected_sources |= occupied_slider_sources(
                pos,
                attack_graph.attackers_to[square]
            );
            affected_sources |= ray_first_sliders(pos, mem, square);
        }

        std::array<Bitboard, SQ_NB> old_attacks{};
        Bitboard sources = affected_sources;
        while (sources != 0ULL) {
            const Square source = static_cast<Square>(
                std::countr_zero(static_cast<u64>(sources))
            );
            sources &= sources - 1;
            old_attacks[source] = attack_graph.attacks_from[source];
        }
        frames[current_ply].affected_attack_sources =
            affected_sources;

        Bitboard dirty_targets = 0ULL;
        sources = affected_sources;
        while (sources != 0ULL) {
            const Square source = static_cast<Square>(
                std::countr_zero(static_cast<u64>(sources))
            );
            sources &= sources - 1;
            const Bitboard old_edges = old_attacks[source];
            const Bitboard new_edges =
                compute_piece_attacks(pos, mem, source);
            const Bitboard removed = old_edges & ~new_edges;
            const Bitboard added = new_edges & ~old_edges;
            Bitboard changed_edges = removed | added;
            bool source_identity_changed = false;
            for (u8 changed = 0;
                 changed < frame.changed_count;
                 ++changed) {
                source_identity_changed =
                    source_identity_changed
                    || frame.changed_squares[changed] == source;
            }
            const Bitboard semantic_edges =
                source_identity_changed
                    ? old_edges | new_edges
                    : changed_edges;
            // Stable edges from an unchanged slider need no work. When the
            // source square's identity changed, the edge feature and incoming
            // tactical summary can change even if the edge bit stayed set.
            dirty_targets |= semantic_edges & pos.occupied;
            while (changed_edges != 0ULL) {
                const Square target = static_cast<Square>(
                    std::countr_zero(static_cast<u64>(changed_edges))
                );
                changed_edges &= changed_edges - 1;
                if ((removed & bb_of(target)) != 0ULL)
                    attack_graph.attackers_to[target] &= ~bb_of(source);
                if ((added & bb_of(target)) != 0ULL)
                    attack_graph.attackers_to[target] |= bb_of(source);
            }
            attack_graph.attacks_from[source] = new_edges;
            // Attack-edge features only exist for occupied targets. Squares
            // whose identity/occupancy changed are handled explicitly below
            // with the union of old and new incoming attackers.
            Bitboard edge_targets =
                semantic_edges & pos.occupied;
            while (edge_targets != 0ULL) {
                const Square target = static_cast<Square>(
                    std::countr_zero(static_cast<u64>(edge_targets))
                );
                edge_targets &= edge_targets - 1;
                refresh_attack_slot_both(
                    pos,
                    mem,
                    EdgeBegin
                        + static_cast<std::size_t>(source * 64 + target)
                );
            }
        }

        for (u8 changed = 0;
             changed < frame.changed_count;
             ++changed) {
            const Square target = frame.changed_squares[changed];
            dirty_targets |= bb_of(target);
            Bitboard incoming =
                old_incoming[changed]
                | attack_graph.attackers_to[target];
            while (incoming != 0ULL) {
                const Square source = static_cast<Square>(
                    std::countr_zero(static_cast<u64>(incoming))
                );
                incoming &= incoming - 1;
                refresh_attack_slot_both(
                    pos,
                    mem,
                    EdgeBegin
                        + static_cast<std::size_t>(source * 64 + target)
                );
            }
        }

        if (type_of(frame.moved_piece) == KING) {
            dirty_targets |= pos.occupied;
        } else {
            dirty_targets |= frame.old_king_ray_targets;
            for (u8 changed = 0;
                 changed < frame.changed_count;
                 ++changed) {
                const Square square =
                    frame.changed_squares[changed];
                for (int color = WHITE;
                     color <= BLACK;
                     ++color) {
                    dirty_targets |= king_ray_target_for_changed(
                        pos,
                        static_cast<Color>(color),
                        square
                    );
                }
            }
        }
        while (dirty_targets != 0ULL) {
            const Square target = static_cast<Square>(
                std::countr_zero(static_cast<u64>(dirty_targets))
            );
            dirty_targets &= dirty_targets - 1;
            std::array<u32, COLOR_NB> status_features{};
            std::array<u32, COLOR_NB> relation_features{};
            compute_attack_summary_pairs(
                pos,
                attack_graph,
                target,
                status_features,
                relation_features
            );
            for (int perspective = WHITE;
                 perspective <= BLACK;
                 ++perspective) {
                const Color color =
                    static_cast<Color>(perspective);
                update_slot(
                    SemanticBranch::Attack,
                    color,
                    StatusBegin + static_cast<std::size_t>(target),
                    status_features[perspective]
                );
                update_slot(
                    SemanticBranch::Attack,
                    color,
                    RelationBegin
                        + static_cast<std::size_t>(target),
                    relation_features[perspective]
                );
            }
        }

        std::array<
            std::array<bool, Layout::AttackPressureSlotCount>,
            COLOR_NB
        > dirty_pressure{};
        const auto mark_pressure_piece =
            [&dirty_pressure](Piece piece) noexcept {
                if (piece == PIECE_NONE)
                    return;
                const Color absolute = color_of(piece);
                const std::size_t type =
                    static_cast<std::size_t>(type_of(piece));
                for (int perspective = WHITE;
                     perspective <= BLACK;
                     ++perspective) {
                    const std::size_t relative =
                        static_cast<std::size_t>(
                            static_cast<int>(absolute) ^ perspective
                        );
                    dirty_pressure[perspective][
                        relative * 6 + type
                    ] = true;
                }
            };
        mark_pressure_piece(frame.moved_piece);
        mark_pressure_piece(frame.captured_piece);
        sources = affected_sources;
        while (sources != 0ULL) {
            const Square source = static_cast<Square>(
                std::countr_zero(static_cast<u64>(sources))
            );
            sources &= sources - 1;
            mark_pressure_piece(piece_on(pos, source));
        }
        if (type_of(frame.moved_piece) == KING) {
            const Color king_color = color_of(frame.moved_piece);
            dirty_pressure[~king_color].fill(true);
        }
        for (int perspective = WHITE;
             perspective <= BLACK;
             ++perspective) {
            const Color color = static_cast<Color>(perspective);
            for (std::size_t pressure = 0;
                 pressure < Layout::AttackPressureSlotCount;
                 ++pressure) {
                if (!dirty_pressure[perspective][pressure])
                    continue;
                const std::size_t index = PressureBegin + pressure;
                update_slot(
                    SemanticBranch::Attack,
                    color,
                    index,
                    compute_attack_slot(
                        pos,
                        mem,
                        attack_graph,
                        color,
                        index
                    )
                );
            }
        }
    }

    [[nodiscard]] static u8 position_bucket_for(
        const Position& pos,
        Color perspective
    ) noexcept {
        Square square = king_square(pos, perspective);
        if (perspective == BLACK)
            square ^= 56;
        return static_cast<u8>(
            (rank_of(square) / 2) * 4 + file_of(square) / 2
        );
    }

    void refresh_position_dirty(
        const Position& pos,
        const SemanticFrame& frame
    ) noexcept {
        CycleScope cycles(CycleKind::PositionSemantic);
        for (int perspective = WHITE;
             perspective <= BLACK;
             ++perspective) {
            const Color color = static_cast<Color>(perspective);
            const bool bucket_changed =
                frame.old_position_bucket[perspective]
                != position_bucket_for(pos, color);
            if (bucket_changed) {
                for (int square = 0; square < SQ_NB; ++square) {
                    update_slot(
                        SemanticBranch::Position,
                        color,
                        static_cast<std::size_t>(square),
                        compute_position_slot(pos, color, square)
                    );
                }
                continue;
            }
            for (u8 changed = 0;
                 changed < frame.changed_count;
                 ++changed) {
                const Square square = frame.changed_squares[changed];
                update_slot(
                    SemanticBranch::Position,
                    color,
                    static_cast<std::size_t>(square),
                    compute_position_slot(pos, color, square)
                );
            }
        }
    }

    void refresh_structure_slot_both(
        const Position& pos,
        std::size_t index
    ) noexcept {
        std::array<u32, COLOR_NB> features{};
        compute_structure_slot_pair(pos, index, features);
        for (int perspective = WHITE;
             perspective <= BLACK;
             ++perspective) {
            const Color color = static_cast<Color>(perspective);
            update_slot(
                SemanticBranch::Structure,
                color,
                index,
                features[perspective]
            );
        }
    }

    void refresh_structure_dirty(
        const Position& pos,
        const SemanticFrame& frame
    ) noexcept {
        CycleScope cycles(CycleKind::StructureSemantic);
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

        const PieceType moved_type = type_of(frame.moved_piece);
        const bool pawn_event =
            moved_type == PAWN
            || move_is_ep(frame.move)
            || move_is_promotion(frame.move)
            || (
                frame.captured_piece != PIECE_NONE
                && type_of(frame.captured_piece) == PAWN
            );

        if (pawn_event) {
            unsigned dirty_files = 0;
            Bitboard dirty_file_squares = 0ULL;
            for (u8 changed = 0;
                 changed < frame.changed_count;
                 ++changed) {
                const int file = file_of(frame.changed_squares[changed]);
                for (int delta = -1; delta <= 1; ++delta) {
                    const int candidate = file + delta;
                    if (candidate >= 0 && candidate < 8)
                        dirty_files |= 1u << candidate;
                }
            }
            for (int file = 0; file < 8; ++file) {
                if ((dirty_files & (1u << file)) != 0)
                    dirty_file_squares |=
                        0x0101010101010101ULL << file;
            }
            for (int color = WHITE; color <= BLACK; ++color) {
                Bitboard pawns =
                    pieces(
                        pos,
                        static_cast<Color>(color),
                        PAWN
                    )
                    & dirty_file_squares;
                while (pawns != 0ULL) {
                    const Square square = static_cast<Square>(
                        std::countr_zero(static_cast<u64>(pawns))
                    );
                    pawns &= pawns - 1;
                    refresh_structure_slot_both(
                        pos,
                        PawnBegin
                            + static_cast<std::size_t>(color * 64 + square)
                    );
                    refresh_structure_slot_both(
                        pos,
                        BlockerBegin
                            + static_cast<std::size_t>(color * 64 + square)
                    );
                }
                // Clear slots for pawns that moved, were captured, promoted,
                // or were removed by en passant.
                for (u8 changed = 0;
                     changed < frame.changed_count;
                     ++changed) {
                    const Square square = frame.changed_squares[changed];
                    refresh_structure_slot_both(
                        pos,
                        PawnBegin
                            + static_cast<std::size_t>(color * 64 + square)
                    );
                    refresh_structure_slot_both(
                        pos,
                        BlockerBegin
                            + static_cast<std::size_t>(color * 64 + square)
                    );
                }
            }
            for (int file = 0; file < 8; ++file) {
                if ((dirty_files & (1u << file)) != 0) {
                    refresh_structure_slot_both(
                        pos,
                        FileBegin + static_cast<std::size_t>(file)
                    );
                }
            }
            for (std::size_t index = IslandBegin;
                 index < OutpostBegin;
                 ++index) {
                refresh_structure_slot_both(pos, index);
            }
            Bitboard outposts =
                pieces(pos, WHITE, KNIGHT)
                | pieces(pos, WHITE, BISHOP)
                | pieces(pos, BLACK, KNIGHT)
                | pieces(pos, BLACK, BISHOP);
            for (u8 changed = 0;
                 changed < frame.changed_count;
                 ++changed) {
                outposts |= bb_of(frame.changed_squares[changed]);
            }
            while (outposts != 0ULL) {
                const Square square = static_cast<Square>(
                    std::countr_zero(static_cast<u64>(outposts))
                );
                outposts &= outposts - 1;
                refresh_structure_slot_both(
                    pos,
                    OutpostBegin + static_cast<std::size_t>(square)
                );
            }
            for (std::size_t index = ComplexBegin;
                 index < BlockerBegin;
                 ++index) {
                refresh_structure_slot_both(pos, index);
            }
            return;
        }

        if (moved_type == KING || move_is_castle(frame.move)) {
            for (std::size_t index = ShelterBegin;
                 index < OutpostBegin;
                 ++index) {
                refresh_structure_slot_both(pos, index);
            }
        }

        for (u8 changed = 0;
             changed < frame.changed_count;
             ++changed) {
            const Square square = frame.changed_squares[changed];
            const Piece current_piece = piece_on(pos, square);
            const bool current_outpost_piece =
                current_piece != PIECE_NONE
                && (
                    type_of(current_piece) == KNIGHT
                    || type_of(current_piece) == BISHOP
                );
            const bool old_outpost_piece =
                (
                    square == from_sq(frame.move)
                    && (
                        moved_type == KNIGHT
                        || moved_type == BISHOP
                    )
                )
                || (
                    square == to_sq(frame.move)
                    && frame.captured_piece != PIECE_NONE
                    && (
                        type_of(frame.captured_piece) == KNIGHT
                        || type_of(frame.captured_piece) == BISHOP
                    )
                );
            if (current_outpost_piece || old_outpost_piece) {
                refresh_structure_slot_both(
                    pos,
                    OutpostBegin + static_cast<std::size_t>(square)
                );
            }
            for (int color = WHITE; color <= BLACK; ++color) {
                const int pawn_square =
                    color == WHITE ? square - 8 : square + 8;
                if (pawn_square < 0 || pawn_square >= SQ_NB)
                    continue;
                const Piece pawn = piece_on(pos, pawn_square);
                if (pawn == PIECE_NONE
                    || type_of(pawn) != PAWN
                    || color_of(pawn) != static_cast<Color>(color)) {
                    continue;
                }
                refresh_structure_slot_both(
                    pos,
                    PawnBegin
                        + static_cast<std::size_t>(
                            color * 64 + pawn_square
                        )
                );
                refresh_structure_slot_both(
                    pos,
                    BlockerBegin
                        + static_cast<std::size_t>(
                            color * 64 + pawn_square
                        )
                );
            }
        }
    }

    void push(
        const Position& pos,
        const memory::Memory& mem,
        Move move
    ) noexcept {
        ensure_initialized(pos, mem);
        ensure_current_updated(pos, mem);
        assert(current_ply + 1 < static_cast<int>(frames.size()));
        ++current_ply;
        SemanticFrame& frame = frames[current_ply];
        frame.undo_begin = undo_top;
        frame.row_begin = row_top;
        frame.old_bucket = material_bucket(pos);
        frame.new_bucket = frame.old_bucket;
        frame.move = move;
        frame.moved_piece = piece_on(pos, from_sq(move));
        frame.captured_piece = PIECE_NONE;
        frame.changed_count = 0;
        const auto add_changed = [&frame](Square square) {
            if (square == NO_SQ)
                return;
            for (u8 index = 0;
                 index < frame.changed_count;
                 ++index) {
                if (frame.changed_squares[index] == square)
                    return;
            }
            assert(frame.changed_count < frame.changed_squares.size());
            if (frame.changed_count < frame.changed_squares.size())
                frame.changed_squares[frame.changed_count++] = square;
        };
        add_changed(from_sq(move));
        add_changed(to_sq(move));
        if (move_is_ep(move)) {
            const Square captured =
                pos.side_to_move == WHITE
                    ? to_sq(move) - 8
                    : to_sq(move) + 8;
            frame.captured_piece = piece_on(pos, captured);
            add_changed(captured);
        } else if (move_is_capture(move)) {
            frame.captured_piece = piece_on(pos, to_sq(move));
        }
        if (move_is_castle(move)) {
            const Square king_from = from_sq(move);
            const bool king_side = to_sq(move) > king_from;
            add_changed(
                king_side ? king_from + 3 : king_from - 4
            );
            add_changed(
                king_side ? king_from + 1 : king_from - 1
            );
        }
        for (int perspective = WHITE;
             perspective <= BLACK;
             ++perspective) {
            frame.old_position_bucket[perspective] =
                position_bucket_for(
                    pos,
                    static_cast<Color>(perspective)
                );
        }
        frame.old_king_ray_targets = 0ULL;
        if (type_of(frame.moved_piece) != KING) {
            for (u8 changed = 0;
                 changed < frame.changed_count;
                 ++changed) {
                for (int color = WHITE;
                     color <= BLACK;
                     ++color) {
                    frame.old_king_ray_targets |=
                        king_ray_target_for_changed(
                            pos,
                            static_cast<Color>(color),
                            frame.changed_squares[changed]
                        );
                }
            }
        }
        frame.affected_attack_sources = 0ULL;
        frame.dirty_branch_mask = 0;
        frame.updated = false;
    }

    void ensure_current_updated(
        const Position& pos,
        const memory::Memory& mem
    ) noexcept {
        if (current_ply == 0 || frames[current_ply].updated)
            return;
        CycleScope cycles(CycleKind::SemanticMake);
        SemanticFrame& frame = frames[current_ply];
        refresh_all_slots(pos, mem, frame);
        frame.new_bucket = material_bucket(pos);
        frame.updated = true;
    }

    void after_make(
        const Position&,
        const memory::Memory&
    ) noexcept {
        assert(current_ply > 0);
        assert(!frames[current_ply].updated);
    }

    void materialize() noexcept {
        CycleScope cycles(CycleKind::Materialisation);
        assert(materialized_ply <= current_ply);
        u8 dirty_mask = 0;
        for (int ply = materialized_ply + 1;
             ply <= current_ply;
             ++ply) {
            dirty_mask = static_cast<u8>(
                dirty_mask | frames[ply].dirty_branch_mask
            );
            const std::size_t begin = frames[ply].row_begin;
            const std::size_t end =
                ply == current_ply
                    ? row_top
                    : frames[ply + 1].row_begin;
            apply_rows(begin, end);
        }
        for (std::size_t branch = 0; branch < 3; ++branch) {
            if ((dirty_mask & (1u << branch)) != 0)
                ++head_cache.branch_epoch[branch];
        }
        materialized_ply = current_ply;
    }

    [[nodiscard]] double forward_cached(
        const Position& pos,
        u64* attack_clips
    ) noexcept {
        CycleScope cycles(CycleKind::SelectedHead);
        const Color stm = static_cast<Color>(pos.side_to_move);
        const Color ntm = ~stm;
        HeadInput input{};
#if MAGNUS_MNUEV2_HAS_AVX2_KERNEL
        if (attack_clips == nullptr
            && !g_force_scalar.load(std::memory_order_relaxed)
            && mnuev2_avx2_available()) {
            build_head_input_avx2(
                position,
                attack,
                structure,
                stm,
                input
            );
        } else
#endif
        {
            std::size_t offset = 0;
            append_branch<
                Layout::PositionAccumulatorWidth,
                Layout::PositionPerspectiveWidth
            >(
                position[stm],
                position[ntm],
                g_network.position_scale,
                input,
                offset
            );
            append_branch<
                Layout::AttackAccumulatorWidth,
                Layout::AttackPerspectiveWidth
            >(
                attack[stm],
                attack[ntm],
                g_network.attack_scale,
                input,
                offset,
                attack_clips
            );
            append_branch<
                Layout::StructureAccumulatorWidth,
                Layout::StructurePerspectiveWidth
            >(
                structure[stm],
                structure[ntm],
                g_network.position_scale,
                input,
                offset
            );
            assert(offset == input.size());
        }

        const std::size_t bucket =
            static_cast<std::size_t>(material_bucket(pos));
        if (head_cache.bucket != static_cast<int>(bucket)) {
            for (auto& branch : head_cache.cached_epoch)
                branch.fill(0);
            head_cache.bucket = static_cast<int>(bucket);
        }

        constexpr std::array<int, 3> Offsets{{
            0,
            Layout::PositionAccumulatorWidth,
            Layout::PositionAccumulatorWidth
                + Layout::AttackAccumulatorWidth
        }};
        constexpr std::array<int, 3> Widths{{
            Layout::PositionAccumulatorWidth,
            Layout::AttackAccumulatorWidth,
            Layout::StructureAccumulatorWidth
        }};
        for (std::size_t branch = 0; branch < 3; ++branch) {
            if (
                head_cache.cached_epoch[branch][stm]
                == head_cache.branch_epoch[branch]
            ) {
                continue;
            }
            auto& partial = head_cache.partial[branch][stm];
#if MAGNUS_MNUEV2_HAS_AVX2_KERNEL
            if (!g_force_scalar.load(std::memory_order_relaxed)
                && mnuev2_avx2_available()) {
                head_partial_avx2(
                    input.data(),
                    Offsets[branch],
                    Widths[branch],
                    bucket,
                    partial
                );
            } else
#endif
            {
                const std::size_t row_base =
                    bucket * Layout::HeadHidden1;
                for (int row = 0;
                     row < Layout::HeadHidden1;
                     ++row) {
                    const std::size_t global_row = row_base + row;
                    const i16* weights =
                        g_network.head1_weights.data()
                        + global_row * Layout::ConcatWidth
                        + Offsets[branch];
                    float sum = 0.0F;
                    for (int column = 0;
                         column < Widths[branch];
                         ++column) {
                        sum += input[
                            Offsets[branch] + column
                        ] * static_cast<float>(weights[column])
                            * g_network.head_scale;
                    }
                    partial[row] = sum;
                }
            }
            head_cache.cached_epoch[branch][stm] =
                head_cache.branch_epoch[branch];
        }

        Head1 hidden1{};
        const std::size_t head1_row_base =
            bucket * Layout::HeadHidden1;
        for (int row = 0; row < Layout::HeadHidden1; ++row) {
            const float sum =
                static_cast<float>(
                    g_network.head1_bias[head1_row_base + row]
                ) * g_network.head_scale
                + head_cache.partial[0][stm][row]
                + head_cache.partial[1][stm][row]
                + head_cache.partial[2][stm][row];
            hidden1[row] = screlu(sum);
        }

#if MAGNUS_MNUEV2_HAS_AVX2_KERNEL
        if (!g_force_scalar.load(std::memory_order_relaxed)
            && mnuev2_avx2_available()) {
            return static_cast<double>(
                dense_tail_avx2(hidden1, bucket)
            );
        }
#endif
        Head2 hidden2{};
        const std::size_t head2_row_base =
            bucket * Layout::HeadHidden2;
        for (int row = 0; row < Layout::HeadHidden2; ++row) {
            const std::size_t global_row = head2_row_base + row;
            float sum =
                static_cast<float>(g_network.head2_bias[global_row])
                * g_network.head_scale;
            const i16* weights = g_network.head2_weights.data()
                + global_row * Layout::HeadHidden1;
            for (int column = 0;
                 column < Layout::HeadHidden1;
                 ++column) {
                sum += hidden1[column]
                    * static_cast<float>(weights[column])
                    * g_network.head_scale;
            }
            hidden2[row] = screlu(sum);
        }
        float output =
            static_cast<float>(g_network.output_bias[bucket])
            * g_network.output_scale;
        const i16* output_weights =
            g_network.output_weights.data()
            + bucket * Layout::HeadHidden2;
        for (int column = 0;
             column < Layout::HeadHidden2;
             ++column) {
            output += hidden2[column]
                * static_cast<float>(output_weights[column])
                * g_network.output_scale;
        }
        return static_cast<double>(output);
    }

    void restore_slot(const SlotUndo& undo) noexcept {
        const SemanticBranch branch =
            static_cast<SemanticBranch>(undo.branch);
        const Color perspective =
            static_cast<Color>(undo.perspective);
        u32& current = slot(branch, perspective, undo.slot);
        if (current != Layout::NoFeature) {
            u8& count = ref(branch, perspective, current);
            assert(count > 0);
            --count;
        }
        if (undo.old_feature != Layout::NoFeature) {
            u8& count = ref(
                branch,
                perspective,
                undo.old_feature
            );
            assert(count != std::numeric_limits<u8>::max());
            ++count;
        }
        current = undo.old_feature;
    }

    void pop(
        const Position& pos,
        const memory::Memory& mem
    ) noexcept {
        CycleScope cycles(CycleKind::SemanticUnmake);
        assert(current_ply > 0);
        const SemanticFrame& frame = frames[current_ply];
        if (!frame.updated) {
            assert(undo_top == frame.undo_begin);
            assert(row_top == frame.row_begin);
            --current_ply;
            assert(materialized_ply <= current_ply);
            return;
        }
        if (materialized_ply == current_ply) {
            apply_rows(frame.row_begin, row_top, true);
            for (std::size_t branch = 0; branch < 3; ++branch) {
                if (
                    (frame.dirty_branch_mask & (1u << branch))
                    != 0
                ) {
                    ++head_cache.branch_epoch[branch];
                }
            }
            --materialized_ply;
        }
        assert(materialized_ply < current_ply);
        for (std::size_t index = undo_top;
             index > frame.undo_begin;
             --index) {
            restore_slot(undo_arena[index - 1]);
        }
        undo_top = frame.undo_begin;
        row_top = frame.row_begin;
        --current_ply;
        Bitboard sources = frame.affected_attack_sources;
        while (sources != 0ULL) {
            const Square source = static_cast<Square>(
                std::countr_zero(static_cast<u64>(sources))
            );
            sources &= sources - 1;
            const Bitboard old_edges = attack_graph.attacks_from[source];
            const Bitboard parent_edges =
                compute_piece_attacks(pos, mem, source);
            Bitboard changed = old_edges ^ parent_edges;
            while (changed != 0ULL) {
                const Square target = static_cast<Square>(
                    std::countr_zero(static_cast<u64>(changed))
                );
                changed &= changed - 1;
                if ((old_edges & bb_of(target)) != 0ULL)
                    attack_graph.attackers_to[target] &= ~bb_of(source);
                if ((parent_edges & bb_of(target)) != 0ULL)
                    attack_graph.attackers_to[target] |= bb_of(source);
            }
            attack_graph.attacks_from[source] = parent_edges;
        }
        assert(materialized_ply <= current_ply);
    }
};

} // namespace

struct AccumulatorStack::Impl {
    static constexpr std::size_t Capacity = 132;
    static constexpr u8 AllPerspectives =
        static_cast<u8>((1u << WHITE) | (1u << BLACK));

    struct alignas(64) State {
        std::array<PositionAccumulator, COLOR_NB> position{};
        std::array<AttackAccumulator, COLOR_NB> attack{};
        std::array<StructureAccumulator, COLOR_NB> structure{};
        EncodedFeatures features{};
        u8 position_features_valid = 0;
        u8 attack_features_valid = 0;
        u8 structure_features_valid = 0;
        u8 position_valid = 0;
        u8 attack_valid = 0;
        u8 structure_valid = 0;
        MoveClass move_class = MoveClass::Quiet;
        u8 affected_pieces = 0;
        bool ordinary_non_pawn_move = false;
    };

    std::array<State, Capacity> states{};
    std::size_t state_count = 1;
    u32 generation = 0;
    Telemetry statistics{};
    SemanticState semantic{};

    [[nodiscard]] static constexpr u8 mask(Color perspective) noexcept {
        return static_cast<u8>(1u << static_cast<unsigned>(perspective));
    }

    void sync_generation() noexcept {
        const u32 current = g_generation.load(std::memory_order_acquire);
        if (generation == current)
            return;
        generation = current;
        state_count = 1;
        states[0].position_features_valid = 0;
        states[0].attack_features_valid = 0;
        states[0].structure_features_valid = 0;
        states[0].position_valid = 0;
        states[0].attack_valid = 0;
        states[0].structure_valid = 0;
        statistics = {};
    }

    void reset() noexcept {
        sync_generation();
        state_count = 1;
        states[0].position_features_valid = 0;
        states[0].attack_features_valid = 0;
        states[0].structure_features_valid = 0;
        states[0].position_valid = 0;
        states[0].attack_valid = 0;
        states[0].structure_valid = 0;
        statistics = {};
    }

    [[nodiscard]] static MoveClass classify_move(
        const Position& pos,
        Move move
    ) noexcept {
        if (move_is_castle(move))
            return MoveClass::Castling;
        if (move_is_ep(move))
            return MoveClass::EnPassant;
        if (move_is_promotion(move))
            return MoveClass::Promotion;
        const PieceType moved = piece_type_on(pos, from_sq(move));
        if (moved == PAWN)
            return move_is_capture(move)
                ? MoveClass::PawnCapture
                : MoveClass::PawnMove;
        if (moved == KING)
            return MoveClass::KingMove;
        return move_is_capture(move)
            ? MoveClass::Capture
            : MoveClass::Quiet;
    }

    void push(const Position& pos, Move move) noexcept {
        CycleScope cycles(CycleKind::SemanticMake);
        sync_generation();
        assert(state_count < states.size());
        if (state_count >= states.size())
            return;
        State& state = states[state_count++];
        state.position_features_valid = 0;
        state.attack_features_valid = 0;
        state.structure_features_valid = 0;
        state.position_valid = 0;
        state.attack_valid = 0;
        state.structure_valid = 0;
        state.move_class = classify_move(pos, move);
        state.affected_pieces = static_cast<u8>(
            1 + (move_is_capture(move) ? 1 : 0)
              + (move_is_castle(move) ? 1 : 0)
        );
        const PieceType moved = piece_type_on(pos, from_sq(move));
        state.ordinary_non_pawn_move =
            moved != PAWN
            && !move_is_capture(move)
            && !move_is_castle(move)
            && !move_is_promotion(move)
            && !move_is_ep(move);
#if MAGNUS_MNUEV2_TELEMETRY
        statistics.affected_pieces += state.affected_pieces;
#endif
    }

    void pop() noexcept {
        CycleScope cycles(CycleKind::SemanticUnmake);
        assert(state_count > 1);
        if (state_count > 1)
            --state_count;
    }

    template<typename ValidMember, typename FeatureValidMember,
             typename AccMember, typename FeatureMember, typename Encode,
             typename Rebuild, typename Diff>
    [[nodiscard]] const auto& ensure_branch(
        const Position& pos,
        const memory::Memory& mem,
        Color perspective,
        ValidMember valid_member,
        FeatureValidMember feature_valid_member,
        AccMember accumulator_member,
        FeatureMember feature_member,
        BranchTelemetry& telemetry,
        Encode encode,
        Rebuild rebuild,
        Diff diff
    ) noexcept {
        sync_generation();
        const u8 perspective_mask = mask(perspective);
        const std::size_t current_index = state_count - 1;
        State& current = states[current_index];
        u8& valid = current.*valid_member;
        auto& destination =
            (current.*accumulator_member)[perspective];
        if ((valid & perspective_mask) != 0) {
#if MAGNUS_MNUEV2_TELEMETRY
            ++telemetry.cache_hits;
#endif
            return destination;
        }
#if MAGNUS_MNUEV2_TELEMETRY
        ++telemetry.cache_misses;
#endif
        u8& feature_valid = current.*feature_valid_member;
        bool encoded = (feature_valid & perspective_mask) != 0;
        if (!encoded) {
            CycleScope cycles(CycleKind::FeatureReconstruction);
            encoded = encode(
                current,
                pos,
                mem,
                perspective,
                (current.features.*feature_member)[perspective]
            );
        }
        assert(encoded);
        (void)encoded;
        feature_valid = static_cast<u8>(
            feature_valid | perspective_mask
        );

        std::size_t source_index = current_index;
        bool found = false;
        while (source_index > 0) {
            --source_index;
            if (((states[source_index].*valid_member)
                 & perspective_mask) != 0) {
                found = true;
                break;
            }
        }
        if (found) {
            const State& source = states[source_index];
            diff(
                (source.features.*feature_member)[perspective],
                (current.features.*feature_member)[perspective],
                (source.*accumulator_member)[perspective],
                destination,
                telemetry,
                current.move_class
            );
        } else {
            rebuild(
                (current.features.*feature_member)[perspective],
                destination
            );
#if MAGNUS_MNUEV2_TELEMETRY
            ++telemetry.full_refreshes;
            ++telemetry.accumulator_rebuilds;
#endif
        }
        valid = static_cast<u8>(valid | perspective_mask);
        return destination;
    }

    [[nodiscard]] const PositionAccumulator& ensure_position(
        const Position& pos,
        const memory::Memory& mem,
        Color perspective
    ) noexcept {
        return ensure_branch(
            pos,
            mem,
            perspective,
            &State::position_valid,
            &State::position_features_valid,
            &State::position,
            &EncodedFeatures::position,
            statistics.position,
            [](State&,
               const Position& current,
               const memory::Memory&,
               Color color,
               PositionFeatureList& features) {
                return encode_position_features(
                    current,
                    color,
                    features
                );
            },
            [](const PositionFeatureList& features,
               PositionAccumulator& accumulator) {
                rebuild_i16(
                    g_network.position_weights,
                    g_network.position_bias,
                    features,
                    accumulator
                );
            },
            [](const PositionFeatureList& old_features,
               const PositionFeatureList& new_features,
               const PositionAccumulator& source,
               PositionAccumulator& destination,
               BranchTelemetry& telemetry,
               MoveClass move_class) {
                apply_diff(
                    old_features,
                    new_features,
                    source,
                    destination,
                    telemetry,
                    move_class,
                    [](PositionAccumulator& accumulator,
                       u32 feature,
                       int sign) {
                        add_i16_row(
                            g_network.position_weights,
                            accumulator,
                            feature,
                            sign
                        );
                    }
                );
            }
        );
    }

    [[nodiscard]] const AttackAccumulator& ensure_attack(
        const Position& pos,
        const memory::Memory& mem,
        Color perspective
    ) noexcept {
        const AttackAccumulator& result = ensure_branch(
            pos,
            mem,
            perspective,
            &State::attack_valid,
            &State::attack_features_valid,
            &State::attack,
            &EncodedFeatures::attack,
            statistics.attack,
            [](State& state,
               const Position& current,
               const memory::Memory& memory,
               Color color,
               AttackFeatureList& features) {
                const bool encoded = encode_attack_features_pair(
                    current,
                    memory,
                    state.features.attack
                );
                if (encoded)
                    state.attack_features_valid = AllPerspectives;
                (void)color;
                (void)features;
                return encoded;
            },
            [](const AttackFeatureList& features,
               AttackAccumulator& accumulator) {
                rebuild_attack(features, accumulator);
            },
            [this](const AttackFeatureList& old_features,
               const AttackFeatureList& new_features,
               const AttackAccumulator& source,
               AttackAccumulator& destination,
               BranchTelemetry& telemetry,
               MoveClass move_class) {
#if MAGNUS_MNUEV2_TELEMETRY
                std::size_t old_index = 0;
                std::size_t new_index = 0;
                const auto count_change = [this](u32 feature) {
                    if (feature >= static_cast<u32>(Layout::AttackStatusEnd)
                        && feature
                            < static_cast<u32>(Layout::AttackEdgeEnd)) {
                        ++statistics.changed_slider_rays;
                    } else {
                        ++statistics.tactical_summary_transitions;
                    }
                };
                while (old_index < old_features.count
                       || new_index < new_features.count) {
                    if (old_index == old_features.count) {
                        count_change(new_features.indices[new_index++]);
                    } else if (new_index == new_features.count) {
                        count_change(old_features.indices[old_index++]);
                    } else {
                        const u32 old_feature =
                            old_features.indices[old_index];
                        const u32 new_feature =
                            new_features.indices[new_index];
                        if (old_feature == new_feature) {
                            ++old_index;
                            ++new_index;
                        } else if (old_feature < new_feature) {
                            count_change(old_feature);
                            ++old_index;
                        } else {
                            count_change(new_feature);
                            ++new_index;
                        }
                    }
                }
#endif
                apply_diff(
                    old_features,
                    new_features,
                    source,
                    destination,
                    telemetry,
                    move_class,
                    [](AttackAccumulator& accumulator,
                       u32 feature,
                       int sign) {
                        add_attack_row(accumulator, feature, sign);
                    }
                );
            }
        );
#if MAGNUS_MNUEV2_TELEMETRY
        for (const i32 value : result) {
            if (statistics.attack_accumulator_samples == 0) {
                statistics.attack_accumulator_min = value;
                statistics.attack_accumulator_max = value;
            } else {
                statistics.attack_accumulator_min =
                    std::min(statistics.attack_accumulator_min, value);
                statistics.attack_accumulator_max =
                    std::max(statistics.attack_accumulator_max, value);
            }
            const u64 magnitude = static_cast<u64>(
                value < 0 ? -static_cast<i64>(value) : value
            );
            const std::size_t bin = std::min<std::size_t>(
                static_cast<std::size_t>(magnitude),
                statistics.attack_abs_histogram.size() - 1
            );
            ++statistics.attack_abs_histogram[bin];
            ++statistics.attack_accumulator_samples;
        }
#endif
        return result;
    }

    [[nodiscard]] const StructureAccumulator& ensure_structure(
        const Position& pos,
        const memory::Memory& mem,
        Color perspective
    ) noexcept {
#if MAGNUS_MNUEV2_TELEMETRY
        const u64 rebuilds_before =
            statistics.structure.full_refreshes;
#endif
        const StructureAccumulator& result = ensure_branch(
            pos,
            mem,
            perspective,
            &State::structure_valid,
            &State::structure_features_valid,
            &State::structure,
            &EncodedFeatures::structure,
            statistics.structure,
            [](State&,
               const Position& current,
               const memory::Memory&,
               Color color,
               StructureFeatureList& features) {
                return encode_structure_features(
                    current,
                    color,
                    features
                );
            },
            [](const StructureFeatureList& features,
               StructureAccumulator& accumulator) {
                rebuild_i16(
                    g_network.structure_weights,
                    g_network.structure_bias,
                    features,
                    accumulator
                );
            },
            [](const StructureFeatureList& old_features,
               const StructureFeatureList& new_features,
               const StructureAccumulator& source,
               StructureAccumulator& destination,
               BranchTelemetry& telemetry,
               MoveClass move_class) {
                apply_diff(
                    old_features,
                    new_features,
                    source,
                    destination,
                    telemetry,
                    move_class,
                    [](StructureAccumulator& accumulator,
                       u32 feature,
                       int sign) {
                        add_i16_row(
                            g_network.structure_weights,
                            accumulator,
                            feature,
                            sign
                        );
                    }
                );
            }
        );
#if MAGNUS_MNUEV2_TELEMETRY
        const State& current = states[state_count - 1];
        if (current.ordinary_non_pawn_move
            && statistics.structure.full_refreshes > rebuilds_before) {
            ++statistics.structure_rebuilds_on_non_pawn_moves;
        }
#endif
        return result;
    }
};

AccumulatorStack::AccumulatorStack() noexcept
    : impl_(std::make_unique<Impl>()) {
    impl_->reset();
}

AccumulatorStack::~AccumulatorStack() = default;
AccumulatorStack::AccumulatorStack(AccumulatorStack&&) noexcept = default;
AccumulatorStack& AccumulatorStack::operator=(
    AccumulatorStack&&
) noexcept = default;

void AccumulatorStack::reset() noexcept {
    impl_->reset();
    impl_->semantic.reset();
}

void AccumulatorStack::push(
    const Position& pos,
    const memory::Memory& mem,
    Move move
) noexcept {
    impl_->semantic.push(pos, mem, move);
}

void AccumulatorStack::after_make(
    const Position& pos,
    const memory::Memory& mem
) noexcept {
    impl_->semantic.after_make(pos, mem);
}

void AccumulatorStack::pop(
    const Position& pos,
    const memory::Memory& mem
) noexcept {
    impl_->semantic.pop(pos, mem);
}

std::size_t AccumulatorStack::size() const noexcept {
    return static_cast<std::size_t>(
        impl_->semantic.current_ply + 1
    );
}

Telemetry AccumulatorStack::telemetry() const noexcept {
    return impl_->semantic.statistics;
}

bool load(const std::string& file_path) {
    clear_network();
    std::vector<u8> bytes;
    if (!load_file_bytes(file_path, bytes))
        return false;
    if (!parse_network(file_path, bytes)) {
        const std::string error = g_network.error;
        clear_network();
        g_network.error = error;
        return false;
    }
    (void)next_generation();
    return true;
}

void unload() noexcept {
    clear_network();
    (void)next_generation();
}

bool loaded() noexcept {
    return g_network.valid();
}

const std::string& path() noexcept {
    return g_network.file_path;
}

const std::string& last_error() noexcept {
    return g_network.error;
}

AttackWeightType attack_weight_type() noexcept {
    return g_network.attack_type;
}

const char* backend_name() noexcept {
    if (g_force_scalar.load(std::memory_order_relaxed))
        return "scalar-forced";
#if MAGNUS_MNUEV2_HAS_AVX2_KERNEL
    if (g_network.attack_type == AttackWeightType::Int8
        && attack_avx2_available())
        return "AVX2-attack-int8";
#endif
    return "scalar";
}

void set_force_scalar(bool force) noexcept {
    g_force_scalar.store(force, std::memory_order_relaxed);
}

bool force_scalar() noexcept {
    return g_force_scalar.load(std::memory_order_relaxed);
}

bool telemetry_enabled() noexcept {
    return MAGNUS_MNUEV2_TELEMETRY != 0;
}

std::size_t network_bytes() noexcept {
    return g_network.file_bytes;
}

std::size_t accumulator_stack_bytes() noexcept {
    return sizeof(AccumulatorStack::Impl);
}

double evaluate_reference_output(
    const Position& pos,
    const memory::Memory& mem
) noexcept {
    if (!loaded())
        return 0.0;
    EncodedFeatures features{};
    if (!encode_all_features(pos, mem, features))
        return 0.0;
    std::array<PositionAccumulator, COLOR_NB> position{};
    std::array<AttackAccumulator, COLOR_NB> attack{};
    std::array<StructureAccumulator, COLOR_NB> structure{};
    rebuild_all(features, position, attack, structure);
    return forward(pos, position, attack, structure);
}

int evaluate_reference(
    const Position& pos,
    const memory::Memory& mem
) noexcept {
    return score_from_output(evaluate_reference_output(pos, mem));
}

int evaluate_incremental(
    const Position& pos,
    const memory::Memory& mem,
    AccumulatorStack& stack
) noexcept {
    if (!loaded())
        return 0;
    telemetry_note_static_eval();
    int cached_score = 0;
    if (stack.impl_->semantic.probe_eval_cache(
            pos.key,
            cached_score
        )) {
        return cached_score;
    }
    telemetry_note_full_head();
    stack.impl_->semantic.ensure_initialized(pos, mem);
    stack.impl_->semantic.ensure_current_updated(pos, mem);
    stack.impl_->semantic.materialize();
    u64 clips = 0;
    const double output = stack.impl_->semantic.forward_cached(
        pos,
#if MAGNUS_MNUEV2_TELEMETRY
        &clips
#else
        nullptr
#endif
    );
#if MAGNUS_MNUEV2_TELEMETRY
    stack.impl_->semantic.statistics.attack_activation_clips += clips;
#else
    (void)clips;
#endif
    const int score = score_from_output(output);
    stack.impl_->semantic.store_eval_cache(pos.key, score);
    return score;
}

bool debug_check_incremental(
    const Position& pos,
    const memory::Memory& mem,
    AccumulatorStack& stack,
    std::ostream& output
) noexcept {
    if (!loaded())
        return false;

    EncodedFeatures features{};
    if (!encode_all_features(pos, mem, features)) {
        output << "info string MNUEv2 feature encoding failed\n";
        return false;
    }
    std::array<PositionAccumulator, COLOR_NB> rebuilt_position{};
    std::array<AttackAccumulator, COLOR_NB> rebuilt_attack{};
    std::array<StructureAccumulator, COLOR_NB> rebuilt_structure{};
    rebuild_all(
        features,
        rebuilt_position,
        rebuilt_attack,
        rebuilt_structure
    );

    stack.impl_->semantic.ensure_initialized(pos, mem);
    stack.impl_->semantic.ensure_current_updated(pos, mem);
    stack.impl_->semantic.materialize();
    for (int perspective = WHITE; perspective <= BLACK; ++perspective) {
        if (stack.impl_->semantic.position[perspective]
            != rebuilt_position[perspective]) {
            output << "info string MNUEv2 Position accumulator mismatch"
                << " perspective " << perspective << '\n';
            return false;
        }
        if (stack.impl_->semantic.attack[perspective]
            != rebuilt_attack[perspective]) {
            output << "info string MNUEv2 Attack accumulator mismatch"
                << " perspective " << perspective << '\n';
            (void)debug_attack_graph_invariant(
                pos,
                mem,
                stack.impl_->semantic.attack_graph,
                output
            );
            for (std::size_t slot_index = 0;
                 slot_index < Layout::AttackSlotCount;
                 ++slot_index) {
                const u32 recomputed = compute_attack_slot(
                    pos,
                    mem,
                    stack.impl_->semantic.attack_graph,
                    static_cast<Color>(perspective),
                    slot_index
                );
                const u32 current =
                    stack.impl_->semantic
                        .attack_slots[perspective][slot_index];
                if (current != recomputed) {
                    output << "info string MNUEv2 Attack slot "
                        << slot_index << " current " << current
                        << " recomputed " << recomputed << '\n';
                    break;
                }
            }
            std::array<u8, Layout::AttackFeatureCount> expected{};
            for (std::size_t index = 0;
                 index < features.attack[perspective].count;
                 ++index) {
                expected[
                    features.attack[perspective].indices[index]
                ] = 1;
            }
            for (std::size_t feature = 0;
                 feature < expected.size();
                 ++feature) {
                const bool active =
                    stack.impl_->semantic
                        .attack_refs[perspective][feature] != 0;
                if (active != (expected[feature] != 0)) {
                    output << "info string MNUEv2 Attack feature "
                        << feature << " expected "
                        << static_cast<int>(expected[feature])
                        << " refcount "
                        << static_cast<int>(
                            stack.impl_->semantic
                                .attack_refs[perspective][feature]
                        )
                        << ' ' << decode_attack_feature(
                            static_cast<u32>(feature)
                        )
                        << '\n';
                    break;
                }
            }
            return false;
        }
        if (stack.impl_->semantic.structure[perspective]
            != rebuilt_structure[perspective]) {
            output << "info string MNUEv2 Structure accumulator mismatch"
                << " perspective " << perspective << '\n';
            std::array<u8, Layout::StructureFeatureCount> expected{};
            for (std::size_t index = 0;
                 index < features.structure[perspective].count;
                 ++index) {
                expected[
                    features.structure[perspective].indices[index]
                ] = 1;
            }
            for (std::size_t feature = 0;
                 feature < expected.size();
                 ++feature) {
                const bool active =
                    stack.impl_->semantic
                        .structure_refs[perspective][feature] != 0;
                if (active != (expected[feature] != 0)) {
                    output << "info string MNUEv2 Structure feature "
                        << feature << " expected "
                        << static_cast<int>(expected[feature])
                        << " refcount "
                        << static_cast<int>(
                            stack.impl_->semantic
                                .structure_refs[perspective][feature]
                        )
                        << ' ' << decode_structure_feature(
                            static_cast<u32>(feature)
                        )
                        << '\n';
                    break;
                }
            }
            return false;
        }
    }

    const int reference = evaluate_reference(pos, mem);
    const int incremental = evaluate_incremental(pos, mem, stack);
    if (reference == incremental)
        return true;
    output << "info string MNUEv2 incremental mismatch reference "
        << reference << " incremental " << incremental << '\n';
    return false;
}

void debug_dump_network(std::ostream& output) {
    output << "info string MNUEv2 loaded " << (loaded() ? 1 : 0) << '\n';
    if (!loaded()) {
        output << "info string MNUEv2 error " << last_error() << '\n';
        return;
    }
    output << "info string MNUEv2 evaluation using " << path() << '\n';
    output << "info string architecture PositionAttackStructure\n";
    output << "info string Position "
        << Layout::PositionFeatureCount << 'x'
        << Layout::PositionAccumulatorWidth << " int16\n";
    output << "info string Position accumulator i32 raw "
        << Layout::PositionAccumulatorWidth
        << " pairwise_per_perspective "
        << Layout::PositionPerspectiveWidth
        << " final_branch " << Layout::PositionAccumulatorWidth
        << '\n';
    output << "info string Attack "
        << Layout::AttackFeatureCount << 'x'
        << Layout::AttackAccumulatorWidth << ' '
        << (attack_weight_type() == AttackWeightType::Int8
            ? "int8"
            : "int16")
        << '\n';
    output << "info string Attack accumulator i32 raw "
        << Layout::AttackAccumulatorWidth
        << " pairwise_per_perspective "
        << Layout::AttackPerspectiveWidth
        << " final_branch " << Layout::AttackAccumulatorWidth
        << '\n';
    output << "info string Structure "
        << Layout::StructureFeatureCount << 'x'
        << Layout::StructureAccumulatorWidth << " int16\n";
    output << "info string Structure accumulator i32 raw "
        << Layout::StructureAccumulatorWidth
        << " pairwise_per_perspective "
        << Layout::StructurePerspectiveWidth
        << " final_branch " << Layout::StructureAccumulatorWidth
        << '\n';
    output << "info string Head " << Layout::ConcatWidth << 'x'
        << Layout::HeadHidden1 << 'x' << Layout::HeadHidden2
        << "x1 buckets " << Layout::OutputBuckets << '\n';
    output << "info string network size "
        << static_cast<double>(network_bytes()) / 1048576.0
        << " MiB accumulator_stack "
        << static_cast<double>(accumulator_stack_bytes()) / 1048576.0
        << " MiB accumulator_stack_bytes "
        << accumulator_stack_bytes()
        << " backend " << backend_name() << '\n';
    output << "info string scales Position/Structure "
        << g_network.position_scale
        << " Attack " << g_network.attack_scale
        << " Head " << g_network.head_scale
        << " Output " << g_network.output_scale
        << " Score " << g_network.score_scale
        << " bucket_mapping_version "
        << Layout::BucketMappingVersion << '\n';
}

void debug_dump_position(
    const Position& pos,
    const memory::Memory& mem,
    std::ostream& output
) {
    debug_dump_features(pos, mem, output);
    EncodedFeatures features{};
    std::array<PositionAccumulator, COLOR_NB> position{};
    std::array<AttackAccumulator, COLOR_NB> attack{};
    std::array<StructureAccumulator, COLOR_NB> structure{};
    ForwardTrace trace{};
    if (encode_all_features(pos, mem, features)) {
        rebuild_all(features, position, attack, structure);
        (void)forward(
            pos,
            position,
            attack,
            structure,
            nullptr,
            &trace
        );
        output << "mnue v2 trace input_sum " << trace.input_sum
            << " hidden1_sum " << trace.hidden1_sum
            << " hidden2_sum " << trace.hidden2_sum << '\n';
    }
    output << "mnue v2 scalar output "
        << evaluate_reference_output(pos, mem) << '\n';
    output << "mnue v2 scalar score "
        << evaluate_reference(pos, mem) << '\n';
}

void print_telemetry(
    const Telemetry& telemetry,
    std::ostream& output
) {
    output << "mnue v2 telemetry enabled "
        << (telemetry_enabled() ? 1 : 0) << '\n';
#if !MAGNUS_MNUEV2_TELEMETRY
    output << "mnue v2 telemetry rebuild with "
        << "MNUEV2_TELEMETRY=1 to collect hot-path counters\n";
    (void)telemetry;
    return;
#else
    constexpr std::array<const char*, MoveClassCount> MoveNames{{
        "quiet",
        "capture",
        "pawn_move",
        "pawn_capture",
        "king_move",
        "castling",
        "promotion",
        "en_passant"
    }};
    const auto percentile = [](const auto& histogram, double fraction) {
        u64 total = 0;
        for (const u64 count : histogram)
            total += count;
        if (total == 0)
            return std::size_t{0};
        const u64 target = static_cast<u64>(
            std::ceil(static_cast<double>(total) * fraction)
        );
        u64 cumulative = 0;
        for (std::size_t index = 0; index < histogram.size(); ++index) {
            cumulative += histogram[index];
            if (cumulative >= target)
                return index;
        }
        return histogram.size() - 1;
    };
    const auto branch = [&]( 
        const char* name,
        const BranchTelemetry& stats
    ) {
        output << "mnue v2 telemetry " << name
            << " rows_added " << stats.rows_added
            << " rows_removed " << stats.rows_removed
            << " unique_rows_changed " << stats.unique_rows_changed
            << " duplicate_deltas " << stats.duplicate_deltas
            << " cancelled_deltas " << stats.cancelled_deltas
            << " full_refreshes " << stats.full_refreshes
            << " cache_hits " << stats.cache_hits
            << " cache_misses " << stats.cache_misses
            << " rebuilds " << stats.accumulator_rebuilds
            << " moves " << stats.moves
            << " average_rows_changed "
            << (stats.moves == 0
                ? 0.0
                : static_cast<double>(stats.unique_rows_changed)
                    / static_cast<double>(stats.moves))
            << " p50 " << percentile(stats.rows_changed_histogram, 0.50)
            << " p90 " << percentile(stats.rows_changed_histogram, 0.90)
            << " p95 " << percentile(stats.rows_changed_histogram, 0.95)
            << " p99 " << percentile(stats.rows_changed_histogram, 0.99)
            << " max_rows_changed " << stats.max_rows_changed
            << '\n';
        for (std::size_t index = 0; index < MoveClassCount; ++index) {
            output << "mnue v2 telemetry " << name
                << " move_type " << MoveNames[index]
                << " updates " << stats.updates_by_move_class[index]
                << " rows " << stats.rows_by_move_class[index]
                << '\n';
        }
    };
    branch("Position", telemetry.position);
    branch("Attack", telemetry.attack);
    branch("Structure", telemetry.structure);
    output << "mnue v2 telemetry Attack accumulator_min "
        << telemetry.attack_accumulator_min
        << " accumulator_max " << telemetry.attack_accumulator_max
        << " p99_abs "
        << percentile(telemetry.attack_abs_histogram, 0.99)
        << " potential_int16_overflow "
        << (
            telemetry.attack_accumulator_min
                    < std::numeric_limits<i16>::min()
                || telemetry.attack_accumulator_max
                    > std::numeric_limits<i16>::max()
            ? 1
            : 0
        )
        << " activation_clips " << telemetry.attack_activation_clips
        << '\n';
    output << "mnue v2 telemetry affected_pieces "
        << telemetry.affected_pieces
        << " changed_slider_rays " << telemetry.changed_slider_rays
        << " tactical_summary_transitions "
        << telemetry.tactical_summary_transitions
        << " structure_rebuilds_on_non_pawn_moves "
        << telemetry.structure_rebuilds_on_non_pawn_moves
        << '\n';
#endif
}

bool debug_attack_kernel_selftest(std::ostream& output) noexcept {
    if (!loaded()) {
        output << "mnue v2 attack kernel selftest: network not loaded\n";
        return false;
    }
    if (g_network.attack_type != AttackWeightType::Int8) {
        output << "mnue v2 attack kernel selftest: int8 network required\n";
        return false;
    }

    u32 rng = 0xC001D00Du;
    u64 checks = 0;
    for (int trial = 0; trial < 4096; ++trial) {
        AttackAccumulator scalar{};
        for (i32& value : scalar) {
            rng = rng * 1664525u + 1013904223u;
            value = static_cast<i32>(rng % 20001u) - 10000;
        }
        AttackAccumulator simd = scalar;
        const AttackAccumulator original = scalar;
        rng = rng * 1664525u + 1013904223u;
        const u32 feature =
            rng % static_cast<u32>(Layout::AttackFeatureCount);
        const int sign = (rng & 0x80000000u) != 0u ? 1 : -1;
        add_attack_row_scalar(scalar, feature, sign);
#if MAGNUS_MNUEV2_HAS_AVX2_KERNEL
        if (attack_avx2_available())
            add_attack_row_avx2(simd, feature, sign);
        else
            add_attack_row_scalar(simd, feature, sign);
#else
        add_attack_row_scalar(simd, feature, sign);
#endif
        ++checks;
        if (scalar != simd) {
            output << "mnue v2 attack kernel mismatch trial "
                << trial << '\n';
            return false;
        }
        add_attack_row(simd, feature, -sign);
        ++checks;
        if (simd != original) {
            output << "mnue v2 attack add/sub restoration mismatch trial "
                << trial << '\n';
            return false;
        }
    }
    output << "mnue v2 attack kernel selftest checks " << checks
        << " backend " << backend_name()
        << " mismatches 0 max_error 0 mean_error 0 ok 1\n";
    return true;
}

bool debug_loader_selftest(std::ostream& output) {
    if (!loaded()) {
        output << "mnue v2 loader selftest: network not loaded\n";
        return false;
    }
    const std::string original_path = g_network.file_path;
    std::vector<u8> bytes;
    if (!load_file_bytes(original_path, bytes))
        return false;

    const auto put_u32 = [&bytes](std::size_t offset, u32 value) {
        for (int shift = 0; shift < 32; shift += 8)
            bytes[offset++] = static_cast<u8>(value >> shift);
    };
    const auto put_u64 = [&bytes](std::size_t offset, u64 value) {
        for (int shift = 0; shift < 64; shift += 8)
            bytes[offset++] = static_cast<u8>(value >> shift);
    };
    const auto rejected = [&](
        const char* name,
        const auto& mutate,
        const auto& restore
    ) {
        mutate();
        clear_network();
        const bool accepted = parse_network("selftest", bytes);
        const std::string error = g_network.error;
        restore();
        output << "mnue v2 loader test " << name
            << " rejected " << (!accepted ? 1 : 0)
            << " error " << error << '\n';
        return !accepted;
    };

    bool ok = true;
    const u8 old_magic = bytes[0];
    ok = rejected(
        "bad_magic",
        [&]() { bytes[0] ^= 0xFFu; },
        [&]() { bytes[0] = old_magic; }
    ) && ok;

    constexpr std::size_t ArchitectureOffset = 12;
    const std::array<u8, 4> old_arch{
        bytes[ArchitectureOffset],
        bytes[ArchitectureOffset + 1],
        bytes[ArchitectureOffset + 2],
        bytes[ArchitectureOffset + 3]
    };
    ok = rejected(
        "wrong_architecture",
        [&]() { put_u32(ArchitectureOffset, 999u); },
        [&]() {
            std::copy(
                old_arch.begin(),
                old_arch.end(),
                bytes.begin()
                    + static_cast<std::ptrdiff_t>(ArchitectureOffset)
            );
        }
    ) && ok;

    constexpr std::size_t AttackScaleOffset = 72;
    const std::array<u8, 4> old_attack_scale{
        bytes[AttackScaleOffset],
        bytes[AttackScaleOffset + 1],
        bytes[AttackScaleOffset + 2],
        bytes[AttackScaleOffset + 3]
    };
    ok = rejected(
        "wrong_quantisation_scale",
        [&]() { put_u32(AttackScaleOffset, 0x3F800000u); },
        [&]() {
            std::copy(
                old_attack_scale.begin(),
                old_attack_scale.end(),
                bytes.begin()
                    + static_cast<std::ptrdiff_t>(AttackScaleOffset)
            );
        }
    ) && ok;

    constexpr std::size_t FirstDtypeOffset = 92 + 24;
    const std::array<u8, 4> old_first_dtype{
        bytes[FirstDtypeOffset],
        bytes[FirstDtypeOffset + 1],
        bytes[FirstDtypeOffset + 2],
        bytes[FirstDtypeOffset + 3]
    };
    ok = rejected(
        "wrong_section_dtype",
        [&]() { put_u32(FirstDtypeOffset, kDtypeI8); },
        [&]() {
            std::copy(
                old_first_dtype.begin(),
                old_first_dtype.end(),
                bytes.begin()
                    + static_cast<std::ptrdiff_t>(FirstDtypeOffset)
            );
        }
    ) && ok;

    const u8 old_payload_byte = bytes.back();
    ok = rejected(
        "wrong_checksum",
        [&]() { bytes.back() ^= 0x01u; },
        [&]() { bytes.back() = old_payload_byte; }
    ) && ok;

    constexpr std::size_t DescriptorStart = 92;
    constexpr std::size_t OffsetInsideDescriptor = 32;
    constexpr std::size_t SecondOffsetField =
        DescriptorStart + kSectionDescriptorBytes
        + OffsetInsideDescriptor;
    u64 second_offset = 0;
    {
        std::size_t cursor = SecondOffsetField;
        (void)read_u64_le(bytes, cursor, second_offset);
    }
    ok = rejected(
        "overlap",
        [&]() { put_u64(SecondOffsetField, kHeaderBytes); },
        [&]() { put_u64(SecondOffsetField, second_offset); }
    ) && ok;

    std::vector<u8> truncated(bytes.begin(), bytes.end() - 1);
    clear_network();
    const bool truncated_accepted = parse_network("selftest", truncated);
    output << "mnue v2 loader test truncated"
        << " rejected " << (!truncated_accepted ? 1 : 0)
        << " error " << g_network.error << '\n';
    ok = !truncated_accepted && ok;

    clear_network();
    const bool restored = parse_network(original_path, bytes);
    if (restored)
        (void)next_generation();
    output << "mnue v2 loader selftest ok "
        << (ok && restored ? 1 : 0) << '\n';
    return ok && restored;
}

bool make_golden_snapshot(
    const Position& pos,
    const memory::Memory& mem,
    GoldenSnapshot& snapshot
) noexcept {
    if (!loaded() || !encode_all_features(pos, mem, snapshot.features))
        return false;
    std::array<PositionAccumulator, COLOR_NB> position{};
    std::array<AttackAccumulator, COLOR_NB> attack{};
    std::array<StructureAccumulator, COLOR_NB> structure{};
    rebuild_all(snapshot.features, position, attack, structure);
    snapshot.material = material_units(pos);
    snapshot.bucket = material_bucket(pos);
    for (int perspective = WHITE; perspective <= BLACK; ++perspective) {
        snapshot.position_hash[perspective] =
            accumulator_hash(position[perspective]);
        snapshot.attack_hash[perspective] =
            accumulator_hash(attack[perspective]);
        snapshot.structure_hash[perspective] =
            accumulator_hash(structure[perspective]);
    }
    snapshot.output = forward(pos, position, attack, structure);
    snapshot.score = score_from_output(snapshot.output);
    return true;
}

} // namespace magnus::mnue::v2
