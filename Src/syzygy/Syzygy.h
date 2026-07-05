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

#pragma once

#include <string_view>

#include "Types.h"

namespace magnus {

struct Position;

namespace memory {
struct Memory;
}

namespace syzygy {

inline constexpr int DEFAULT_PROBE_DEPTH = 1;
inline constexpr int MIN_PROBE_DEPTH = 1;
inline constexpr int MAX_PROBE_DEPTH = 100;
inline constexpr int DEFAULT_PROBE_LIMIT = 7;
inline constexpr int MIN_PROBE_LIMIT = 0;
inline constexpr int MAX_PROBE_LIMIT = 7;

enum class Wdl : int {
    Loss = -2,
    BlessedLoss = -1,
    Draw = 0,
    CursedWin = 1,
    Win = 2
};

struct RootProbe {
    Move moves[256]{};
    int move_count = 0;
    int rank = 0;
    Wdl wdl = Wdl::Draw;
    bool used_dtz = false;
};

[[nodiscard]] bool init(std::string_view path) noexcept;
void shutdown() noexcept;

[[nodiscard]] int max_cardinality() noexcept;

[[nodiscard]] bool probe_wdl(
    const Position& pos,
    int probe_limit,
    Wdl& result
) noexcept;

[[nodiscard]] bool rank_root_moves(
    const Position& pos,
    const memory::Memory& mem,
    int probe_limit,
    bool use_rule50,
    const Move* allowed_moves,
    int allowed_move_count,
    RootProbe& result,
    bool rank_against_all_legal = false,
    bool rank_dtz = false
) noexcept;

} // namespace syzygy
} // namespace magnus
