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

#include "Time.h"
#include "board/Position.h"

#include <algorithm>
#include <cmath>

/* ===== 繁體中文註釋 =====
 * 本檔案是 MagnusChessX Thinking 西洋棋引擎的一部分。
 * 實作詳情請參閱對應的 .h 標頭檔案。
 */

namespace magnus::timeman {

namespace {

[[nodiscard]] inline int game_ply_from_position(const Position& pos) noexcept {
    const int fullmove = std::max(1, pos.fullmove_number);
    return std::max(
        0,
        (fullmove - 1) * 2 + (pos.side_to_move == BLACK ? 1 : 0)
    );
}

} // namespace

/*
 * 時間管理實作
 * build_limits() — 從 UCI go 參數計算軟/硬時間限制
 */
void TimeManager::new_game() noexcept {
    original_time_adjust_ = -1.0;
    state_ = search::TimeManagementState{};
}

void TimeManager::set_move_overhead_ms(int value) noexcept {
    const int clamped = std::clamp(
        value,
        MIN_MOVE_OVERHEAD_MS,
        MAX_MOVE_OVERHEAD_MS
    );
    if (move_overhead_ms_ == clamped)
        return;

    move_overhead_ms_ = clamped;
    original_time_adjust_ = -1.0;
}

int TimeManager::move_overhead_ms() const noexcept {
    return move_overhead_ms_;
}

bool TimeManager::build_limits(
    const Position& pos,
    const GoParams& params,
    search::SearchLimits& limits
) noexcept {
    limits.depth = search::MAX_SEARCH_DEPTH;
    limits.node_limit = 0;
    limits.soft_time_ms = 0;
    limits.hard_time_ms = 0;
    limits.ponder = params.ponder;
    limits.infinite = false;
    limits.use_time_management = false;
    limits.time_state = nullptr;

    if (params.depth > 0)
        limits.depth = params.depth;

    limits.node_limit = params.nodes;
    limits.infinite = params.infinite;

    if (params.movetime > 0) {
        limits.soft_time_ms = params.movetime;
        limits.hard_time_ms = params.movetime;
        return true;
    }

    const Color side = static_cast<Color>(pos.side_to_move);
    const int remaining = side == WHITE ? params.wtime : params.btime;
    const int increment = side == WHITE ? params.winc : params.binc;

    if (!limits.infinite && remaining > 0) {
        limits.use_time_management = true;
        limits.time_state = &state_;
        const int ply = game_ply_from_position(pos);
        const i64 safe_remaining = std::max(1, remaining);
        const i64 safe_increment = std::max(0, increment);

        int centi_mtg =
            params.movestogo > 0 ? std::min(params.movestogo * 100, 5000) : 5051;
        if (remaining < 1000)
            centi_mtg = std::max(1, static_cast<int>(double(remaining) * 5.051));

        const i64 overhead = move_overhead_ms_;
        const i64 time_left = std::max<i64>(
            1,
            safe_remaining
                + (
                    safe_increment * (centi_mtg - 100)
                    - overhead * (200 + centi_mtg)
                ) / 100
        );

        double opt_scale = 1.0;
        double max_scale = 1.0;

        if (params.movestogo == 0) {
            if (original_time_adjust_ < 0.0)
                original_time_adjust_ =
                    0.3128 * std::log10(double(time_left)) - 0.4354;

            const double log_time_in_sec =
                std::log10(double(safe_remaining) / 1000.0);
            const double opt_constant =
                std::min(0.0032116 + 0.000321123 * log_time_in_sec, 0.00508017);
            const double max_constant =
                std::max(3.3977 + 3.03950 * log_time_in_sec, 2.94761);

            opt_scale =
                std::min(
                    0.0121431
                        + std::pow(double(ply) + 2.94693, 0.461073) * opt_constant,
                    0.213035 * double(safe_remaining) / double(time_left)
                )
                * original_time_adjust_;
            max_scale = std::min(6.67704, max_constant + double(ply) / 11.9847);
        } else {
            const double mtg = double(centi_mtg) / 100.0;
            opt_scale = std::min(
                (0.88 + double(ply) / 116.4) / mtg,
                0.88 * double(safe_remaining) / double(time_left)
            );
            max_scale = 1.3 + 0.11 * mtg;
        }

        int optimum = std::max(
            1,
            static_cast<int>(opt_scale * double(time_left))
        );
        const int maximum = std::max(
            1,
            static_cast<int>(
                std::min(
                    0.825179 * double(safe_remaining) - double(overhead),
                    max_scale * double(optimum)
                )
            ) - 10
        );

        if (params.ponder)
            optimum += optimum / 4;

        // The hard deadline is the safety cap. Never publish a soft budget
        // beyond it, especially at very short time controls.
        optimum = std::min(optimum, maximum);

        limits.soft_time_ms = optimum;
        limits.hard_time_ms = maximum;
    }

    if (!limits.infinite &&
        limits.depth == search::MAX_SEARCH_DEPTH &&
        limits.node_limit == 0 &&
        limits.soft_time_ms == 0 &&
        limits.hard_time_ms == 0) {
        return false;
    }

    return true;
}

} // namespace magnus::timeman
