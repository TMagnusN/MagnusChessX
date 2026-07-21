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

#include "Search.h"
#include "Types.h"

namespace magnus {
struct Position;
}

/*
 * Time management module — Time Manager
 *
 * Converts raw time parameters from the UCI go command into standardized
 * SearchLimits, and computes the time budget based on the clock, increment,
 * moves-to-go, and current ply.
 *
 * Supports two layers of time control:
 *   1. Hard limit (hard_time_ms) — the absolute ceiling that is never exceeded
 *   2. Soft limit (soft_time_ms) — the optimal stopping point computed by the
 *      time management algorithm
 *
 * Initial soft/hard budgets are computed by the time management formula;
 * during-search adjustments for falling eval, depth factor, and best-move
 * instability reside in Search.cpp.
 */
namespace magnus::timeman {

inline constexpr int DEFAULT_MOVE_OVERHEAD_MS = 10;
inline constexpr int MIN_MOVE_OVERHEAD_MS = 0;
inline constexpr int MAX_MOVE_OVERHEAD_MS = 5000;

/*
 * GoParams — standardized time parameters parsed from the UCI "go" command
 *
 * All time units are in milliseconds (ms).
 */
struct GoParams {
    int depth = 0;          // fixed depth limit (go depth N)
    u64 nodes = 0;          // fixed node count limit (go nodes N)
    int movetime = 0;       // fixed time limit (go movetime N)
    int wtime = 0;          // white's remaining time (go wtime N)
    int btime = 0;          // black's remaining time (go btime N)
    int winc = 0;           // white's increment (go winc N)
    int binc = 0;           // black's increment (go binc N)
    int movestogo = 0;      // moves to go (go movestogo N), 0 = unknown
    bool ponder = false;    // whether in ponder mode
    bool infinite = false;  // whether in infinite search mode
};

/*
 * TimeManager — time manager
 *
 * One instance per UciSession; retains only the cross-move raw time adjustment
 * value and the user-configured Move Overhead.
 */
class TimeManager {
public:
    TimeManager() = default;

    // new_game — reset cross-move time adjustment state
    void new_game() noexcept;

    void set_move_overhead_ms(int value) noexcept;
    [[nodiscard]] int move_overhead_ms() const noexcept;

    /*
     * build_limits — construct SearchLimits from GoParams
     *
     * Computes soft/hard time limits based on the current position and time
     * parameters. Returns false if the parameters are invalid (missing required
     * time controls).
     */
    [[nodiscard]] bool build_limits(
        const Position& pos,
        const GoParams& params,
        search::SearchLimits& limits
    ) noexcept;

private:
    double original_time_adjust_ = -1.0;
    int move_overhead_ms_ = DEFAULT_MOVE_OVERHEAD_MS;
    search::TimeManagementState state_{};
};

} // namespace magnus::timeman
