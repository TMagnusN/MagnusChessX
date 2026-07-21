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

#include "Memory.h"
#include "board/MoveGen.h"

namespace magnus {
struct Position;
}

/*
 * SEE (Static Exchange Evaluation)
 *
 * Without actually making moves on the board, simulates a series of consecutive
 * captures on a target square and computes the net material gain (in centipawns)
 * for a capture move.
 *
 * Core algorithm:
 *   1. The side to move makes the capture and gains the victim's value
 *   2. The opponent responds with the Least Valuable Attacker (LVA)
 *   3. The sides alternate, each time responding with the LVA
 *   4. The exchange terminates when one side runs out of attackers or a king
 *      is exposed
 *   5. Backpropagate from the deepest level using minimax to compute the net
 *      value of the optimal exchange sequence
 *
 * Two main uses:
 *   a) Move ordering — good captures (positive SEE) are ordered before bad
 *      captures (negative SEE)
 *   b) Pruning decisions — captures with negative SEE can be safely skipped
 *      at shallow depths
 *
 * This module provides four public functions:
 *   see_value()      — full SEE computation with legality verification (slow path)
 *   see_value_fast() — fast path, assumes the caller has confirmed the move is legal
 *   see_ge()         — threshold check with legality verification
 *   see_ge_fast()    — fast-path threshold check (with early-exit optimization)
 */
namespace magnus::search {

// Shared piece value table (centipawns) — used for MVV-LVA ordering and capture-gain estimation
// Pawn=100, Knight=320, Bishop=330, Rook=500, Queen=900, King=0 (kings do not participate in MVV-LVA ordering)
constexpr int piece_order_value[PIECE_TYPE_NB] = {
    100, 320, 330, 500, 900, 0
};

/*
 * see_value — compute the full static exchange evaluation for a capture move
 *
 * Only valid for capture moves; returns 0 directly for non-captures.
 * This version verifies move legality and is suitable for non-hot paths
 * (root node, observation output).
 *
 * Returns: net exchange gain in centipawns (cp)
 */
[[nodiscard]] int see_value(
    const Position& pos,
    const memory::Memory& mem,
    Move move
) noexcept;

/*
 * see_value_fast — fast-path SEE computation
 *
 * Assumes the caller has already ensured that move is a legal capture
 * (verified by assert). Used in hot paths such as MovePicker move scoring
 * and qsearch capture ordering.
 */
[[nodiscard]] int see_value_fast(
    const Position& pos,
    const memory::Memory& mem,
    Move move
) noexcept;

/*
 * see_ge — determine whether a capture move's SEE value is >= threshold
 *
 * Compared to see_value(), this version has early-exit optimization:
 * computation stops as soon as it is determined that the exchange result
 * cannot possibly reach the threshold.
 * Widely used for capture pruning, late move pruning, and similar decisions.
 */
[[nodiscard]] bool see_ge(
    const Position& pos,
    const memory::Memory& mem,
    Move move,
    int threshold
) noexcept;

/*
 * see_ge_fast — fast-path threshold check
 *
 * Same as see_ge(), but assumes move is a legal capture.
 * This is the most commonly used SEE call form in the search hot loop.
 */
[[nodiscard]] bool see_ge_fast(
    const Position& pos,
    const memory::Memory& mem,
    Move move,
    int threshold
) noexcept;

} // namespace magnus::search
