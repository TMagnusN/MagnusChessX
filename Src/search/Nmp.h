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

#include "TT.h"

/*
 * NMP (Null Move Pruning)
 *
 * One of the most effective pruning techniques in a chess engine.
 * Core idea: give the opponent a "free turn" (null move). If the opponent
 * still cannot push the score below beta even with an extra move, then the
 * current position is very likely a cut-node and we can immediately return beta
 * without searching all moves.
 *
 * Null-move pruning conditions:
 *   1. Non-PV node (to avoid affecting principal variation accuracy)
 *   2. The side to move is not in check (otherwise a null move would skip
 *      the obligation to respond to the check)
 *   3. The side to move has sufficient non-pawn material (to avoid
 *      mis-pruning in Zugzwang positions)
 *   4. The static evaluation is already significantly above beta
 *
 * When the null-move search returns a score >= beta, there are two outcomes:
 *   a) Shallow depth: return the score directly (no verification needed)
 *   b) Deep depth: requires a verification search at the original depth to
 *      ensure correctness
 *
 * NmpNodeContext — input context for null-move decisions
 * NmpDecision    — null-move decision output (eligible, reduction, verification params)
 */
namespace magnus::search {

/*
 * NmpNodeContext — complete node information needed for null-move pruning decisions
 *
 * Filled in by pvs() before performing null-move pruning; contains all parameters
 * that influence the decision.
 */
struct NmpNodeContext {
    int depth = 0;                      // current search depth
    int ply = 0;                        // half-move count from the root
    int alpha = 0;                      // current alpha bound
    int beta = 0;                       // current beta bound (the null move aims to exceed this)
    int static_eval = 0;                // static evaluation of the current node
    int tt_score = 0;                   // score stored in the TT (used for bound adjustment)
    int nmp_min_ply = 0;                // minimum ply for which NMP is disabled after verification
    bool allow_null = false;            // whether null moves are allowed (may be temporarily disabled by Singular Extension)
    bool pv_node = false;               // whether this is a PV node (null-move pruning is not performed in PV nodes)
    bool cut_node = false;              // whether this is a cut-node (cut-nodes allow more aggressive reduction)
    bool checked = false;               // whether the side to move is in check (null move is illegal when in check)
    bool improving = false;             // whether the static evaluation is improving
    bool exclusion_search = false;      // whether this is an exclusion search (singular-detection exclusion searches should not null-move)
    bool tt_hit = false;                // whether the TT was hit (affects null-move threshold)
    bool tt_move_present = false;       // whether the TT has a move
    bool material_ok = false;           // whether there is sufficient non-pawn material for a null move (prevent Zugzwang)
    memory::Bound tt_bound = memory::BOUND_NONE; // TT bound type
};

/*
 * NmpDecision — the final null-move pruning decision
 *
 * Encapsulates all output parameters of the null-move computation.
 */
struct NmpDecision {
    bool eligible = false;              // whether null-move pruning is triggered
    bool requires_verification = false; // whether a verification search is required (deep null moves need it)
    int eval_gate = 0;                  // evaluation gate: static_eval must exceed this for null move to trigger
    int eval_margin = 0;                // evaluation margin: static_eval - beta
    int reduction = 0;                  // reduction in ply for the null-move search
    int null_depth = 0;                 // depth used for the null-move search (depth - 1 - reduction)
    int verify_depth = 0;               // depth used for the verification search (if required)
    int verify_min_ply = 0;             // minimum ply to disable NMP after verification (prevent consecutive null moves)
};

/*
 * nmp_disabled_for_ply — check whether null-move pruning is disabled at the current ply
 *
 * After a previous null-move verification fails, NMP is disabled over a
 * specific ply range.
 */
[[nodiscard]] bool nmp_disabled_for_ply(int ply, int nmp_min_ply) noexcept;

/*
 * decide_null_move — compute the null-move pruning decision
 *
 * Determines whether null-move pruning should be attempted and with what
 * parameters, based on the node context.
 * Returns eligible=false if the node does not meet null-move conditions.
 */
[[nodiscard]] NmpDecision decide_null_move(const NmpNodeContext& node) noexcept;

} // namespace magnus::search
