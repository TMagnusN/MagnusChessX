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

/*
 * LMR (Late Move Reduction)
 *
 * One of the most critical pruning techniques in a chess engine.
 * Core idea: in non-PV nodes, moves ordered later in the list are very likely
 * to be worse than the first few moves, so they can be searched at a reduced
 * depth, saving a huge number of nodes.
 *
 * The reduction amount is computed dynamically using fixed-point arithmetic:
 *   base reduction = log(depth) * log(move_index)   (implemented by table lookup)
 *   then adjusted according to node type, improvement status, TT status,
 *   history scores, etc.
 *
 * If the reduced search unexpectedly exceeds alpha, a re-search is triggered
 * to confirm the result.
 *
 * Three core structures:
 *   LmrNodeContext — node-level context (depth, PV/non-PV, in-check, ...)
 *   LmrMoveContext — move-level context (type, history scores, SEE value, ...)
 *   LmrDecision     — LMR decision output (eligible, reduction amount, re-search depth)
 */
namespace magnus::search {

/*
 * LmrNodeContext — node-level information needed for LMR decisions
 *
 * Describes the basic state of the current node, used to determine the
 * baseline reduction amount. Filled in by pvs() before the search loop.
 */
struct LmrNodeContext {
    int depth = 0;                  // current search depth (already includes IIR / hindsight adjustments)
    int alpha = 0;                  // current alpha bound
    int beta = 0;                   // current beta bound
    int ply = 0;                    // half-move count from the root
    bool pv_node = false;           // whether this is a PV node (beta - alpha > 1)
    bool cut_node = false;          // whether this is a cut-node (expected to produce a quick cutoff)
    bool all_node = false;          // whether this is an all-node (expected no cutoff)
    bool checked = false;           // whether the side to move is in check
    bool improving = false;         // whether the static evaluation is improving
    bool exclusion_search = false;  // whether this is a singular/exclusion verification search
    bool mate_window = false;       // whether the window is near mate score
    bool tt_move_present = false;   // whether the TT has a move (affects reduction amount)
    bool tt_move_is_capture = false;// whether the TT move is a capture
    int static_eval = 0;            // static evaluation of the current node
    int move_extension = 0;         // extension / negative-extension amount for the move
    int next_ply_cutoff_count = 0;  // cutoff count at the next ply (used to adjust reduction)
    int parent_reduction_fp = 0;    // parent node's reduction amount (fixed-point, for consecutive reduction)
    int tt_depth = 0;               // TT entry depth (used for LMR confidence weighting)
    int tt_bound = 0;              // TT bound type (EXACT > LOWER > UPPER > NONE)
};

/*
 * LmrMoveContext — move-level information needed for LMR decisions
 *
 * Describes all attributes of the move currently being evaluated.
 * Includes move type, history heuristic scores, SEE value, etc.
 */
struct LmrMoveContext {
    Move move = 0;                  // the move itself
    int move_index = 0;             // position in the MovePicker (0-based)
    int reduction_index = 0;        // reduction index: move_index for quiets, (capture_count-1) for captures
    bool is_tt_move = false;        // whether this is the TT move (TT moves must not be reduced)
    bool quiet = false;             // whether this is a quiet move (non-capture, non-promotion)
    bool capture = false;           // whether this is a capture move (including capture promotions)
    bool simple_capture = false;    // whether this is a simple capture (capture but not promotion)
    bool bad_capture = false;       // bad capture as determined by MovePicker/SEE
    bool gives_check = false;       // whether this move gives check
    bool recapture = false;         // whether this is a recapture (target square same as previous move)
    bool promotion = false;         // whether this is a promotion
    int ordering_score = 0;         // MovePicker ordering score
    int quiet_history_score = 0;    // quiet-move history heuristic score (includes pawn history)
    int continuation_score = 0;     // continuation history score (based on preceding move pairs)
    int countermove_bonus = 0;      // countermove bonus (direct response to the previous move)
    int capture_history_score = 0;  // capture history score
    int see_value = 0;              // static exchange evaluation value (captures only)
    int see_bias_term = 0;          // SEE bias term (dynamic adjustment based on depth + SEE)
};

/*
 * LmrDecision — the final LMR decision
 *
 * Encapsulates all outputs of the reduction computation:
 *   - Whether the reduction takes effect (eligible)
 *   - The reduction amount in ply (final_reduction)
 *   - The statistical score used for history updates and child-node adjustment (stat_score)
 *   - The fixed-point reduction amount used for re-search depth calculation (final_reduction_fp)
 */
struct LmrDecision {
    int stat_score = 0;             // composite history statistical score (used for child-node adjustment and history updates)
    int base_reduction_fp = 0;      // base reduction amount (fixed-point format, FP_ONE_PLY = 1024)
    int final_reduction_fp = 0;     // final reduction amount (fixed-point format, after all adjustments)
    int final_reduction = 0;        // final reduction in integer ply (converted from final_reduction_fp)
    bool eligible = false;          // whether the reduction takes effect (final_reduction > 0)
};

/*
 * decide_lmr — compute the LMR reduction decision
 *
 * Based on the node context and move attributes, determines how much depth
 * reduction should be applied to this move.
 * If the move does not meet reduction conditions (PV node, TT move,
 * insufficient depth, etc.), returns a decision with eligible=false.
 */
[[nodiscard]] LmrDecision decide_lmr(
    const LmrNodeContext& node,
    const LmrMoveContext& move
) noexcept;

/*
 * lmr_research_depth — compute the LMR re-search depth
 *
 * When the reduced search unexpectedly exceeds alpha, a re-search at a larger
 * depth is needed to confirm the result.
 * This function determines the re-search depth based on the reduction amount,
 * score gap, and current best score.
 * If the score significantly exceeds alpha, the re-search may use a depth
 * deeper than the original.
 */
[[nodiscard]] int lmr_research_depth(
    const LmrDecision& decision,
    int full_depth,         // original full depth
    int score,              // score returned by the LMR search
    int alpha,              // current alpha
    int best_score          // best score so far
) noexcept;

} // namespace magnus::search
