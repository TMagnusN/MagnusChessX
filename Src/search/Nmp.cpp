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

#include "Nmp.h"

#include <algorithm>

#include "Search.h"

/*
 * NMP (空步剪枝) 實作 — Null Move Pruning
 *
 * 核心思想：給對方一個「免費回合」。
 * 若對方多走一步仍無法將局面降到 beta 以下，則當前節點很可能截斷，
 * 可以直接回傳 beta 而無需搜索所有著法。
 *
 * 注意事項：
 *   1. 被將軍時不可空步（否則跳過應將）
 *   2. Zugzwang 局面需特別處理（僅有王+兵時禁用空步）
 *   3. 深層空步需要驗證搜索以免誤剪
 *
 * NMP 減免量 = 2 + depth/4 + eval_margin/96 + cut_node + !improving + !tt_move
 */
namespace magnus::search {


bool nmp_disabled_for_ply(int ply, int nmp_min_ply) noexcept {
    return nmp_min_ply != 0 && ply < nmp_min_ply;
}

NmpDecision decide_null_move(const NmpNodeContext& node) noexcept {
    NmpDecision decision{};
    decision.eval_gate =
        node.beta
        - NMP_STATIC_DEPTH_SLOPE * node.depth
        + NMP_STATIC_BASE
        - (node.improving ? NMP_IMPROVING_MARGIN : 0);
    decision.eval_margin = node.static_eval - node.beta;

    if (!node.allow_null ||
        node.pv_node ||
        node.checked ||
        node.exclusion_search ||
        node.depth < 3 ||
        !node.material_ok ||
        nmp_disabled_for_ply(node.ply, node.nmp_min_ply) ||
        node.static_eval < decision.eval_gate) {
        return decision;
    }

    if (node.tt_hit &&
        node.tt_bound == memory::BOUND_UPPER &&
        node.tt_score < node.beta) {
        return decision;
    }

    int reduction = NMP_MIN_REDUCTION + node.depth / 4;
    reduction += std::clamp(decision.eval_margin / NMP_EVAL_BUCKET, 0, 3);
    if (node.cut_node)
        ++reduction;
    if (!node.improving)
        ++reduction;
    if (!node.tt_move_present)
        ++reduction;
    if (node.tt_hit &&
        node.tt_bound == memory::BOUND_LOWER &&
        node.tt_score >= node.beta) {
        --reduction;
    }

    reduction = std::clamp(
        reduction,
        NMP_MIN_REDUCTION,
        std::max(NMP_MIN_REDUCTION, node.depth - 2)
    );

    decision.eligible = true;
    decision.reduction = reduction;
    decision.null_depth = std::max(0, node.depth - 1 - reduction);
    decision.requires_verification =
        node.depth >= NMP_VERIFICATION_MIN_DEPTH &&
        node.nmp_min_ply == 0;
    decision.verify_depth = decision.null_depth;
    decision.verify_min_ply = node.ply + std::max(
        NMP_VERIFICATION_MIN_SPAN,
        (3 * std::max(1, decision.null_depth)) / 4
    );
    return decision;
}

} // namespace magnus::search
