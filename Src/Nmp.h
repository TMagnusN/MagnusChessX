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
 * NMP (空步剪枝) — Null Move Pruning
 *
 * 搜尋引擎中最有效的剪枝技術之一。
 * 核心思想：給對方一個「免費回合」（空步），若對方即使多走一步
 * 仍無法將局面降到 beta 以下，則當前局面很可能是一個截斷節點，
 * 可以直接回傳 beta，無需完整搜索所有著法。
 *
 * 空步剪枝的條件：
 *   1. 非 PV 節點（避免影響主變例精度）
 *   2. 走子方未被將軍（否則空步會跳過應將）
 *   3. 走子方有足夠的非兵材料（避免 Zugzwang 局面誤剪）
 *   4. 靜態評估已顯著高於 beta
 *
 * 當空步搜索回傳分數 >= beta 時有兩種處理：
 *   a) 淺層：直接回傳分數（無需驗證）
 *   b) 深層：需要驗證搜索（verify search），以原始深度重新搜索確保正確性
 *
 * NmpNodeContext — 空步決策的輸入上下文
 * NmpDecision    — 空步決策的輸出（是否觸發、減免量、驗證參數）
 */
namespace magnus::search {

/*
 * NmpNodeContext — 空步剪枝決策所需的完整節點資訊
 *
 * 由 pvs() 在進行空步剪枝前填充，包含所有影響決策的參數。
 */
struct NmpNodeContext {
    int depth = 0;                      // 當前搜索深度
    int ply = 0;                        // 從根節點算起的半步數
    int alpha = 0;                      // 當前 alpha 邊界
    int beta = 0;                       // 當前 beta 邊界（空步的目標是超過此值）
    int static_eval = 0;                // 當前節點的靜態評估值
    int tt_score = 0;                   // TT 中儲存的分數（用於邊界調整）
    int nmp_min_ply = 0;                // NMP 驗證後的最小禁用 ply
    bool allow_null = false;            // 是否允許空步（可能被 Singular Extension 暫時禁用）
    bool pv_node = false;               // 是否為 PV 節點（PV 節點不進行空步剪枝）
    bool cut_node = false;              // 是否為 cut-node（cut-node 允許更激進的減免）
    bool checked = false;               // 走子方是否被將軍（被將時不可空步）
    bool improving = false;             // 靜態評估是否在改善中
    bool exclusion_search = false;      // 是否為排除搜索（奇異檢測中的排除搜索不應空步）
    bool tt_hit = false;                // TT 是否命中（影響空步閾值）
    bool tt_move_present = false;       // TT 中是否有著法
    bool material_ok = false;           // 是否有足夠的非兵材料進行空步（防止 Zugzwang）
    memory::Bound tt_bound = memory::BOUND_NONE; // TT 邊界類型
};

/*
 * NmpDecision — 空步剪枝的最終決策
 *
 * 封裝了空步計算的所有輸出參數。
 */
struct NmpDecision {
    bool eligible = false;              // 是否觸發空步剪枝
    bool requires_verification = false; // 是否需要驗證搜索（深層空步需要）
    int eval_gate = 0;                  // 評估門檻：static_eval 必須超過此值才能觸發空步
    int eval_margin = 0;                // 評估餘量：static_eval - beta
    int reduction = 0;                  // 空步搜索的減免 ply 數
    int null_depth = 0;                 // 空步搜索使用的深度（depth - 1 - reduction）
    int verify_depth = 0;               // 驗證搜索使用的深度（若需要驗證）
    int verify_min_ply = 0;             // 驗證後的最小禁用 ply（防止連續空步）
};

/*
 * nmp_disabled_for_ply — 檢查當前 ply 是否被禁用空步
 *
 * 當之前的空步驗證失敗後，會在特定 ply 範圍內禁用空步。
 */
[[nodiscard]] bool nmp_disabled_for_ply(int ply, int nmp_min_ply) noexcept;

/*
 * decide_null_move — 計算空步剪枝決策
 *
 * 根據節點上下文判斷是否應進行空步剪枝，以及使用什麼參數。
 * 若節點不符合空步條件，回傳 eligible=false。
 */
[[nodiscard]] NmpDecision decide_null_move(const NmpNodeContext& node) noexcept;

} // namespace magnus::search
