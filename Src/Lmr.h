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
 * LMR (延遲著法減免) — Late Move Reduction
 *
 * 搜尋引擎中最關鍵的剪枝技術之一。
 * 核心思想：在非 PV 節點中，排序靠後的著法很可能不如前面幾個著法好，
 * 因此可以用減少的深度來搜索它們，節省大量節點。
 *
 * 減免量由固定點 (fixed-point) 算術動態計算：
 *   基礎減免 = log(depth) * log(move_index)   （查表實現）
 *   然後根據節點類型、改善狀態、TT 狀態、歷史分數等進行調整
 *
 * 若減免後的搜索意外地超過 alpha，則觸發 re-search 以確認結果。
 *
 * 三個核心結構體：
 *   LmrNodeContext — 節點層面的上下文（深度、PV/非PV、是否被將軍…）
 *   LmrMoveContext — 著法層面的上下文（類型、歷史分數、SEE 值…）
 *   LmrDecision     — LMR 決策輸出（是否減免、減免量、re-search 深度）
 */
namespace magnus::search {

/*
 * LmrNodeContext — LMR 決策所需的節點層面資訊
 *
 * 描述當前節點的基本狀態，用於決定減免的基準量。
 * 由 pvs() 在搜尋迴圈開始前填充。
 */
struct LmrNodeContext {
    int depth = 0;                  // 當前搜索深度（已含 IIR / hindsight 調整）
    int alpha = 0;                  // 當前 alpha 邊界
    int beta = 0;                   // 當前 beta 邊界
    int ply = 0;                    // 從根節點算起的半步數
    bool pv_node = false;           // 是否為 PV 節點（beta - alpha > 1）
    bool cut_node = false;          // 是否為 cut-node（預期快速截斷）
    bool all_node = false;          // 是否為 all-node（預期無截斷）
    bool checked = false;           // 走子方是否被將軍
    bool improving = false;         // 靜態評估是否在改善中
    bool exclusion_search = false;  // 是否為 singular/exclusion 驗證搜索
    bool mate_window = false;       // 是否接近 mate score window
    bool tt_move_present = false;   // TT 中是否有著法（影響減免量）
    bool tt_move_is_capture = false;// TT 著法是否為捕獲
    int static_eval = 0;            // 當前節點靜態評估
    int move_extension = 0;         // 該著法的延伸/負延伸量
    int next_ply_cutoff_count = 0;  // 下一層的截斷次數（用於調整減免）
    int parent_reduction_fp = 0;    // 父節點的減免量（固定點格式，用於連續減免）
    int tt_depth = 0;               // TT 條目深度（用於 LMR 信心加權）
    int tt_bound = 0;              // TT 邊界類型（EXACT>LOWER>UPPER>NONE）
};

/*
 * LmrMoveContext — LMR 決策所需的著法層面資訊
 *
 * 描述當前正在評估的著法的所有屬性。
 * 包含著法類型、歷史啟發式分數、SEE 值等。
 */
struct LmrMoveContext {
    Move move = 0;                  // 著法本身
    int move_index = 0;             // 在 MovePicker 中的位置（0-based）
    int reduction_index = 0;        // 減免索引：安靜著法用 move_index，捕獲用 (capture_count-1)
    bool is_tt_move = false;        // 是否為 TT 著法（TT 著法不可減免）
    bool quiet = false;             // 是否為安靜著法（非捕獲、非升變）
    bool capture = false;           // 是否為捕獲著法（含升變捕獲）
    bool simple_capture = false;    // 是否為簡單捕獲（捕獲但非升變）
    bool bad_capture = false;       // MovePicker/SEE 判定的壞捕獲
    bool gives_check = false;       // 是否將軍
    bool recapture = false;         // 是否為反吃（目標格與前一步相同）
    bool promotion = false;         // 是否為升變
    int ordering_score = 0;         // MovePicker 排序分數
    int quiet_history_score = 0;    // 安靜著法的歷史啟發式分數（含兵歷史）
    int continuation_score = 0;     // 延續歷史分數（基於前序著法對）
    int countermove_bonus = 0;      // 反著獎勵（對前一步的直接回應）
    int capture_history_score = 0;  // 捕獲歷史分數
    int see_value = 0;              // 靜態交換評估值（僅捕獲）
    int see_bias_term = 0;          // SEE 偏置項（基於深度+SEE 的動態調整）
};

/*
 * LmrDecision — LMR 的最終決策
 *
 * 封裝了減免計算的所有輸出：
 *   - 減免是否生效（eligible）
 *   - 減免的 ply 數（final_reduction）
 *   - 用於歷史更新和子節點調整的統計分數（stat_score）
 *   - 用於 re-search 深度計算的固定點減免量（final_reduction_fp）
 */
struct LmrDecision {
    int stat_score = 0;             // 綜合歷史統計分數（用於子節點調整和歷史更新）
    int base_reduction_fp = 0;      // 基礎減免量（固定點格式，FP_ONE_PLY = 1024）
    int final_reduction_fp = 0;     // 最終減免量（固定點格式，經所有調整後）
    int final_reduction = 0;        // 最終減免的整數 ply 數（從 final_reduction_fp 換算）
    bool eligible = false;          // 減免是否生效（final_reduction > 0）
};

/*
 * decide_lmr — 計算 LMR 減免決策
 *
 * 根據節點上下文和著法屬性，決定該著法應減免多少深度。
 * 若著法不符合減免條件（PV 節點、TT 著法、深度不足等），
 * 回傳 eligible=false 的決策。
 */
[[nodiscard]] LmrDecision decide_lmr(
    const LmrNodeContext& node,
    const LmrMoveContext& move
) noexcept;

/*
 * lmr_research_depth — 計算 LMR re-search 深度
 *
 * 當減免後的搜索意外超過 alpha 時，需要以更大的深度重新搜索。
 * 此函數根據減免量、分數差距、當前最佳分數決定 re-search 的深度。
 * 若分數大幅超過 alpha，可能使用比原始深度更深的 re-search。
 */
[[nodiscard]] int lmr_research_depth(
    const LmrDecision& decision,
    int full_depth,         // 原始完整深度
    int score,              // LMR 搜索回傳的分數
    int alpha,              // 當前 alpha
    int best_score          // 迄今為止的最佳分數
) noexcept;

} // namespace magnus::search
