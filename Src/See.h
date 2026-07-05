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
 * SEE (靜態交換評估) — Static Exchange Evaluation
 *
 * 在不實際走棋的情況下，模擬目標格上的連續兌子交換，
 * 計算捕獲著法的淨材料收益（以厘兵為單位）。
 *
 * 核心演算法：
 *   1. 起始方走捕獲，獲得 victim 的價值
 *   2. 對方用最小價值的攻擊子（LVA, Least Valuable Attacker）反吃
 *   3. 交替進行，每次用 LVA 回應
 *   4. 當一方不再有攻擊子、或國王被暴露時終止
 *   5. 從最深層回溯，使用 minimax 計算最優交換序列的淨值
 *
 * 兩個主要用途：
 *   a) 著法排序 — 好的捕獲（正 SEE）排在壞的捕獲（負 SEE）之前
 *   b) 剪枝決策 — 負 SEE 的捕獲在淺層可以被安全跳過
 *
 * 本模組提供四個公開函數：
 *   see_value()      — 完整計算捕獲的 SEE 值（慢速路徑，含驗證）
 *   see_value_fast() — 快速路徑，假設調用者已確認著法合法
 *   see_ge()         — 判斷捕獲是否達到指定閾值（含驗證）
 *   see_ge_fast()    — 快速路徑的閾值判斷（含提前退出優化）
 */
namespace magnus::search {

// 共用棋子價值表（厘兵）— 用於 MVV-LVA 排序與捕獲 gain 估算
// 兵=100, 馬=320, 象=330, 車=500, 后=900, 王=0（王不參與 MVV-LVA 排序）
constexpr int piece_order_value[PIECE_TYPE_NB] = {
    100, 320, 330, 500, 900, 0
};

/*
 * see_value — 計算捕獲著法的完整靜態交換評估值
 *
 * 僅對捕獲著法有效；對非捕獲著法直接回傳 0。
 * 此版本會驗證著法的合法性，適用於非熱路徑（根節點、觀測輸出）。
 *
 * 回傳值：以厘兵 (cp) 為單位的淨交換收益
 */
[[nodiscard]] int see_value(
    const Position& pos,
    const memory::Memory& mem,
    Move move
) noexcept;

/*
 * see_value_fast — 快速路徑的 SEE 計算
 *
 * 假設調用者已確保 move 是合法的捕獲著法（通過 assert 驗證）。
 * 用於 MovePicker 著法評分、qsearch 捕獲排序等熱路徑。
 */
[[nodiscard]] int see_value_fast(
    const Position& pos,
    const memory::Memory& mem,
    Move move
) noexcept;

/*
 * see_ge — 判斷捕獲著法的 SEE 值是否 >= threshold
 *
 * 相比 see_value()，此版本有提前退出優化：
 * 當確定交換結果不可能達到閾值時立即停止計算。
 * 廣泛用於捕獲剪枝、late move pruning 等決策。
 */
[[nodiscard]] bool see_ge(
    const Position& pos,
    const memory::Memory& mem,
    Move move,
    int threshold
) noexcept;

/*
 * see_ge_fast — 快速路徑的閾值判斷
 *
 * 與 see_ge() 相同，但假設 move 為合法捕獲著法。
 * 是搜索熱迴圈中最常用的 SEE 調用形式。
 */
[[nodiscard]] bool see_ge_fast(
    const Position& pos,
    const memory::Memory& mem,
    Move move,
    int threshold
) noexcept;

} // namespace magnus::search
