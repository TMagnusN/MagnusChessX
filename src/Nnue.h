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

#include <string>

#include "Types.h"

namespace magnus {
struct Position;
}

/*
 * NNUE 公開介面 — 高效更新的神經網路 (Efficiently Updatable Neural Network)
 *
 * 架構概述：
 *   Chess768 雙視角編碼（2×6×64 = 768 個輸入特徵）
 *   → 128 個隱藏層神經元（雙視角累加器）
 *   → 單一輸出神經元（評分，非 cp 單位）
 *
 * 增量更新：
 *   累加器儲存在 Position 結構體中（每個視角一個），
 *   透過 on_piece_added / on_piece_removed / on_piece_moved 鉤子
 *   在 make/unmake 期間進行增量更新（O(隱藏層大小) 而非 O(輸入×隱藏層)）。
 *
 * 分數轉換鏈：
 *   原始輸出 → search_score() → 內部搜尋分數
 *            → search_score_to_cp() → UCI 厘兵顯示
 *            → search_score_to_wdl() → 勝率和三元組
 *
 * 網路檔案格式：
 *   原生 .nnue (Magnus 格式, 含標頭)
 *   Bullet .bin  (Rust simple.rs 量化格式, 無標頭)
 */
namespace magnus::nnue {

// ============================================================
// 分數轉換參數結構體
// ============================================================

// WinRateParams — 勝率模型的 Sigmoid 參數 (a=中心, b=斜率)
struct WinRateParams { double a; double b; };

// WdlTriplet — 勝/和/負機率（千分比，總和=1000）
struct WdlTriplet { int win; int draw; int loss; };

// ============================================================
// 網路生命週期管理
// ============================================================
bool load(const std::string& path);     // 載入 NNUE 網路檔案
void unload() noexcept;                 // 卸載當前網路
bool loaded() noexcept;                 // 網路是否已載入
const std::string& path() noexcept;     // 當前載入的檔案路徑
const std::string& description() noexcept; // 網路架構描述
const std::string& last_error() noexcept;  // 最後一次錯誤訊息

// ============================================================
// 局面評估
// ============================================================
int eval(const Position& pos) noexcept;         // 原始 NNUE 輸出（非 cp）
WinRateParams win_rate_params(const Position& pos) noexcept;
int to_cp(int v, const Position& pos) noexcept; // 原始值 → cp
int win_rate_model(int v, const Position& pos) noexcept; // 原始值 → 勝率(0-1000)
int search_score(int v, const Position& pos) noexcept;   // 原始值 → 搜尋分數
int search_score_to_cp(int score, const Position& pos) noexcept;
int search_score_to_winrate(int score, const Position& pos) noexcept;
WdlTriplet search_score_to_wdl(int score, const Position& pos) noexcept;

// ============================================================
// 增量累加器更新鉤子 — 由 Position 的 make/unmake 調用
// ============================================================
void on_position_cleared(Position& pos) noexcept;
void on_piece_added(Position& pos, Color color, PieceType pt, Square sq) noexcept;
void on_piece_removed(Position& pos, Color color, PieceType pt, Square sq) noexcept;
void on_piece_moved(Position& pos, Color color, PieceType pt, Square from, Square to) noexcept;

} // namespace magnus::nnue
