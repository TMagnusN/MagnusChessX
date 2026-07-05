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
 * 時間管理模組 — Time Manager
 *
 * 將 UCI go 命令的原始時間參數轉換為標準化的 SearchLimits，
 * 並根據棋鐘、增量、剩餘著數與當前 ply 計算時間預算。
 *
 * 支援兩層時間控制：
 *   1. 硬限制 (hard_time_ms)：無論如何都不會超過的絕對上限
 *   2. 軟限制 (soft_time_ms)：時間管理演算法計算的最佳停止點
 *
 * 初始 soft/hard 預算由 Stockfish 風格公式計算；搜尋期間的
 * falling-eval、深度因子與最佳著法不穩定性調整位於 Search.cpp。
 */
namespace magnus::timeman {

inline constexpr int DEFAULT_MOVE_OVERHEAD_MS = 10;
inline constexpr int MIN_MOVE_OVERHEAD_MS = 0;
inline constexpr int MAX_MOVE_OVERHEAD_MS = 5000;

/*
 * GoParams — 從 UCI "go" 命令解析出的標準化時間參數
 *
 * 所有時間單位均為毫秒 (ms)。
 */
struct GoParams {
    int depth = 0;          // 固定深度限制（go depth N）
    u64 nodes = 0;          // 固定節點數限制（go nodes N）
    int movetime = 0;       // 固定時間限制（go movetime N）
    int wtime = 0;          // 白方剩餘時間（go wtime N）
    int btime = 0;          // 黑方剩餘時間（go btime N）
    int winc = 0;           // 白方增量（go winc N）
    int binc = 0;           // 黑方增量（go binc N）
    int movestogo = 0;      // 剩餘著法數（go movestogo N），0=未知
    bool ponder = false;    // 是否為沉思模式
    bool infinite = false;  // 是否為無限搜尋
};

/*
 * TimeManager — 時間管理器
 *
 * 每個 UciSession 創建一個實例，只保留跨著需要的原始時間調整值
 * 與使用者設定的 Move Overhead。
 */
class TimeManager {
public:
    TimeManager() = default;

    // new_game — 重設跨著時間調整狀態
    void new_game() noexcept;

    void set_move_overhead_ms(int value) noexcept;
    [[nodiscard]] int move_overhead_ms() const noexcept;

    /*
     * build_limits — 從 GoParams 構建 SearchLimits
     *
     * 根據當前局面和時間參數計算軟/硬時間限制。
     * 回傳 false 表示參數無效（缺少必要的時間控制）。
     */
    [[nodiscard]] bool build_limits(
        const Position& pos,
        const GoParams& params,
        search::SearchLimits& limits
    ) noexcept;

private:
    double original_time_adjust_ = -1.0;
    int move_overhead_ms_ = DEFAULT_MOVE_OVERHEAD_MS;
};

} // namespace magnus::timeman
