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

/*
 * UCI 前端 — MagnusChessX Thinking 的通用西洋棋介面 (Universal Chess Interface)
 *
 * 本模組實作完整的 UCI 協定前端，負責：
 *   1. UCI 命令解析（go / position / setoption / stop / ponderhit / quit / uci / isready / ucinewgame）
 *   2. FEN 字串解析與局面建構（startpos / fen + moves）
 *   3. MNUE P2/P2Pro 網路檔案的載入與管理
 *   4. 搜尋線程的生命週期管理（啟動 / 停止 / 沉思 / ponderhit）
 *   5. UCI info 字串輸出（深度、分數、PV、節點數、nps、hashfull）
 *   6. 基於最近搜尋結果的沉思著法 (ponder move) 提取
 *
 * 唯一的公開入口點是 run_uci()，啟動標準輸入/輸出 UCI 迴圈。
 */
namespace magnus {

/*
 * run_uci — UCI 命令迴圈入口點
 *
 * 初始化 UciSession（記憶體、局面、MNUE），輸出橫幅和 UCI ID，
 * 然後進入阻塞的 stdin 命令迴圈。接收到 "quit" 命令時回傳。
 *
 * 回傳值：0 表示正常退出
 */
int run_uci();

// Minimal command-line tools kept in the formal engine build.
int run_bench(int argc, char** argv);

} // namespace magnus
