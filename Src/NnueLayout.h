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
 * NNUE 網路佈局常數 — 在評估器和 Position 狀態之間共享
 *
 * Chess768 雙視角編碼方案：
 *   輸入層：768 個特徵
 *     = 2 個顏色視角 × 6 種棋子類型 × 64 個方格
 *     每個 (perspective, piece_type, square) 三元組對應一個特徵索引
 *
 *   隱藏層：128 個神經元
 *     = 雙視角累加器的大小
 *     Position 結構體為每個視角維護一個累加器（共 2×128 = 256 個 i16）
 *
 *   激活函數：Clipped ReLU → Square
 *     screlu(x) = clamp(x, 0, 255)^2
 *     輸入裁剪到 [0, 255]，輸出平方（最大 65025）
 *
 *   網路架構：768 → 128 → 1
 *     輸入特徵權重：w0[768][128]（i16，每個特徵對 128 個隱藏神經元的貢獻）
 *     隱藏層偏置：b0[128]（i16）
 *     輸出權重：w1[256]（i16，前半部為我方視角權重，後半部為對方視角權重）
 *     輸出偏置：b1（i16）
 *
 *   AVX2 加速：
 *     每次處理 16 個隱藏神經元（128/16 = 8 輪）
 *     累加器對齊到 64 位元組以支援 SIMD 載入
 */
namespace magnus::nnue {

constexpr int kInputSize = 768;         // Chess768 編碼的輸入特徵總數
constexpr int kHiddenSize = 128;        // 隱藏層神經元數量（必須為 16 的倍數以支援 AVX2）
constexpr int kActivationClip = 255;    // screlu 激活函數的裁剪上限

} // namespace magnus::nnue
