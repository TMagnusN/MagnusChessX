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
 * NNUE Network Layout Constants — shared between evaluator and Position state
 *
 * Chess768 dual-perspective encoding scheme:
 *   Input layer: 768 features
 *     = 2 color perspectives x 6 piece types x 64 squares
 *     Each (perspective, piece_type, square) triple maps to one feature index
 *
 *   Hidden layer: 128 neurons
 *     = size of the dual-perspective accumulators
 *     The Position struct maintains one accumulator per perspective (2x128 = 256 i16 total)
 *
 *   Activation function: Clipped ReLU --> Square
 *     screlu(x) = clamp(x, 0, 255)^2
 *     Input clipped to [0, 255], output squared (max 65025)
 *
 *   Network architecture: 768 --> 128 --> 1
 *     Input feature weights: w0[768][128] (i16, per-feature contribution to 128 hidden neurons)
 *     Hidden-layer bias: b0[128] (i16)
 *     Output weights: w1[256] (i16, first half = own-perspective weights, second half = opponent-perspective weights)
 *     Output bias: b1 (i16)
 *
 *   AVX2 acceleration:
 *     Processes 16 hidden neurons at a time (128/16 = 8 rounds)
 *     Accumulators aligned to 64 bytes for SIMD loads
 */
namespace magnus::nnue {

constexpr int kInputSize = 768;         // Total input features for Chess768 encoding
constexpr int kHiddenSize = 128;        // Number of hidden-layer neurons (must be a multiple of 16 for AVX2)
constexpr int kActivationClip = 255;    // Upper clipping bound for the screlu activation function

} // namespace magnus::nnue
