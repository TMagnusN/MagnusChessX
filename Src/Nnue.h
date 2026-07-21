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
 * NNUE Public Interface — Efficiently Updatable Neural Network
 *
 * Architecture overview:
 *   Chess768 dual-perspective encoding (2x6x64 = 768 input features)
 *   --> 128 hidden-layer neurons (dual-perspective accumulators)
 *   --> single output neuron (score, not in cp units)
 *
 * Incremental updates:
 *   Accumulators are stored in the Position struct (one per perspective),
 *   updated incrementally via the on_piece_added / on_piece_removed / on_piece_moved hooks
 *   during make/unmake (O(hidden size) instead of O(input x hidden)).
 *
 * Score conversion chain:
 *   raw output --> search_score() --> internal search score
 *              --> search_score_to_cp() --> UCI centipawn display
 *              --> search_score_to_wdl() --> win rate and WDL triplet
 *
 * Network file formats:
 *   Native .nnue (Magnus format, with header)
 *   Bullet  .bin  (Rust simple.rs quantized format, no header)
 */
namespace magnus::nnue {

// ============================================================
// Score conversion parameter structs
// ============================================================

// WinRateParams — Sigmoid parameters for the win-rate model (a=center, b=slope)
struct WinRateParams { double a; double b; };

// WdlTriplet — win/draw/loss probabilities (in per mille, sum = 1000)
struct WdlTriplet { int win; int draw; int loss; };

// ============================================================
// Network lifecycle management
// ============================================================
bool load(const std::string& path);     // Load NNUE network file
void unload() noexcept;                 // Unload the current network
bool loaded() noexcept;                 // Whether a network is loaded
const std::string& path() noexcept;     // Path of the currently loaded file
const std::string& description() noexcept; // Network architecture description
const std::string& last_error() noexcept;  // Most recent error message

// ============================================================
// Position evaluation
// ============================================================
int eval(const Position& pos) noexcept;         // Raw NNUE output (not cp)
WinRateParams win_rate_params(const Position& pos) noexcept;
int to_cp(int v, const Position& pos) noexcept; // Raw value --> cp
int win_rate_model(int v, const Position& pos) noexcept; // Raw value --> win rate (0-1000)
int search_score(int v, const Position& pos) noexcept;   // Raw value --> search score
int search_score_to_cp(int score, const Position& pos) noexcept;
int search_score_to_winrate(int score, const Position& pos) noexcept;
WdlTriplet search_score_to_wdl(int score, const Position& pos) noexcept;

// ============================================================
// Incremental accumulator update hooks — called by Position's make/unmake
// ============================================================
void on_position_cleared(Position& pos) noexcept;
void on_piece_added(Position& pos, Color color, PieceType pt, Square sq) noexcept;
void on_piece_removed(Position& pos, Color color, PieceType pt, Square sq) noexcept;
void on_piece_moved(Position& pos, Color color, PieceType pt, Square from, Square to) noexcept;

} // namespace magnus::nnue
