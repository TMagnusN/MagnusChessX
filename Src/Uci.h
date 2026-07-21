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
 * UCI Frontend — Universal Chess Interface for MagnusChessX Thinking
 *
 * This module implements a complete UCI protocol frontend, responsible for:
 *   1. UCI command parsing (go / position / setoption / stop / ponderhit / quit / uci / isready / ucinewgame)
 *   2. FEN string parsing and position construction (startpos / fen + moves)
 *   3. Loading and managing MNUE P2/P2Pro network files
 *   4. Search thread lifecycle management (start / stop / ponder / ponderhit)
 *   5. UCI info string output (depth, score, PV, node count, nps, hashfull)
 *   6. Extracting the ponder move from the most recent search results
 *
 * The sole public entry point is run_uci(), which starts the stdin/stdout UCI loop.
 */
namespace magnus {

/*
 * run_uci — UCI command loop entry point
 *
 * Initializes the UciSession (memory, position, MNUE), outputs the banner and UCI ID,
 * then enters a blocking stdin command loop. Returns upon receiving the "quit" command.
 *
 * Return value: 0 indicates normal exit
 */
int run_uci();

// Minimal command-line tools kept in the formal engine build.
int run_bench(int argc, char** argv);

} // namespace magnus
