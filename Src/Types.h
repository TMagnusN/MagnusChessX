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

#include <cstdint>

/*
 * Global Type Definitions — Foundation type system for MagnusChessX Thinking
 *
 * All modules (move generation, search, evaluation, hashing) share the
 * scalar aliases, enumerations, and bit-operation helper functions defined here.
 *
 * Design principles:
 *   - Use fixed-width integer types (u64/i16 etc.) to ensure cross-platform consistency
 *   - Move is packed into 16 bits (u16), containing source/destination/promotion info
 *   - Square uses int (0-63) for convenient array indexing
 *   - All helper functions are constexpr, supporting compile-time evaluation
 */
namespace magnus {

// ============================================================
// Fixed-width integer aliases — ensure consistent bit-width across platforms
// ============================================================
using u8  = std::uint8_t;       //  8-bit unsigned (used for TT age/flags)
using u16 = std::uint16_t;      // 16-bit unsigned (Move packed format)
using u32 = std::uint32_t;      // 32-bit unsigned (TT tag32, NNUE generation)
using u64 = std::uint64_t;      // 64-bit unsigned (Bitboard, Key, node count)
using i8  = std::int8_t;        //  8-bit signed
using i16 = std::int16_t;       // 16-bit signed (TT score/eval/depth, NNUE weights)
using i32 = std::int32_t;       // 32-bit signed (move ordering score, LMR computation)
using i64 = std::int64_t;       // 64-bit signed (observational statistics accumulation)

// ============================================================
// Core domain type aliases
// ============================================================
using Bitboard = u64;           // Bitboard: each bit represents one square
using Key      = u64;           // Zobrist hash key (64-bit)
using Move     = u16;           // Move packing (16-bit: source 6 + destination 6 + flags 4)
using Square   = int;           // Square index (0-63, a1=0, h8=63)
using Score    = int;           // Score (centipawn cp units)

// ============================================================
// Board dimension constants
// ============================================================
constexpr int SQ_NB    = 64;    // Total number of squares
constexpr int FILE_NB  = 8;     // Number of files (a-h)
constexpr int RANK_NB  = 8;     // Number of ranks (1-8)
constexpr int COLOR_NB = 2;     // Number of colors (white/black)
constexpr int PIECE_NB = 6;     // Number of piece types (pawn/knight/bishop/rook/queen/king)

// ============================================================
// Color enumeration — side-to-move identifier
// ============================================================
enum Color : int {
    WHITE = 0,                  // White
    BLACK = 1,                  // Black
    COLOR_NONE = 2              // No color (for empty squares or error state)
};

// ============================================================
// PieceType enumeration — piece kind without color information
// ============================================================
enum PieceType : int {
    PAWN = 0,                   // Pawn
    KNIGHT = 1,                 // Knight
    BISHOP = 2,                 // Bishop
    ROOK = 3,                   // Rook
    QUEEN = 4,                  // Queen
    KING = 5,                   // King
    PIECE_TYPE_NB = 6,          // Total number of piece types (for array sizing)
    PIECE_TYPE_NONE = 7         // No type (empty square)
};

// ============================================================
// Piece enumeration — concrete piece with color (12 types + none)
// ============================================================
enum Piece : int {
    W_PAWN   = 0,               // White Pawn
    W_KNIGHT = 1,               // White Knight
    W_BISHOP = 2,               // White Bishop
    W_ROOK   = 3,               // White Rook
    W_QUEEN  = 4,               // White Queen
    W_KING   = 5,               // White King

    B_PAWN   = 6,               // Black Pawn
    B_KNIGHT = 7,               // Black Knight
    B_BISHOP = 8,               // Black Bishop
    B_ROOK   = 9,               // Black Rook
    B_QUEEN  = 10,              // Black Queen
    B_KING   = 11,              // Black King

    PIECE_NONE = 12             // No piece
};

// ============================================================
// Castling rights enumeration — bit flags
// ============================================================
enum CastlingRight : int {
    NO_CASTLING = 0,            // No castling rights
    WHITE_OO    = 1 << 0,       // White kingside castling
    WHITE_OOO   = 1 << 1,       // White queenside castling
    BLACK_OO    = 1 << 2,       // Black kingside castling
    BLACK_OOO   = 1 << 3,       // Black queenside castling

    WHITE_CASTLING = WHITE_OO | WHITE_OOO,   // All white castling
    BLACK_CASTLING = BLACK_OO | BLACK_OOO,   // All black castling
    ANY_CASTLING   = WHITE_CASTLING | BLACK_CASTLING  // All castling
};

constexpr Square NO_SQ = -1;    // Represents "no square" (used when en-passant does not exist)

// ============================================================
// Bit-operation helper functions (all constexpr, zero runtime overhead)
// ============================================================

// bb_of — get the bitmask for a given square (1ULL << sq)
constexpr Bitboard bb_of(Square sq) noexcept {
    return 1ULL << sq;
}

// file_of — extract the file index from a square (0=a, 7=h)
constexpr int file_of(Square sq) noexcept {
    return sq & 7;
}

// rank_of — extract the rank index from a square (0=1st rank, 7=8th rank)
constexpr int rank_of(Square sq) noexcept {
    return sq >> 3;
}

// on_board — check whether a square is within the board range (0-63)
constexpr bool on_board(Square sq) noexcept {
    return sq >= 0 && sq < 64;
}

// operator~ — flip color (white-->black, black-->white)
constexpr Color operator~(Color c) noexcept {
    return static_cast<Color>(static_cast<int>(c) ^ 1);
}

// make_piece — combine color and piece type into a full piece
constexpr Piece make_piece(Color c, PieceType pt) noexcept {
    return static_cast<Piece>(static_cast<int>(pt) + (c == WHITE ? 0 : 6));
}

// color_of — extract the color from a piece
constexpr Color color_of(Piece pc) noexcept {
    return pc == PIECE_NONE ? COLOR_NONE
                            : (pc < 6 ? WHITE : BLACK);
}

// type_of — extract the piece type from a piece (strips color)
constexpr PieceType type_of(Piece pc) noexcept {
    return pc == PIECE_NONE ? PIECE_TYPE_NONE
                            : static_cast<PieceType>(static_cast<int>(pc) % 6);
}

// is_ok — validate that an enum value is within legal bounds
constexpr bool is_ok(Color c) noexcept {
    return c == WHITE || c == BLACK;
}

constexpr bool is_ok(PieceType pt) noexcept {
    return pt >= PAWN && pt <= KING;
}

constexpr bool is_ok(Piece pc) noexcept {
    return pc >= W_PAWN && pc <= B_KING;
}

constexpr bool is_ok(Square sq) noexcept {
    return on_board(sq);
}

} // namespace magnus
