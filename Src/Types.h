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
 * 全域型別定義 — MagnusChessX Thinking 的基礎型別系統
 *
 * 所有模組（著法生成、搜尋、評估、雜湊）共享此處定義的
 * 純量別名、列舉型別、以及位元運算輔助函數。
 *
 * 設計原則：
 *   - 使用固定寬度整數型別（u64/i16 等）確保跨平台一致性
 *   - Move 封裝為 16 位元（u16），包含起點/終點/升變資訊
 *   - Square 使用 int（0-63），方便陣列索引
 *   - 所有輔助函數為 constexpr，支援編譯期計算
 */
namespace magnus {

// ============================================================
// 固定寬度整數別名 — 確保跨平台位元寬度一致
// ============================================================
using u8  = std::uint8_t;       //  8 位元無符號（用於 TT age/flags）
using u16 = std::uint16_t;      // 16 位元無符號（Move 封裝格式）
using u32 = std::uint32_t;      // 32 位元無符號（TT tag32、NNUE generation）
using u64 = std::uint64_t;      // 64 位元無符號（Bitboard、Key、節點計數）
using i8  = std::int8_t;        //  8 位元有符號
using i16 = std::int16_t;       // 16 位元有符號（TT score/eval/depth、NNUE 權重）
using i32 = std::int32_t;       // 32 位元有符號（著法排序分數、LMR 計算）
using i64 = std::int64_t;       // 64 位元有符號（觀測統計累加）

// ============================================================
// 核心領域型別別名
// ============================================================
using Bitboard = u64;           // 位元棋盤：每個位元代表一個方格
using Key      = u64;           // Zobrist 雜湊鍵（64 位元）
using Move     = u16;           // 著法封裝（16 位元：起點 6 + 終點 6 + 旗標 4）
using Square   = int;           // 方格索引（0-63，a1=0, h8=63）
using Score    = int;           // 評分（厘兵 cp 單位）

// ============================================================
// 棋盤維度常數
// ============================================================
constexpr int SQ_NB    = 64;    // 總方格數
constexpr int FILE_NB  = 8;     // 列數（a-h）
constexpr int RANK_NB  = 8;     // 行數（1-8）
constexpr int COLOR_NB = 2;     // 顏色數（白/黑）
constexpr int PIECE_NB = 6;     // 棋子類型數（兵/馬/象/車/后/王）

// ============================================================
// 顏色列舉 — 走子方標識
// ============================================================
enum Color : int {
    WHITE = 0,                  // 白方
    BLACK = 1,                  // 黑方
    COLOR_NONE = 2              // 無顏色（用於空方格或錯誤狀態）
};

// ============================================================
// 棋子類型列舉 — 不含顏色資訊的棋子種類
// ============================================================
enum PieceType : int {
    PAWN = 0,                   // 兵
    KNIGHT = 1,                 // 馬
    BISHOP = 2,                 // 象
    ROOK = 3,                   // 車
    QUEEN = 4,                  // 后
    KING = 5,                   // 王
    PIECE_TYPE_NB = 6,          // 棋子類型總數（用於陣列大小）
    PIECE_TYPE_NONE = 7         // 無類型（空方格）
};

// ============================================================
// 棋子列舉 — 含顏色資訊的具體棋子（12 種 + 空）
// ============================================================
enum Piece : int {
    W_PAWN   = 0,               // 白兵
    W_KNIGHT = 1,               // 白馬
    W_BISHOP = 2,               // 白象
    W_ROOK   = 3,               // 白車
    W_QUEEN  = 4,               // 白后
    W_KING   = 5,               // 白王

    B_PAWN   = 6,               // 黑兵
    B_KNIGHT = 7,               // 黑馬
    B_BISHOP = 8,               // 黑象
    B_ROOK   = 9,               // 黑車
    B_QUEEN  = 10,              // 黑后
    B_KING   = 11,              // 黑王

    PIECE_NONE = 12             // 無棋子
};

// ============================================================
// 王車易位權限列舉 — 位元旗標
// ============================================================
enum CastlingRight : int {
    NO_CASTLING = 0,            // 無易位權限
    WHITE_OO    = 1 << 0,       // 白方短易位（Kingside）
    WHITE_OOO   = 1 << 1,       // 白方長易位（Queenside）
    BLACK_OO    = 1 << 2,       // 黑方短易位
    BLACK_OOO   = 1 << 3,       // 黑方長易位

    WHITE_CASTLING = WHITE_OO | WHITE_OOO,   // 白方所有易位
    BLACK_CASTLING = BLACK_OO | BLACK_OOO,   // 黑方所有易位
    ANY_CASTLING   = WHITE_CASTLING | BLACK_CASTLING  // 所有易位
};

constexpr Square NO_SQ = -1;    // 表示「無方格」（用於 en-passant 不存在時）

// ============================================================
// 位元運算輔助函數（全部 constexpr，零運行時開銷）
// ============================================================

// bb_of — 取得指定方格的位元遮罩（1ULL << sq）
constexpr Bitboard bb_of(Square sq) noexcept {
    return 1ULL << sq;
}

// file_of — 從方格索引提取列號（0=a, 7=h）
constexpr int file_of(Square sq) noexcept {
    return sq & 7;
}

// rank_of — 從方格索引提取行號（0=1st rank, 7=8th rank）
constexpr int rank_of(Square sq) noexcept {
    return sq >> 3;
}

// on_board — 檢查方格是否在棋盤範圍內（0-63）
constexpr bool on_board(Square sq) noexcept {
    return sq >= 0 && sq < 64;
}

// operator~ — 顏色反轉（白→黑，黑→白）
constexpr Color operator~(Color c) noexcept {
    return static_cast<Color>(static_cast<int>(c) ^ 1);
}

// make_piece — 從顏色和棋子類型組合出完整的棋子
constexpr Piece make_piece(Color c, PieceType pt) noexcept {
    return static_cast<Piece>(static_cast<int>(pt) + (c == WHITE ? 0 : 6));
}

// color_of — 從棋子提取顏色
constexpr Color color_of(Piece pc) noexcept {
    return pc == PIECE_NONE ? COLOR_NONE
                            : (pc < 6 ? WHITE : BLACK);
}

// type_of — 從棋子提取棋子類型（去除顏色資訊）
constexpr PieceType type_of(Piece pc) noexcept {
    return pc == PIECE_NONE ? PIECE_TYPE_NONE
                            : static_cast<PieceType>(static_cast<int>(pc) % 6);
}

// is_ok — 驗證列舉值是否在合法範圍內
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
