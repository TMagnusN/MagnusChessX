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

#include <charconv>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <streambuf>

#include "board/MoveGen.h"
#include "board/Position.h"
#include "Search.h"

namespace magnus {

[[nodiscard]] inline bool parse_int(std::string_view sv, int& value) noexcept {
    const char* first = sv.data();
    const char* last = sv.data() + sv.size();
    const auto [ptr, ec] = std::from_chars(first, last, value);
    return ec == std::errc{} && ptr == last;
}

[[nodiscard]] inline bool parse_square(std::string_view sv, Square& sq) noexcept {
    if (sv.size() != 2)
        return false;

    const char file = sv[0];
    const char rank = sv[1];
    if (file < 'a' || file > 'h' || rank < '1' || rank > '8')
        return false;

    sq = static_cast<Square>((rank - '1') * 8 + (file - 'a'));
    return true;
}

[[nodiscard]] inline bool parse_piece_char(
    char c,
    Color& color,
    PieceType& piece_type
) noexcept {
    color = std::isupper(static_cast<unsigned char>(c)) ? WHITE : BLACK;

    switch (static_cast<char>(std::tolower(static_cast<unsigned char>(c)))) {
        case 'p': piece_type = PAWN; return true;
        case 'n': piece_type = KNIGHT; return true;
        case 'b': piece_type = BISHOP; return true;
        case 'r': piece_type = ROOK; return true;
        case 'q': piece_type = QUEEN; return true;
        case 'k': piece_type = KING; return true;
        default: return false;
    }
}

[[nodiscard]] inline bool move_matches_uci(Move move, std::string_view token) noexcept {
    if (token.size() != 4 && token.size() != 5)
        return false;

    if (token[0] != static_cast<char>('a' + file_of(from_sq(move))) ||
        token[1] != static_cast<char>('1' + rank_of(from_sq(move))) ||
        token[2] != static_cast<char>('a' + file_of(to_sq(move))) ||
        token[3] != static_cast<char>('1' + rank_of(to_sq(move)))) {
        return false;
    }

    if (!move_is_promotion(move))
        return token.size() == 4;

    if (token.size() != 5)
        return false;

    switch (promo_piece(move)) {
        case KNIGHT: return token[4] == 'n';
        case BISHOP: return token[4] == 'b';
        case ROOK:   return token[4] == 'r';
        case QUEEN:  return token[4] == 'q';
        default:     return false;
    }
}

[[nodiscard]] inline bool find_uci_move(
    const Position& pos,
    const memory::Memory& mem,
    std::string_view token,
    Move& move
) noexcept {
    MoveList list{};
    generate_legal(pos, mem, list);

    for (int i = 0; i < list.size; ++i) {
        if (move_matches_uci(list.moves[i], token)) {
            move = list.moves[i];
            return true;
        }
    }

    return false;
}

[[nodiscard]] inline bool legal_move_exists(
    const Position& pos,
    const memory::Memory& mem,
    Move move
) noexcept {
    if (move_is_none(move))
        return false;

    MoveList list{};
    generate_legal(pos, mem, list);

    for (int i = 0; i < list.size; ++i) {
        if (list.moves[i] == move)
            return true;
    }

    return false;
}

inline void set_start_position(Position& pos) noexcept {
    position_clear(pos);

    pos.side_to_move = WHITE;
    pos.ep_sq = NO_SQ;
    pos.castling_rights = ANY_CASTLING;
    pos.halfmove_clock = 0;
    pos.fullmove_number = 1;

    for (int sq = 8; sq < 16; ++sq)
        position_put_piece(pos, WHITE, PAWN, sq);
    for (int sq = 48; sq < 56; ++sq)
        position_put_piece(pos, BLACK, PAWN, sq);

    position_put_piece(pos, WHITE, ROOK, 0);
    position_put_piece(pos, WHITE, KNIGHT, 1);
    position_put_piece(pos, WHITE, BISHOP, 2);
    position_put_piece(pos, WHITE, QUEEN, 3);
    position_put_piece(pos, WHITE, KING, 4);
    position_put_piece(pos, WHITE, BISHOP, 5);
    position_put_piece(pos, WHITE, KNIGHT, 6);
    position_put_piece(pos, WHITE, ROOK, 7);

    position_put_piece(pos, BLACK, ROOK, 56);
    position_put_piece(pos, BLACK, KNIGHT, 57);
    position_put_piece(pos, BLACK, BISHOP, 58);
    position_put_piece(pos, BLACK, QUEEN, 59);
    position_put_piece(pos, BLACK, KING, 60);
    position_put_piece(pos, BLACK, BISHOP, 61);
    position_put_piece(pos, BLACK, KNIGHT, 62);
    position_put_piece(pos, BLACK, ROOK, 63);
}

class PvTrackingStreamBuf final : public std::streambuf {
public:
    explicit PvTrackingStreamBuf(std::streambuf* sink) noexcept
        : sink_(sink) {}

    ~PvTrackingStreamBuf() override {
        flush_pending_line();
    }

    [[nodiscard]] std::string_view last_pv() const noexcept {
        return last_pv_;
    }

protected:
    int overflow(int ch) override {
        if (ch == traits_type::eof())
            return sync() == 0 ? traits_type::not_eof(ch) : traits_type::eof();

        const char c = static_cast<char>(ch);
        append_char(c);
        if (sink_ == nullptr)
            return ch;

        const int result = sink_->sputc(c);
        if (c == '\n')
            sink_->pubsync();
        return result;
    }

    std::streamsize xsputn(
        const char* s,
        std::streamsize count
    ) override {
        bool saw_newline = false;
        for (std::streamsize i = 0; i < count; ++i)
        {
            append_char(s[i]);
            saw_newline = saw_newline || s[i] == '\n';
        }

        if (sink_ == nullptr)
            return count;

        const std::streamsize written = sink_->sputn(s, count);
        if (saw_newline)
            sink_->pubsync();
        return written;
    }

    int sync() override {
        flush_pending_line();
        return sink_ != nullptr ? sink_->pubsync() : 0;
    }

private:
    void append_char(char c) {
        if (c == '\n') {
            flush_pending_line();
            return;
        }

        if (c != '\r')
            line_buffer_.push_back(c);
    }

    void flush_pending_line() {
        if (line_buffer_.empty())
            return;

        process_line();
        line_buffer_.clear();
    }

    void process_line() {
        const std::string_view line{line_buffer_};
        if (line.rfind("info ", 0) != 0)
            return;

        constexpr std::string_view pv_marker = " pv ";
        const std::size_t pv_pos = line.find(pv_marker);
        if (pv_pos == std::string_view::npos)
            return;

        const std::size_t pv_begin = pv_pos + pv_marker.size();
        if (pv_begin >= line.size())
            return;

        last_pv_.assign(line.substr(pv_begin));
    }

    std::streambuf* sink_ = nullptr;
    std::string line_buffer_{};
    std::string last_pv_{};
};

[[nodiscard]] inline std::string ponder_move_from_last_pv(
    const Position& root,
    const memory::Memory& mem,
    Move best_move,
    std::string_view last_pv
) noexcept {
    if (move_is_none(best_move) || last_pv.empty())
        return {};

    std::istringstream pv_stream{std::string(last_pv)};
    std::string best_token;
    std::string ponder_token;
    if (!(pv_stream >> best_token >> ponder_token))
        return {};

    Move pv_best_move = 0;
    if (!find_uci_move(root, mem, best_token, pv_best_move) ||
        pv_best_move != best_move) {
        return {};
    }

    Position after_best{};
    position_copy_without_accumulators(after_best, root);
    do_move_copy(after_best, best_move, mem.tables);

    Move ponder_move = 0;
    if (!find_uci_move(after_best, mem, ponder_token, ponder_move))
        return {};

    return search::move_to_uci(ponder_move);
}

[[nodiscard]] inline std::string ponder_move_from_search_result(
    const Position& root,
    memory::Memory& mem,
    const search::SearchResult& result
) noexcept {
    const Move best_move = result.best_move;
    if (move_is_none(best_move))
        return {};

    Position after_best{};
    position_copy_without_accumulators(after_best, root);
    do_move_copy(after_best, best_move, mem.tables);

    if (result.pv_length > 1 &&
        result.pv[0] == best_move &&
        legal_move_exists(after_best, mem, result.pv[1])) {
        return search::move_to_uci(result.pv[1]);
    }

    const memory::TTProbe probe = memory::tt_probe(mem.tt, after_best.key);
    const Move tt_move = probe.hit ? static_cast<Move>(probe.data.move) : Move(0);
    if (legal_move_exists(after_best, mem, tt_move))
        return search::move_to_uci(tt_move);

    return {};
}

[[nodiscard]] inline bool parse_fen(
    Position& pos,
    const memory::Memory& mem,
    std::string_view fen
) noexcept {
    std::istringstream iss{std::string(fen)};

    std::string board_part;
    std::string stm_part;
    std::string castling_part;
    std::string ep_part;
    std::string halfmove_part = "0";
    std::string fullmove_part = "1";

    if (!(iss >> board_part >> stm_part >> castling_part >> ep_part))
        return false;

    iss >> halfmove_part >> fullmove_part;

    position_clear(pos);

    int rank = 7;
    int file = 0;

    for (char c : board_part) {
        if (c == '/') {
            if (file != 8 || rank == 0)
                return false;

            --rank;
            file = 0;
            continue;
        }

        if (c >= '1' && c <= '8') {
            file += c - '0';
            if (file > 8)
                return false;
            continue;
        }

        Color color = WHITE;
        PieceType piece_type = PAWN;
        if (!parse_piece_char(c, color, piece_type) || file >= 8)
            return false;

        position_put_piece(pos, color, piece_type, rank * 8 + file);
        ++file;
    }

    if (rank != 0 || file != 8)
        return false;

    if (stm_part == "w") pos.side_to_move = WHITE;
    else if (stm_part == "b") pos.side_to_move = BLACK;
    else return false;

    pos.castling_rights = NO_CASTLING;
    if (castling_part != "-") {
        for (char c : castling_part) {
            switch (c) {
                case 'K': pos.castling_rights |= WHITE_OO; break;
                case 'Q': pos.castling_rights |= WHITE_OOO; break;
                case 'k': pos.castling_rights |= BLACK_OO; break;
                case 'q': pos.castling_rights |= BLACK_OOO; break;
                default: return false;
            }
        }
    }

    pos.ep_sq = NO_SQ;
    if (ep_part != "-" && !parse_square(ep_part, pos.ep_sq))
        return false;

    if (!parse_int(halfmove_part, pos.halfmove_clock) || pos.halfmove_clock < 0)
        return false;

    if (!parse_int(fullmove_part, pos.fullmove_number) || pos.fullmove_number <= 0)
        return false;

    position_refresh_key(pos, mem.tables);
    return position_has_valid_kings(pos) && position_board_matches_bitboards(pos);
}

} // namespace magnus
