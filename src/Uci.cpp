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

#include "Uci.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "board/Attack.h"
#include "board/MoveGen.h"

#include "Common.h"
#include "Memory.h"
#include "mnue/Mnue.h"
#include "mnue/MnueX2K16PawnNetwork.h"
#include "Search.h"
#include "syzygy/Syzygy.h"
#include "Time.h"

#ifdef _WIN32
#include <windows.h>
#endif

/*
 * MagnusChessX Thinking UCI å‰ç«¯ â€” Universal Chess Interface å¯¦ä½œ
 *
 * å®Œæ•´çš„ UCI å”å®šå¯¦ä½œï¼ŒåŒ…å«ä»¥ä¸‹åŠŸèƒ½æ¨¡çµ„ï¼š
 *
 * 1. UCI å‘½ä»¤è§£æžèˆ‡æ´¾ç™¼
 *    - uci / isready / ucinewgame / quit â€” æ¨™æº– UCI æ¡æ‰‹
 *    - position [startpos|fen ...] [moves ...] â€” å±€é¢è¨­å®š
 *    - go [depth|movetime|wtime|btime|...] â€” æœå°‹æŽ§åˆ¶
 *    - stop / ponderhit â€” éžåŒæ­¥æœå°‹æŽ§åˆ¶
 *    - setoption name [Hash|Threads|MNUEfile|...] value â€” å¼•æ“Žé¸é …
 *
 * 2. FEN è§£æž (parse_fen)
 *    - é©—è­‰æ£‹ç›¤çµæ§‹ã€èµ°å­æ–¹ã€æ˜“ä½æ¬Šã€éŽè·¯å…µ
 *    - é€šéŽæ¨™æº– piece-placement API å»ºæ§‹å±€é¢
 *
 * 3. æœå°‹ç·šç¨‹ç®¡ç† (UciSession)
 *    - åˆä½œå¼åœæ­¢ (stop_requested åŽŸå­è®Šé‡)
 *    - æ²‰æ€æ¨¡å¼æ”¯æ´ (ponder / ponderhit æ™‚é–“è¿½è¹¤)
 *    - MNUE P2/P2Pro evaluator è¼‰å…¥
 *
 * 4. PvTrackingStreamBuf â€” è‡ªè¨‚ streambuf
 *    - æ””æˆªæœå°‹è¼¸å‡ºçš„ "info ... pv ..." è¡Œ
 *    - æå– PV ç”¨æ–¼æ²‰æ€è‘—æ³•è¨ˆç®—
 */

namespace magnus {

namespace {

constexpr int DEFAULT_UCI_HASH_MB = 16;
constexpr int DEFAULT_UCI_THREADS = 1;
constexpr int MAX_UCI_THREADS = 512;
constexpr int DEFAULT_UCI_MULTIPV = 1;
constexpr int MAX_UCI_MULTIPV = 256;
constexpr int DEFAULT_UCI_CONTEMPT = 0;
constexpr int MIN_UCI_CONTEMPT = -10000;
constexpr int MAX_UCI_CONTEMPT = 10000;
constexpr int DEFAULT_BENCH_DEPTH = 12;
constexpr int MAX_SEARCH_THREADS = 512;

constexpr std::uint32_t X2K16_MNUE_MAGIC = 0x45554E4D;

[[nodiscard]] std::uint32_t read_u32_le(
    const std::array<unsigned char, 16>& bytes,
    int field
) noexcept {
    const int offset = field * 4;
    return static_cast<std::uint32_t>(bytes[static_cast<std::size_t>(offset)])
        | (static_cast<std::uint32_t>(bytes[static_cast<std::size_t>(offset + 1)]) << 8)
        | (static_cast<std::uint32_t>(bytes[static_cast<std::size_t>(offset + 2)]) << 16)
        | (static_cast<std::uint32_t>(bytes[static_cast<std::size_t>(offset + 3)]) << 24);
}

[[nodiscard]] bool looks_like_x2k16_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;

    std::array<unsigned char, 16> header{};
    in.read(
        reinterpret_cast<char*>(header.data()),
        static_cast<std::streamsize>(header.size())
    );
    if (in.gcount() < 12)
        return false;

    return read_u32_le(header, 0) == X2K16_MNUE_MAGIC
        && read_u32_le(header, 2)
            == static_cast<std::uint32_t>(mnue::x2k16::Layout::ArchId);
}

[[nodiscard]] const char* search_eval_kind_name(
    search::SearchEvalKind kind
) noexcept {
    switch (kind) {
        case search::SearchEvalKind::P2:
            return "p2";
        case search::SearchEvalKind::X2K16:
            return "x2-k16-pawn-q8-a384";
        case search::SearchEvalKind::None:
            return "none";
    }
    return "none";
}

constexpr std::array<std::string_view, 50> SEARCH_BENCH_FENS{{
    "rnb1k2r/pp2bp1p/2p1pp2/q7/8/1P6/PBPPQPPP/2KR1BNR w kq - 4 9",
    "1r1qk1nr/pppn1ppp/3p4/3Pp1b1/2P5/2N2Q1P/PP2PPP1/R1B1KB1R w KQk - 3 9",
    "r3kbnr/pp3ppp/2n5/2P1pq2/N1Pp2b1/5N2/PP1BPPPP/R2QKB1R w KQkq - 0 9",
    "rnb1k2r/pppp2pp/8/8/2P2Bn1/2q4N/P3PPPP/R2QKB1R w KQkq - 0 9",
    "r2qk2r/ppp1bppp/2n1bn2/8/2NPp3/2P5/PP2BPPP/RNBQ1RK1 w kq - 1 9",
    "r1b1kb1r/1pqnppp1/p2p1n1p/8/3NP1PP/P1N5/1PP2P2/R1BQKB1R w KQkq - 1 9",
    "rnb1kb1r/pp3pp1/1qpnp2p/3p4/3PP2B/P1N2P1N/1PP3PP/R2QKB1R w KQkq - 0 9",
    "r1bqr1k1/pppp1ppp/2n5/3np1N1/4P3/2P5/PPP2PPP/R1BQ1RK1 w - - 0 9",
    "r1bqkb1r/3n1ppp/p1p1pn2/1p6/8/5NP1/PPQPPPBP/RNB2RK1 w kq - 0 9",
    "r2qk2r/ppp1bppp/2np2n1/3N4/2BPPpb1/5N2/PPP3PP/R1BQ1RK1 w kq - 4 9",
    "r1bqkbnr/3n1ppp/p3p3/2p5/Pp1P4/4PN2/1P2BPPP/RNBQ1RK1 w kq - 0 9",
    "r1bqk2r/ppp1bppp/2n1p3/3pP3/2PP1B2/2PQ4/P4PPP/R3KBNR w KQkq d6 0 9",
    "r1bqk1nr/1ppn1pb1/p2p2pp/4p3/P2PP3/2NB1N2/1PP2PPP/R1BQ1RK1 w kq - 0 9",
    "r1bq1rk1/pp1pppbp/5np1/4n3/2PN4/1PN3P1/P3PPBP/R1BQK2R w KQ - 1 9",
    "rn1qkb1r/1p2npp1/4p2p/p2pPb2/3P4/P1N5/1P2NPPP/R1BQKB1R w KQkq - 2 9",
    "r1bqk2r/1p1nbpp1/p2p1n1p/2pPp3/2P5/P1N1PN1P/1P3PP1/R1BQKB1R w KQkq - 1 9",
    "r1bqk2r/2p1bpp1/p1np1n2/1p2p2p/3PP3/2N1BP2/PPPQN1PP/2KR1B1R w kq - 0 9",
    "r1bqk2r/p2nppbp/2pp2p1/1p2Pn2/3P1P2/2N1BN2/PPPQ2PP/R3KB1R w KQkq - 3 9",
    "r1b1k2r/pp1n1ppp/2p1pn2/q2p4/1bPP4/1PN2NP1/P2BPPBP/R2QK2R w KQkq - 3 9",
    "r1bq1bnr/p1p4p/1pk2p2/3pp1pQ/3P4/4P1B1/PPP2PPP/RN2K1NR w KQ - 0 9",
    "rn1qkb1r/1bp2ppp/p3pn2/8/Pp1PP3/1B3P2/1P2N1PP/RNBQK2R w KQkq - 0 9",
    "rnbq1rk1/1p2ppbp/2p3p1/p2n4/3P4/2NB1N1P/PPP2PP1/R1BQ1RK1 w - - 0 9",
    "rn1qkb1r/1p3p2/p1p1pn1p/2Pp1bp1/3P1B2/2N1PN2/PP2BPPP/R2QK2R w KQkq - 0 9",
    "r1bqk2r/pp1n1pp1/3p1n1p/2pPp3/1bP1P3/2N1BP2/PP4PP/R2QKBNR w KQkq - 2 9",
    "r1bqkb1r/1p2np1p/p1npp1p1/8/3NPP2/2NBB3/PPP3PP/R2QK2R w KQkq - 0 9",
    "rnbq1rk1/p1p2pp1/1p3p1p/8/1bBP4/2N1P3/PP2NPPP/R2QK2R w KQ - 0 9",
    "r1bqkb1r/pp4pp/2p1p3/3pnp1n/2PP4/2NBPN2/PP3PPP/R2QK2R w KQkq - 0 9",
    "rn1q1rk1/p1ppbppp/b3pn2/1p6/2PP4/1P3NP1/P2BPPBP/RN1Q1RK1 w - - 0 9",
    "rnbq1rk1/p3npbp/1pp1p1p1/3p4/2PP1B2/2NBPN2/PP3PPP/2RQK2R w K - 0 9",
    "rnbqk2r/1pp2pbp/p2p2p1/3pP3/3P1P2/3B1N2/PPP3PP/R1BQK2R w KQkq - 0 9",
    "rn1q1rk1/pbp1ppbp/1p3np1/3p4/3PPP2/2N1BB1P/PPP3P1/R2QK1NR w KQ - 0 9",
    "rnbqk2r/pp2p1bp/2pn1pp1/3pN3/3P4/2P3P1/PP1NPPBP/R1BQ1RK1 w kq - 0 9",
    "r1bq1rk1/p1pp1ppp/1pn5/4P3/2P1n3/P3PN2/1P1B1PPP/R2QKB1R w KQ - 0 9",
    "r1bqkbnr/ppp2p2/2npp3/8/2PP1P1p/3NP1pP/PP4P1/RNBQKB1R w KQkq - 0 9",
    "r2qk2r/ppp3pp/2n1b3/3n4/1b6/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 9",
    "r1bqk2r/pppnn1b1/3pp1pp/5p1P/4PP2/2PP1N2/PP2B1P1/RNBQK2R w KQkq - 1 9",
    "r1b1kb1r/1p1p1ppp/p1q1pn2/8/4P3/1P1B4/P1P2PPP/RNBQ1RK1 w kq - 0 9",
    "r1bqk1nr/pp1p1ppp/1b6/8/1n2P3/1N1B4/PP3PPP/RNBQK2R w KQkq - 5 9",
    "r2q1rk1/ppp1ppb1/2np1np1/6Bp/3PP1bP/2PQ1N2/PP1N1PP1/R3KB1R w KQ - 5 9",
    "rn1qkb1r/1bpp2pp/p3p3/3n1p2/Pp1P4/4PNB1/1PPNBPPP/R2QK2R w KQkq - 0 9",
    "r1b1kbnr/1pq2pp1/p1np3p/4p3/2B1P3/5N2/PPP2PPP/RNBQR1K1 w kq - 2 9",
    "r1b1kb1r/pp3ppp/1q2pn2/2pP4/2pn4/2N2NP1/PP2PPBP/R1BQ1RK1 w kq - 0 9",
    "r3kb1r/ppp1p2p/2np1np1/5q2/3P4/5N2/PPP2PPP/RNBQ1RK1 w kq - 0 9",
    "r1bqnrk1/pp1nbppp/3pp3/2p3B1/3PP3/2PB1N1P/PP3PP1/RN1Q1RK1 w - - 1 9",
    "r1bqr1k1/pp1nbppp/4pn2/2pp4/3P1B1P/2PBPN2/PP1N1PP1/R2QK2R w KQ - 1 9",
    "r1b1kb1r/1pq2ppp/p1np1n2/2p1p3/P3P3/2N2NP1/1PPP1PBP/R1BQR1K1 w kq - 0 9",
    "rnbq1rk1/1p2ppb1/2pp1npp/p7/P2PP3/2N1BN2/1PP1BPPP/R2Q1RK1 w - - 0 9",
    "r1bqk2r/pp1nppbp/2np4/6B1/2P5/2NQPN2/PP3PPP/R3KB1R w KQkq - 1 9",
    "rnbq1rk1/1p2ppb1/p1p2n1p/3p2p1/2PP4/2N1PNBP/PP3PP1/R2QKB1R w KQ - 1 9",
    "rn2k2r/ppq2p1p/2ppbp2/2b1p3/2B1P2N/3P4/PPP2PPP/RN1Q1RK1 w kq - 2 9",
}};

[[nodiscard]] const std::string& get_executable_dir() {
    static const std::string dir = []() -> std::string {
#ifdef _WIN32
        char buf[MAX_PATH];
        const DWORD len = GetModuleFileNameA(nullptr, buf, sizeof(buf));
        if (len > 0 && len < sizeof(buf))
            return std::filesystem::path(buf).parent_path().string();
#endif
        return std::filesystem::current_path().string();
    }();
    return dir;
}

[[nodiscard]] std::string resolve_file_path(const std::string& filename) {
    std::error_code ec;

    // 1. Absolute path or CWD-relative â€” if it already exists, use as-is.
    if (std::filesystem::exists(filename, ec) && !ec)
        return filename;

    // 2. Relative to executable directory.
    const std::filesystem::path exe_dir = get_executable_dir();
    const std::filesystem::path exe_parent = exe_dir.parent_path();
    const std::filesystem::path search_dirs[] = {
        exe_dir,
        exe_dir / "build",
        exe_parent,
        exe_parent.parent_path(),
        exe_parent / "build",
        exe_parent.parent_path() / "build"
    };

    for (const std::filesystem::path& dir : search_dirs) {
        const std::filesystem::path candidate =
            (dir / filename).lexically_normal();
        if (std::filesystem::exists(candidate, ec) && !ec)
            return candidate.string();
    }

    // 3. Give up â€” fall back to the original filename so the error message
    //    still shows what was requested.
    return filename;
}

struct PositionHistory {
    Key keys[search::MAX_GAME_HISTORY]{};
    int count = 0;
    int head = 0; // circular-buffer write cursor
};

inline void clear_position_history(PositionHistory& history) noexcept {
    history.count = 0;
    history.head = 0;
}

inline void push_position_history(
    PositionHistory& history,
    Key key
) noexcept {
    if (history.count < search::MAX_GAME_HISTORY) {
        history.keys[history.count] = key;
        ++history.count;
        return;
    }
    // Circular overwrite: O(1) instead of O(n) shift.
    history.keys[history.head] = key;
    history.head = (history.head + 1) % search::MAX_GAME_HISTORY;
}

[[nodiscard]] constexpr const char* go_usage_hint() noexcept {
    return "go <depth/movetime/nodes/ponder>";
}

[[nodiscard]] constexpr const char* go_usage_examples() noexcept {
    return "examples: go depth 8 | go movetime 1000 | go nodes 50000 | go wtime 15000 btime 15000 | go ponder wtime 15000 btime 15000";
}

[[nodiscard]] constexpr const char* bench_usage_hint() noexcept {
    return "bench";
}

[[nodiscard]] char display_piece_char(Piece pc) noexcept {
    switch (pc) {
        case W_PAWN:   return 'P';
        case W_KNIGHT: return 'N';
        case W_BISHOP: return 'B';
        case W_ROOK:   return 'R';
        case W_QUEEN:  return 'Q';
        case W_KING:   return 'K';
        case B_PAWN:   return 'p';
        case B_KNIGHT: return 'n';
        case B_BISHOP: return 'b';
        case B_ROOK:   return 'r';
        case B_QUEEN:  return 'q';
        case B_KING:   return 'k';
        default:       return '.';
    }
}

[[nodiscard]] std::string display_square(Square sq) {
    if (!is_ok(sq))
        return "-";

    std::string result;
    result.push_back(static_cast<char>('a' + file_of(sq)));
    result.push_back(static_cast<char>('1' + rank_of(sq)));
    return result;
}

[[nodiscard]] std::string display_castling_rights(int rights) {
    std::string result;
    if ((rights & WHITE_OO) != 0)
        result.push_back('K');
    if ((rights & WHITE_OOO) != 0)
        result.push_back('Q');
    if ((rights & BLACK_OO) != 0)
        result.push_back('k');
    if ((rights & BLACK_OOO) != 0)
        result.push_back('q');
    return result.empty() ? "-" : result;
}

[[nodiscard]] std::string display_fen(const Position& pos) {
    std::string fen;

    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const Square sq = rank * 8 + file;
            const Piece pc = piece_on(pos, sq);
            if (pc == PIECE_NONE) {
                ++empty;
                continue;
            }

            if (empty != 0) {
                fen.push_back(static_cast<char>('0' + empty));
                empty = 0;
            }
            fen.push_back(display_piece_char(pc));
        }

        if (empty != 0)
            fen.push_back(static_cast<char>('0' + empty));
        if (rank != 0)
            fen.push_back('/');
    }

    fen += pos.side_to_move == WHITE ? " w " : " b ";
    fen += display_castling_rights(pos.castling_rights);
    fen.push_back(' ');
    fen += display_square(pos.ep_sq);
    fen.push_back(' ');
    fen += std::to_string(pos.halfmove_clock);
    fen.push_back(' ');
    fen += std::to_string(pos.fullmove_number);
    return fen;
}

[[nodiscard]] std::string display_key_hex(Key key) {
    std::ostringstream oss;
    oss << "0x"
        << std::uppercase
        << std::hex
        << std::setw(16)
        << std::setfill('0')
        << key;
    return oss.str();
}

void display_position_snapshot(
    const Position& pos,
    std::ostream& out,
    std::string_view changed = {}
) {
    if (!changed.empty())
        out << "info string changed -> " << changed << '\n';
    out << "info string pos -> " << display_fen(pos) << '\n';
    out << "info string hash -> " << pos.key << " (" << display_key_hex(pos.key) << ")\n";
}

void display_position(const Position& pos, std::ostream& out) {
    out << '\n';
    for (int rank = 7; rank >= 0; --rank) {
        out << "  +---+---+---+---+---+---+---+---+\n";
        out << (rank + 1) << " |";
        for (int file = 0; file < 8; ++file) {
            const Square sq = rank * 8 + file;
            out << ' ' << display_piece_char(piece_on(pos, sq)) << " |";
        }
        out << '\n';
    }

    out << "  +---+---+---+---+---+---+---+---+\n";
    out << "    a   b   c   d   e   f   g   h\n";
    out << "Fen: " << display_fen(pos) << '\n';
}

[[nodiscard]] std::string_view trim_ascii(std::string_view text) noexcept {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t'))
        text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' ||
                             text.back() == '\r' || text.back() == '\n'))
        text.remove_suffix(1);
    return text;
}

[[nodiscard]] std::string_view arrow_arguments(std::string_view args) noexcept {
    args = trim_ascii(args);
    if (args.size() >= 2 && args[0] == '-' && args[1] == '>')
        args.remove_prefix(2);
    return trim_ascii(args);
}

[[nodiscard]] bool command_starts_with(
    std::string_view line,
    std::string_view command
) noexcept {
    return line == command ||
           (line.size() > command.size() &&
            line.substr(0, command.size()) == command &&
            line[command.size()] == ' ');
}

[[nodiscard]] std::string_view command_arguments(
    std::string_view line,
    std::string_view command
) noexcept {
    return line.size() > command.size() ? line.substr(command.size() + 1) : std::string_view{};
}

[[nodiscard]] inline long long steady_now_ms() noexcept {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

[[nodiscard]] bool parse_u64(std::string_view sv, u64& value) noexcept {
    const char* first = sv.data();
    const char* last = sv.data() + sv.size();
    const auto [ptr, ec] = std::from_chars(first, last, value);
    return ec == std::errc{} && ptr == last;
}

[[nodiscard]] bool parse_bool(std::string_view sv, bool& value) noexcept {
    const auto equals_ci = [sv](std::string_view expected) noexcept {
        if (sv.size() != expected.size())
            return false;

        for (std::size_t i = 0; i < sv.size(); ++i) {
            const char lhs = static_cast<char>(
                std::tolower(static_cast<unsigned char>(sv[i]))
            );
            if (lhs != expected[i])
                return false;
        }
        return true;
    };

    if (equals_ci("true") || sv == "1") {
        value = true;
        return true;
    }

    if (equals_ci("false") || sv == "0") {
        value = false;
        return true;
    }

    return false;
}

[[nodiscard]] bool apply_move_list(
    Position& pos,
    const memory::Memory& mem,
    std::istringstream& iss,
    PositionHistory& history
) noexcept {
    std::string move_token;
    while (iss >> move_token) {
        Move move = 0;
        if (!find_uci_move(pos, mem, move_token, move))
            return false;

        const Key prev_key = pos.key;
        do_move_copy(pos, move, mem.tables);
        if (pos.halfmove_clock == 0)
            clear_position_history(history);
        else
            push_position_history(history, prev_key);
    }

    return true;
}

[[nodiscard]] bool set_position_from_command(
    Position& pos,
    const memory::Memory& mem,
    std::string_view command,
    PositionHistory& history
) noexcept {
    std::istringstream iss{std::string(command)};
    std::string token;

    if (!(iss >> token))
        return false;

    if (token == "startpos") {
        set_start_position(pos);
        position_refresh_key(pos, mem.tables);
    } else if (token == "fen") {
        std::string fen;
        std::string part;

        for (int i = 0; i < 6; ++i) {
            if (!(iss >> part))
                return false;

            if (i != 0)
                fen.push_back(' ');
            fen += part;
        }

        if (!parse_fen(pos, mem, fen))
            return false;
    } else {
        return false;
    }

    clear_position_history(history);

    if (!(iss >> token))
        return true;

    if (token != "moves")
        return false;

    return apply_move_list(pos, mem, iss, history);
}


[[nodiscard]] inline int white_pov_score(const Position& pos, int stm_score) noexcept {
    return pos.side_to_move == WHITE ? stm_score : -stm_score;
}

[[nodiscard]] inline int white_pov_winrate(const Position& pos, int stm_wr) noexcept {
    return pos.side_to_move == WHITE ? stm_wr : (1000 - stm_wr);
}

[[nodiscard]] inline mnue::WdlTriplet white_pov_wdl(
    const Position& pos,
    mnue::WdlTriplet stm_wdl
) noexcept {
    if (pos.side_to_move == WHITE)
        return stm_wdl;

    return {
        .win = stm_wdl.loss,
        .draw = stm_wdl.draw,
        .loss = stm_wdl.win
    };
}

[[nodiscard]] bool parse_go_command(
    const Position& pos,
    const memory::Memory& mem,
    timeman::TimeManager& time_manager,
    std::string_view command,
    search::SearchLimits& limits
) noexcept {
    // Convert the UCI go command into normalized parameters, then let the time
    // manager derive the final soft/hard budgets.
    std::istringstream iss{std::string(command)};
    std::string token;

    timeman::GoParams params{};
    bool has_limit = false;
    limits.root_move_count = 0;

    iss >> token; // go

    const auto parse_next_int = [&](int& value, bool require_positive = false) noexcept {
        std::string raw;
        return (iss >> raw) &&
               parse_int(raw, value) &&
               (!require_positive || value > 0);
    };

    const auto parse_next_u64 = [&](u64& value, bool require_positive = false) noexcept {
        std::string raw;
        return (iss >> raw) &&
               parse_u64(raw, value) &&
               (!require_positive || value > 0);
    };

    while (iss >> token) {
        if (token == "depth") {
            if (!parse_next_int(params.depth, true))
                return false;
            has_limit = true;
        }
        else if (token == "nodes") {
            if (!parse_next_u64(params.nodes, true))
                return false;
            has_limit = true;
        }
        else if (token == "movetime") {
            if (!parse_next_int(params.movetime, true))
                return false;
            has_limit = true;
        }
        else if (token == "wtime") {
            if (!parse_next_int(params.wtime))
                return false;
            has_limit = true;
        }
        else if (token == "btime") {
            if (!parse_next_int(params.btime))
                return false;
            has_limit = true;
        }
        else if (token == "winc") {
            if (!parse_next_int(params.winc))
                return false;
            has_limit = true;
        }
        else if (token == "binc") {
            if (!parse_next_int(params.binc))
                return false;
            has_limit = true;
        }
        else if (token == "movestogo") {
            if (!parse_next_int(params.movestogo))
                return false;
            has_limit = true;
        }
        else if (token == "ponder") {
            params.ponder = true;
        }
        else if (token == "infinite") {
            params.infinite = true;
            has_limit = true;
        }
        else if (token == "searchmoves") {
            std::string move_token;
            while (iss >> move_token) {
                Move move = 0;
                if (!find_uci_move(pos, mem, move_token, move))
                    return false;

                bool duplicate = false;
                for (int i = 0; i < limits.root_move_count; ++i) {
                    if (limits.root_moves[i] == move) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate)
                    continue;

                if (limits.root_move_count >= 256)
                    return false;
                limits.root_moves[limits.root_move_count++] = move;
            }

            if (limits.root_move_count == 0)
                return false;
            break;
        }
    }

    if (!has_limit)
        return false;

    return time_manager.build_limits(pos, params, limits);
}

struct SearchBenchResult {
    search::SearchResult search{};
    u64 time_ms = 0;
    double seconds = 0.0;
    u64 nps = 0;
    std::string ponder{};
};

[[nodiscard]] SearchBenchResult benchmark_search_position(
    const Position& pos,
    memory::Memory& mem,
    const search::SearchLimits& limits,
    std::ostream* out
) {
    using clock = std::chrono::steady_clock;

    memory::memory_clear_hash(mem);

    const auto start = clock::now();
    SearchBenchResult result;
    if (out != nullptr) {
        PvTrackingStreamBuf pv_tracking_buf(out->rdbuf());
        std::ostream tracked_out(&pv_tracking_buf);
        result.search =
            search::iterative_deepening(pos, mem, limits, &tracked_out);
        tracked_out.flush();
    } else {
        result.search = search::iterative_deepening(pos, mem, limits, nullptr);
    }

    if (limits.recover_ponder_pv)
        result.ponder = ponder_move_from_search_result(pos, mem, result.search);

    const auto end = clock::now();
    result.time_ms = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start
        ).count()
    );
    result.seconds = std::chrono::duration<double>(end - start).count();
    result.nps = result.seconds > 0.0
        ? static_cast<u64>(
            static_cast<double>(result.search.nodes) / result.seconds
        )
        : 0ULL;
    return result;
}

[[nodiscard]] bool run_compact_search_bench(
    memory::Memory& mem,
    int depth,
    std::size_t hash_mb,
    std::size_t threads,
    std::ostream& out,
    search::SearchEvalKind eval_kind = search::SearchEvalKind::P2,
    bool quiet = false
) {
    const int search_threads = std::clamp<int>(
        static_cast<int>(threads),
        1,
        MAX_SEARCH_THREADS
    );

    u64 total_nodes = 0;
    double total_seconds = 0.0;
    u64 checksum = 0;

    if (!quiet) {
        out << "--------------------------------------------------\n";
        out << "          Nodes       Elapsed             NPS\n";
        out << "--------------------------------------------------\n";
    }

    for (std::size_t i = 0; i < SEARCH_BENCH_FENS.size(); ++i) {
        Position bench_pos{};
        const std::string_view fen = SEARCH_BENCH_FENS[i];
        if (!parse_fen(bench_pos, mem, fen))
            return false;

        search::SearchLimits limits{};
        limits.depth = depth;
        limits.thread_count = search_threads;
        limits.thread_id = 0;
        limits.report_info = false;
        limits.recover_ponder_pv = false;
        limits.eval_kind = eval_kind;

        const SearchBenchResult res =
            benchmark_search_position(bench_pos, mem, limits, nullptr);

        total_nodes += res.search.nodes;
        total_seconds += res.seconds;
        checksum = (checksum * 1315423911ULL)
            ^ res.search.nodes
            ^ (static_cast<u64>(res.search.best_move) << 32)
            ^ static_cast<u64>(static_cast<unsigned>(res.search.score));

        if (!quiet) {
            out << std::setw(3) << i
                << std::setw(12) << res.search.nodes
                << std::setw(13) << std::fixed << std::setprecision(3)
                << res.seconds << "s"
                << std::setw(16) << res.nps << " N/s\n";
        }
    }

    const u64 total_nps = total_seconds > 0.0
        ? static_cast<u64>(static_cast<double>(total_nodes) / total_seconds)
        : 0ULL;

    if (!quiet) {
        out << "--------------------------------------------------\n";
        out << std::setw(15) << total_nodes
            << std::setw(13) << std::fixed << std::setprecision(3)
            << total_seconds << "s"
            << std::setw(16) << total_nps << " N/s\n";
        out << "--------------------------------------------------\n";
        out << "depth " << depth
            << " hash " << hash_mb
            << " threads " << search_threads
            << " evaluator "
            << (eval_kind == search::SearchEvalKind::X2K16
                ? search_eval_kind_name(eval_kind)
                : mnue::p2_eval_name())
            << " checksum " << checksum
            << '\n';
    }

    out << "Bench: " << total_nodes << " nodes " << total_nps << " nps\n";
    return true;
}

[[nodiscard]] u64 basic_perft(Position& pos, const memory::Memory& mem, int depth) {
    if (depth == 0)
        return 1;

    MoveList list{};
    generate_legal(pos, mem, list);
    if (depth == 1)
        return static_cast<u64>(list.size);

    u64 nodes = 0;
    for (int i = 0; i < list.size; ++i) {
        StateInfo st{};
        const Move move = list.moves[i];
        make_move(pos, move, mem.tables, st);
        nodes += basic_perft(pos, mem, depth - 1);
        unmake_move(pos, move, mem.tables, st);
    }
    return nodes;
}

[[nodiscard]] bool run_basic_perft_root(
    Position& pos,
    const memory::Memory& mem,
    int depth,
    std::ostream& out
) {
    if (depth < 0)
        return false;

    if (depth == 0) {
        out << "nodes 1\n";
        return true;
    }

    MoveList list{};
    generate_legal(pos, mem, list);

    u64 total = 0;
    for (int i = 0; i < list.size; ++i) {
        StateInfo st{};
        const Move move = list.moves[i];
        make_move(pos, move, mem.tables, st);
        const u64 nodes = basic_perft(pos, mem, depth - 1);
        unmake_move(pos, move, mem.tables, st);

        total += nodes;
        out << search::move_to_uci(move) << ' ' << nodes << '\n';
    }

    out << "nodes " << total << '\n';
    return true;
}

[[nodiscard]] bool ensure_cli_mnue_loaded(std::ostream& out) {
    if (mnue::p2_loaded())
        return true;

    if (mnue::p2_embedded_available() && mnue::load_p2_embedded())
        return true;

    out << "info string no MNUE network available\n";
    const std::string& error = mnue::last_error();
    if (!error.empty())
        out << "info string mnue error: " << error << '\n';
    return false;
}

struct UciSession {
    memory::Memory mem{};
    Position pos{};
    PositionHistory position_history{};
    timeman::TimeManager time_manager{};
    bool enable_ponder = false;
    bool full_pv = false;
    bool use_msv_smp = false;
    bool msv_info = false;
    int threads = DEFAULT_UCI_THREADS;
    int multipv = DEFAULT_UCI_MULTIPV;
    int contempt = DEFAULT_UCI_CONTEMPT;
    int syzygy_probe_depth = syzygy::DEFAULT_PROBE_DEPTH;
    int syzygy_probe_limit = syzygy::DEFAULT_PROBE_LIMIT;
    bool syzygy_50_move_rule = true;
    std::string eval_file_p2{}; // empty â†’ use embedded
    search::SearchEvalKind active_eval_kind = search::SearchEvalKind::P2;
    std::string syzygy_path{};
    std::atomic<bool> stop_requested{false};
    std::atomic<bool> search_running{false};
    std::atomic<bool> ponder_search{false};
    std::atomic<bool> pondering{false};
    std::atomic<bool> ponder_hit_received{false};
    std::atomic<int> ponder_time_offset_ms{0};
    std::atomic<long long> search_start_ms{0};
    std::thread search_thread;

    UciSession() {
        memory::memory_init(mem, DEFAULT_UCI_HASH_MB, 8, 2);
        // attack_init_backend deferred to first command that needs it.
        set_start_position(pos);
        position_refresh_key(pos, mem.tables);
        (void)ensure_eval_loaded(nullptr);
    }

    [[nodiscard]] bool mnue_active() const noexcept {
        switch (active_eval_kind) {
            case search::SearchEvalKind::P2:
                return mnue::p2_loaded();
            case search::SearchEvalKind::X2K16:
                return mnue::x2k16::loaded();
            case search::SearchEvalKind::None:
                return false;
        }
        return false;
    }

    [[nodiscard]] std::string mnue_name() const {
        switch (active_eval_kind) {
            case search::SearchEvalKind::P2:
                return mnue::p2_loaded() ? mnue::p2_path() : std::string{};
            case search::SearchEvalKind::X2K16:
                return mnue::x2k16::loaded()
                    ? mnue::x2k16::path()
                    : std::string{};
            case search::SearchEvalKind::None:
                return {};
        }
        return {};
    }

    void ensure_attack_ready() {
        attack_init_backend(mem);
    }

    ~UciSession() {
        stop_search();
        syzygy::shutdown();
        memory::memory_free(mem);
    }

    void join_finished_search() {
        if (search_thread.joinable() &&
            !search_running.load(std::memory_order_acquire)) {
            search_thread.join();
        }
    }

    void stop_search() {
        stop_requested.store(true, std::memory_order_release);
        pondering.store(false, std::memory_order_release);
        if (search_thread.joinable())
            search_thread.join();
        ponder_search.store(false, std::memory_order_release);
        pondering.store(false, std::memory_order_release);
        ponder_hit_received.store(false, std::memory_order_release);
        ponder_time_offset_ms.store(0, std::memory_order_release);
        search_start_ms.store(0, std::memory_order_release);
        search_running.store(false, std::memory_order_release);
    }

    void handle_ponderhit() noexcept {
        if (!search_running.load(std::memory_order_acquire) ||
            !ponder_search.load(std::memory_order_acquire) ||
            !pondering.load(std::memory_order_acquire)) {
            return;
        }

        const long long start_ms = search_start_ms.load(std::memory_order_acquire);
        const long long now_ms = steady_now_ms();
        const long long elapsed = start_ms > 0 ? std::max(0LL, now_ms - start_ms) : 0LL;
        const int offset_ms = static_cast<int>(
            std::min<long long>(elapsed, std::numeric_limits<int>::max())
        );

        ponder_time_offset_ms.store(offset_ms, std::memory_order_release);
        ponder_hit_received.store(true, std::memory_order_release);
        pondering.store(false, std::memory_order_release);
    }

    void emit_banner(std::ostream& out) const {
        out << "MagnusChessX Thinking 0.1 by the Theodore Magnus Øen Nidhar";
        out << std::endl;
    }

    void emit_uci_id(std::ostream& out) const {
        out << "id name MagnusChessX Thinking 0.1\n";

        out << "id author Theodore Magnus Øen Nidhar\n";
        out << "option name Hash type spin default " << DEFAULT_UCI_HASH_MB
            << " min 1 max 1048576\n";
        out << "option name Threads type spin default 1 min 1 max " << MAX_UCI_THREADS << "\n";
        out << "option name MultiPV type spin default " << DEFAULT_UCI_MULTIPV
            << " min " << DEFAULT_UCI_MULTIPV
            << " max " << MAX_UCI_MULTIPV << "\n";
        out << "option name Contempt type spin default " << DEFAULT_UCI_CONTEMPT
            << " min " << MIN_UCI_CONTEMPT
            << " max " << MAX_UCI_CONTEMPT << "\n";
        out << "option name Move Overhead type spin default "
            << timeman::DEFAULT_MOVE_OVERHEAD_MS
            << " min " << timeman::MIN_MOVE_OVERHEAD_MS
            << " max " << timeman::MAX_MOVE_OVERHEAD_MS << "\n";
        out << "option name Ponder type check default false\n";
        out << "option name SyzygyPath type string default <empty>\n";
        out << "option name SyzygyProbeDepth type spin default "
            << syzygy::DEFAULT_PROBE_DEPTH
            << " min " << syzygy::MIN_PROBE_DEPTH
            << " max " << syzygy::MAX_PROBE_DEPTH << "\n";
        out << "option name Syzygy50MoveRule type check default true\n";
        out << "option name SyzygyProbeLimit type spin default "
            << syzygy::DEFAULT_PROBE_LIMIT
            << " min " << syzygy::MIN_PROBE_LIMIT
            << " max " << syzygy::MAX_PROBE_LIMIT << "\n";
        out << "option name MNUEfile type string default " << mnue::kEmbeddedP2Filename << "\n";
        out << "uciok" << std::endl;
    }

    void reset_new_game() {
        memory::memory_clear_hash(mem);
        set_start_position(pos);
        position_refresh_key(pos, mem.tables);
        clear_position_history(position_history);
        time_manager.new_game();
    }

    void copy_history_to_limits(search::SearchLimits& limits) const {
        limits.game_history_count = position_history.count;
        if (position_history.count < search::MAX_GAME_HISTORY) {
            for (int i = 0; i < position_history.count; ++i)
                limits.game_history_keys[i] = position_history.keys[i];
            return;
        }
        // Circular buffer wrap: copy [head..end] then [0..head-1].
        const int head = position_history.head;
        int dst = 0;
        for (int i = head; i < search::MAX_GAME_HISTORY; ++i)
            limits.game_history_keys[dst++] = position_history.keys[i];
        for (int i = 0; i < head; ++i)
            limits.game_history_keys[dst++] = position_history.keys[i];
    }

    [[nodiscard]] bool ensure_mnue_loaded(std::ostream* out) {
        // External file explicitly set by user â€” load from disk.
        if (!eval_file_p2.empty()) {
            const std::string resolved = resolve_file_path(eval_file_p2);

            if (looks_like_x2k16_file(resolved)) {
                if (active_eval_kind == search::SearchEvalKind::X2K16
                    && mnue::x2k16::loaded()
                    && mnue::x2k16::path() == resolved) {
                    return true;
                }

                active_eval_kind = search::SearchEvalKind::X2K16;
                mnue::unload_p2();
                if (!mnue::x2k16::load(resolved)) {
                    if (out) {
                        *out << "info string failed to load MNUE-X2-K16-pawn-Q8-A384: "
                             << mnue::x2k16::last_error() << '\n';
                    }
                    return false;
                }

                if (out) {
                    *out << "info string loaded MNUE-X2-K16-pawn-Q8-A384 "
                         << mnue::x2k16::path() << '\n';
                    *out << "info string MNUE-X2-K16-pawn-Q8-A384 backend: "
                        << mnue::x2k16::backend_name() << '\n';
                }
                return true;
            }

            if (active_eval_kind == search::SearchEvalKind::P2
                && mnue::p2_loaded()
                && mnue::p2_path() == resolved) {
                return true;
            }

            active_eval_kind = search::SearchEvalKind::P2;
            mnue::x2k16::unload();
            if (mnue::load_p2(resolved)) {
                if (out)
                    *out << "info string loaded " << mnue::p2_eval_name()
                         << ' ' << resolved << '\n';
                return true;
            }

            if (out)
                *out << "info string failed to load mnue: "
                     << mnue::last_error() << '\n';
            return false;
        }

        // Default: compile-time embedded P2/P2Pro network.
        active_eval_kind = search::SearchEvalKind::P2;
        mnue::x2k16::unload();
        if (mnue::p2_loaded() && mnue::p2_path() == mnue::kEmbeddedP2Filename)
            return true;

        if (mnue::p2_embedded_available() && mnue::load_p2_embedded()) {
            if (out)
                *out << "info string loaded embedded " << mnue::p2_eval_name() << '\n';
            return true;
        }

        if (out)
            *out << "info string no MNUE network available\n";
        return false;
    }

    [[nodiscard]] bool ensure_eval_loaded(std::ostream* out) {
        return ensure_mnue_loaded(out);
    }

    void handle_setoption(std::ostream& out, std::string_view command) {
        std::istringstream iss{std::string(command)};
        std::string token;
        std::string name;
        std::string value;

        iss >> token; // setoption
        iss >> token; // name

        while (iss >> token) {
            if (token == "value")
                break;

            if (!name.empty())
                name.push_back(' ');
            name += token;
        }

        std::getline(iss, value);
        if (!value.empty() && value.front() == ' ')
            value.erase(value.begin());

        if (name == "Hash") {
            int mb = 0;
            if (parse_int(value, mb) && mb > 0) {
                // Safety cap: prevent OOM crash from -fno-exceptions + huge allocation.
                constexpr std::size_t MAX_HASH_MB = 1ULL << 20; // 1 TB
                const std::size_t clamped_mb = std::min(
                    static_cast<std::size_t>(mb), MAX_HASH_MB);
                memory::tt_resize_mb(mem.tt, clamped_mb);
            }
        }
        else if (name == "Threads") {
            int parsed_threads = 0;
            if (parse_int(value, parsed_threads))
                threads = std::clamp(parsed_threads, 1, MAX_UCI_THREADS);
        }
        else if (name == "MultiPV") {
            int parsed_multipv = 0;
            if (parse_int(value, parsed_multipv))
                multipv = std::clamp(parsed_multipv, DEFAULT_UCI_MULTIPV, MAX_UCI_MULTIPV);
        }
        else if (name == "Contempt") {
            int parsed_contempt = 0;
            if (parse_int(value, parsed_contempt)) {
                const int clamped = std::clamp(parsed_contempt, MIN_UCI_CONTEMPT, MAX_UCI_CONTEMPT);
                if (contempt != clamped)
                    memory::memory_clear_hash(mem);
                contempt = clamped;
            }
        }
        else if (name == "Move Overhead") {
            int parsed_overhead = 0;
            if (parse_int(value, parsed_overhead))
                time_manager.set_move_overhead_ms(parsed_overhead);
        }
        else if (name == "Ponder") {
            bool parsed = false;
            if (parse_bool(value, parsed))
                enable_ponder = parsed;
        }
        else if (name == "SyzygyPath") {
            syzygy_path = value == "<empty>" ? std::string{} : value;
            if (!syzygy::init(syzygy_path)) {
                out << "info string failed to initialize Syzygy tablebases\n";
            } else if (syzygy::max_cardinality() > 0) {
                out << "info string Syzygy tablebases loaded up to "
                    << syzygy::max_cardinality() << " pieces\n";
            } else {
                out << "info string no Syzygy tablebases found\n";
            }
            memory::memory_clear_hash(mem);
        }
        else if (name == "SyzygyProbeDepth") {
            int parsed_depth = 0;
            if (parse_int(value, parsed_depth)) {
                syzygy_probe_depth = std::clamp(
                    parsed_depth,
                    syzygy::MIN_PROBE_DEPTH,
                    syzygy::MAX_PROBE_DEPTH
                );
            }
        }
        else if (name == "Syzygy50MoveRule") {
            bool parsed = false;
            if (parse_bool(value, parsed))
                syzygy_50_move_rule = parsed;
        }
        else if (name == "SyzygyProbeLimit") {
            int parsed_limit = 0;
            if (parse_int(value, parsed_limit)) {
                syzygy_probe_limit = std::clamp(
                    parsed_limit,
                    syzygy::MIN_PROBE_LIMIT,
                    syzygy::MAX_PROBE_LIMIT
                );
            }
        }
        else if (name == "MNUEfile") {
            // "<embedded>" or empty â†’ revert to built-in network.
            if (value.empty() || value == "<embedded>") {
                eval_file_p2.clear();
                active_eval_kind = search::SearchEvalKind::P2;
                mnue::x2k16::unload();
                mnue::unload_p2();
                memory::memory_clear_hash(mem);
                if (!ensure_eval_loaded(&out))
                    out << "info string eval unavailable\n";
                return;
            }
            const std::string resolved = resolve_file_path(value);
            std::error_code ec;
            if (!std::filesystem::exists(resolved, ec) || ec) {
                out << "info string MNUEfile not found: " << value << '\n';
                return;
            }
            eval_file_p2 = resolved;
            memory::memory_clear_hash(mem);

            if (!ensure_eval_loaded(&out))
                out << "info string eval unavailable\n";
        }
    }

    [[nodiscard]] bool ensure_search_eval_ready(
        std::ostream& out,
        const char* error_message
    ) {
        if (!ensure_eval_loaded(&out)) {
            out << error_message << '\n';
            return false;
        }
        return true;
    }


    void handle_eval(std::ostream& out) {
        if (!ensure_eval_loaded(&out)) {
            out << "info string eval unavailable\n";
            return;
        }

        int raw_stm = 0;
        std::string eval_name;
        std::string eval_path;
        if (active_eval_kind == search::SearchEvalKind::X2K16) {
            raw_stm = mnue::x2k16::evaluate_lut(pos, mem);
            eval_name = "x2-k16-pawn-q8-a384";
            eval_path = mnue::x2k16::path();
        } else {
            raw_stm = mnue::eval_p2(pos);
            eval_name = mnue::p2_short_name();
            eval_path = mnue::p2_path();
        }
        const int search_stm = mnue::search_score(raw_stm, pos);
        const int search_cp_stm = mnue::search_score_to_cp(search_stm, pos);
        const int winrate_stm = mnue::win_rate_model(raw_stm, pos);
        const int raw_white = white_pov_score(pos, raw_stm);
        const int search_white = white_pov_score(pos, search_stm);
        const int search_cp_white = white_pov_score(pos, search_cp_stm);
        const int winrate_white = white_pov_winrate(pos, winrate_stm);
        const mnue::WdlTriplet wdl_stm =
            mnue::search_score_to_wdl(search_stm, pos);
        const mnue::WdlTriplet wdl_white =
            white_pov_wdl(pos, wdl_stm);

        out << "info string eval " << eval_name << '\n';
        out << "info string mnue " << eval_name << " path " << eval_path << '\n';
        if (active_eval_kind == search::SearchEvalKind::X2K16)
            out << "info string mnue " << eval_name << " backend "
                << mnue::x2k16::backend_name() << '\n';
        out << "info string mnue " << eval_name << " material " << mnue::material_units(pos) << '\n';
        out << "info string mnue " << eval_name << " raw " << raw_white << '\n';
        out << "info string mnue " << eval_name << " search " << search_white << '\n';
        out << "info string mnue " << eval_name << " searchcp " << search_cp_white << '\n';
        out << "info string mnue " << eval_name << " winrate "
            << winrate_white << '\n';
        out << "info string mnue " << eval_name << " wdl "
            << wdl_white.win << ' '
            << wdl_white.draw << ' '
            << wdl_white.loss << '\n';
        out << "info string mnue " << eval_name << " raw_stm " << raw_stm << '\n';
        out << "info string mnue " << eval_name << " raw_white " << raw_white << '\n';
        out << "info string mnue " << eval_name << " search_stm " << search_stm << '\n';
        out << "info string mnue " << eval_name << " search_white " << search_white << '\n';
        out << "info string mnue " << eval_name << " searchcp_stm " << search_cp_stm << '\n';
        out << "info string mnue " << eval_name << " searchcp_white " << search_cp_white << '\n';
        out << "info string mnue " << eval_name << " winrate_stm " << winrate_stm << '\n';
        out << "info string mnue " << eval_name << " winrate_white " << winrate_white << '\n';
        out << "info string mnue " << eval_name << " white_expectation " << winrate_white << '\n';
        out << "info string mnue " << eval_name << " wdl_stm "
            << wdl_stm.win << ' '
            << wdl_stm.draw << ' '
            << wdl_stm.loss << '\n';
        out << "info string mnue " << eval_name << " wdl_white "
            << wdl_white.win << ' '
            << wdl_white.draw << ' '
            << wdl_white.loss << '\n';
    }

    void handle_x2k16_load(std::string_view line, std::ostream& out) {
        const std::string_view argument =
            trim_ascii(command_arguments(line, "mnuex2k16load"));
        if (argument.empty()) {
            out << "info string usage: mnuex2k16load <path>\n";
            return;
        }

        const std::string requested{argument};
        const std::string resolved = resolve_file_path(requested);
        std::error_code ec;
        if (!std::filesystem::exists(resolved, ec) || ec) {
            out << "info string MNUE-X2-K16-pawn-Q8-A384 file not found: "
                << requested << '\n';
            return;
        }

        if (!mnue::x2k16::load(resolved)) {
            out << "info string failed to load MNUE-X2-K16-pawn-Q8-A384: "
                << mnue::x2k16::last_error() << '\n';
            return;
        }

        out << "info string loaded MNUE-X2-K16-pawn-Q8-A384 "
            << mnue::x2k16::path() << '\n';
        out << "info string MNUE-X2-K16-pawn-Q8-A384 backend: "
            << mnue::x2k16::backend_name() << '\n';
    }

    void handle_x2k16_info(std::ostream& out) {
        mnue::x2k16::debug_dump_network(out);
    }

    void handle_x2k16_eval(std::ostream& out) {
        if (!mnue::x2k16::loaded()) {
            out << "info string MNUE-X2-K16-pawn-Q8-A384 unavailable: "
                << mnue::x2k16::last_error() << '\n';
            return;
        }

        const int raw_stm = mnue::x2k16::evaluate_lut(pos, mem);
        const int reference_raw_stm = mnue::x2k16::evaluate_reference(pos, mem);
        out << "info string MNUE-X2-K16-pawn-Q8-A384 eval backend "
            << mnue::x2k16::backend_name() << '\n';
        out << "info string MNUE-X2-K16-pawn-Q8-A384 eval stm "
            << raw_stm << '\n';
        out << "info string MNUE-X2-K16-pawn-Q8-A384 eval white "
            << white_pov_score(pos, raw_stm) << '\n';
        out << "info string MNUE-X2-K16-pawn-Q8-A384 eval reference_stm "
            << reference_raw_stm << '\n';
        out << "info string MNUE-X2-K16-pawn-Q8-A384 eval reference_diff "
            << (raw_stm - reference_raw_stm) << '\n';
    }

    void handle_x2k16_dump(std::ostream& out) {
        mnue::x2k16::debug_dump_features(pos, out);
    }

    void handle_x2k16_debug(std::ostream& out) {
        mnue::x2k16::debug_dump_evaluation(pos, mem, out);
    }

    void handle_x2k16_compare(std::ostream& out) {
        mnue::x2k16::debug_compare_evaluation(pos, mem, out);
    }

    void handle_x2k16_profile(std::ostream& out) {
        mnue::x2k16::debug_profile_evaluation(pos, mem, out);
    }

    void handle_x2k16_stress(std::string_view line, std::ostream& out) {
        int positions = 1000;
        const std::string_view argument =
            trim_ascii(command_arguments(line, "mnuex2k16stress"));
        if (!argument.empty() && !parse_int(argument, positions)) {
            out << "info string usage: mnuex2k16stress [positions]\n";
            return;
        }
        mnue::x2k16::debug_stress_evaluation(pos, mem, positions, out);
    }

    void handle_x2k16_inc_compare(std::ostream& out) {
        mnue::x2k16::debug_incremental_compare(pos, mem, out);
    }

    void handle_x2k16_inc_stress(std::string_view line, std::ostream& out) {
        int positions = 1000;
        const std::string_view argument =
            trim_ascii(command_arguments(line, "mnuex2k16incstress"));
        if (!argument.empty() && !parse_int(argument, positions)) {
            out << "info string usage: mnuex2k16incstress [positions]\n";
            return;
        }
        mnue::x2k16::debug_incremental_stress(pos, mem, positions, out);
    }

    void handle_bench(std::string_view line, std::ostream& out) {
        ensure_attack_ready();
        if (line != "bench") {
            out << "info string usage: " << bench_usage_hint() << '\n';
            return;
        }

        if (!ensure_search_eval_ready(out, "info string mnue unavailable, bench failed"))
            return;

        if (!run_compact_search_bench(
                mem,
                DEFAULT_BENCH_DEPTH,
                DEFAULT_UCI_HASH_MB,
                static_cast<std::size_t>(threads),
                out,
                active_eval_kind
            ))
            out << "info string bench failed\n";
    }

    void handle_position(std::string_view line, std::ostream& out) {
        ensure_attack_ready();
        if (!set_position_from_command(
                pos,
                mem,
                command_arguments(line, "position"),
                position_history
            )) {
            out << "info string invalid position command\n";
            return;
        }

        display_position_snapshot(pos, out, "position");
    }

    void handle_fen_shortcut(
        std::string_view line,
        std::string_view command,
        std::ostream& out
    ) {
        std::string_view fen = arrow_arguments(command_arguments(line, command));
        if (fen.empty()) {
            if (command == "fen")
                out << "fen -> " << display_fen(pos) << '\n';
            else
                display_position_snapshot(pos, out);
            return;
        }

        if (!parse_fen(pos, mem, fen)) {
            out << "invalid -> " << command << '\n';
            out << "usage -> " << command << " [->] <fen>\n";
            return;
        }

        clear_position_history(position_history);
        display_position_snapshot(pos, out, command);
    }

    void handle_go(std::string_view line, std::ostream& out) {
        ensure_attack_ready();
        if (!ensure_search_eval_ready(out, "info string mnue unavailable, search aborted"))
            return;

        search::SearchLimits limits{};
        if (!parse_go_command(pos, mem, time_manager, line, limits)) {
            out << "info string usage: " << go_usage_hint() << '\n';
            out << "info string " << go_usage_examples() << '\n';
            return;
        }

        if (!mnue_active()) {
            out << "info string mnue unavailable, search aborted\n";
            return;
        }

        std::ostringstream desc;
        if (active_eval_kind == search::SearchEvalKind::X2K16) {
            desc << "info string MNUE evaluation using "
                << mnue_name()
                << " (X2-K16-pawn-Q8-A384, " << std::fixed << std::setprecision(1)
                << (static_cast<double>(mnue::x2k16::network_bytes()) / (1024.0 * 1024.0))
                << " MiB, ("
                << mnue::x2k16::Layout::OutputBuckets << ','
                << mnue::x2k16::Layout::InputBuckets << ','
                << mnue::x2k16::Layout::PieceHiddenSize << '/'
                << mnue::x2k16::Layout::AttackHiddenSize << '/'
                << mnue::x2k16::Layout::PawnPairHiddenSize << ','
                << mnue::x2k16::Layout::PieceInputSize << '+'
                << mnue::x2k16::Layout::AttackInputSize << '+'
                << mnue::x2k16::Layout::PawnPairInputSize
                << "), " << mnue::x2k16::backend_name() << ")\n";
        } else {
            desc << "info string MNUE evaluation using "
                << mnue_name()
                << " (" << mnue::p2_arch_name()
                << ", " << std::fixed << std::setprecision(1)
                << (static_cast<double>(mnue::p2_file_bytes()) / (1024.0 * 1024.0))
                << " MiB, (1,"
                << mnue::p2_output_buckets() << ','
                << mnue::p2_input_buckets() << ','
                << mnue::p2_hidden_size() << ','
                << mnue::p2_input_size() << "), "
                << mnue::eval_simd_name() << ")\n";
        }
        std::cout << desc.str();

        limits.contempt = contempt;
        limits.full_pv = full_pv;
        limits.use_msv_smp = use_msv_smp;
        limits.msv_info = msv_info;
        limits.multipv = multipv;
        limits.stop = &stop_requested;
        limits.pondering = &pondering;
        limits.ponder_time_offset_ms = &ponder_time_offset_ms;
        limits.thread_count = threads;
        limits.thread_id = 0;
        limits.report_info = true;
        limits.eval_kind = active_eval_kind;
        limits.recover_ponder_pv = enable_ponder || limits.ponder;
        limits.syzygy_probe_depth = syzygy_probe_depth;
        limits.syzygy_probe_limit = syzygy_probe_limit;
        limits.syzygy_50_move_rule = syzygy_50_move_rule;
        copy_history_to_limits(limits);
        stop_requested.store(false, std::memory_order_release);
        ponder_search.store(limits.ponder, std::memory_order_release);
        pondering.store(limits.ponder, std::memory_order_release);
        ponder_hit_received.store(false, std::memory_order_release);
        ponder_time_offset_ms.store(0, std::memory_order_release);
        search_start_ms.store(steady_now_ms(), std::memory_order_release);

        const Position root = pos;
        search_running.store(true, std::memory_order_release);
        search_thread = std::thread([this, root, limits]() {
            PvTrackingStreamBuf pv_tracking_buf(std::cout.rdbuf());
            std::ostream tracked_out(&pv_tracking_buf);
            const search::SearchResult result =
                search::iterative_deepening(root, mem, limits, &tracked_out);
            tracked_out.flush();

            const std::string ponder = ponder_move_from_search_result(
                root,
                mem,
                result
            );

            std::cout << "bestmove " << search::move_to_uci(result.best_move);
            if (enable_ponder && !ponder.empty())
                std::cout << " ponder " << ponder;
            std::cout << std::endl;
            ponder_search.store(false, std::memory_order_release);
            pondering.store(false, std::memory_order_release);
            ponder_hit_received.store(false, std::memory_order_release);
            ponder_time_offset_ms.store(0, std::memory_order_release);
            search_start_ms.store(0, std::memory_order_release);
            search_running.store(false, std::memory_order_release);
        });
    }

    void handle_perft(std::string_view line, std::ostream& out) {
        ensure_attack_ready();

        std::istringstream iss{std::string(line)};
        std::string command;
        std::string depth_text;
        std::string extra;
        int depth = -1;
        if (!(iss >> command) ||
            !(iss >> depth_text) ||
            (iss >> extra) ||
            !parse_int(depth_text, depth) ||
            depth < 0) {
            out << "info string usage: perft <depth>\n";
            return;
        }

        if (!run_basic_perft_root(pos, mem, depth, out))
            out << "info string perft failed\n";
    }

    [[nodiscard]] bool process_command(
        std::string_view line,
        std::ostream& out
    ) {
        join_finished_search();

        if (line == "uci") {
            emit_uci_id(out);
            return true;
        }

        if (line == "isready") {
            if (!ensure_eval_loaded(&out))
                out << "info string eval unavailable\n";
            out << "readyok\n";
            return true;
        }

        if (line == "stop") {
            stop_search();
            return true;
        }

        if (line == "ponderhit") {
            handle_ponderhit();
            return true;
        }

        if (line == "quit") {
            stop_search();
            return false;
        }

        if (search_running.load(std::memory_order_acquire)) {
            out << "info string search busy, send stop first\n";
            return true;
        }

        if (line == "ucinewgame") {
            reset_new_game();
            display_position_snapshot(pos, out, "newgame");
            return true;
        }

        if (command_starts_with(line, "setoption")) {
            handle_setoption(out, line);
            return true;
        }



        if (line == "eval") {
            handle_eval(out);
            return true;
        }

        if (command_starts_with(line, "mnuex2k16load")) {
            handle_x2k16_load(line, out);
            return true;
        }

        if (line == "mnuex2k16info") {
            handle_x2k16_info(out);
            return true;
        }

        if (line == "mnuex2k16eval") {
            handle_x2k16_eval(out);
            return true;
        }

        if (line == "mnuex2k16dump") {
            handle_x2k16_dump(out);
            return true;
        }

        if (line == "mnuex2k16debug") {
            handle_x2k16_debug(out);
            return true;
        }

        if (line == "mnuex2k16compare") {
            handle_x2k16_compare(out);
            return true;
        }

        if (line == "mnuex2k16profile") {
            handle_x2k16_profile(out);
            return true;
        }

        if (command_starts_with(line, "mnuex2k16stress")) {
            handle_x2k16_stress(line, out);
            return true;
        }

        if (line == "mnuex2k16inccompare") {
            handle_x2k16_inc_compare(out);
            return true;
        }

        if (command_starts_with(line, "mnuex2k16incstress")) {
            handle_x2k16_inc_stress(line, out);
            return true;
        }

        if (line == "d") {
            display_position(pos, out);
            return true;
        }


        if (command_starts_with(line, "pos")) {
            handle_fen_shortcut(line, "pos", out);
            return true;
        }

        if (command_starts_with(line, "fen")) {
            handle_fen_shortcut(line, "fen", out);
            return true;
        }

        if (command_starts_with(line, "hash") || command_starts_with(line, "key")) {
            const std::string_view command = command_starts_with(line, "hash") ? "hash" : "key";
            if (!arrow_arguments(command_arguments(line, command)).empty())
                out << "readonly -> hash is derived from the current position\n";
            out << "info string hash -> " << pos.key << " (" << display_key_hex(pos.key) << ")\n";
            return true;
        }

        if (command_starts_with(line, "bench")) {
            handle_bench(line, out);
            return true;
        }

        if (command_starts_with(line, "position")) {
            handle_position(line, out);
            return true;
        }

        if (command_starts_with(line, "go")) {
            handle_go(line, out);
            return true;
        }

        if (command_starts_with(line, "perft")) {
            handle_perft(line, out);
            return true;
        }

        return true;
    }
};

} // namespace

int run_bench(int argc, char** argv) {
    const auto print_usage = []() {
        std::cerr
            << "usage: MagnusChessXThinking bench [depth=12] [hash_mb=16] [threads=1]\n"
            << "       MagnusChessXThinking perft <depth>\n";
    };

    if (argc <= 1) {
        print_usage();
        return 1;
    }

    const std::string_view command{argv[1]};
    if (command == "bench") {
        int depth = DEFAULT_BENCH_DEPTH;
        int hash_mb = DEFAULT_UCI_HASH_MB;
        int thread_count = DEFAULT_UCI_THREADS;

        const auto parse_optional = [&](int index, int& value) noexcept {
            return argc <= index || parse_int(argv[index], value);
        };

        if (argc > 5 ||
            !parse_optional(2, depth) ||
            !parse_optional(3, hash_mb) ||
            !parse_optional(4, thread_count) ||
            depth <= 0 ||
            depth > search::MAX_SEARCH_DEPTH ||
            hash_mb <= 0 ||
            thread_count <= 0) {
            print_usage();
            return 1;
        }

        const int clamped_threads =
            std::clamp(thread_count, 1, MAX_SEARCH_THREADS);

        memory::Memory mem{};
        memory::memory_init(mem, static_cast<std::size_t>(hash_mb), 8, 2);
        attack_init_backend(mem);

        if (!ensure_cli_mnue_loaded(std::cout)) {
            memory::memory_free(mem);
            return 1;
        }

        const bool ok = run_compact_search_bench(
            mem,
            depth,
            static_cast<std::size_t>(hash_mb),
            static_cast<std::size_t>(clamped_threads),
            std::cout,
            search::SearchEvalKind::P2,
            true
        );
        memory::memory_free(mem);
        return ok ? 0 : 1;
    }

    if (command == "perft") {
        int depth = -1;
        if (argc != 3 || !parse_int(argv[2], depth) || depth < 0) {
            print_usage();
            return 1;
        }

        memory::Memory mem{};
        memory::memory_init(mem, DEFAULT_UCI_HASH_MB, 8, 2);
        attack_init_backend(mem);

        Position pos{};
        set_start_position(pos);
        position_refresh_key(pos, mem.tables);

        const bool ok = run_basic_perft_root(pos, mem, depth, std::cout);
        memory::memory_free(mem);
        return ok ? 0 : 1;
    }

    print_usage();
    return 1;
}

int run_uci() {
    // UCI command loop with cooperative stop support.
    std::cout << std::unitbuf;
    UciSession session{};
    session.emit_banner(std::cout);

    constexpr std::size_t MAX_UCI_LINE = 8192;
    std::string line;
    line.reserve(MAX_UCI_LINE);
    while (true) {
        line.clear();
        int ch = 0;
        while ((ch = std::cin.get()) != std::char_traits<char>::eof() && ch != '\n') {
            if (line.size() < MAX_UCI_LINE)
                line.push_back(static_cast<char>(ch));
        }
        if (ch == std::char_traits<char>::eof() && line.empty())
            break;
        if (line.size() >= MAX_UCI_LINE)
            std::cout << "info string line too long, ignoring\n";
        else if (!session.process_command(line, std::cout))
            break;
    }

    return 0;
}

} // namespace magnus
