#include "Common.h"
#include "board/MoveGen.h"
#include "board/Position.h"
#include "search/Search.h"

#include <cstdlib>
#include <iostream>

namespace {

using namespace magnus;

[[noreturn]] void fail(const char* label) {
    std::cerr << "search reference test failed: " << label << '\n';
    std::exit(1);
}

void expect(bool condition, const char* label) {
    if (!condition)
        fail(label);
}

void test_plies_from_null() {
    memory::Memory mem{};
    memory::memory_init(mem, 1, 1, 1);

    Position pos{};
    set_start_position(pos);
    position_refresh_key(pos, mem.tables);
    expect(pos.plies_from_null == 0, "start/FEN initialization");
    const Key start_key = pos.key;

    const Move e2e4 = magnus::make_move(12, 28, MOVE_DOUBLE_PUSH);
    StateInfo e2e4_state{};
    make_move(pos, e2e4, mem.tables, e2e4_state);
    expect(pos.plies_from_null == 1, "pawn move does not reset");

    Position copied{};
    position_copy_without_accumulators(copied, pos);
    expect(copied.plies_from_null == 1, "position copy");

    StateInfo null_state{};
    make_null_move(pos, mem.tables, null_state);
    expect(pos.plies_from_null == 0, "null resets");
    unmake_null_move(pos, null_state);
    expect(pos.plies_from_null == 1, "null undo restores");

    const Move d7d5 = magnus::make_move(51, 35, MOVE_DOUBLE_PUSH);
    StateInfo d7d5_state{};
    make_move(pos, d7d5, mem.tables, d7d5_state);
    const Move e4d5 = magnus::make_move(28, 35, MOVE_CAPTURE);
    StateInfo capture_state{};
    make_move(pos, e4d5, mem.tables, capture_state);
    expect(pos.plies_from_null == 3, "capture does not reset");
    unmake_move(pos, e4d5, mem.tables, capture_state);
    expect(pos.plies_from_null == 2, "capture undo");
    unmake_move(pos, d7d5, mem.tables, d7d5_state);
    unmake_move(pos, e2e4, mem.tables, e2e4_state);
    expect(pos.plies_from_null == 0 && pos.key == start_key, "normal undo round trip");

    Position copied_move{};
    set_start_position(copied_move);
    position_refresh_key(copied_move, mem.tables);
    do_move_copy(copied_move, e2e4, mem.tables);
    expect(copied_move.plies_from_null == 1, "do_move_copy");
    position_clear(copied_move);
    expect(copied_move.plies_from_null == 0, "FEN clear path");

    memory::memory_free(mem);
}

void test_shuffling_truth_table() {
    Position pos{};
    pos.halfmove_clock = 10;
    pos.plies_from_null = 7;

    const Move move4 = magnus::make_move(0, 1, MOVE_QUIET);
    const Move move2 = magnus::make_move(1, 2, MOVE_QUIET);
    const Move move = magnus::make_move(2, 3, MOVE_QUIET);
    const auto shuffling = [&](Move candidate, int ply = 20) {
        return search::stockfish_is_shuffling(candidate, pos, move2, move4, ply);
    };

    expect(shuffling(move), "two-leg return");
    expect(!shuffling(magnus::make_move(2, 3, MOVE_CAPTURE)), "capture stage");
    expect(!shuffling(magnus::make_move(2, 3, MOVE_PROMO_Q)), "queen promotion stage");
    expect(shuffling(magnus::make_move(2, 3, MOVE_PROMO_N)), "underpromotion not capture stage");

    pos.halfmove_clock = 9;
    expect(!shuffling(move), "rule50 threshold");
    pos.halfmove_clock = 10;
    pos.plies_from_null = 6;
    expect(!shuffling(move), "null distance threshold");
    pos.plies_from_null = 7;
    expect(!shuffling(move, 19), "ply threshold");
    expect(!shuffling(magnus::make_move(3, 4, MOVE_QUIET)), "return squares");
}

void test_depth_transitions() {
    const int original_node_depth = 10;
    const int move_base_depth = original_node_depth - 1;
    const int extended_node_depth = original_node_depth + 1;

    expect(
        search::stockfish_singular_child_depth(move_base_depth, 1) == 10,
        "current TT child depth is frozen"
    );
    expect(extended_node_depth - 1 == 10, "later move inherits +1");
    expect(
        search::stockfish_depth_after_alpha_improvement(extended_node_depth, false) == 9,
        "alpha improvement applies -2"
    );
    expect(
        search::stockfish_depth_after_alpha_improvement(14, false) == 14,
        "alpha depth upper boundary"
    );
    expect(
        search::stockfish_fail_high_softbound(150, 100, 9) == 145,
        "fail-high softbound uses final depth"
    );
}

void test_tt_trust_required_gap() {
    constexpr int base_gap = 53;

    expect(search::tt_trust_required_gap(base_gap, 511) == 53, "+511 truncates to zero");
    expect(search::tt_trust_required_gap(base_gap, -511) == 53, "-511 truncates to zero");
    expect(search::tt_trust_required_gap(base_gap, 512) == 52, "+512 lowers gap by one");
    expect(search::tt_trust_required_gap(base_gap, -512) == 54, "-512 raises gap by one");
    expect(search::tt_trust_required_gap(base_gap, 4096) == 45, "+4096 reaches clamp");
    expect(search::tt_trust_required_gap(base_gap, -4096) == 61, "-4096 reaches clamp");
    expect(search::tt_trust_required_gap(base_gap, 8192) == 45, "+8192 stays clamped");
    expect(search::tt_trust_required_gap(base_gap, -8192) == 61, "-8192 stays clamped");
    expect(search::tt_trust_required_gap(1, 8192) == 1, "required gap floor");
}

void test_persistent_worker_slots() {
    Position pos{};
    pos.side_to_move = WHITE;
    search::SearchSessionState session{};
    session.ensure_workers(3);
    session.worker(2).update_trust(pos, 256);
    const int learned = session.worker(2).trust(pos);
    expect(learned > 0, "worker trust learns");

    const u64 first_sequence = session.begin_search(3);
    expect(first_sequence == 1, "search sequence starts at one");
    expect(session.worker(2).search_sequence == 1, "worker search sequence");
    session.ensure_workers(1);
    expect(session.worker(2).trust(pos) == learned, "disabled worker slot retained");

    session.new_game();
    expect(session.worker(2).trust(pos) == 0, "new game clears trust");
    expect(session.worker(2).game_epoch == 1, "new game advances epoch");
}

} // namespace

int main() {
    test_plies_from_null();
    test_shuffling_truth_table();
    test_depth_transitions();
    test_tt_trust_required_gap();
    test_persistent_worker_slots();
    std::cout << "search reference state and depth transitions ok\n";
    return 0;
}
