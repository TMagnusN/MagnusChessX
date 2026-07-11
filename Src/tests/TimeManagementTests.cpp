#include "board/Position.h"
#include "search/Time.h"

#include <cstdlib>
#include <iostream>

namespace {

using magnus::Position;
using magnus::search::SearchLimits;
using magnus::timeman::GoParams;
using magnus::timeman::TimeManager;

[[noreturn]] void fail(const char* label, int actual, int expected) {
    std::cerr << label << ": expected " << expected << ", got " << actual << '\n';
    std::exit(1);
}

void expect_equal(const char* label, int actual, int expected) {
    if (actual != expected)
        fail(label, actual, expected);
}

SearchLimits build(
    TimeManager& manager,
    Position& position,
    int remaining,
    int increment = 0,
    int moves_to_go = 0
) {
    GoParams params{};
    params.wtime = remaining;
    params.btime = remaining;
    params.winc = increment;
    params.binc = increment;
    params.movestogo = moves_to_go;

    SearchLimits limits{};
    if (!manager.build_limits(position, params, limits)) {
        std::cerr << "build_limits rejected " << remaining << "ms\n";
        std::exit(1);
    }
    if (!limits.use_time_management || limits.time_state == nullptr) {
        std::cerr << "managed clock did not publish persistent state\n";
        std::exit(1);
    }
    if (limits.soft_time_ms < 1 || limits.soft_time_ms > limits.hard_time_ms) {
        std::cerr << "invalid soft/hard ordering\n";
        std::exit(1);
    }
    return limits;
}

} // namespace

int main() {
    Position position{};
    position.side_to_move = magnus::WHITE;
    position.fullmove_number = 1;

    TimeManager manager{};
    SearchLimits limits = build(manager, position, 60000);
    expect_equal("60+0 soft", limits.soft_time_ms, 1155);
    expect_equal("60+0 hard", limits.hard_time_ms, 7701);

    manager.new_game();
    limits = build(manager, position, 10000, 100);
    expect_equal("10+0.1 soft", limits.soft_time_ms, 224);
    expect_equal("10+0.1 hard", limits.hard_time_ms, 1431);

    manager.new_game();
    limits = build(manager, position, 60000, 0, 40);
    expect_equal("40/60 soft", limits.soft_time_ms, 1310);
    expect_equal("40/60 hard", limits.hard_time_ms, 7457);

    manager.new_game();
    limits = build(manager, position, 500);
    expect_equal("subsecond soft", limits.soft_time_ms, 1);
    expect_equal("subsecond hard", limits.hard_time_ms, 1);

    manager.set_move_overhead_ms(100);
    position.fullmove_number = 16;
    limits = build(manager, position, 10000);
    expect_equal("overhead soft", limits.soft_time_ms, 101);
    expect_equal("overhead hard", limits.hard_time_ms, 664);

    GoParams fixed{};
    fixed.movetime = 250;
    SearchLimits fixed_limits{};
    if (!manager.build_limits(position, fixed, fixed_limits))
        return 1;
    expect_equal("movetime soft", fixed_limits.soft_time_ms, 250);
    expect_equal("movetime hard", fixed_limits.hard_time_ms, 250);
    if (fixed_limits.use_time_management || fixed_limits.time_state != nullptr)
        return 1;

    manager.new_game();
    position.fullmove_number = 1;
    limits = build(manager, position, 10000, 100);
    expect_equal("newgame restores soft", limits.soft_time_ms, 141);
    expect_equal("newgame preserves overhead hard", limits.hard_time_ms, 897);

    std::cout << "time management calculations ok\n";
    return 0;
}
