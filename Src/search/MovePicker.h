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

#include "History.h"
#include "board/MoveGen.h"

#include "board/Position.h"

namespace magnus::search {

enum class MoveStage : std::uint8_t {
    TT_MOVE = 0,
    GEN_CAPTURES,
    GOOD_CAPTURES,
    GEN_QUIETS,
    KILLER_1,
    KILLER_2,
    QUIETS,
    BAD_CAPTURES,
    DONE
};

struct QuietControl {
    // Enables node-driven late quiet suppression in the picker. This is not a
    // global hard skip; it only demotes low-value late quiets to cheaper
    // history-only ordering.
    bool skip_quiets = false;
    int quiet_limit = 0;
    int history_floor = 0;
    int keep_top_history = 0;
};

/*
MovePicker emits legal moves in staged order:
TT move -> good captures -> killers -> quiets -> bad captures.
The picker lazily builds each stage on first use so early cutoffs do not pay
for later generation and scoring work.
*/
/*
 * MovePicker — staged lazy move generator
 * Stage order: TT move -> good captures -> killer 1 -> killer 2 -> quiets -> bad captures
 * Each stage is lazily built on first use; early cutoffs avoid paying for
 * generation/scoring of later stages.
 * QuietControl suppresses late quiet moves (history ordering only, no full scoring).
 * ScoredEntry stores the capture list for observation/tool reuse (via public accessors).
 */
class MovePicker {
public:
    MovePicker(
        Position& pos,
        const memory::Memory& mem,
        const GenInfo& info,
        const HistoryTables& history,
        Move tt_move,
        int ply,
        Move prev_move,
        ContinuationHistoryContext prev2,
        ContinuationHistoryContext prev4,
        ContinuationHistoryContext prev8,
        int depth,
        QuietControl quiet_control = {}
    ) noexcept;

    [[nodiscard]] Move next() noexcept;

    [[nodiscard]] int last_score() const noexcept { return last_score_; }
    [[nodiscard]] bool last_was_capture() const noexcept { return last_was_capture_; }
    [[nodiscard]] bool last_was_bad_capture() const noexcept { return last_was_bad_capture_; }
    [[nodiscard]] int last_see_value() const noexcept { return last_see_value_; }
    [[nodiscard]] bool last_quiet_in_skip_band() const noexcept { return last_quiet_in_skip_band_; }
    [[nodiscard]] bool last_quiet_suppressed() const noexcept { return last_quiet_suppressed_; }
    [[nodiscard]] int quiet_generated() const noexcept { return quiet_generated_; }
    [[nodiscard]] int quiet_scored() const noexcept { return quiet_scored_; }
    [[nodiscard]] int quiet_suppressed() const noexcept { return quiet_suppressed_; }

    struct ScoredEntry {
        Move move = 0;
        int score = 0;
        int see_value = 0;
        bool quiet_in_skip_band = false;
        bool quiet_suppressed = false;
    };

    // Capture list accessors for observation / tooling reuse.
    [[nodiscard]] int good_capture_count() const noexcept { return good_size_; }
    [[nodiscard]] int bad_capture_count() const noexcept { return bad_size_; }
    [[nodiscard]] const ScoredEntry* good_captures() const noexcept { return good_caps_; }
    [[nodiscard]] const ScoredEntry* bad_captures() const noexcept { return bad_caps_; }

private:

    Position& pos_;
    const memory::Memory& mem_;
    const GenInfo& info_;
    const HistoryTables& history_;

    Move tt_move_ = Move(0);
    Move killer1_ = Move(0);
    Move killer2_ = Move(0);
    Move prev_move_ = Move(0);
    ContinuationHistoryContext prev2_{};
    ContinuationHistoryContext prev4_{};
    ContinuationHistoryContext prev8_{};
    int depth_ = 0;
    QuietControl quiet_control_{};

    MoveStage stage_ = MoveStage::TT_MOVE;

    ScoredEntry good_caps_[MAX_MOVES]{};
    ScoredEntry bad_caps_[MAX_MOVES]{};
    ScoredEntry quiets_[MAX_MOVES]{};
    int good_size_ = 0;
    int bad_size_ = 0;
    int quiet_size_ = 0;
    int good_idx_ = 0;
    int bad_idx_ = 0;
    int quiet_idx_ = 0;

    bool tt_ready_ = false;
    bool tt_prepared_ = false;
    bool tt_legal_ = false;
    int tt_score_ = 0;
    int tt_see_value_ = 0;
    bool tt_bad_capture_ = false;

    bool captures_built_ = false;
    bool quiets_built_ = false;

    bool killer1_ready_ = false;
    bool killer2_ready_ = false;
    Move killer1_move_ = Move(0);
    Move killer2_move_ = Move(0);
    int killer1_score_ = 0;
    int killer2_score_ = 0;

    int last_score_ = 0;
    bool last_was_capture_ = false;
    bool last_was_bad_capture_ = false;
    int last_see_value_ = 0;
    bool last_quiet_in_skip_band_ = false;
    bool last_quiet_suppressed_ = false;
    int quiet_generated_ = 0;
    int quiet_scored_ = 0;
    int quiet_suppressed_ = 0;

private:
    void prepare_tt_move() noexcept;
    void build_capture_stage() noexcept;
    void build_quiet_stage() noexcept;
    void add_capture(Move move) noexcept;
    void add_quiet(
        Move move,
        int history_score,
        bool quiet_in_skip_band,
        bool quiet_suppressed
    ) noexcept;

    [[nodiscard]] int score_capture(Move move, int see_value) const noexcept;
    [[nodiscard]] int score_quiet(Move move) const noexcept;

    [[nodiscard]] ScoredEntry pick_best_entry(
        ScoredEntry* list,
        int size,
        int& index
    ) noexcept;

    inline void set_last(
        int score,
        bool capture,
        bool bad_capture,
        int see_value,
        bool quiet_in_skip_band = false,
        bool quiet_suppressed = false
    ) noexcept {
        last_score_ = score;
        last_was_capture_ = capture;
        last_was_bad_capture_ = bad_capture;
        last_see_value_ = see_value;
        last_quiet_in_skip_band_ = quiet_in_skip_band;
        last_quiet_suppressed_ = quiet_suppressed;
    }
};

} // namespace magnus::search
