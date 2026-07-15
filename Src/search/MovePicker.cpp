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

#include "MovePicker.h"
#include "board/MoveGen.h"


#include <algorithm>
#include <limits>

#include "See.h"

/* ===== 繁體中文註釋 =====
 * 本檔案是 MagnusChessX Thinking 西洋棋引擎的一部分。
 * 實作詳情請參閱對應的 .h 標頭檔案。
 */

namespace magnus::search {

namespace {

constexpr int MAX_TOP_HISTORY_QUIETS = 8;

[[nodiscard]] inline int mvv_lva_capture_term(
    const Position& pos,
    Move move
) noexcept {
    const PieceType attacker = piece_type_on(pos, from_sq(move));
    const PieceType victim = move_is_ep(move) ? PAWN : piece_type_on(pos, to_sq(move));
    const int attacker_value = is_ok(attacker) ? piece_order_value[attacker] : 0;
    const int victim_value = is_ok(victim) ? piece_order_value[victim] : 0;
    return victim_value * 32 - attacker_value;
}

} // namespace

/*
 * MovePicker 實作 — 分階段惰性著法產生與排序
 * prepare_tt_move() — 驗證 TT 著法合法性並計算評分
 * build_capture_stage() — 生成捕獲、計算 SEE、分為好/壞捕獲
 * build_quiet_stage() — 生成安靜著法、排序、應用 QuietControl 抑制
 * add_capture/add_quiet() — 將著法加入對應列表（處理殺手著法檢測）
 * pick_best_entry() — 惰性選擇排序（O(n²) 部分排序，每步選最佳）
 * score_capture() — MVV-LVA + 捕獲歷史 + SEE 偏差項
 * score_quiet() — 安靜著法歷史排序分數（含延續歷史 + 反著獎勵）
 */
MovePicker::MovePicker(
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
    QuietControl quiet_control
) noexcept
    : pos_(pos)
    , mem_(mem)
    , info_(info)
    , history_(history)
    , tt_move_(tt_move)
    , killer1_(history.killer_fast(ply, 0))
    , killer2_(history.killer_fast(ply, 1))
    , prev_move_(prev_move)
    , prev2_(prev2)
    , prev4_(prev4)
    , prev8_(prev8)
    , depth_(depth)
    , quiet_control_(quiet_control) {}

Move MovePicker::next() noexcept {
    while (true) {
        switch (stage_) {
            case MoveStage::TT_MOVE:
                prepare_tt_move();
                stage_ = MoveStage::GEN_CAPTURES;
                if (tt_ready_) {
                    tt_ready_ = false;
                    set_last(
                        tt_score_,
                        move_is_capture(tt_move_),
                        tt_bad_capture_,
                        tt_see_value_
                    );
                    return tt_move_;
                }
                break;

            case MoveStage::GEN_CAPTURES:
                build_capture_stage();
                stage_ = MoveStage::GOOD_CAPTURES;
                break;

            case MoveStage::GOOD_CAPTURES:
                if (good_idx_ < good_size_) {
                    const ScoredEntry e = pick_best_entry(good_caps_, good_size_, good_idx_);
                    set_last(e.score, true, false, e.see_value);
                    return e.move;
                }
                stage_ = MoveStage::GEN_QUIETS;
                break;

            case MoveStage::GEN_QUIETS:
                build_quiet_stage();
                stage_ = MoveStage::KILLER_1;
                break;

            case MoveStage::KILLER_1:
                stage_ = MoveStage::KILLER_2;
                if (killer1_ready_) {
                    killer1_ready_ = false;
                    set_last(killer1_score_, false, false, 0);
                    return killer1_move_;
                }
                break;

            case MoveStage::KILLER_2:
                stage_ = MoveStage::QUIETS;
                if (killer2_ready_) {
                    killer2_ready_ = false;
                    set_last(killer2_score_, false, false, 0);
                    return killer2_move_;
                }
                break;

            case MoveStage::QUIETS:
                while (quiet_idx_ < quiet_size_) {
                    const ScoredEntry e = pick_best_entry(quiets_, quiet_size_, quiet_idx_);
                    if (e.move == killer1_move_ || e.move == killer2_move_)
                        continue;
                    set_last(
                        e.score,
                        false,
                        false,
                        0,
                        e.quiet_in_skip_band,
                        e.quiet_suppressed
                    );
                    return e.move;
                }
                stage_ = MoveStage::BAD_CAPTURES;
                break;

            case MoveStage::BAD_CAPTURES:
                if (bad_idx_ < bad_size_) {
                    const ScoredEntry e = pick_best_entry(bad_caps_, bad_size_, bad_idx_);
                    set_last(e.score, true, true, e.see_value);
                    return e.move;
                }
                stage_ = MoveStage::DONE;
                break;

            case MoveStage::DONE:
                return Move(0);
        }
    }
}

void MovePicker::prepare_tt_move() noexcept {
    if (tt_prepared_)
        return;

    tt_prepared_ = true;
    if (move_is_none(tt_move_))
        return;

    if (!pseudo_legal_fast(pos_, mem_, info_, tt_move_))
        return;

    if (!legal(pos_, mem_, tt_move_))
        return;

    tt_legal_ = true;
    tt_ready_ = true;
    if (move_is_capture(tt_move_)) {
        tt_see_value_ = search::see_value_fast(pos_, mem_, tt_move_);
        tt_bad_capture_ = tt_see_value_ < 0;
        tt_score_ = score_capture(tt_move_, tt_see_value_);
        return;
    }

    tt_see_value_ = 0;
    tt_bad_capture_ = false;
    tt_score_ = score_quiet(tt_move_);
}

void MovePicker::build_capture_stage() noexcept {
    if (captures_built_)
        return;

    captures_built_ = true;
    MoveList list;
    Move* end = generate_pseudo_captures_only(pos_, mem_, info_, list.moves);
    list.size = static_cast<int>(end - list.moves);

    for (int i = 0; i < list.size; ++i) {
        const Move move = list.moves[i];
        if (tt_legal_ && move == tt_move_)
            continue;
        if (!legal_fast(pos_, mem_, info_, move))
            continue;
        add_capture(move);
    }
}

void MovePicker::build_quiet_stage() noexcept {
    if (quiets_built_)
        return;

    quiets_built_ = true;
    MoveList list;
    Move* end = generate_pseudo_quiets(pos_, mem_, info_, list.moves);
    list.size = static_cast<int>(end - list.moves);

    if (!quiet_control_.skip_quiets) {
        for (int i = 0; i < list.size; ++i) {
            const Move move = list.moves[i];
            if (tt_legal_ && move == tt_move_)
                continue;
            if (!legal_fast(pos_, mem_, info_, move))
                continue;
            add_quiet(move, history_.quiet_value_fast(pos_, move), false, false);
        }
        return;
    }

    struct QuietCandidate {
        Move move = 0;
        int history_score = 0;
        bool countermove = false;
        bool top_history = false;
    };

    QuietCandidate candidates[MAX_MOVES]{};
    int candidate_count = 0;

    const Move countermove = history_.countermove_fast(pos_, prev_move_);
    const int keep_top_history =
        std::clamp(quiet_control_.keep_top_history, 0, MAX_TOP_HISTORY_QUIETS);
    int top_indices[MAX_TOP_HISTORY_QUIETS]{};
    int top_scores[MAX_TOP_HISTORY_QUIETS]{};
    for (int i = 0; i < MAX_TOP_HISTORY_QUIETS; ++i) {
        top_indices[i] = -1;
        top_scores[i] = std::numeric_limits<int>::min();
    }

    for (int i = 0; i < list.size; ++i) {
        const Move move = list.moves[i];
        if (tt_legal_ && move == tt_move_)
            continue;
        if (!legal_fast(pos_, mem_, info_, move))
            continue;

        QuietCandidate candidate{};
        candidate.move = move;
        candidate.history_score = history_.quiet_value_fast(pos_, move);
        candidate.countermove = !move_is_none(countermove) && move == countermove;

        const int candidate_index = candidate_count;
        candidates[candidate_count++] = candidate;

        if (keep_top_history == 0 ||
            candidate.countermove ||
            move == killer1_ ||
            move == killer2_) {
            continue;
        }

        int insert_at = keep_top_history;
        for (int j = 0; j < keep_top_history; ++j) {
            if (candidate.history_score > top_scores[j]) {
                insert_at = j;
                break;
            }
        }

        if (insert_at >= keep_top_history)
            continue;

        for (int j = keep_top_history - 1; j > insert_at; --j) {
            top_scores[j] = top_scores[j - 1];
            top_indices[j] = top_indices[j - 1];
        }

        top_scores[insert_at] = candidate.history_score;
        top_indices[insert_at] = candidate_index;
    }

    for (int i = 0; i < keep_top_history; ++i) {
        if (top_indices[i] >= 0)
            candidates[top_indices[i]].top_history = true;
    }

    const int quiet_limit = std::max(0, quiet_control_.quiet_limit);
    int late_quiet_count = 0;
    for (int i = 0; i < candidate_count; ++i) {
        const QuietCandidate& candidate = candidates[i];
        const bool exempt =
            candidate.countermove ||
            candidate.top_history;
        const bool quiet_in_skip_band =
            !exempt &&
            late_quiet_count >= quiet_limit;
        bool quiet_suppressed = false;
        if (quiet_in_skip_band &&
            candidate.history_score <= quiet_control_.history_floor &&
            !move_gives_check(pos_, mem_, candidate.move)) {
            quiet_suppressed = true;
        }

        add_quiet(
            candidate.move,
            candidate.history_score,
            quiet_in_skip_band,
            quiet_suppressed
        );

        if (!exempt)
            ++late_quiet_count;
    }
}

void MovePicker::add_capture(Move move) noexcept {
    const int see_value = search::see_value_fast(pos_, mem_, move);
    const ScoredEntry entry{
        move,
        score_capture(move, see_value),
        see_value
    };

    if (see_value >= 0) {
        good_caps_[good_size_] = entry;
        ++good_size_;
    } else {
        bad_caps_[bad_size_] = entry;
        ++bad_size_;
    }
}

void MovePicker::add_quiet(
    Move move,
    int history_score,
    bool quiet_in_skip_band,
    bool quiet_suppressed
) noexcept {
    ++quiet_generated_;

    const bool full_score = !quiet_suppressed;
    const int score = full_score ? score_quiet(move) : history_score;
    if (full_score)
        ++quiet_scored_;
    else
        ++quiet_suppressed_;

    if (move == killer1_) {
        killer1_ready_ = true;
        killer1_move_ = move;
        killer1_score_ = score;
        return;
    }

    if (move == killer2_ && move != killer1_move_) {
        killer2_ready_ = true;
        killer2_move_ = move;
        killer2_score_ = score;
        return;
    }

    quiets_[quiet_size_] = ScoredEntry{
        move,
        score,
        0,
        quiet_in_skip_band,
        quiet_suppressed
    };
    ++quiet_size_;
}

int MovePicker::score_capture(Move move, int see_value) const noexcept {
    return mvv_lva_capture_term(pos_, move)
        + history_.capture_ordering_score_fast(pos_, move, depth_, see_value);
}

int MovePicker::score_quiet(Move move) const noexcept {
    return history_.quiet_ordering_score_fast(
        pos_,
        move,
        prev_move_,
        prev2_,
        prev4_,
        prev8_
    );
}

MovePicker::ScoredEntry MovePicker::pick_best_entry(
    ScoredEntry* list,
    int size,
    int& index
) noexcept {
    int best = index;
    for (int i = index + 1; i < size; ++i) {
        if (list[i].score > list[best].score)
            best = i;
    }

    if (best != index) {
        const ScoredEntry tmp = list[index];
        list[index] = list[best];
        list[best] = tmp;
    }

    return list[index++];
}

} // namespace magnus::search
