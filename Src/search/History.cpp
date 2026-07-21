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

#include "History.h"

#include <algorithm>

namespace magnus::search {

/*
 * History Heuristic Implementation
 * depth_class() — maps depth to Shallow/MediumLow/MediumHigh/Deep categories
 * classify_see() — classifies SEE values (LossBig/LossSmall/Equal/GainSmall/GainBig/Promo/Check)
 * history_bonus/penalty() — computes history weight updates (bonus=d², penalty=4d²)
 * see_immediate_term() — SEE immediate scoring term (scaled by Weak/Medium/Strong presets)
 */
DepthClass depth_class(int depth) noexcept {
    if (depth <= 3) return DepthClass::Shallow;
    if (depth <= 6) return DepthClass::MediumLow;
    if (depth <= 10) return DepthClass::MediumHigh;
    return DepthClass::Deep;
}

SeeClass classify_see(int see_value, bool gives_check, bool is_promotion) noexcept {
    if (is_promotion) return SeeClass::Promo;
    if (gives_check) return SeeClass::Check;
    if (see_value <= -100) return SeeClass::LossBig;
    if (see_value < 0) return SeeClass::LossSmall;
    if (see_value == 0) return SeeClass::Equal;
    if (see_value < 200) return SeeClass::GainSmall;
    return SeeClass::GainBig;
}

SeeClass classify_see_bias(int see_value) noexcept {
    if (see_value < MAGNUS_SEE_BIAS_BAD_THRESHOLD)
        return SeeClass::LossSmall;
    if (see_value > MAGNUS_SEE_BIAS_GOOD_BIG_THRESHOLD)
        return SeeClass::GainBig;
    if (see_value > MAGNUS_SEE_BIAS_EQ_THRESHOLD)
        return SeeClass::GainSmall;
    return SeeClass::Equal;
}

int history_bonus(int depth) noexcept {
    const int d = std::max(1, depth);
    return d * d;
}

int history_penalty(int depth) noexcept {
    return history_bonus(depth) * 4;
}

int see_immediate_term(int see_value, SeeScalePreset preset) noexcept {
    switch (preset) {
        case SeeScalePreset::Weak:
            return std::clamp(see_value / 4, -75, 75);
        case SeeScalePreset::Medium:
            return std::clamp(see_value / 2, -150, 150);
        case SeeScalePreset::Strong:
            return std::clamp(see_value, -300, 300);
        default:
            return std::clamp(see_value / 2, -150, 150);
    }
}

void HistoryTables::clear() noexcept {
    killers = {};
    quiet = {};
    capture = {};
    countermove = {};
    for (auto& table : continuation)
        table = {};
    see_bias = {};
    pawn_history = {};
}

} // namespace magnus::search
