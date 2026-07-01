use std::sync::OnceLock;

use bulletformat::ChessBoard;

use crate::board::{
    BoardView, PIECE_TYPES, SQUARES, colour_of, pop_lsb, relative_colour, relative_square, type_of,
};

use super::{
    DIAGONALS, FeaturePair, KING_STEPS, KNIGHT_STEPS, ORTHOGONALS, attackers_to, attacks_from,
    leaper_attacks, pawn_attacks, slider_attacks,
};

const STATES: usize = 64;
const STATUS_COUNT: usize = 2 * PIECE_TYPES * SQUARES * STATES;
const VICTIM_CLASSES: usize = 2 * PIECE_TYPES;
const EDGE_COUNT: usize = 90_048;
const RELATION_STATES: usize = 16;
const RELATION_COUNT: usize = 2 * PIECE_TYPES * SQUARES * RELATION_STATES;
const PRESSURE_BINS: usize = 9;
const PRESSURE_COUNT: usize = 2 * PIECE_TYPES * 16 * PRESSURE_BINS;
pub const ATTACK_FEATURE_COUNT: usize = STATUS_COUNT + EDGE_COUNT + RELATION_COUNT + PRESSURE_COUNT;

const STATUS_ENEMY_PAWN: u8 = 1 << 0;
const STATUS_ENEMY_KNIGHT: u8 = 1 << 1;
const STATUS_ENEMY_DIAGONAL: u8 = 1 << 2;
const STATUS_ENEMY_ORTHOGONAL: u8 = 1 << 3;
const STATUS_ENEMY_KING: u8 = 1 << 4;
const STATUS_DEFENDED: u8 = 1 << 5;

struct EdgeIndex {
    base: [[[usize; SQUARES]; PIECE_TYPES]; 2],
}
static EDGE_INDEX: OnceLock<EdgeIndex> = OnceLock::new();

fn empty_attacks(colour: usize, piece_type: usize, square: u8) -> u64 {
    match piece_type {
        0 => pawn_attacks(colour, square),
        1 => leaper_attacks(square, &KNIGHT_STEPS),
        2 => slider_attacks(square, 0, &DIAGONALS),
        3 => slider_attacks(square, 0, &ORTHOGONALS),
        4 => slider_attacks(square, 0, &DIAGONALS) | slider_attacks(square, 0, &ORTHOGONALS),
        5 => leaper_attacks(square, &KING_STEPS),
        _ => 0,
    }
}

fn edge_table() -> &'static EdgeIndex {
    EDGE_INDEX.get_or_init(|| {
        let mut table = EdgeIndex {
            base: [[[0; SQUARES]; PIECE_TYPES]; 2],
        };
        let mut size = 0;
        for colour in 0..2 {
            for piece in 0..PIECE_TYPES {
                for square in 0..SQUARES {
                    table.base[colour][piece][square] = size;
                    size += empty_attacks(colour, piece, square as u8).count_ones() as usize
                        * VICTIM_CLASSES;
                }
            }
        }
        assert_eq!(size, EDGE_COUNT);
        table
    })
}

fn status_index(colour: usize, piece: usize, square: usize, status: u8) -> usize {
    (((colour * PIECE_TYPES + piece) * SQUARES + square) * STATES) + usize::from(status)
}

fn edge_index(ac: usize, at: usize, from: usize, vc: usize, vt: usize, to: usize) -> usize {
    let attacks = empty_attacks(ac, at, from as u8);
    let before = if to == 0 { 0 } else { (1_u64 << to) - 1 };
    let slot = (attacks & before).count_ones() as usize;
    STATUS_COUNT + edge_table().base[ac][at][from] + slot * VICTIM_CLASSES + vc * PIECE_TYPES + vt
}

fn relation_index(colour: usize, piece: usize, square: usize, flags: u8) -> usize {
    STATUS_COUNT
        + EDGE_COUNT
        + (((colour * PIECE_TYPES + piece) * SQUARES + square) * RELATION_STATES)
        + usize::from(flags)
}

fn pressure_index(colour: usize, piece: usize, zone: usize, count: usize) -> usize {
    STATUS_COUNT
        + EDGE_COUNT
        + RELATION_COUNT
        + (((colour * PIECE_TYPES + piece) * 16 + zone) * PRESSURE_BINS)
        + count.min(PRESSURE_BINS - 1)
}

fn tactical_status(view: &BoardView, piece: u8, square: u8) -> u8 {
    let own = colour_of(piece);
    let enemy = own ^ 1;
    let enemy_attackers = attackers_to(view, square, enemy);
    let mut status = 0;
    let mut bb = enemy_attackers;
    while bb != 0 {
        let from = pop_lsb(&mut bb);
        status |= match type_of(view.pieces[usize::from(from)]) {
            0 => STATUS_ENEMY_PAWN,
            1 => STATUS_ENEMY_KNIGHT,
            2 => STATUS_ENEMY_DIAGONAL,
            3 => STATUS_ENEMY_ORTHOGONAL,
            4 => STATUS_ENEMY_DIAGONAL | STATUS_ENEMY_ORTHOGONAL,
            5 => STATUS_ENEMY_KING,
            _ => 0,
        };
    }
    if attackers_to(view, square, own) != 0 {
        status |= STATUS_DEFENDED;
    }
    status
}

fn pinned_to_king(view: &BoardView, piece: u8, square: u8, king: u8) -> bool {
    let own = colour_of(piece);
    if type_of(piece) == 5 || own != colour_of(view.pieces[usize::from(king)]) {
        return false;
    }
    let df = (king & 7) as i8 - (square & 7) as i8;
    let dr = (king >> 3) as i8 - (square >> 3) as i8;
    let aligned = df == 0 || dr == 0 || df.abs() == dr.abs();
    if !aligned {
        return false;
    }
    let sf = df.signum();
    let sr = dr.signum();
    let mut f = (square & 7) as i8 - sf;
    let mut r = (square >> 3) as i8 - sr;
    while (0..8).contains(&f) && (0..8).contains(&r) {
        let sq = (r * 8 + f) as u8;
        if let Some(attacker) = view.piece_on(sq) {
            if colour_of(attacker) == (own ^ 1) {
                let pt = type_of(attacker);
                return if sf == 0 || sr == 0 {
                    pt == 3 || pt == 4
                } else {
                    pt == 2 || pt == 4
                };
            }
            return false;
        }
        f -= sf;
        r -= sr;
    }
    false
}

fn relation_flags(view: &BoardView, piece: u8, square: u8, own_king: u8) -> u8 {
    let own = colour_of(piece);
    let enemy = own ^ 1;
    let attackers = attackers_to(view, square, enemy);
    let defenders = attackers_to(view, square, own);
    let hanging = attackers != 0 && defenders == 0;
    let overloaded = attackers.count_ones() >= 2 && defenders.count_ones() == 1;
    let pinned = pinned_to_king(view, piece, square, own_king);
    let victim_value = [1, 3, 3, 5, 9, 100][type_of(piece)];
    let mut low_value = false;
    let mut bb = attackers;
    while bb != 0 {
        let from = pop_lsb(&mut bb);
        low_value |= [1, 3, 3, 5, 9, 100][type_of(view.pieces[usize::from(from)])] < victim_value;
    }
    hanging as u8 | ((overloaded as u8) << 1) | ((pinned as u8) << 2) | ((low_value as u8) << 3)
}

fn encode_perspective(pos: &ChessBoard, view: &BoardView, ntm: bool) -> Vec<usize> {
    let mut out = Vec::new();
    for (piece, square) in *pos {
        let colour = relative_colour(colour_of(piece), ntm);
        let pt = type_of(piece);
        let rel_sq = relative_square(square, ntm);
        out.push(status_index(
            colour,
            pt,
            rel_sq,
            tactical_status(view, piece, square),
        ));
        let piece_king = if colour_of(piece) == 0 {
            pos.our_ksq()
        } else {
            pos.opp_ksq() ^ 56
        };
        out.push(relation_index(
            colour,
            pt,
            rel_sq,
            relation_flags(view, piece, square, piece_king),
        ));

        let mut targets = attacks_from(piece, square, view.occupied) & view.occupied;
        while targets != 0 {
            let to = pop_lsb(&mut targets);
            let victim = view.pieces[usize::from(to)];
            out.push(edge_index(
                colour,
                pt,
                rel_sq,
                relative_colour(colour_of(victim), ntm),
                type_of(victim),
                relative_square(to, ntm),
            ));
        }
    }

    let enemy_king_absolute = if ntm {
        pos.our_ksq()
    } else {
        pos.opp_ksq() ^ 56
    };
    let enemy_king_relative = if ntm {
        enemy_king_absolute ^ 56
    } else {
        enemy_king_absolute
    };
    let zone =
        (usize::from(enemy_king_relative >> 3) / 2) * 4 + usize::from(enemy_king_relative & 7) / 2;
    let ring = leaper_attacks(enemy_king_absolute, &KING_STEPS) | (1_u64 << enemy_king_absolute);
    let mut counts = [[0_usize; PIECE_TYPES]; 2];
    for (piece, square) in *pos {
        let hits = (attacks_from(piece, square, view.occupied) & ring).count_ones() as usize;
        counts[relative_colour(colour_of(piece), ntm)][type_of(piece)] += hits;
    }
    for (colour, row) in counts.iter().enumerate() {
        for (piece, &count) in row.iter().enumerate() {
            out.push(pressure_index(colour, piece, zone, count));
        }
    }
    out
}

pub fn encode_attack_features(pos: &ChessBoard, view: &BoardView) -> Result<FeaturePair, String> {
    FeaturePair::checked(
        encode_perspective(pos, view, false),
        encode_perspective(pos, view, true),
        ATTACK_FEATURE_COUNT,
    )
}

pub fn decode_attack_feature(index: usize) -> Option<String> {
    if index >= ATTACK_FEATURE_COUNT {
        return None;
    }
    if index < STATUS_COUNT {
        return Some(format!("attack tactical_status raw_index={index}"));
    }
    if index < STATUS_COUNT + EDGE_COUNT {
        return Some(format!(
            "attack occupied_edge raw_index={}",
            index - STATUS_COUNT
        ));
    }
    if index < STATUS_COUNT + EDGE_COUNT + RELATION_COUNT {
        return Some(format!(
            "attack relation_flags raw_index={}",
            index - STATUS_COUNT - EDGE_COUNT
        ));
    }
    Some(format!(
        "attack king_pressure raw_index={}",
        index - STATUS_COUNT - EDGE_COUNT - RELATION_COUNT
    ))
}
