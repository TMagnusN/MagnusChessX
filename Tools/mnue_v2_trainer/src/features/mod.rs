mod attack;
mod position;
mod structure;

pub use attack::{ATTACK_FEATURE_COUNT, decode_attack_feature, encode_attack_features};
pub use position::{POSITION_FEATURE_COUNT, decode_position_feature, encode_position_features};
pub use structure::{STRUCTURE_FEATURE_COUNT, decode_structure_feature, encode_structure_features};

use std::collections::BTreeSet;

use bulletformat::ChessBoard;

use crate::board::BoardView;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeaturePair {
    pub stm: Vec<usize>,
    pub ntm: Vec<usize>,
}

impl FeaturePair {
    pub fn checked(
        mut stm: Vec<usize>,
        mut ntm: Vec<usize>,
        vocabulary: usize,
    ) -> Result<Self, String> {
        // Sparse rows are binary features. Duplicate semantic emissions are
        // deduplicated deterministically rather than counted more than once.
        stm = BTreeSet::from_iter(stm).into_iter().collect();
        ntm = BTreeSet::from_iter(ntm).into_iter().collect();
        if let Some(index) = stm.iter().chain(&ntm).copied().find(|&i| i >= vocabulary) {
            return Err(format!(
                "feature index {index} is outside vocabulary {vocabulary}"
            ));
        }
        Ok(Self { stm, ntm })
    }
}

#[derive(Clone, Debug)]
pub struct EncodedPosition {
    pub position: FeaturePair,
    pub attack: FeaturePair,
    pub structure: FeaturePair,
}

pub fn encode_all(pos: &ChessBoard) -> Result<EncodedPosition, String> {
    let view = BoardView::new(pos)?;
    Ok(EncodedPosition {
        position: encode_position_features(pos, &view)?,
        attack: encode_attack_features(pos, &view)?,
        structure: encode_structure_features(pos, &view)?,
    })
}

const KNIGHT_STEPS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];
const KING_STEPS: [(i8, i8); 8] = [
    (1, 1),
    (1, 0),
    (1, -1),
    (0, 1),
    (0, -1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
];
const DIAGONALS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ORTHOGONALS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

pub(crate) fn leaper_attacks(square: u8, steps: &[(i8, i8)]) -> u64 {
    let file = (square & 7) as i8;
    let rank = (square >> 3) as i8;
    let mut attacks = 0_u64;
    for &(df, dr) in steps {
        let f = file + df;
        let r = rank + dr;
        if (0..8).contains(&f) && (0..8).contains(&r) {
            attacks |= 1_u64 << (r * 8 + f);
        }
    }
    attacks
}

pub(crate) fn pawn_attacks(colour: usize, square: u8) -> u64 {
    let file = (square & 7) as i8;
    let rank = (square >> 3) as i8;
    if !(1..=6).contains(&rank) {
        return 0;
    }
    let target_rank = rank + if colour == 0 { 1 } else { -1 };
    if !(0..8).contains(&target_rank) {
        return 0;
    }
    let mut attacks = 0;
    for df in [-1, 1] {
        let f = file + df;
        if (0..8).contains(&f) {
            attacks |= 1_u64 << (target_rank * 8 + f);
        }
    }
    attacks
}

pub(crate) fn slider_attacks(square: u8, occupied: u64, directions: &[(i8, i8)]) -> u64 {
    let file = (square & 7) as i8;
    let rank = (square >> 3) as i8;
    let mut attacks = 0;
    for &(df, dr) in directions {
        let mut f = file + df;
        let mut r = rank + dr;
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let target = (r * 8 + f) as u8;
            let b = 1_u64 << target;
            attacks |= b;
            if occupied & b != 0 {
                break;
            }
            f += df;
            r += dr;
        }
    }
    attacks
}

pub(crate) fn attacks_from(piece: u8, square: u8, occupied: u64) -> u64 {
    let colour = usize::from((piece & 8) != 0);
    match piece & 7 {
        0 => pawn_attacks(colour, square),
        1 => leaper_attacks(square, &KNIGHT_STEPS),
        2 => slider_attacks(square, occupied, &DIAGONALS),
        3 => slider_attacks(square, occupied, &ORTHOGONALS),
        4 => {
            slider_attacks(square, occupied, &DIAGONALS)
                | slider_attacks(square, occupied, &ORTHOGONALS)
        }
        5 => leaper_attacks(square, &KING_STEPS),
        _ => 0,
    }
}

pub(crate) fn attackers_to(view: &BoardView, square: u8, colour: usize) -> u64 {
    let target = 1_u64 << square;
    let mut attackers = 0_u64;
    let mut pieces = view.by_colour[colour];
    while pieces != 0 {
        let from = pieces.trailing_zeros() as u8;
        pieces &= pieces - 1;
        let piece = view.pieces[usize::from(from)];
        if attacks_from(piece, from, view.occupied) & target != 0 {
            attackers |= 1_u64 << from;
        }
    }
    attackers
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(mut x: Vec<usize>) -> Vec<usize> {
        x.sort_unstable();
        x
    }

    #[test]
    fn all_families_are_bounded_and_deterministic() {
        let position: ChessBoard =
            "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1 | 0 | 0.5"
                .parse()
                .unwrap();
        let first = encode_all(&position).unwrap();
        let second = encode_all(&position).unwrap();
        assert_eq!(first.position, second.position);
        assert_eq!(first.attack, second.attack);
        assert_eq!(first.structure, second.structure);
        assert!(
            first
                .position
                .stm
                .iter()
                .all(|&x| x < POSITION_FEATURE_COUNT)
        );
        assert!(first.attack.stm.iter().all(|&x| x < ATTACK_FEATURE_COUNT));
        assert!(
            first
                .structure
                .stm
                .iter()
                .all(|&x| x < STRUCTURE_FEATURE_COUNT)
        );
    }

    #[test]
    fn side_to_move_flip_swaps_all_perspectives() {
        let white: ChessBoard =
            "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1 | 0 | 0.5"
                .parse()
                .unwrap();
        let black: ChessBoard =
            "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R b KQkq - 0 1 | 0 | 0.5"
                .parse()
                .unwrap();
        let w = encode_all(&white).unwrap();
        let b = encode_all(&black).unwrap();
        assert_eq!(sorted(w.position.stm), sorted(b.position.ntm));
        assert_eq!(sorted(w.position.ntm), sorted(b.position.stm));
        assert_eq!(sorted(w.attack.stm), sorted(b.attack.ntm));
        assert_eq!(sorted(w.attack.ntm), sorted(b.attack.stm));
        assert_eq!(sorted(w.structure.stm), sorted(b.structure.ntm));
        assert_eq!(sorted(w.structure.ntm), sorted(b.structure.stm));
    }
}
