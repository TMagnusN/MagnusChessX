use bulletformat::ChessBoard;

use crate::board::{
    BoardView, PIECE_TYPES, SQUARES, colour_of, relative_colour, relative_square, type_of,
};

use super::FeaturePair;

const KING_BUCKETS: usize = 16;
pub const POSITION_FEATURE_COUNT: usize = KING_BUCKETS * 2 * PIECE_TYPES * SQUARES;

fn king_bucket(square: u8) -> usize {
    (usize::from(square >> 3) / 2) * 4 + usize::from(square & 7) / 2
}

fn index(bucket: usize, colour: usize, piece_type: usize, square: usize) -> usize {
    (((bucket * 2 + colour) * PIECE_TYPES + piece_type) * SQUARES) + square
}

pub fn encode_position_features(
    pos: &ChessBoard,
    _view: &BoardView,
) -> Result<FeaturePair, String> {
    let mut stm = Vec::with_capacity(pos.occ().count_ones() as usize);
    let mut ntm = Vec::with_capacity(stm.capacity());
    let stm_bucket = king_bucket(pos.our_ksq());
    let ntm_bucket = king_bucket(pos.opp_ksq());
    for (piece, square) in *pos {
        let colour = colour_of(piece);
        let piece_type = type_of(piece);
        stm.push(index(
            stm_bucket,
            colour,
            piece_type,
            relative_square(square, false),
        ));
        ntm.push(index(
            ntm_bucket,
            relative_colour(colour, true),
            piece_type,
            relative_square(square, true),
        ));
    }
    FeaturePair::checked(stm, ntm, POSITION_FEATURE_COUNT)
}

pub fn decode_position_feature(index_value: usize) -> Option<String> {
    if index_value >= POSITION_FEATURE_COUNT {
        return None;
    }
    let square = index_value % SQUARES;
    let x = index_value / SQUARES;
    let piece = x % PIECE_TYPES;
    let x = x / PIECE_TYPES;
    let colour = x % 2;
    let bucket = x / 2;
    Some(format!(
        "position king_bucket={bucket} rel_colour={colour} piece={piece} square={square}"
    ))
}
