use bulletformat::ChessBoard;

use crate::{board::type_of, config::OUTPUT_BUCKETS};

/// Material units use conventional piece values P=1, N/B=3, R=5, Q=9.
/// The eleven inclusive upper bounds were selected from quantiles of one
/// million accepted records from the current shuffled training corpus.
pub const MATERIAL_BUCKET_MAX: [u8; OUTPUT_BUCKETS - 1] =
    [7, 11, 13, 17, 21, 27, 33, 41, 50, 59, 69];

pub fn material_units(pos: &ChessBoard) -> u8 {
    (*pos)
        .into_iter()
        .map(|(piece, _)| match type_of(piece) {
            0 => 1,
            1 | 2 => 3,
            3 => 5,
            4 => 9,
            _ => 0,
        })
        .sum()
}

pub fn material_bucket(pos: &ChessBoard) -> u8 {
    let units = material_units(pos);
    MATERIAL_BUCKET_MAX.partition_point(|&upper| units > upper) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_is_monotonic_and_reaches_all_buckets() {
        let mut previous = 0;
        let mut reached = [false; OUTPUT_BUCKETS];
        for units in 0..=78 {
            let bucket = MATERIAL_BUCKET_MAX.partition_point(|&upper| units > upper);
            assert!(bucket >= previous && bucket < OUTPUT_BUCKETS);
            reached[bucket] = true;
            previous = bucket;
        }
        assert!(reached.into_iter().all(|x| x));
    }
}
