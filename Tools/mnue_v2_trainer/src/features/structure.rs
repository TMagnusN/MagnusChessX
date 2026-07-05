use bulletformat::ChessBoard;

use crate::board::{
    BoardView, SQUARES, bit, colour_of, file_of, rank_of, relative_colour, relative_square, type_of,
};

use super::{FeaturePair, pawn_attacks};

const PAWN_STATES: usize = 256;
const PAWN_COUNT: usize = 2 * SQUARES * PAWN_STATES;
const FILE_COUNT: usize = 8 * 4;
const ISLAND_COUNT: usize = 2 * 9;
const CENTER_COUNT: usize = 9;
const SHELTER_COUNT: usize = 2 * 16 * 9;
const OUTPOST_COUNT: usize = 2 * 2 * SQUARES * 4;
const COMPLEX_COUNT: usize = 2 * 9 * 9;
const BLOCKER_COUNT: usize = 2 * SQUARES * 3;
pub const STRUCTURE_FEATURE_COUNT: usize = PAWN_COUNT
    + FILE_COUNT
    + ISLAND_COUNT
    + CENTER_COUNT
    + SHELTER_COUNT
    + OUTPOST_COUNT
    + COMPLEX_COUNT
    + BLOCKER_COUNT;

fn pawn_index(colour: usize, square: usize, flags: u8) -> usize {
    ((colour * SQUARES + square) * PAWN_STATES) + usize::from(flags)
}
fn file_index(file: usize, state: usize) -> usize {
    PAWN_COUNT + file * 4 + state
}
fn island_index(colour: usize, count: usize) -> usize {
    PAWN_COUNT + FILE_COUNT + colour * 9 + count.min(8)
}
fn center_index(count: usize) -> usize {
    PAWN_COUNT + FILE_COUNT + ISLAND_COUNT + count.min(8)
}
fn shelter_index(colour: usize, zone: usize, count: usize) -> usize {
    PAWN_COUNT
        + FILE_COUNT
        + ISLAND_COUNT
        + CENTER_COUNT
        + ((colour * 16 + zone) * 9)
        + count.min(8)
}
fn outpost_index(colour: usize, piece: usize, square: usize, flags: usize) -> usize {
    PAWN_COUNT
        + FILE_COUNT
        + ISLAND_COUNT
        + CENTER_COUNT
        + SHELTER_COUNT
        + (((colour * 2 + piece) * SQUARES + square) * 4)
        + flags
}
fn complex_index(colour: usize, light: usize, dark: usize) -> usize {
    PAWN_COUNT
        + FILE_COUNT
        + ISLAND_COUNT
        + CENTER_COUNT
        + SHELTER_COUNT
        + OUTPOST_COUNT
        + ((colour * 9 + light.min(8)) * 9)
        + dark.min(8)
}
fn blocker_index(colour: usize, square: usize, state: usize) -> usize {
    PAWN_COUNT
        + FILE_COUNT
        + ISLAND_COUNT
        + CENTER_COUNT
        + SHELTER_COUNT
        + OUTPOST_COUNT
        + COMPLEX_COUNT
        + ((colour * SQUARES + square) * 3)
        + state
}

fn files_mask(pawns: u64) -> u8 {
    let mut mask = 0;
    for file in 0..8 {
        if pawns & (0x0101_0101_0101_0101_u64 << file) != 0 {
            mask |= 1 << file;
        }
    }
    mask
}

fn islands(mask: u8) -> usize {
    (mask & !(mask << 1)).count_ones() as usize
}

fn ahead_mask(colour: usize, square: u8, adjacent: bool) -> u64 {
    let file = file_of(square) as i8;
    let rank = rank_of(square) as i8;
    let mut mask = 0;
    let files: &[i8] = if adjacent { &[-1, 0, 1] } else { &[0] };
    for &df in files {
        let f = file + df;
        if !(0..8).contains(&f) {
            continue;
        }
        let ranks: Box<dyn Iterator<Item = i8>> = if colour == 0 {
            Box::new((rank + 1)..8)
        } else {
            Box::new((0..rank).rev())
        };
        for r in ranks {
            mask |= 1_u64 << (r * 8 + f);
        }
    }
    mask
}

fn pawn_flags(view: &BoardView, colour: usize, square: u8) -> u8 {
    let own = view.by_type[colour][0];
    let enemy = view.by_type[colour ^ 1][0];
    let file = file_of(square);
    let mut adjacent_files = 0_u64;
    if file > 0 {
        adjacent_files |= 0x0101_0101_0101_0101_u64 << (file - 1);
    }
    if file < 7 {
        adjacent_files |= 0x0101_0101_0101_0101_u64 << (file + 1);
    }
    let same_file = 0x0101_0101_0101_0101_u64 << file;
    let isolated = own & adjacent_files == 0;
    let doubled = (own & same_file).count_ones() > 1;
    let passed = enemy & ahead_mask(colour, square, true) == 0;
    let enemy_ahead = (enemy & ahead_mask(colour, square, true)).count_ones();
    let own_support = (own & ahead_mask(colour, square, true) & adjacent_files).count_ones();
    let candidate = !passed && own_support >= enemy_ahead;
    let connected = own & pawn_attacks(colour ^ 1, square) != 0;
    let chain = own & pawn_attacks(colour, square) != 0;
    let next = if colour == 0 {
        square.checked_add(8)
    } else {
        square.checked_sub(8)
    };
    let blocked = next.is_some_and(|sq| view.occupied & bit(sq) != 0);
    let backward = !isolated
        && !connected
        && !chain
        && !passed
        && next.is_some_and(|sq| pawn_attacks(colour ^ 1, sq) & enemy != 0);
    isolated as u8
        | ((doubled as u8) << 1)
        | ((passed as u8) << 2)
        | ((candidate as u8) << 3)
        | ((connected as u8) << 4)
        | ((chain as u8) << 5)
        | ((backward as u8) << 6)
        | ((blocked as u8) << 7)
}

fn encode_perspective(pos: &ChessBoard, view: &BoardView, ntm: bool) -> Vec<usize> {
    let mut out = Vec::new();
    let pawns = [view.by_type[0][0], view.by_type[1][0]];
    for (colour, &pawn_bb) in pawns.iter().enumerate() {
        let mut bb = pawn_bb;
        while bb != 0 {
            let square = bb.trailing_zeros() as u8;
            bb &= bb - 1;
            out.push(pawn_index(
                relative_colour(colour, ntm),
                relative_square(square, ntm),
                pawn_flags(view, colour, square),
            ));
        }
    }

    let rel_pawns = if ntm {
        [pawns[1].swap_bytes(), pawns[0].swap_bytes()]
    } else {
        pawns
    };
    for file in 0..8 {
        let mask = 0x0101_0101_0101_0101_u64 << file;
        let state =
            usize::from(rel_pawns[0] & mask != 0) | (usize::from(rel_pawns[1] & mask != 0) << 1);
        out.push(file_index(file, state));
    }
    out.push(island_index(0, islands(files_mask(rel_pawns[0]))));
    out.push(island_index(1, islands(files_mask(rel_pawns[1]))));

    let centre = [27_u8, 28, 35, 36, 19, 20, 43, 44]
        .into_iter()
        .filter(|&sq| (rel_pawns[0] | rel_pawns[1]) & bit(sq) != 0)
        .count();
    out.push(center_index(centre));

    for (colour, &relative_pawns) in rel_pawns.iter().enumerate() {
        let king = if colour == 0 {
            if ntm { pos.opp_ksq() } else { pos.our_ksq() }
        } else if ntm {
            pos.our_ksq()
        } else {
            pos.opp_ksq()
        };
        let zone = (rank_of(king) / 2) * 4 + file_of(king) / 2;
        let kf = file_of(king) as i8;
        let kr = rank_of(king) as i8;
        let mut shelter = 0;
        for df in -1..=1 {
            for dr in 1..=3 {
                let f = kf + df;
                let r = kr + if colour == 0 { dr } else { -dr };
                if (0..8).contains(&f)
                    && (0..8).contains(&r)
                    && relative_pawns & bit((r * 8 + f) as u8) != 0
                {
                    shelter += 1;
                }
            }
        }
        out.push(shelter_index(colour, zone, shelter));
    }

    for (piece, square) in *pos {
        let pt = type_of(piece);
        if !(1..=2).contains(&pt) {
            continue;
        }
        let colour = colour_of(piece);
        let rel_colour = relative_colour(colour, ntm);
        let rel_sq = relative_square(square, ntm);
        let enemy_pawns = view.by_type[colour ^ 1][0];
        let own_pawns = view.by_type[colour][0];
        let cannot_be_chased = enemy_pawns & ahead_mask(colour ^ 1, square, true) == 0;
        let pawn_supported = pawn_attacks(colour ^ 1, square) & own_pawns != 0;
        out.push(outpost_index(
            rel_colour,
            pt - 1,
            rel_sq,
            usize::from(cannot_be_chased) | (usize::from(pawn_supported) << 1),
        ));
    }

    for (colour, &relative_pawns) in rel_pawns.iter().enumerate() {
        let mut light = 0;
        let mut dark = 0;
        let mut bb = relative_pawns;
        while bb != 0 {
            let sq = bb.trailing_zeros() as u8;
            bb &= bb - 1;
            if (file_of(sq) + rank_of(sq)).is_multiple_of(2) {
                light += 1;
            } else {
                dark += 1;
            }
        }
        out.push(complex_index(colour, light, dark));
    }

    for (colour, &pawn_bb) in pawns.iter().enumerate() {
        let mut bb = pawn_bb;
        while bb != 0 {
            let square = bb.trailing_zeros() as u8;
            bb &= bb - 1;
            if pawn_flags(view, colour, square) & (1 << 2) == 0 {
                continue;
            }
            let next = if colour == 0 {
                square.checked_add(8)
            } else {
                square.checked_sub(8)
            };
            let state = next
                .and_then(|sq| view.piece_on(sq))
                .map_or(0, |p| 1 + usize::from(colour_of(p) != colour));
            out.push(blocker_index(
                relative_colour(colour, ntm),
                relative_square(square, ntm),
                state,
            ));
        }
    }
    out
}

pub fn encode_structure_features(
    pos: &ChessBoard,
    view: &BoardView,
) -> Result<FeaturePair, String> {
    FeaturePair::checked(
        encode_perspective(pos, view, false),
        encode_perspective(pos, view, true),
        STRUCTURE_FEATURE_COUNT,
    )
}

pub fn decode_structure_feature(index: usize) -> Option<String> {
    if index >= STRUCTURE_FEATURE_COUNT {
        return None;
    }
    let family = if index < PAWN_COUNT {
        "pawn_state"
    } else if index < PAWN_COUNT + FILE_COUNT {
        "file_state"
    } else if index < PAWN_COUNT + FILE_COUNT + ISLAND_COUNT {
        "pawn_islands"
    } else if index < PAWN_COUNT + FILE_COUNT + ISLAND_COUNT + CENTER_COUNT {
        "centre_openness"
    } else if index < PAWN_COUNT + FILE_COUNT + ISLAND_COUNT + CENTER_COUNT + SHELTER_COUNT {
        "king_shelter"
    } else if index
        < PAWN_COUNT + FILE_COUNT + ISLAND_COUNT + CENTER_COUNT + SHELTER_COUNT + OUTPOST_COUNT
    {
        "outpost"
    } else if index < STRUCTURE_FEATURE_COUNT - BLOCKER_COUNT {
        "colour_complex"
    } else {
        "passed_blocker"
    };
    Some(format!("structure {family} raw_index={index}"))
}
