use bulletformat::ChessBoard;

pub const PIECE_TYPES: usize = 6;
pub const SQUARES: usize = 64;
pub const WHITE: usize = 0;
pub const BLACK: usize = 1;

#[derive(Clone, Debug)]
pub struct BoardView {
    pub pieces: [u8; SQUARES],
    pub by_colour: [u64; 2],
    pub by_type: [[u64; PIECE_TYPES]; 2],
    pub occupied: u64,
}

impl BoardView {
    pub fn new(pos: &ChessBoard) -> Result<Self, String> {
        validate_position(pos)?;
        let mut view = Self {
            pieces: [u8::MAX; SQUARES],
            by_colour: [0; 2],
            by_type: [[0; PIECE_TYPES]; 2],
            occupied: pos.occ(),
        };
        for (piece, square) in *pos {
            let colour = colour_of(piece);
            let piece_type = type_of(piece);
            let b = bit(square);
            view.pieces[usize::from(square)] = piece;
            view.by_colour[colour] |= b;
            view.by_type[colour][piece_type] |= b;
        }
        Ok(view)
    }

    pub fn piece_on(&self, square: u8) -> Option<u8> {
        let piece = self.pieces[usize::from(square)];
        (piece != u8::MAX).then_some(piece)
    }
}

pub fn decode_record(bytes: &[u8; 32]) -> ChessBoard {
    let mut pcs = [0_u8; 16];
    pcs.copy_from_slice(&bytes[8..24]);
    ChessBoard {
        occ: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
        pcs,
        score: i16::from_le_bytes(bytes[24..26].try_into().unwrap()),
        result: bytes[26],
        ksq: bytes[27],
        opp_ksq: bytes[28],
        extra: bytes[29..32].try_into().unwrap(),
    }
}

pub fn validate_position(pos: &ChessBoard) -> Result<(), String> {
    let count = pos.occ().count_ones() as usize;
    if !(2..=32).contains(&count) {
        return Err(format!("occupied count {count} is outside 2..=32"));
    }
    if pos.result > 2 {
        return Err(format!("result {} is outside 0..=2", pos.result));
    }
    if pos.our_ksq() >= 64 || pos.opp_ksq() >= 64 {
        return Err("king square outside board".to_string());
    }

    let mut kings = [0_usize; 2];
    let mut seen = 0_usize;
    for (piece, square) in *pos {
        if square >= 64 || type_of_raw(piece) >= PIECE_TYPES || piece & 0b0110_0000 != 0 {
            return Err(format!("invalid piece nibble {piece} or square {square}"));
        }
        if type_of(piece) == 5 {
            kings[colour_of(piece)] += 1;
        }
        seen += 1;
    }
    if seen != count || kings != [1, 1] {
        return Err(format!(
            "piece count/kings invalid: seen={seen}, kings={kings:?}"
        ));
    }
    Ok(())
}

#[inline]
pub const fn type_of_raw(piece: u8) -> usize {
    (piece & 7) as usize
}

#[inline]
pub const fn type_of(piece: u8) -> usize {
    type_of_raw(piece)
}

#[inline]
pub const fn colour_of(piece: u8) -> usize {
    ((piece & 8) != 0) as usize
}

#[inline]
pub const fn bit(square: u8) -> u64 {
    1_u64 << square
}

#[inline]
pub const fn file_of(square: u8) -> usize {
    (square & 7) as usize
}

#[inline]
pub const fn rank_of(square: u8) -> usize {
    (square >> 3) as usize
}

#[inline]
pub const fn relative_square(square: u8, ntm: bool) -> usize {
    (if ntm { square ^ 56 } else { square }) as usize
}

#[inline]
pub const fn relative_colour(colour: usize, ntm: bool) -> usize {
    colour ^ ntm as usize
}

pub fn pop_lsb(bb: &mut u64) -> u8 {
    let square = bb.trailing_zeros() as u8;
    *bb &= *bb - 1;
    square
}
