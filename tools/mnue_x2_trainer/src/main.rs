/*
MNUE-X2 smoke trainer for MagnusChessX Thinking.

The architecture takes the useful design principles observed in modern
dual-accumulator NNUEs, while using an independently defined feature set:

  * i16 king-bucketed piece features, including kings;
  * i8 occupied attack-edge features;
  * both branches sum before pairwise CReLU;
  * u8 sparse activation -> i8 bucketed first layer;
  * float 16 -> 32 -> 1 tail.

Run:

    cargo run --features cuda
*/

use std::{
    fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::OnceLock,
};

use bullet_lib::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    nn::optimiser::{AdamW, AdamWParams},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};
use bulletformat::ChessBoard;

const INPUT_BUCKETS: usize = 10;
const OUTPUT_BUCKETS: usize = 8;
const RELATIVE_COLOURS: usize = 2;
const PIECE_TYPES: usize = 6;
const VICTIM_CLASSES: usize = RELATIVE_COLOURS * PIECE_TYPES;
const SQUARES: usize = 64;

const PIECE_INPUTS: usize = INPUT_BUCKETS * RELATIVE_COLOURS * PIECE_TYPES * SQUARES;
const ATTACK_INPUTS: usize = 90_048;
const INPUT_SIZE: usize = PIECE_INPUTS + ATTACK_INPUTS;

const HIDDEN_SIZE: usize = 768;
const L1_SIZE: usize = 16;
const L2_SIZE: usize = 32;
// 32 occupied piece features plus at most eight occupied attack targets per
// piece gives a conservative hard upper bound of 288.
const MAX_ACTIVE: usize = 320;

const SCALE: i32 = 400;
const QA: i16 = 255;
const THREAT_QUANT: i16 = 64;
const THREAT_RESCALE: i32 = 4;
const L1_QUANT: i16 = 64;

const TRAIN_DATA: &str = "D:/NNUE/data/train_3b_shuffled.data";
const OUTPUT_DIR: &str = "D:/NNUE/runs/mnue_x2_edges_300m";
const NET_DIR: &str = "D:/NNUE/nets/mnue";
const NET_ID: &str = "mnue_x2_edges_300m";

const BATCH_SIZE: usize = 16_384;
const BATCHES_PER_SUPERBATCH: usize = 6104;
const START_SUPERBATCH: usize = 1;
const END_SUPERBATCH: usize = 3;
const TRAIN_DATA_FILES: [&str; 1] = [TRAIN_DATA];

const MNUE_MAGIC: u32 = 0x4555_4E4D;
const MNUE_VERSION: u32 = 3;
const MNUE_ARCH_X2: u32 = 6;
const MNUE_HEADER_BYTES: u32 = 80;
const MNUE_FEATURE_VERSION: u32 = 1;

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

#[derive(Clone, Copy, Debug, Default)]
struct MnueX2Input;

#[derive(Clone, Copy, Debug, Default)]
struct MnueX2Output;

#[derive(Clone, Copy, Debug)]
struct BoardView {
    pieces: [u8; SQUARES],
    occupied: u64,
}

impl BoardView {
    fn from_position(pos: &ChessBoard) -> Self {
        let mut view = Self {
            pieces: [u8::MAX; SQUARES],
            occupied: pos.occ(),
        };
        for (piece, square) in (*pos).into_iter() {
            view.pieces[usize::from(square)] = piece;
        }
        view
    }

    fn piece_on(&self, square: u8) -> u8 {
        let piece = self.pieces[usize::from(square)];
        assert_ne!(piece, u8::MAX);
        piece
    }
}

struct AttackIndex {
    // Base offset before the target slot and victim class dimensions.
    base: [[[usize; SQUARES]; PIECE_TYPES]; RELATIVE_COLOURS],
    size: usize,
}

static ATTACK_INDEX: OnceLock<AttackIndex> = OnceLock::new();

fn attack_index_table() -> &'static AttackIndex {
    ATTACK_INDEX.get_or_init(|| {
        let mut table = AttackIndex {
            base: [[[0; SQUARES]; PIECE_TYPES]; RELATIVE_COLOURS],
            size: 0,
        };

        for relative_colour in 0..RELATIVE_COLOURS {
            for piece_type in 0..PIECE_TYPES {
                for square in 0..SQUARES {
                    table.base[relative_colour][piece_type][square] = table.size;
                    let attacks = empty_board_attacks(relative_colour, piece_type, square as u8);
                    table.size += attacks.count_ones() as usize * VICTIM_CLASSES;
                }
            }
        }

        assert_eq!(table.size, ATTACK_INPUTS);
        table
    })
}

#[inline]
fn file_of(square: u8) -> usize {
    usize::from(square & 7)
}

#[inline]
fn rank_of(square: u8) -> usize {
    usize::from(square >> 3)
}

#[inline]
fn bit(square: u8) -> u64 {
    1_u64 << square
}

#[inline]
fn mirror_for_king(relative_king: u8) -> bool {
    file_of(relative_king) >= 4
}

#[inline]
fn relative_square(square: u8, ntm: bool, mirror: bool) -> usize {
    let vertically_normalised = if ntm { square ^ 56 } else { square };
    usize::from(if mirror {
        vertically_normalised ^ 7
    } else {
        vertically_normalised
    })
}

fn king_bucket10(relative_king: u8) -> usize {
    let file = file_of(relative_king).min(7 - file_of(relative_king));
    match rank_of(relative_king) {
        0 => file,
        1 => 4 + file,
        2 | 3 => 8,
        _ => 9,
    }
}

#[inline]
fn piece_feature_index(
    bucket: usize,
    relative_colour: usize,
    piece_type: usize,
    square: usize,
) -> usize {
    (((bucket * RELATIVE_COLOURS + relative_colour) * PIECE_TYPES + piece_type) * SQUARES) + square
}

fn leaper_attacks(square: u8, steps: &[(i8, i8)]) -> u64 {
    let file = (square & 7) as i8;
    let rank = (square >> 3) as i8;
    let mut attacks = 0_u64;

    for &(df, dr) in steps {
        let target_file = file + df;
        let target_rank = rank + dr;
        if (0..8).contains(&target_file) && (0..8).contains(&target_rank) {
            attacks |= bit((target_rank * 8 + target_file) as u8);
        }
    }
    attacks
}

fn pawn_attacks(colour: usize, square: u8) -> u64 {
    let file = (square & 7) as i8;
    let rank = (square >> 3) as i8;
    if !(1..=6).contains(&rank) {
        return 0;
    }
    let target_rank = rank + if colour == 0 { 1 } else { -1 };
    let mut attacks = 0_u64;

    if !(0..8).contains(&target_rank) {
        return attacks;
    }
    for df in [-1, 1] {
        let target_file = file + df;
        if (0..8).contains(&target_file) {
            attacks |= bit((target_rank * 8 + target_file) as u8);
        }
    }
    attacks
}

fn slider_attacks(square: u8, occupied: u64, directions: &[(i8, i8)]) -> u64 {
    let file = (square & 7) as i8;
    let rank = (square >> 3) as i8;
    let mut attacks = 0_u64;

    for &(df, dr) in directions {
        let mut target_file = file + df;
        let mut target_rank = rank + dr;
        while (0..8).contains(&target_file) && (0..8).contains(&target_rank) {
            let target = (target_rank * 8 + target_file) as u8;
            let target_bit = bit(target);
            attacks |= target_bit;
            if occupied & target_bit != 0 {
                break;
            }
            target_file += df;
            target_rank += dr;
        }
    }
    attacks
}

fn empty_board_attacks(relative_colour: usize, piece_type: usize, square: u8) -> u64 {
    match piece_type {
        0 => pawn_attacks(relative_colour, square),
        1 => leaper_attacks(square, &KNIGHT_STEPS),
        2 => slider_attacks(square, 0, &DIAGONALS),
        3 => slider_attacks(square, 0, &ORTHOGONALS),
        4 => slider_attacks(square, 0, &DIAGONALS) | slider_attacks(square, 0, &ORTHOGONALS),
        5 => leaper_attacks(square, &KING_STEPS),
        _ => unreachable!(),
    }
}

fn occupied_attacks(piece: u8, square: u8, occupied: u64) -> u64 {
    let colour = usize::from((piece & 8) != 0);
    (match piece & 7 {
        0 => pawn_attacks(colour, square),
        1 => leaper_attacks(square, &KNIGHT_STEPS),
        2 => slider_attacks(square, occupied, &DIAGONALS),
        3 => slider_attacks(square, occupied, &ORTHOGONALS),
        4 => {
            slider_attacks(square, occupied, &DIAGONALS)
                | slider_attacks(square, occupied, &ORTHOGONALS)
        }
        5 => leaper_attacks(square, &KING_STEPS),
        _ => unreachable!(),
    }) & occupied
}

fn attack_feature_index(
    attacker_colour: usize,
    attacker_type: usize,
    attacker_square: usize,
    victim_colour: usize,
    victim_type: usize,
    victim_square: usize,
) -> usize {
    let empty_attacks = empty_board_attacks(attacker_colour, attacker_type, attacker_square as u8);
    let victim_bit = 1_u64 << victim_square;
    assert_ne!(empty_attacks & victim_bit, 0);

    let before = if victim_square == 0 {
        0
    } else {
        (1_u64 << victim_square) - 1
    };
    let target_slot = (empty_attacks & before).count_ones() as usize;
    let victim_class = victim_colour * PIECE_TYPES + victim_type;

    PIECE_INPUTS
        + attack_index_table().base[attacker_colour][attacker_type][attacker_square]
        + target_slot * VICTIM_CLASSES
        + victim_class
}

impl SparseInputType for MnueX2Input {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        INPUT_SIZE
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let view = BoardView::from_position(pos);
        let stm_mirror = mirror_for_king(pos.our_ksq());
        let ntm_mirror = mirror_for_king(pos.opp_ksq());
        let stm_bucket = king_bucket10(pos.our_ksq());
        let ntm_bucket = king_bucket10(pos.opp_ksq());

        for (piece, square) in (*pos).into_iter() {
            let piece_type = usize::from(piece & 7);
            let stm_colour = usize::from((piece & 8) != 0);
            let ntm_colour = stm_colour ^ 1;
            let stm_square = relative_square(square, false, stm_mirror);
            let ntm_square = relative_square(square, true, ntm_mirror);

            f(
                piece_feature_index(stm_bucket, stm_colour, piece_type, stm_square),
                piece_feature_index(ntm_bucket, ntm_colour, piece_type, ntm_square),
            );
        }

        for (attacker, from) in (*pos).into_iter() {
            let attacker_type = usize::from(attacker & 7);
            let stm_attacker_colour = usize::from((attacker & 8) != 0);
            let ntm_attacker_colour = stm_attacker_colour ^ 1;
            let stm_from = relative_square(from, false, stm_mirror);
            let ntm_from = relative_square(from, true, ntm_mirror);

            let mut targets = occupied_attacks(attacker, from, view.occupied);
            while targets != 0 {
                let to = targets.trailing_zeros() as u8;
                targets &= targets - 1;

                let victim = view.piece_on(to);
                let victim_type = usize::from(victim & 7);
                let stm_victim_colour = usize::from((victim & 8) != 0);
                let ntm_victim_colour = stm_victim_colour ^ 1;
                let stm_to = relative_square(to, false, stm_mirror);
                let ntm_to = relative_square(to, true, ntm_mirror);

                f(
                    attack_feature_index(
                        stm_attacker_colour,
                        attacker_type,
                        stm_from,
                        stm_victim_colour,
                        victim_type,
                        stm_to,
                    ),
                    attack_feature_index(
                        ntm_attacker_colour,
                        attacker_type,
                        ntm_from,
                        ntm_victim_colour,
                        victim_type,
                        ntm_to,
                    ),
                );
            }
        }
    }

    fn shorthand(&self) -> String {
        "mnue-x2-edges-97728".to_string()
    }

    fn description(&self) -> String {
        "MNUE-X2 king-bucketed pieces plus occupied attack edges".to_string()
    }
}

impl OutputBuckets<ChessBoard> for MnueX2Output {
    const BUCKETS: usize = OUTPUT_BUCKETS;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        match pos.occ().count_ones() {
            0..=5 => 0,
            6..=8 => 1,
            9..=11 => 2,
            12..=14 => 3,
            15..=17 => 4,
            18..=20 => 5,
            21..=24 => 6,
            _ => 7,
        }
    }
}

fn main() {
    assert_eq!(PIECE_INPUTS, 7_680);
    assert_eq!(attack_index_table().size, ATTACK_INPUTS);
    assert_eq!(INPUT_SIZE, 97_728);

    fs::create_dir_all(OUTPUT_DIR).expect("failed to create output directory");
    fs::create_dir_all(NET_DIR).expect("failed to create network directory");

    let piece_weight_count = PIECE_INPUTS * HIDDEN_SIZE;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(MnueX2Input)
        .output_buckets(MnueX2Output)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(move |_, weights| weights[..piece_weight_count].to_vec())
                .round()
                .quantise::<i16>(QA),
            SavedFormat::id("l0w")
                .transform(move |_, weights| weights[piece_weight_count..].to_vec())
                .round()
                .quantise::<i8>(THREAT_QUANT),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w")
                .transpose()
                .round()
                .quantise::<i8>(L1_QUANT),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w").transpose(),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w").transpose(),
            SavedFormat::id("l3b"),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", INPUT_SIZE, HIDDEN_SIZE);
            l0.init_with_effective_input_size(96);

            let l1 = builder.new_affine("l1", HIDDEN_SIZE, OUTPUT_BUCKETS * L1_SIZE);
            let l2 = builder.new_affine("l2", L1_SIZE, OUTPUT_BUCKETS * L2_SIZE);
            let l3 = builder.new_affine("l3", L2_SIZE, OUTPUT_BUCKETS);

            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hidden = stm_hidden.concat(ntm_hidden);
            let hidden = l1.forward(hidden).select(output_buckets).crelu();
            let hidden = l2.forward(hidden).select(output_buckets).crelu();
            l3.forward(hidden).select(output_buckets)
        });

    let bounded = AdamWParams {
        min_weight: -1.98,
        max_weight: 1.98,
        ..Default::default()
    };
    trainer.optimiser.set_params_for_weight("l0w", bounded);
    trainer.optimiser.set_params_for_weight("l1w", bounded);

    let schedule = TrainingSchedule {
        net_id: NET_ID.to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: BATCH_SIZE,
            batches_per_superbatch: BATCHES_PER_SUPERBATCH,
            start_superbatch: START_SUPERBATCH,
            end_superbatch: END_SUPERBATCH,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.75 },
        lr_scheduler: lr::ConstantLR { value: 0.001 },
        save_rate: 1,
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: OUTPUT_DIR,
        batch_queue_size: 64,
    };
    let data_loader = loader::DirectSequentialDataLoader::new(&TRAIN_DATA_FILES);
    trainer.run(&schedule, &settings, &data_loader);

    for superbatch in START_SUPERBATCH..=END_SUPERBATCH {
        let checkpoint = PathBuf::from(OUTPUT_DIR)
            .join(format!("{NET_ID}-{superbatch}"))
            .join("quantised.bin");
        if !checkpoint.exists() {
            continue;
        }

        let output = PathBuf::from(NET_DIR).join(format!("{NET_ID}-{superbatch}.mnue"));
        match write_headered_mnue(&checkpoint, &output) {
            Ok(()) => println!("Wrote {}", output.display()),
            Err(error) => eprintln!("failed to export superbatch {superbatch}: {error}"),
        }
    }
}

fn expected_payload_bytes() -> usize {
    PIECE_INPUTS * HIDDEN_SIZE * size_of::<i16>()
        + ATTACK_INPUTS * HIDDEN_SIZE * size_of::<i8>()
        + HIDDEN_SIZE * size_of::<i16>()
        + OUTPUT_BUCKETS * L1_SIZE * HIDDEN_SIZE * size_of::<i8>()
        + OUTPUT_BUCKETS * L1_SIZE * size_of::<f32>()
        + OUTPUT_BUCKETS * L2_SIZE * L1_SIZE * size_of::<f32>()
        + OUTPUT_BUCKETS * L2_SIZE * size_of::<f32>()
        + OUTPUT_BUCKETS * L2_SIZE * size_of::<f32>()
        + OUTPUT_BUCKETS * size_of::<f32>()
}

fn write_headered_mnue(input_quantised: &Path, output_mnue: &Path) -> io::Result<()> {
    let mut payload = Vec::new();
    fs::File::open(input_quantised)?.read_to_end(&mut payload)?;

    let expected = expected_payload_bytes();
    if payload.len() < expected || payload.len() - expected >= 64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unexpected quantised payload size: got {}, expected {} plus less than 64 padding bytes",
                payload.len(),
                expected
            ),
        ));
    }
    payload.truncate(expected);

    let mut output = fs::File::create(output_mnue)?;
    for field in [
        MNUE_MAGIC,
        MNUE_VERSION,
        MNUE_ARCH_X2,
        MNUE_HEADER_BYTES,
        PIECE_INPUTS as u32,
        ATTACK_INPUTS as u32,
        HIDDEN_SIZE as u32,
        INPUT_BUCKETS as u32,
        OUTPUT_BUCKETS as u32,
        L1_SIZE as u32,
        L2_SIZE as u32,
        SCALE as u32,
        u32::from(QA as u16),
        u32::from(THREAT_QUANT as u16),
        THREAT_RESCALE as u32,
        u32::from(L1_QUANT as u16),
        MNUE_FEATURE_VERSION,
        0,
        0,
        0,
    ] {
        output.write_all(&field.to_le_bytes())?;
    }
    output.write_all(&payload)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feature_pairs(position: &ChessBoard) -> Vec<(usize, usize)> {
        let mut features = Vec::new();
        MnueX2Input.map_features(position, |stm, ntm| {
            features.push((stm, ntm));
        });
        features
    }

    #[test]
    fn attack_table_has_expected_size() {
        assert_eq!(attack_index_table().size, ATTACK_INPUTS);
    }

    #[test]
    fn start_position_features_are_bounded() {
        let position: ChessBoard =
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.5"
                .parse()
                .unwrap();
        let features = feature_pairs(&position);
        assert!(features.len() > 32);
        assert!(features.len() <= MAX_ACTIVE);
        assert!(
            features
                .iter()
                .all(|&(stm, ntm)| { stm < INPUT_SIZE && ntm < INPUT_SIZE })
        );
    }

    #[test]
    fn side_to_move_normalisation_swaps_perspectives() {
        let white: ChessBoard =
            "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1 | 0 | 0.5"
                .parse()
                .unwrap();
        let black: ChessBoard =
            "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R b KQkq - 0 1 | 0 | 0.5"
                .parse()
                .unwrap();

        let mut white_features = feature_pairs(&white);
        let mut black_features = feature_pairs(&black);
        white_features.sort_unstable();
        black_features.sort_unstable();

        let mut swapped: Vec<_> = white_features
            .into_iter()
            .map(|(stm, ntm)| (ntm, stm))
            .collect();
        swapped.sort_unstable();
        assert_eq!(swapped, black_features);
    }
}
