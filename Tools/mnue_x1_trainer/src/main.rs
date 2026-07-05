/*
MNUE-X1 smoke trainer for MagnusChessX Thinking.

Run from this directory:

    cargo run --features cuda -- --data PATH
*/

use std::{
    env, fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process,
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

const INPUT_BUCKETS: usize = 16;
const OUTPUT_BUCKETS: usize = 8;
const RELATIVE_COLOURS: usize = 2;
const NON_KING_PIECES: usize = 5;
const PIECE_TYPES: usize = 6;
const SQUARES: usize = 64;
const TACTICAL_STATES: usize = 64;

const PIECE_INPUTS: usize = INPUT_BUCKETS * RELATIVE_COLOURS * NON_KING_PIECES * SQUARES;
const TACTICAL_INPUTS: usize = RELATIVE_COLOURS * PIECE_TYPES * SQUARES * TACTICAL_STATES;
const INPUT_SIZE: usize = PIECE_INPUTS + TACTICAL_INPUTS;

const HIDDEN_SIZE: usize = 768;
const L1_SIZE: usize = 16;
const L2_SIZE: usize = 32;

const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

const DEFAULT_OUTPUT_DIR: &str = "runs/mnue_x1_768_300m";
const DEFAULT_NET_DIR: &str = "nets/mnue";
const DEFAULT_NET_ID: &str = "mnue_x1_768_300m";

const BATCH_SIZE: usize = 16_384;
const BATCHES_PER_SUPERBATCH: usize = 6104;
const START_SUPERBATCH: usize = 1;
const END_SUPERBATCH: usize = 3;

const MNUE_MAGIC: u32 = 0x4555_4E4D;
const MNUE_VERSION: u32 = 2;
const MNUE_ARCH_X1: u32 = 5;
const MNUE_HEADER_BYTES: u32 = 64;
const MNUE_FEATURE_VERSION: u32 = 1;
const MNUE_FLAGS: u32 = 0;

const STATUS_ENEMY_PAWN: u8 = 1 << 0;
const STATUS_ENEMY_KNIGHT: u8 = 1 << 1;
const STATUS_ENEMY_DIAGONAL: u8 = 1 << 2;
const STATUS_ENEMY_ORTHOGONAL: u8 = 1 << 3;
const STATUS_ENEMY_KING: u8 = 1 << 4;
const STATUS_DEFENDED: u8 = 1 << 5;

const DIAGONALS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ORTHOGONALS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
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

#[derive(Clone, Debug)]
struct Config {
    data: Vec<String>,
    output_dir: String,
    net_dir: String,
    net_id: String,
}

impl Config {
    fn from_args() -> Self {
        let mut data = Vec::new();
        let mut output_dir = DEFAULT_OUTPUT_DIR.to_string();
        let mut net_dir = DEFAULT_NET_DIR.to_string();
        let mut net_id = DEFAULT_NET_ID.to_string();

        let args = env::args().skip(1).collect::<Vec<_>>();
        let mut idx = 0;
        while idx < args.len() {
            let arg = &args[idx];
            if arg == "-h" || arg == "--help" {
                print_usage();
                process::exit(0);
            }

            let Some(key_value) = arg.strip_prefix("--") else {
                die(format!("unexpected positional argument: {arg}"));
            };
            let (key, value) = if let Some((key, value)) = key_value.split_once('=') {
                (key, value.to_string())
            } else {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    die(format!("missing value for --{key_value}"));
                };
                (key_value, value.clone())
            };

            match key {
                "data" | "train-data" => data.push(parse_path_arg(key, &value)),
                "output-dir" => output_dir = parse_path_arg(key, &value),
                "net-dir" => net_dir = parse_path_arg(key, &value),
                "net-id" => net_id = parse_path_arg(key, &value),
                _ => die(format!("unknown option --{key}")),
            }

            idx += 1;
        }

        if data.is_empty() {
            die("--data must be supplied at least once");
        }

        Self {
            data,
            output_dir,
            net_dir,
            net_id,
        }
    }
}

fn parse_path_arg(option: &str, value: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        die(format!("--{option} must not be empty"));
    }
    value.to_string()
}

fn print_usage() {
    println!(
        "Usage: cargo run --manifest-path Tools/mnue_x1_trainer/Cargo.toml --release --features cuda -- --data PATH [options]\n\
         \n\
         Options:\n\
           --data PATH        Required training .data path; repeatable\n\
           --output-dir PATH  Checkpoint/output directory. Default: {DEFAULT_OUTPUT_DIR}\n\
           --net-dir PATH     Exported .mnue directory. Default: {DEFAULT_NET_DIR}\n\
           --net-id NAME      Net id and checkpoint prefix. Default: {DEFAULT_NET_ID}"
    );
}

fn die(message: impl std::fmt::Display) -> ! {
    eprintln!("error: {message}");
    eprintln!("try --help for usage");
    process::exit(2);
}

#[derive(Clone, Copy, Debug, Default)]
struct MnueX1Input;

#[derive(Clone, Copy, Debug, Default)]
struct MnueX1Output;

#[derive(Clone, Copy, Debug, Default)]
struct BoardView {
    colours: [u64; 2],
    pieces: [[u64; PIECE_TYPES]; 2],
    occupied: u64,
}

impl BoardView {
    fn from_position(pos: &ChessBoard) -> Self {
        let mut view = Self {
            occupied: pos.occ(),
            ..Self::default()
        };

        for (piece, square) in (*pos).into_iter() {
            let colour = usize::from((piece & 8) != 0);
            let piece_type = usize::from(piece & 7);
            assert!(piece_type < PIECE_TYPES);

            let bit = 1_u64 << square;
            view.colours[colour] |= bit;
            view.pieces[colour][piece_type] |= bit;
        }

        view
    }
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
fn king_zone16(square: u8) -> usize {
    (rank_of(square) / 2) * 4 + file_of(square) / 2
}

#[inline]
fn piece_feature_index(
    bucket: usize,
    relative_colour: usize,
    piece_type: usize,
    relative_square: usize,
) -> usize {
    (((bucket * RELATIVE_COLOURS + relative_colour) * NON_KING_PIECES + piece_type) * SQUARES)
        + relative_square
}

#[inline]
fn tactical_feature_index(
    relative_colour: usize,
    piece_type: usize,
    relative_square: usize,
    status: u8,
) -> usize {
    let relative_class = relative_colour * PIECE_TYPES + piece_type;
    PIECE_INPUTS
        + ((relative_class * SQUARES + relative_square) * TACTICAL_STATES)
        + usize::from(status)
}

#[inline]
fn bit(square: u8) -> u64 {
    1_u64 << square
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

fn king_attacks(square: u8) -> u64 {
    let file = (square & 7) as i8;
    let rank = (square >> 3) as i8;
    let mut attacks = 0_u64;

    for df in -1..=1 {
        for dr in -1..=1 {
            if df == 0 && dr == 0 {
                continue;
            }

            let target_file = file + df;
            let target_rank = rank + dr;
            if (0..8).contains(&target_file) && (0..8).contains(&target_rank) {
                attacks |= bit((target_rank * 8 + target_file) as u8);
            }
        }
    }

    attacks
}

fn pawn_attacks_from(colour: usize, square: u8) -> u64 {
    let file = (square & 7) as i8;
    let rank = (square >> 3) as i8;
    let rank_delta = if colour == 0 { 1 } else { -1 };
    let target_rank = rank + rank_delta;
    let mut attacks = 0_u64;

    if !(0..8).contains(&target_rank) {
        return attacks;
    }

    for file_delta in [-1, 1] {
        let target_file = file + file_delta;
        if (0..8).contains(&target_file) {
            attacks |= bit((target_rank * 8 + target_file) as u8);
        }
    }

    attacks
}

fn pawn_attackers_to(colour: usize, square: u8) -> u64 {
    pawn_attacks_from(colour ^ 1, square)
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

#[inline]
fn pawn_attackers(view: &BoardView, square: u8, by: usize) -> u64 {
    pawn_attackers_to(by, square) & view.pieces[by][0]
}

#[inline]
fn knight_attackers(view: &BoardView, square: u8, by: usize) -> u64 {
    leaper_attacks(square, &KNIGHT_STEPS) & view.pieces[by][1]
}

#[inline]
fn diagonal_attackers(view: &BoardView, square: u8, by: usize) -> u64 {
    slider_attacks(square, view.occupied, &DIAGONALS) & (view.pieces[by][2] | view.pieces[by][4])
}

#[inline]
fn orthogonal_attackers(view: &BoardView, square: u8, by: usize) -> u64 {
    slider_attacks(square, view.occupied, &ORTHOGONALS) & (view.pieces[by][3] | view.pieces[by][4])
}

#[inline]
fn king_attackers(view: &BoardView, square: u8, by: usize) -> u64 {
    king_attacks(square) & view.pieces[by][5]
}

fn has_any_attacker(view: &BoardView, square: u8, by: usize) -> bool {
    pawn_attackers(view, square, by)
        | knight_attackers(view, square, by)
        | diagonal_attackers(view, square, by)
        | orthogonal_attackers(view, square, by)
        | king_attackers(view, square, by)
        != 0
}

fn tactical_status(view: &BoardView, piece: u8, square: u8) -> u8 {
    let victim_colour = usize::from((piece & 8) != 0);
    let enemy = victim_colour ^ 1;
    let mut status = 0_u8;

    if pawn_attackers(view, square, enemy) != 0 {
        status |= STATUS_ENEMY_PAWN;
    }
    if knight_attackers(view, square, enemy) != 0 {
        status |= STATUS_ENEMY_KNIGHT;
    }
    if diagonal_attackers(view, square, enemy) != 0 {
        status |= STATUS_ENEMY_DIAGONAL;
    }
    if orthogonal_attackers(view, square, enemy) != 0 {
        status |= STATUS_ENEMY_ORTHOGONAL;
    }
    if king_attackers(view, square, enemy) != 0 {
        status |= STATUS_ENEMY_KING;
    }
    if has_any_attacker(view, square, victim_colour) {
        status |= STATUS_DEFENDED;
    }

    status
}

impl SparseInputType for MnueX1Input {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        INPUT_SIZE
    }

    fn max_active(&self) -> usize {
        64
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let view = BoardView::from_position(pos);
        let stm_bucket = king_zone16(pos.our_ksq());
        let ntm_bucket = king_zone16(pos.opp_ksq());

        for (piece, square) in (*pos).into_iter() {
            let piece_type = usize::from(piece & 7);
            let stm_colour = usize::from((piece & 8) != 0);
            let ntm_colour = stm_colour ^ 1;
            let stm_square = usize::from(square);
            let ntm_square = usize::from(square ^ 56);

            if piece_type < NON_KING_PIECES {
                let stm = piece_feature_index(stm_bucket, stm_colour, piece_type, stm_square);
                let ntm = piece_feature_index(ntm_bucket, ntm_colour, piece_type, ntm_square);
                f(stm, ntm);
            }

            let status = tactical_status(&view, piece, square);
            let stm = tactical_feature_index(stm_colour, piece_type, stm_square, status);
            let ntm = tactical_feature_index(ntm_colour, piece_type, ntm_square, status);
            f(stm, ntm);
        }
    }

    fn shorthand(&self) -> String {
        "mnue-x1-59392".to_string()
    }

    fn description(&self) -> String {
        "MNUE-X1 piece plus per-piece tactical-state features".to_string()
    }
}

impl OutputBuckets<ChessBoard> for MnueX1Output {
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
    assert_eq!(PIECE_INPUTS, 10_240);
    assert_eq!(TACTICAL_INPUTS, 49_152);
    assert_eq!(INPUT_SIZE, 59_392);

    if dump_requested_position() {
        return;
    }

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(MnueX1Input)
        .output_buckets(MnueX1Output)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w")
                .transpose()
                .round()
                .quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA),
            SavedFormat::id("l2w")
                .transpose()
                .round()
                .quantise::<i16>(QB),
            SavedFormat::id("l2b").round().quantise::<i16>(QA),
            SavedFormat::id("l3w")
                .transpose()
                .round()
                .quantise::<i16>(QB),
            SavedFormat::id("l3b")
                .round()
                .quantise::<i32>(i32::from(QA) * i32::from(QB)),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", INPUT_SIZE, HIDDEN_SIZE);
            l0.init_with_effective_input_size(62);

            let l1 = builder.new_affine("l1", HIDDEN_SIZE, OUTPUT_BUCKETS * L1_SIZE);
            let l2 = builder.new_affine("l2", L1_SIZE, OUTPUT_BUCKETS * L2_SIZE);
            let l3 = builder.new_affine("l3", L2_SIZE, OUTPUT_BUCKETS);

            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hidden = stm_hidden.concat(ntm_hidden);

            let hidden = l1.forward(hidden).select(output_buckets).screlu();
            let hidden = l2.forward(hidden).select(output_buckets).screlu();
            l3.forward(hidden).select(output_buckets)
        });

    let bounded = AdamWParams {
        min_weight: -1.98,
        max_weight: 1.98,
        ..Default::default()
    };
    trainer.optimiser.set_params_for_weight("l0w", bounded);
    trainer.optimiser.set_params_for_weight("l1w", bounded);
    trainer.optimiser.set_params_for_weight("l2w", bounded);
    trainer.optimiser.set_params_for_weight("l3w", bounded);

    if let Some((checkpoint, fens)) = checkpoint_eval_request() {
        trainer.load_from_checkpoint(&checkpoint);
        for fen in fens {
            let input = if fen.contains('|') {
                fen
            } else {
                format!("{fen} | 0 | 0.5")
            };
            let score = SCALE as f32 * trainer.eval(&input);
            println!("mnue x1 float score {score:.6} fen {input}");
        }
        return;
    }

    let config = Config::from_args();
    fs::create_dir_all(&config.output_dir).expect("failed to create output directory");
    fs::create_dir_all(&config.net_dir).expect("failed to create network directory");

    let schedule = TrainingSchedule {
        net_id: config.net_id.clone(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: BATCH_SIZE,
            batches_per_superbatch: BATCHES_PER_SUPERBATCH,
            start_superbatch: START_SUPERBATCH,
            end_superbatch: END_SUPERBATCH,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.75 },
        // The 300M run validates the pipeline. Do not tune a long-run decay
        // schedule until this architecture has passed engine-side testing.
        lr_scheduler: lr::ConstantLR { value: 0.001 },
        save_rate: 1,
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: &config.output_dir,
        batch_queue_size: 64,
    };

    let path_refs = config.data.iter().map(String::as_str).collect::<Vec<_>>();
    let data_loader = loader::DirectSequentialDataLoader::new(&path_refs);
    trainer.run(&schedule, &settings, &data_loader);

    for superbatch in START_SUPERBATCH..=END_SUPERBATCH {
        let checkpoint = PathBuf::from(&config.output_dir)
            .join(format!("{}-{superbatch}", config.net_id))
            .join("quantised.bin");
        if !checkpoint.exists() {
            continue;
        }

        let output =
            PathBuf::from(&config.net_dir).join(format!("{}-{superbatch}.mnue", config.net_id));
        match write_headered_mnue(&checkpoint, &output) {
            Ok(()) => println!("Wrote {}", output.display()),
            Err(error) => eprintln!("failed to export superbatch {superbatch}: {error}"),
        }
    }
}

fn checkpoint_eval_request() -> Option<(String, Vec<String>)> {
    let mut args = std::env::args();
    let _program = args.next();
    if args.next().as_deref() != Some("--eval-checkpoint") {
        return None;
    }

    let checkpoint = args
        .next()
        .expect("usage: --eval-checkpoint <checkpoint-dir> \"<fen>\"");
    let fens: Vec<_> = args.collect();
    assert!(
        !fens.is_empty(),
        "usage: --eval-checkpoint <checkpoint-dir> \"<fen>\"..."
    );
    Some((checkpoint, fens))
}

fn dump_requested_position() -> bool {
    let mut args = std::env::args();
    let _program = args.next();
    if args.next().as_deref() != Some("--dump-fen") {
        return false;
    }

    let Some(mut fen) = args.next() else {
        eprintln!("usage: mnue_x1_trainer --dump-fen \"<fen>\"");
        return true;
    };
    if !fen.contains('|') {
        fen.push_str(" | 0 | 0.5");
    }

    let position: ChessBoard = fen.parse().expect("invalid FEN for --dump-fen");
    let mut stm = Vec::new();
    let mut ntm = Vec::new();
    MnueX1Input.map_features(&position, |stm_index, ntm_index| {
        stm.push(stm_index);
        ntm.push(ntm_index);
    });
    stm.sort_unstable();
    ntm.sort_unstable();

    println!("mnue x1 features");
    println!("output_bucket {}", MnueX1Output.bucket(&position));
    println!(
        "perspective 0 input_bucket {} count {}",
        king_zone16(position.our_ksq()),
        stm.len()
    );
    print!("indices");
    for index in stm {
        print!(" {index}");
    }
    println!();
    println!(
        "perspective 1 input_bucket {} count {}",
        king_zone16(position.opp_ksq()),
        ntm.len()
    );
    print!("indices");
    for index in ntm {
        print!(" {index}");
    }
    println!();
    true
}

fn expected_payload_bytes() -> usize {
    INPUT_SIZE * HIDDEN_SIZE * size_of::<i16>()
        + HIDDEN_SIZE * size_of::<i16>()
        + OUTPUT_BUCKETS * L1_SIZE * HIDDEN_SIZE * size_of::<i16>()
        + OUTPUT_BUCKETS * L1_SIZE * size_of::<i16>()
        + OUTPUT_BUCKETS * L2_SIZE * L1_SIZE * size_of::<i16>()
        + OUTPUT_BUCKETS * L2_SIZE * size_of::<i16>()
        + OUTPUT_BUCKETS * L2_SIZE * size_of::<i16>()
        + OUTPUT_BUCKETS * size_of::<i32>()
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
    write_u32(&mut output, MNUE_MAGIC)?;
    write_u32(&mut output, MNUE_VERSION)?;
    write_u32(&mut output, MNUE_ARCH_X1)?;
    write_u32(&mut output, MNUE_HEADER_BYTES)?;
    write_u32(&mut output, INPUT_SIZE as u32)?;
    write_u32(&mut output, HIDDEN_SIZE as u32)?;
    write_u32(&mut output, INPUT_BUCKETS as u32)?;
    write_u32(&mut output, OUTPUT_BUCKETS as u32)?;
    write_u32(&mut output, L1_SIZE as u32)?;
    write_u32(&mut output, L2_SIZE as u32)?;
    write_i32(&mut output, SCALE)?;
    write_i32(&mut output, i32::from(QA))?;
    write_i32(&mut output, i32::from(QB))?;
    write_u32(&mut output, MNUE_FEATURE_VERSION)?;
    write_u32(&mut output, MNUE_FLAGS)?;
    write_u32(&mut output, 0)?;
    output.write_all(&payload)?;
    Ok(())
}

fn write_u32<W: Write>(output: &mut W, value: u32) -> io::Result<()> {
    output.write_all(&value.to_le_bytes())
}

fn write_i32<W: Write>(output: &mut W, value: i32) -> io::Result<()> {
    output.write_all(&value.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feature_pairs(position: &ChessBoard) -> Vec<(usize, usize)> {
        let mut features = Vec::new();
        MnueX1Input.map_features(position, |stm, ntm| {
            features.push((stm, ntm));
        });
        features
    }

    #[test]
    fn start_position_has_expected_active_count_and_bucket() {
        let position: ChessBoard =
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.5"
                .parse()
                .unwrap();

        let features = feature_pairs(&position);
        assert_eq!(features.len(), 62);
        assert!(
            features
                .iter()
                .all(|&(stm, ntm)| { stm < INPUT_SIZE && ntm < INPUT_SIZE })
        );
        assert_eq!(MnueX1Output.bucket(&position), 7);
    }

    #[test]
    fn kings_only_have_two_tactical_features() {
        let position: ChessBoard = "8/8/8/8/8/8/4k3/4K3 w - - 0 1 | 0 | 0.5".parse().unwrap();

        let features = feature_pairs(&position);
        assert_eq!(features.len(), 2);
        assert!(
            features
                .iter()
                .all(|&(stm, ntm)| { stm >= PIECE_INPUTS && ntm >= PIECE_INPUTS })
        );
        assert_eq!(MnueX1Output.bucket(&position), 0);
    }

    #[test]
    fn side_to_move_normalisation_swaps_feature_perspectives() {
        let white: ChessBoard = "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1 | 0 | 0.5"
            .parse()
            .unwrap();
        let black: ChessBoard = "4k3/8/8/3q4/4P3/8/8/4K3 b - - 0 1 | 0 | 0.5"
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
