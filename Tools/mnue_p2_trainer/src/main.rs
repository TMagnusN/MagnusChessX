/*
Generic MNUE-P2 trainer for MagnusChess.

Supported architectures:
    p2      = 1 x 32 x 16 x 1024 x 10240
    p2-k32  = 1 x 64 x 32 x 1024 x 20480

Meaning:
    output buckets     = phase_bucket2 x stm_king_bucket
    input buckets      = king_bucket
    hidden size        = 1024
    sparse input size  = input_buckets x 2 x 5 x 64

Input feature:
    input_bucket x relative_color2 x non_king_piece_type5 x relative_square64

This is NOT Chess768. It implements a custom Bullet SparseInputType matching
the MagnusChess engine-side MNUE-P2 and MNUE-P2-K32 layouts.

Run:
    cargo run --manifest-path Tools/mnue_p2_trainer/Cargo.toml --release --features cuda -- \
        --arch p2-k32 --name mnue_p2k32_100m_5sb --data PATH --positions 100000000 \
        --batch-size 65536 --threads 12 --superbatches 5 \
        --lr 0.001 --lr-final 0.0001 --l0-init-scale 0.25 --save-sb 1

Output:
    runs/<name>/<name>-<superbatch>/quantised.bin
    nets/mnue/<name>-<superbatch>.mnue

The .mnue files are headered for the MagnusChess MNUE loader.
*/

use std::{
    env, fmt, fs,
    io::{self, Read, Write},
    marker::PhantomData,
    path::{Path, PathBuf},
    process,
};

use bullet_lib::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    nn::{Affine, InitSettings, ModelBuilder, Shape, optimiser::AdamW},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};
use bulletformat::ChessBoard;
use mnue_binpack_loader::ResamplingSfBinpackLoader;
use mnue_data_schedule::{
    DEFAULT_CHUNK_SHUFFLE_SEED, DEFAULT_CHUNK_VIRTUAL_EPOCHS, DataSchedule, DataScheduleOptions,
    InputDataKind, build_data_schedule, input_data_kind_name, print_startup_log,
};

const RELATIVE_COLORS: usize = 2;
const NON_KING_PIECES: usize = 5;
const SQUARES: usize = 64;

const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

const DEFAULT_RUNS_DIR: &str = "runs";
const DEFAULT_NET_DIR: &str = "nets/mnue";

const DEFAULT_P2_NAME: &str = "mnue_p2_medium";
const DEFAULT_P2K32_NAME: &str = "mnue_p2k32";
const DEFAULT_POSITIONS: u128 = 40_003_174_400;
const DEFAULT_BATCH_SIZE: usize = 16_384;
const DEFAULT_THREADS: usize = 4;
const DEFAULT_SUPERBATCHES: usize = 400;
const DEFAULT_LR: f32 = 0.001;
const DEFAULT_LR_FINAL: f32 = 0.0000625;
const DEFAULT_L0_INIT_SCALE: f32 = 1.0;
const DEFAULT_SAVE_RATE: usize = 1;
const BATCH_QUEUE_SIZE: usize = 64;
const BINPACK_BUFFER_MB: usize = 256;

const MNUE_MAGIC: u32 = 0x45554E4D; // file bytes: "MNUE"
const MNUE_VERSION: u32 = 1;
const MNUE_ARCH_P2: u32 = 2;

#[derive(Clone, Copy, Debug)]
enum Arch {
    P2,
    P2K32,
}

impl Arch {
    fn default_name(self) -> &'static str {
        match self {
            Self::P2 => DEFAULT_P2_NAME,
            Self::P2K32 => DEFAULT_P2K32_NAME,
        }
    }

    fn cli_name(self) -> &'static str {
        match self {
            Self::P2 => "p2",
            Self::P2K32 => "p2-k32",
        }
    }
}

impl Default for Arch {
    fn default() -> Self {
        Self::P2
    }
}

trait P2Spec: Copy + Send + Sync + 'static {
    const ARCH: Arch;
    const INPUT_BUCKETS: usize;
    const OUTPUT_BUCKETS: usize;
    const INPUT_SIZE: usize;
    const HIDDEN_SIZE: usize;
    const SHORTHAND: &'static str;
    const DESCRIPTION: &'static str;

    fn king_bucket(ksq: u8) -> usize;
}

#[derive(Clone, Copy)]
struct P2Spec16;

impl P2Spec for P2Spec16 {
    const ARCH: Arch = Arch::P2;
    const INPUT_BUCKETS: usize = 16;
    const OUTPUT_BUCKETS: usize = 32;
    const INPUT_SIZE: usize = Self::INPUT_BUCKETS * RELATIVE_COLORS * NON_KING_PIECES * SQUARES;
    const HIDDEN_SIZE: usize = 1024;
    const SHORTHAND: &'static str = "mnue-p2-10240";
    const DESCRIPTION: &'static str = "MNUE-P2 non-king 16-bucket 10240-feature input";

    fn king_bucket(ksq: u8) -> usize {
        king_zone16(ksq)
    }
}

#[derive(Clone, Copy)]
struct P2K32Spec;

impl P2Spec for P2K32Spec {
    const ARCH: Arch = Arch::P2K32;
    const INPUT_BUCKETS: usize = 32;
    const OUTPUT_BUCKETS: usize = 64;
    const INPUT_SIZE: usize = Self::INPUT_BUCKETS * RELATIVE_COLORS * NON_KING_PIECES * SQUARES;
    const HIDDEN_SIZE: usize = 1024;
    const SHORTHAND: &'static str = "mnue-p2-k32-20480";
    const DESCRIPTION: &'static str = "MNUE-P2-K32 non-king 32-bucket 20480-feature input";

    fn king_bucket(ksq: u8) -> usize {
        king_zone32(ksq)
    }
}

#[derive(Clone, Copy)]
struct MnueP2Input<S>(PhantomData<S>);

impl<S> Default for MnueP2Input<S> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<S: P2Spec> fmt::Debug for MnueP2Input<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(S::SHORTHAND)
    }
}

#[derive(Clone, Copy)]
struct MnueP2Output<S>(PhantomData<S>);

impl<S> Default for MnueP2Output<S> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<S: P2Spec> fmt::Debug for MnueP2Output<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(S::ARCH.cli_name())
    }
}

#[derive(Debug, Clone)]
struct Config {
    arch: Arch,
    name: String,
    data: Vec<String>,
    data_dirs: Vec<String>,
    chunk_shuffle_seed: u64,
    chunk_virtual_epochs: usize,
    chunk_sample: Option<usize>,
    chunk_resample_on_exhaustion: bool,
    positions: u128,
    batch_size: usize,
    threads: usize,
    superbatches: usize,
    lr: f32,
    lr_final: f32,
    l0_init_scale: f32,
    save_rate: usize,
    output_dir: String,
    net_dir: String,
}

impl Config {
    fn from_args() -> Self {
        let mut arch = Arch::default();
        let mut name: Option<String> = None;
        let mut data = Vec::new();
        let mut data_dirs = Vec::new();
        let mut chunk_shuffle_seed = DEFAULT_CHUNK_SHUFFLE_SEED;
        let mut chunk_virtual_epochs = DEFAULT_CHUNK_VIRTUAL_EPOCHS;
        let mut chunk_sample = None;
        let mut chunk_resample_on_exhaustion = false;
        let mut positions = DEFAULT_POSITIONS;
        let mut batch_size = DEFAULT_BATCH_SIZE;
        let mut threads = DEFAULT_THREADS;
        let mut superbatches = DEFAULT_SUPERBATCHES;
        let mut lr = DEFAULT_LR;
        let mut lr_final = DEFAULT_LR_FINAL;
        let mut l0_init_scale = DEFAULT_L0_INIT_SCALE;
        let mut save_rate = DEFAULT_SAVE_RATE;
        let mut output_dir = None;
        let mut net_dir = DEFAULT_NET_DIR.to_string();

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

            if key_value == "chunk-resample-on-exhaustion" {
                chunk_resample_on_exhaustion = true;
                idx += 1;
                continue;
            }

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
                "arch" => arch = parse_arch(&value),
                "name" => name = Some(parse_name(&value)),
                "data" | "train-data" => {
                    data.push(parse_path_arg(key, &value));
                }
                "data-dir" => data_dirs.push(parse_path_arg(key, &value)),
                "chunk-shuffle-seed" => chunk_shuffle_seed = parse_u64(key, &value),
                "chunk-virtual-epochs" => chunk_virtual_epochs = parse_usize(key, &value),
                "chunk-sample" => chunk_sample = Some(parse_usize(key, &value)),
                "positions" => positions = parse_u128(key, &value),
                "batch-size" => batch_size = parse_usize(key, &value),
                "threads" => threads = parse_usize(key, &value),
                "superbatches" => superbatches = parse_usize(key, &value),
                "lr" => lr = parse_f32(key, &value),
                "lr-final" => lr_final = parse_f32(key, &value),
                "l0-init-scale" => l0_init_scale = parse_f32(key, &value),
                "save-rate" | "save-sb" => save_rate = parse_usize(key, &value),
                "output-dir" => output_dir = Some(value),
                "net-dir" => net_dir = value,
                _ => die(format!("unknown option --{key}")),
            }

            idx += 1;
        }

        require_nonzero_u128("positions", positions);
        require_nonzero_usize("batch-size", batch_size);
        require_nonzero_usize("threads", threads);
        require_nonzero_usize("superbatches", superbatches);
        require_nonzero_usize("save-sb", save_rate);
        require_positive_f32("lr", lr);
        require_positive_f32("lr-final", lr_final);
        require_positive_f32("l0-init-scale", l0_init_scale);

        let name = name.unwrap_or_else(|| arch.default_name().to_string());
        let output_dir = output_dir.unwrap_or_else(|| format!("{DEFAULT_RUNS_DIR}/{name}"));

        Self {
            arch,
            name,
            data,
            data_dirs,
            chunk_shuffle_seed,
            chunk_virtual_epochs,
            chunk_sample,
            chunk_resample_on_exhaustion,
            positions,
            batch_size,
            threads,
            superbatches,
            lr,
            lr_final,
            l0_init_scale,
            save_rate,
            output_dir,
            net_dir,
        }
    }

    fn total_batches(&self) -> usize {
        let total_batches = div_ceil_u128(self.positions, self.batch_size as u128);
        usize::try_from(total_batches)
            .unwrap_or_else(|_| die("positions / batch-size is too large for usize"))
    }

    fn batches_per_superbatch(&self) -> usize {
        div_ceil_usize(self.total_batches(), self.superbatches)
    }

    fn effective_positions(&self) -> u128 {
        self.batch_size as u128 * self.batches_per_superbatch() as u128 * self.superbatches as u128
    }

    fn data_schedule_options(&self) -> DataScheduleOptions {
        DataScheduleOptions {
            explicit_paths: self.data.iter().map(PathBuf::from).collect(),
            data_dirs: self.data_dirs.iter().map(PathBuf::from).collect(),
            chunk_shuffle_seed: self.chunk_shuffle_seed,
            chunk_virtual_epochs: self.chunk_virtual_epochs,
            chunk_sample: self.chunk_sample,
            chunk_resample_on_exhaustion: self.chunk_resample_on_exhaustion,
        }
    }

    fn print_summary<S: P2Spec>(&self, data_schedule: &DataSchedule) {
        println!("MNUE-P2 generic run");
        println!("Architecture           : {}", self.arch.cli_name());
        println!("Name                   : {}", self.name);
        println!(
            "Training data format   : {}",
            input_data_kind_name(data_schedule.kind)
        );
        println!("Output dir             : {}", self.output_dir);
        println!("Net dir                : {}", self.net_dir);
        println!("Input buckets          : {}", S::INPUT_BUCKETS);
        println!("Output buckets         : {}", S::OUTPUT_BUCKETS);
        println!("Input size             : {}", S::INPUT_SIZE);
        println!("Hidden size            : {}", S::HIDDEN_SIZE);
        println!("Requested positions    : {}", self.positions);
        println!("Effective positions    : {}", self.effective_positions());
        println!("Batch size             : {}", self.batch_size);
        println!("Superbatches           : {}", self.superbatches);
        println!("Batches/superbatch     : {}", self.batches_per_superbatch());
        println!("Threads                : {}", self.threads);
        println!("LR                     : {}", self.lr);
        println!("LR final               : {}", self.lr_final);
        println!("L0 init scale          : {}", self.l0_init_scale);
        println!("Save every superbatches: {}", self.save_rate);
    }
}

fn print_usage() {
    println!(
        "Usage: cargo run --manifest-path Tools/mnue_p2_trainer/Cargo.toml --release --features cuda -- [options]\n\
         \n\
         Options:\n\
           --arch p2|p2-k32           Network architecture. Default: p2\n\
           --name NAME                 Net id and default output directory name.\n\
           --data PATH                 Required unless --data-dir is used. Training .data or .binpack path; repeatable.\n\
           --data-dir DIR              Required unless --data is used. Directory of .binpack chunks; repeatable.\n\
           --chunk-shuffle-seed N      Virtual chunk shuffle seed. Default: {DEFAULT_CHUNK_SHUFFLE_SEED}\n\
           --chunk-virtual-epochs N    Number of virtual chunk schedule passes. Default: {DEFAULT_CHUNK_VIRTUAL_EPOCHS}\n\
           --chunk-sample N            Chunks sampled without replacement per virtual pass. Default: all\n\
           --chunk-resample-on-exhaustion  Rebuild a new virtual chunk pool after the current one is exhausted.\n\
           --positions N               Total training positions. Default: {DEFAULT_POSITIONS}\n\
           --batch-size N              Positions per batch. Default: {DEFAULT_BATCH_SIZE}\n\
           --threads N                 CPU data-prep threads. Default: {DEFAULT_THREADS}\n\
           --superbatches N            Split total positions into N superbatches. Default: {DEFAULT_SUPERBATCHES}\n\
           --lr X                      Initial learning rate. Default: {DEFAULT_LR}\n\
           --lr-final X                Final exponential-decay learning rate. Default: {DEFAULT_LR_FINAL}\n\
           --l0-init-scale X           Multiplier for l0 weight init stdev. Default: {DEFAULT_L0_INIT_SCALE}\n\
           --save-sb N                 Save one checkpoint every N superbatches. Default: {DEFAULT_SAVE_RATE}\n\
           --save-rate N               Alias for --save-sb.\n\
           --output-dir PATH           Default: {DEFAULT_RUNS_DIR}/NAME\n\
           --net-dir PATH              Default: {DEFAULT_NET_DIR}"
    );
}

fn parse_arch(value: &str) -> Arch {
    match value.trim().to_ascii_lowercase().as_str() {
        "p2" => Arch::P2,
        "p2-k32" | "p2k32" | "k32" => Arch::P2K32,
        _ => die("--arch must be p2 or p2-k32"),
    }
}

fn parse_name(value: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        die("--name must not be empty");
    }
    if value.contains('/') || value.contains('\\') {
        die("--name must be a plain net id, not a path");
    }
    value.to_string()
}

fn parse_path_arg(key: &str, value: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        die(format!("--{key} must not be empty"));
    }
    value.to_string()
}

fn accept_all_binpack(_: &loader::sfbinpack::TrainingDataEntry) -> bool {
    true
}

fn parse_usize(key: &str, value: &str) -> usize {
    normalize_number(value)
        .parse()
        .unwrap_or_else(|_| die(format!("--{key} must be a positive integer")))
}

fn parse_u128(key: &str, value: &str) -> u128 {
    normalize_number(value)
        .parse()
        .unwrap_or_else(|_| die(format!("--{key} must be a positive integer")))
}

fn parse_u64(key: &str, value: &str) -> u64 {
    normalize_number(value)
        .parse()
        .unwrap_or_else(|_| die(format!("--{key} must be a non-negative integer")))
}

fn parse_f32(key: &str, value: &str) -> f32 {
    value
        .parse()
        .unwrap_or_else(|_| die(format!("--{key} must be a positive float")))
}

fn normalize_number(value: &str) -> String {
    value.replace('_', "")
}

fn require_nonzero_usize(key: &str, value: usize) {
    if value == 0 {
        die(format!("--{key} must be at least 1"));
    }
}

fn require_nonzero_u128(key: &str, value: u128) {
    if value == 0 {
        die(format!("--{key} must be at least 1"));
    }
}

fn require_positive_f32(key: &str, value: f32) {
    if !value.is_finite() || value <= 0.0 {
        die(format!("--{key} must be a finite positive float"));
    }
}

fn div_ceil_u128(lhs: u128, rhs: u128) -> u128 {
    lhs.div_ceil(rhs)
}

fn div_ceil_usize(lhs: usize, rhs: usize) -> usize {
    lhs.div_ceil(rhs)
}

fn die(message: impl fmt::Display) -> ! {
    eprintln!("error: {message}");
    eprintln!("run with --help for usage");
    process::exit(2);
}

fn new_scaled_affine<'a>(
    builder: &'a ModelBuilder,
    id: &str,
    input_size: usize,
    output_size: usize,
    init_scale: f32,
) -> Affine<'a> {
    let weight_id = format!("{id}w");
    let bias_id = format!("{id}b");
    let stdev = (2.0 / input_size as f32).sqrt() * init_scale;
    let weights = builder.new_weights(
        &weight_id,
        Shape::new(output_size, input_size),
        InitSettings::Normal { mean: 0.0, stdev },
    );
    let bias = builder.new_weights(&bias_id, Shape::new(output_size, 1), InitSettings::Zeroed);

    Affine { weights, bias }
}

#[inline]
fn file_of(sq: u8) -> usize {
    usize::from(sq & 7)
}

#[inline]
fn rank_of(sq: u8) -> usize {
    usize::from(sq >> 3)
}

#[inline]
fn king_zone16(ksq: u8) -> usize {
    let file_group = file_of(ksq) / 2; // 0..3
    let rank_group = rank_of(ksq) / 2; // 0..3
    rank_group * 4 + file_group // 0..15
}

#[inline]
fn king_zone32(ksq: u8) -> usize {
    let file = file_of(ksq);
    let file_group = file.min(7 - file); // 0..3, horizontal mirror
    let rank = rank_of(ksq); // 0..7
    rank * 4 + file_group // 0..31
}

#[inline]
fn phase_bucket2(pos: &ChessBoard) -> usize {
    // Crude non-pawn-material phase.
    //
    // Bullet ChessBoard is side-to-move normalised:
    // piece & 7 gives P,N,B,R,Q,K = 0,1,2,3,4,5.
    let mut npm = 0usize;

    for (piece, _) in (*pos).into_iter() {
        match piece & 7 {
            1 | 2 => npm += 1, // knight/bishop
            3 => npm += 2,     // rook
            4 => npm += 4,     // queen
            _ => {}
        }
    }

    // Opening/middlegame vs simplified/endgame.
    usize::from(npm < 10)
}

#[inline]
fn non_king_piece_index(piece: u8) -> Option<usize> {
    match piece & 7 {
        0 => Some(0), // pawn
        1 => Some(1), // knight
        2 => Some(2), // bishop
        3 => Some(3), // rook
        4 => Some(4), // queen
        5 => None,    // king is not an input feature
        _ => None,
    }
}

#[inline]
fn mnue_feature_index(bucket: usize, rel_color: usize, piece_idx: usize, rel_sq: usize) -> usize {
    (((bucket * RELATIVE_COLORS + rel_color) * NON_KING_PIECES + piece_idx) * SQUARES) + rel_sq
}

impl<S: P2Spec> SparseInputType for MnueP2Input<S> {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        S::INPUT_SIZE
    }

    fn max_active(&self) -> usize {
        // There are at most 30 non-king pieces, but 32 keeps the same safe upper bound as Chess768.
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        // In Bullet's ChessBoard, our_ksq is already side-to-move relative.
        // opp_ksq is already flipped into opponent perspective.
        let stm_bucket = S::king_bucket(pos.our_ksq());
        let ntm_bucket = S::king_bucket(pos.opp_ksq());

        for (piece, square) in (*pos).into_iter() {
            let Some(piece_idx) = non_king_piece_index(piece) else {
                continue;
            };

            // piece colour bit is relative to side-to-move:
            // 0 = own, 1 = opponent.
            let stm_rel_color = usize::from((piece & 8) != 0);
            let ntm_rel_color = 1 ^ stm_rel_color;

            let stm_sq = usize::from(square);
            let ntm_sq = usize::from(square ^ 56);

            let stm = mnue_feature_index(stm_bucket, stm_rel_color, piece_idx, stm_sq);
            let ntm = mnue_feature_index(ntm_bucket, ntm_rel_color, piece_idx, ntm_sq);

            f(stm, ntm);
        }
    }

    fn shorthand(&self) -> String {
        S::SHORTHAND.to_string()
    }

    fn description(&self) -> String {
        S::DESCRIPTION.to_string()
    }
}

impl<S: P2Spec> OutputBuckets<ChessBoard> for MnueP2Output<S> {
    const BUCKETS: usize = S::OUTPUT_BUCKETS;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let phase = phase_bucket2(pos); // 0..1
        let zone = S::king_bucket(pos.our_ksq());
        (phase * S::INPUT_BUCKETS + zone) as u8
    }
}

fn main() {
    let config = Config::from_args();
    match config.arch {
        Arch::P2 => run_training::<P2Spec16>(config),
        Arch::P2K32 => run_training::<P2K32Spec>(config),
    }
}

fn run_training<S: P2Spec>(config: Config) {
    debug_assert_eq!(
        S::INPUT_SIZE,
        S::INPUT_BUCKETS * RELATIVE_COLORS * NON_KING_PIECES * SQUARES
    );
    debug_assert_eq!(S::OUTPUT_BUCKETS, S::INPUT_BUCKETS * 2);

    let batches_per_superbatch = config.batches_per_superbatch();
    let l0_init_scale = config.l0_init_scale;

    let data_schedule = build_data_schedule(config.data_schedule_options())
        .unwrap_or_else(|error| die(error.to_string()));
    config.print_summary::<S>(&data_schedule);
    print_startup_log(&data_schedule);

    fs::create_dir_all(&config.output_dir).expect("failed to create output dir");
    fs::create_dir_all(&config.net_dir).expect("failed to create net dir");

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(MnueP2Input::<S>::default())
        .output_buckets(MnueP2Output::<S>::default())
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            // Output-bucketed weights must be transposed so each output bucket's
            // weights are contiguous for fast engine-side inference.
            SavedFormat::id("l1w")
                .round()
                .quantise::<i16>(QB)
                .transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = new_scaled_affine(builder, "l0", S::INPUT_SIZE, S::HIDDEN_SIZE, l0_init_scale);
            let l1 = builder.new_affine("l1", 2 * S::HIDDEN_SIZE, S::OUTPUT_BUCKETS);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden = stm_hidden.concat(ntm_hidden);

            l1.forward(hidden).select(output_buckets)
        });

    let schedule = TrainingSchedule {
        net_id: config.name.clone(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: config.batch_size,
            batches_per_superbatch,
            start_superbatch: 1,
            end_superbatch: config.superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.75 },
        lr_scheduler: lr::ExponentialDecayLR {
            initial_lr: config.lr,
            final_lr: config.lr_final,
            final_superbatch: config.superbatches,
        },
        save_rate: config.save_rate,
    };

    let settings = LocalSettings {
        threads: config.threads,
        test_set: None,
        output_directory: &config.output_dir,
        batch_queue_size: BATCH_QUEUE_SIZE,
    };

    let path_strings = data_schedule.path_strings();
    let path_refs = path_strings.iter().map(String::as_str).collect::<Vec<_>>();
    match data_schedule.kind {
        InputDataKind::Direct => {
            let data_loader = loader::DirectSequentialDataLoader::new(&path_refs);
            trainer.run(&schedule, &settings, &data_loader);
        }
        InputDataKind::SfBinpack => {
            if data_schedule.resample_on_exhaustion {
                let data_loader = ResamplingSfBinpackLoader::new(
                    data_schedule.chunk_path_strings(),
                    BINPACK_BUFFER_MB,
                    config.threads,
                    accept_all_binpack,
                    data_schedule.seed,
                    data_schedule.virtual_epochs,
                    data_schedule.chunk_sample,
                );
                trainer.run(&schedule, &settings, &data_loader);
            } else {
                let data_loader = loader::SfBinpackLoader::new_concat_multiple(
                    &path_refs,
                    BINPACK_BUFFER_MB,
                    config.threads,
                    accept_all_binpack,
                );
                trainer.run(&schedule, &settings, &data_loader);
            }
        }
    }

    for sb in 1..=config.superbatches {
        let checkpoint_dir =
            PathBuf::from(&config.output_dir).join(format!("{}-{sb}", config.name));
        let quantised = checkpoint_dir.join("quantised.bin");

        if !quantised.exists() {
            continue;
        }

        let out = PathBuf::from(&config.net_dir).join(format!("{}-{sb}.mnue", config.name));

        if let Err(err) = write_headered_mnue::<S>(&quantised, &out) {
            eprintln!("warning: failed to write headered MNUE for superbatch {sb}: {err}");
        } else {
            println!("Wrote {}", out.display());
        }
    }
}

fn write_headered_mnue<S: P2Spec>(input_quantised: &Path, output_mnue: &Path) -> io::Result<()> {
    let mut payload = Vec::new();
    fs::File::open(input_quantised)?.read_to_end(&mut payload)?;

    let expected_payload_bytes = S::INPUT_SIZE * S::HIDDEN_SIZE * 2
        + S::HIDDEN_SIZE * 2
        + S::OUTPUT_BUCKETS * 2 * S::HIDDEN_SIZE * 2
        + S::OUTPUT_BUCKETS * 2;

    if payload.len() != expected_payload_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unexpected quantised payload size: got {}, expected {}",
                payload.len(),
                expected_payload_bytes
            ),
        ));
    }

    let mut out = fs::File::create(output_mnue)?;

    write_u32(&mut out, MNUE_MAGIC)?;
    write_u32(&mut out, MNUE_VERSION)?;
    write_u32(&mut out, MNUE_ARCH_P2)?;
    write_u32(&mut out, S::INPUT_SIZE as u32)?;
    write_u32(&mut out, S::HIDDEN_SIZE as u32)?;
    write_u32(&mut out, S::INPUT_BUCKETS as u32)?;
    write_u32(&mut out, S::OUTPUT_BUCKETS as u32)?;
    write_i32(&mut out, SCALE)?;
    write_i32(&mut out, i32::from(QA))?;
    write_i32(&mut out, i32::from(QB))?;

    out.write_all(&payload)?;
    Ok(())
}

fn write_u32<W: Write>(out: &mut W, x: u32) -> io::Result<()> {
    out.write_all(&x.to_le_bytes())
}

fn write_i32<W: Write>(out: &mut W, x: i32) -> io::Result<()> {
    out.write_all(&x.to_le_bytes())
}
