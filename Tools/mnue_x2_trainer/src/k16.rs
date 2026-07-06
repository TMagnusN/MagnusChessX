use std::{
    fs,
    io::{self, BufReader, Read},
    path::{Path, PathBuf},
    sync::OnceLock,
    thread,
};

use bullet_compiler::tensor::TValue;
use bullet_gpu::runtime::Device;
use bullet_lib::nn::{
    ExecutionContext, InitSettings, ModelBuilder, ModelNode, Shape,
    optimiser::{AdamWOptimiser, AdamWParams, Optimiser},
};
use bullet_lib::value::loader::{
    DataLoader as DirectDataLoader, DirectSequentialDataLoader, SfBinpackLoader,
};
use bullet_trainer::{
    Trainer,
    model::save::ModelWeights,
    run::{
        dataloader::{DataLoader, DataLoadingError, PreparedBatchHost},
        schedule::{TrainingSchedule, TrainingSteps},
    },
};
use bulletformat::{BulletFormat, ChessBoard};
use mnue_binpack_loader::ResamplingSfBinpackLoader;
use mnue_data_schedule::{
    DEFAULT_CHUNK_SHUFFLE_SEED, DEFAULT_CHUNK_VIRTUAL_EPOCHS, DataScheduleOptions, InputDataKind,
    build_data_schedule, input_data_kind_name, print_startup_log,
};

pub const ARCH_NAME: &str = "x2-k16-pawn-q8-a384";

const INPUT_BUCKETS: usize = 16;
const OUTPUT_BUCKETS: usize = 8;
const RELATIVE_COLOURS: usize = 2;
const PIECE_TYPES: usize = 6;
const VICTIM_CLASSES: usize = RELATIVE_COLOURS * PIECE_TYPES;
const SQUARES: usize = 64;
const PAWN_SQUARES: usize = 48;
const PAWN_TOKENS: usize = RELATIVE_COLOURS * PAWN_SQUARES;

const PIECE_INPUTS: usize = INPUT_BUCKETS * RELATIVE_COLOURS * PIECE_TYPES * SQUARES;
const ATTACK_INPUTS: usize = 90_048;
const PAWN_PAIR_INPUTS: usize = PAWN_TOKENS * (PAWN_TOKENS - 1) / 2;

const PIECE_HIDDEN: usize = 768;
const ATTACK_HIDDEN: usize = 384;
const PAWN_HIDDEN: usize = 768;
const MERGED_HIDDEN: usize = 768;
const PAIRWISE: usize = MERGED_HIDDEN / 2;
const HEAD_INPUT: usize = PAIRWISE * 2;
const L1_SIZE: usize = 16;
const L2_SIZE: usize = 32;

const PIECE_MAX_ACTIVE: usize = 32;
const ATTACK_MAX_ACTIVE: usize = 320;
const PAWN_PAIR_MAX_ACTIVE: usize = 120;

const SCALE: u32 = 400;
const QA: u32 = 255;
const PIECE_QUANT: i16 = 64;
const PIECE_RESCALE: u32 = 4;
const ATTACK_QUANT: i16 = 64;
const ATTACK_RESCALE: u32 = 4;
const PAWN_PAIR_QUANT: i16 = 128;
const PAWN_PAIR_RESCALE: u32 = 2;
const L1_QUANT: i16 = 64;

const MNUE_MAGIC: u32 = 0x4555_4E4D;
const MNUE_VERSION: u32 = 3;
const MNUE_ARCH: u32 = 8;
const MNUE_HEADER_BYTES: u32 = 112;
const MNUE_FEATURE_VERSION: u32 = 2;

const DEFAULT_OUTPUT_DIR: &str = "runs/mnue_x2_k16_pawn_q8_a384";
const DEFAULT_EXPORT: &str = "nets/mnue/mnue_x2_k16_pawn_q8_a384.mnue";
const DEFAULT_BATCH_SIZE: usize = 16_384;
const DEFAULT_PREP_THREADS: usize = 4;
const DEFAULT_FULL_SUPERBATCHES: usize = 3;
const DEFAULT_POSITION_SUPERBATCHES: usize = 4;
const DEFAULT_BATCHES_PER_SUPERBATCH: usize = 6104;
const DEFAULT_LR: f32 = 0.001;
const DEFAULT_LR_FINAL: f32 = 0.0001;
const DEFAULT_DATA_STATS_LIMIT: u64 = 1_000_000;
const BINPACK_BUFFER_MB: usize = 256;
const BINPACK_INSPECT_BUFFER_MB: usize = 1;

#[derive(Clone, Copy)]
struct CalibrationCase {
    name: &'static str,
    fen: &'static str,
    move_description: Option<&'static str>,
}

const CALIBRATION_CASES: [CalibrationCase; 8] = [
    CalibrationCase {
        name: "startpos_w",
        fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        move_description: None,
    },
    CalibrationCase {
        name: "startpos_b",
        fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
        move_description: Some("startpos side-to-move flipped to black"),
    },
    CalibrationCase {
        name: "after_e4",
        fen: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        move_description: Some("after 1.e4"),
    },
    CalibrationCase {
        name: "after_e4_c5",
        fen: "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        move_description: Some("after 1.e4 c5"),
    },
    CalibrationCase {
        name: "white_queen_missing",
        fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
        move_description: Some("queen_swing remove white queen from d1"),
    },
    CalibrationCase {
        name: "black_queen_missing",
        fen: "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        move_description: Some("queen_swing remove black queen from d8"),
    },
    CalibrationCase {
        name: "low_material_pawn",
        fen: "8/8/8/3k4/3P4/4K3/8/8 w - - 0 1",
        move_description: None,
    },
    CalibrationCase {
        name: "kiwipete",
        fen: "r3k2r/p1ppqpb1/bn2pnp1/2P5/1p2P3/2N2N2/PP1PBPPP/R2Q1RK1 w kq - 0 1",
        move_description: None,
    },
];

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

#[derive(Clone, Debug)]
struct Cli {
    arch: String,
    data: Vec<PathBuf>,
    data_dirs: Vec<PathBuf>,
    chunk_shuffle_seed: u64,
    chunk_virtual_epochs: usize,
    chunk_sample: Option<usize>,
    chunk_resample_on_exhaustion: bool,
    output_dir: PathBuf,
    export: PathBuf,
    positions: u64,
    batch_size: usize,
    threads: usize,
    superbatches: Option<usize>,
    lr: f32,
    lr_final: f32,
    dry_run: bool,
    dump_fen: Option<String>,
    eval_mnue: Option<PathBuf>,
    eval_fens: Vec<String>,
    data_stats_limit: u64,
    strict_calibration_gate: bool,
    calibration_only: bool,
    calibration_l0_scale_sweep: bool,
    calibration_l0_scales: Vec<f32>,
    load_weights: Option<PathBuf>,
    l0_init_scale: f32,
    piece_l0_init_scale: Option<f32>,
    attack_l0_init_scale: Option<f32>,
    pawn_l0_init_scale: Option<f32>,
    head_init_scale: f32,
    l1_init_scale: Option<f32>,
    l2_init_scale: Option<f32>,
    l3_init_scale: Option<f32>,
    l0b_lr_scale: f32,
    l0b_freeze_steps: u64,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            arch: "x2".to_string(),
            data: Vec::new(),
            data_dirs: Vec::new(),
            chunk_shuffle_seed: DEFAULT_CHUNK_SHUFFLE_SEED,
            chunk_virtual_epochs: DEFAULT_CHUNK_VIRTUAL_EPOCHS,
            chunk_sample: None,
            chunk_resample_on_exhaustion: false,
            output_dir: PathBuf::from(DEFAULT_OUTPUT_DIR),
            export: PathBuf::from(DEFAULT_EXPORT),
            positions: 0,
            batch_size: DEFAULT_BATCH_SIZE,
            threads: DEFAULT_PREP_THREADS,
            superbatches: None,
            lr: DEFAULT_LR,
            lr_final: DEFAULT_LR_FINAL,
            dry_run: false,
            dump_fen: None,
            eval_mnue: None,
            eval_fens: Vec::new(),
            data_stats_limit: DEFAULT_DATA_STATS_LIMIT,
            strict_calibration_gate: false,
            calibration_only: false,
            calibration_l0_scale_sweep: false,
            calibration_l0_scales: default_l0_scale_sweep_values(),
            load_weights: None,
            l0_init_scale: 1.0,
            piece_l0_init_scale: None,
            attack_l0_init_scale: None,
            pawn_l0_init_scale: None,
            head_init_scale: 1.0,
            l1_init_scale: None,
            l2_init_scale: None,
            l3_init_scale: None,
            l0b_lr_scale: 1.0,
            l0b_freeze_steps: 0,
        }
    }
}

impl Cli {
    fn parse<I, S>(args: I) -> Result<Option<Self>, String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut args = args.into_iter().map(Into::into);
        let _program = args.next();
        let rest: Vec<String> = args.collect();
        if rest.is_empty() {
            return Ok(None);
        }
        if !rest.iter().any(|x| {
            matches!(
                x.as_str(),
                "--arch"
                    | "--dump-fen"
                    | "--dry-run"
                    | "--positions"
                    | "--date"
                    | "--data"
                    | "--data-dir"
                    | "--chunk-shuffle-seed"
                    | "--chunk-virtual-epochs"
                    | "--chunk-sample"
                    | "--chunk-resample-on-exhaustion"
                    | "--output-dir"
                    | "--export"
                    | "--batch-size"
                    | "--threads"
                    | "--superbatches"
                    | "--lr"
                    | "--lr-final"
                    | "--eval-mnue"
                    | "--eval-fen"
                    | "--data-stats"
                    | "--strict-calibration-gate"
                    | "--calibration-only"
                    | "--calibration-l0-scale-sweep"
                    | "--calibration-l0-scales"
                    | "--load-weights"
                    | "--l0-init-scale"
                    | "--piece-l0-init-scale"
                    | "--attack-l0-init-scale"
                    | "--pawn-l0-init-scale"
                    | "--head-init-scale"
                    | "--l1-init-scale"
                    | "--l2-init-scale"
                    | "--l3-init-scale"
                    | "--l0b-lr-scale"
                    | "--l0b-freeze-steps"
                    | "--help"
                    | "-h"
            )
        }) {
            return Ok(None);
        }

        let mut cli = Self::default();
        let mut iter = rest.into_iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--arch" => cli.arch = next_value(&mut iter, &arg)?,
                "--data" => {
                    cli.data.push(PathBuf::from(next_value(&mut iter, &arg)?));
                }
                "--data-dir" => {
                    cli.data_dirs
                        .push(PathBuf::from(next_value(&mut iter, &arg)?));
                }
                "--chunk-shuffle-seed" => {
                    cli.chunk_shuffle_seed = next_value(&mut iter, &arg)?
                        .parse()
                        .map_err(|_| "invalid --chunk-shuffle-seed".to_string())?;
                }
                "--chunk-virtual-epochs" => {
                    cli.chunk_virtual_epochs = next_value(&mut iter, &arg)?
                        .parse()
                        .map_err(|_| "invalid --chunk-virtual-epochs".to_string())?;
                }
                "--chunk-sample" => {
                    cli.chunk_sample = Some(
                        next_value(&mut iter, &arg)?
                            .parse()
                            .map_err(|_| "invalid --chunk-sample".to_string())?,
                    );
                }
                "--chunk-resample-on-exhaustion" => cli.chunk_resample_on_exhaustion = true,
                "--output-dir" => cli.output_dir = PathBuf::from(next_value(&mut iter, &arg)?),
                "--export" => cli.export = PathBuf::from(next_value(&mut iter, &arg)?),
                "--positions" | "--date" => {
                    cli.positions = next_value(&mut iter, &arg)?
                        .parse()
                        .map_err(|_| format!("invalid {arg}"))?;
                }
                "--batch-size" => {
                    cli.batch_size = next_value(&mut iter, &arg)?
                        .parse()
                        .map_err(|_| "invalid --batch-size".to_string())?;
                    if cli.batch_size == 0 {
                        return Err("--batch-size must be positive".to_string());
                    }
                }
                "--threads" => {
                    cli.threads = next_value(&mut iter, &arg)?
                        .parse()
                        .map_err(|_| "invalid --threads".to_string())?;
                    if cli.threads == 0 {
                        return Err("--threads must be positive".to_string());
                    }
                }
                "--superbatches" => {
                    let superbatches = next_value(&mut iter, &arg)?
                        .parse()
                        .map_err(|_| "invalid --superbatches".to_string())?;
                    if superbatches == 0 {
                        return Err("--superbatches must be positive".to_string());
                    }
                    cli.superbatches = Some(superbatches);
                }
                "--lr" => {
                    cli.lr = next_value(&mut iter, &arg)?
                        .parse()
                        .map_err(|_| "invalid --lr".to_string())?;
                    if !cli.lr.is_finite() || cli.lr <= 0.0 {
                        return Err("--lr must be a positive finite value".to_string());
                    }
                }
                "--lr-final" => {
                    cli.lr_final = next_value(&mut iter, &arg)?
                        .parse()
                        .map_err(|_| "invalid --lr-final".to_string())?;
                    if !cli.lr_final.is_finite() || cli.lr_final <= 0.0 {
                        return Err("--lr-final must be a positive finite value".to_string());
                    }
                }
                "--dry-run" => cli.dry_run = true,
                "--dump-fen" => cli.dump_fen = Some(next_value(&mut iter, &arg)?),
                "--eval-mnue" => cli.eval_mnue = Some(PathBuf::from(next_value(&mut iter, &arg)?)),
                "--eval-fen" => cli.eval_fens.push(next_value(&mut iter, &arg)?),
                "--data-stats" => {
                    cli.data_stats_limit = next_value(&mut iter, &arg)?
                        .parse()
                        .map_err(|_| "invalid --data-stats".to_string())?;
                }
                "--strict-calibration-gate" => cli.strict_calibration_gate = true,
                "--calibration-only" => cli.calibration_only = true,
                "--calibration-l0-scale-sweep" => cli.calibration_l0_scale_sweep = true,
                "--calibration-l0-scales" => {
                    cli.calibration_l0_scales = parse_l0_scale_list(&mut iter, &arg)?;
                }
                "--load-weights" => {
                    cli.load_weights = Some(PathBuf::from(next_value(&mut iter, &arg)?));
                }
                "--l0-init-scale" => {
                    cli.l0_init_scale = parse_nonnegative_f32(&mut iter, &arg)?;
                }
                "--piece-l0-init-scale" => {
                    cli.piece_l0_init_scale = Some(parse_nonnegative_f32(&mut iter, &arg)?);
                }
                "--attack-l0-init-scale" => {
                    cli.attack_l0_init_scale = Some(parse_nonnegative_f32(&mut iter, &arg)?);
                }
                "--pawn-l0-init-scale" => {
                    cli.pawn_l0_init_scale = Some(parse_nonnegative_f32(&mut iter, &arg)?);
                }
                "--head-init-scale" => {
                    cli.head_init_scale = parse_nonnegative_f32(&mut iter, &arg)?;
                }
                "--l1-init-scale" => {
                    cli.l1_init_scale = Some(parse_nonnegative_f32(&mut iter, &arg)?);
                }
                "--l2-init-scale" => {
                    cli.l2_init_scale = Some(parse_nonnegative_f32(&mut iter, &arg)?);
                }
                "--l3-init-scale" => {
                    cli.l3_init_scale = Some(parse_nonnegative_f32(&mut iter, &arg)?);
                }
                "--l0b-lr-scale" => {
                    cli.l0b_lr_scale = parse_nonnegative_f32(&mut iter, &arg)?;
                }
                "--l0b-freeze-steps" => {
                    cli.l0b_freeze_steps = next_value(&mut iter, &arg)?
                        .parse()
                        .map_err(|_| "invalid --l0b-freeze-steps".to_string())?;
                }
                "-h" | "--help" => return Err(Self::help().to_string()),
                _ => return Err(format!("unknown argument: {arg}\n{}", Self::help())),
            }
        }
        if cli.lr_final > cli.lr {
            return Err("--lr-final must be less than or equal to --lr".to_string());
        }
        Ok(Some(cli))
    }

    const fn help() -> &'static str {
        "MNUE-X2 trainer\n\
         --arch x2-k16-pawn-q8-a384\n\
         --dump-fen FEN            dump branch feature ids for the selected arch\n\
         --data PATH               required unless --data-dir is used; repeatable\n\
         --data-dir DIR            required unless --data is used; repeatable .binpack directory\n\
         --chunk-shuffle-seed N    virtual chunk shuffle seed; default 1\n\
         --chunk-virtual-epochs N  virtual chunk schedule passes; default 1024\n\
         --chunk-sample N          chunks sampled without replacement per virtual pass\n\
         --chunk-resample-on-exhaustion rebuild a new virtual chunk pool after the current one is exhausted\n\
         --output-dir PATH         checkpoint/output directory\n\
         --export PATH             exported .mnue path\n\
         --positions N             accepted-position limit; 0 uses default schedule\n\
         --batch-size N            training batch size\n\
         --threads N               CPU feature-prepare threads; default 4\n\
         --superbatches N          training superbatches; default 4 with --positions, 3 otherwise\n\
         --lr X                    starting learning rate; default 0.001\n\
         --lr-final X              final cosine-decay learning rate; default 0.0001\n\
         --eval-mnue PATH          scalar-forward an exported arch-8 .mnue\n\
         --eval-fen FEN            FEN for --eval-mnue; repeatable\n\
         --data-stats N            inspect N raw records for target statistics; 0 disables\n\
         --strict-calibration-gate fail export on fixed-FEN calibration failures\n\
         --calibration-only        load/build trainer and print calibration without training/export\n\
         --load-weights PATH       load trainer weights.bin; directory paths append weights.bin\n\
         --calibration-l0-scale-sweep print calibration-only temporary L0 scale sweep\n\
         --calibration-l0-scales CSV override sweep values; default 1,0.5,0.25,0.125,0.0625\n\
         --l0-init-scale X         scale all L0 branch initialisation stddevs; default 1.0\n\
         --piece-l0-init-scale X   override piece L0 initialisation scale\n\
         --attack-l0-init-scale X  override attack L0 initialisation scale\n\
         --pawn-l0-init-scale X    override pawn-pair L0 initialisation scale\n\
         --head-init-scale X       scale all head initialisation stddevs; default 1.0\n\
         --l1-init-scale X         override L1 initialisation scale\n\
         --l2-init-scale X         override L2 initialisation scale\n\
         --l3-init-scale X         override L3 initialisation scale\n\
         --l0b-lr-scale X          accepted for ablation docs; per-weight LR is not implemented\n\
         --l0b-freeze-steps N      keep l0b zero for the first N optimiser updates\n\
         --dry-run                 one batch train/export smoke test"
    }
}

fn next_value<I: Iterator<Item = String>>(args: &mut I, option: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value after {option}"))
}

fn parse_nonnegative_f32<I: Iterator<Item = String>>(
    args: &mut I,
    option: &str,
) -> Result<f32, String> {
    let value: f32 = next_value(args, option)?
        .parse()
        .map_err(|_| format!("invalid {option}"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{option} must be a non-negative finite value"));
    }
    Ok(value)
}

fn default_l0_scale_sweep_values() -> Vec<f32> {
    vec![1.0, 0.5, 0.25, 0.125, 0.0625]
}

fn parse_l0_scale_list<I: Iterator<Item = String>>(
    args: &mut I,
    option: &str,
) -> Result<Vec<f32>, String> {
    let text = next_value(args, option)?;
    let mut values = Vec::new();
    for part in text.split(',') {
        let value: f32 = part
            .trim()
            .parse()
            .map_err(|_| format!("invalid {option} entry: {part}"))?;
        if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "{option} entries must be non-negative finite values"
            ));
        }
        values.push(value);
    }
    if values.is_empty() {
        return Err(format!("{option} requires at least one scale"));
    }
    Ok(values)
}

#[derive(Clone, Copy, Debug)]
struct InitScales {
    piece_l0: f32,
    attack_l0: f32,
    pawn_l0: f32,
    l1: f32,
    l2: f32,
    l3: f32,
}

impl Cli {
    fn init_scales(&self) -> InitScales {
        InitScales {
            piece_l0: self.piece_l0_init_scale.unwrap_or(self.l0_init_scale),
            attack_l0: self.attack_l0_init_scale.unwrap_or(self.l0_init_scale),
            pawn_l0: self.pawn_l0_init_scale.unwrap_or(self.l0_init_scale),
            l1: self.l1_init_scale.unwrap_or(self.head_init_scale),
            l2: self.l2_init_scale.unwrap_or(self.head_init_scale),
            l3: self.l3_init_scale.unwrap_or(self.head_init_scale),
        }
    }

    fn data_schedule_options(&self) -> DataScheduleOptions {
        DataScheduleOptions {
            explicit_paths: self.data.clone(),
            data_dirs: self.data_dirs.clone(),
            chunk_shuffle_seed: self.chunk_shuffle_seed,
            chunk_virtual_epochs: self.chunk_virtual_epochs,
            chunk_sample: self.chunk_sample,
            chunk_resample_on_exhaustion: self.chunk_resample_on_exhaustion,
        }
    }
}

fn piece_l0_base_std() -> f32 {
    (2.0 / 96.0_f32).sqrt()
}

fn attack_l0_base_std() -> f32 {
    (2.0 / 96.0_f32).sqrt()
}

fn pawn_l0_base_std() -> f32 {
    (2.0 / 32.0_f32).sqrt()
}

fn l1_base_std() -> f32 {
    (2.0 / HEAD_INPUT as f32).sqrt()
}

fn l2_base_std() -> f32 {
    (2.0 / L1_SIZE as f32).sqrt()
}

fn l3_base_std() -> f32 {
    (2.0 / L2_SIZE as f32).sqrt()
}

pub fn maybe_run_from_args() -> bool {
    let cli = match Cli::parse(std::env::args()) {
        Ok(Some(cli)) => cli,
        Ok(None) => return false,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };

    if cli.arch != ARCH_NAME {
        if cli.arch == "x2" && cli.dump_fen.is_none() && cli.eval_mnue.is_none() {
            return false;
        }
        eprintln!("unsupported --arch {}; use {ARCH_NAME}", cli.arch);
        std::process::exit(2);
    }

    if let Some(path) = cli.eval_mnue.as_deref() {
        eval_mnue(path, &cli.eval_fens).unwrap_or_else(|error| {
            eprintln!("{error}");
            std::process::exit(1);
        });
        return true;
    }

    if let Some(fen) = cli.dump_fen.as_deref() {
        dump_fen(fen).unwrap_or_else(|error| {
            eprintln!("{error}");
            std::process::exit(2);
        });
        return true;
    }

    run_training(cli).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(1);
    });
    true
}

fn path_strings(paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect()
}

fn accept_all_binpack(_: &bullet_lib::value::loader::sfbinpack::TrainingDataEntry) -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FeaturePair {
    stm: Vec<usize>,
    ntm: Vec<usize>,
}

impl FeaturePair {
    fn checked(
        mut stm: Vec<usize>,
        mut ntm: Vec<usize>,
        vocabulary: usize,
    ) -> Result<Self, String> {
        stm.sort_unstable();
        stm.dedup();
        ntm.sort_unstable();
        ntm.dedup();
        if let Some(index) = stm.iter().chain(&ntm).copied().find(|&x| x >= vocabulary) {
            return Err(format!(
                "feature index {index} is outside vocabulary {vocabulary}"
            ));
        }
        Ok(Self { stm, ntm })
    }
}

#[derive(Clone, Debug)]
struct EncodedPosition {
    piece: FeaturePair,
    attack: FeaturePair,
    pawn_pair: FeaturePair,
    bucket: u8,
}

#[derive(Clone, Copy, Debug)]
struct BoardView {
    pieces: [u8; SQUARES],
    occupied: u64,
}

impl BoardView {
    fn new(pos: &ChessBoard) -> Result<Self, String> {
        let mut view = Self {
            pieces: [u8::MAX; SQUARES],
            occupied: pos.occ(),
        };
        let mut count = 0_usize;
        let mut kings = [0_usize; 2];
        for (piece, square) in *pos {
            let piece_type = usize::from(piece & 7);
            let colour = usize::from((piece & 8) != 0);
            if usize::from(square) >= SQUARES || piece_type >= PIECE_TYPES {
                return Err(format!("invalid piece {piece} square {square}"));
            }
            if piece_type == 5 {
                kings[colour] += 1;
            }
            view.pieces[usize::from(square)] = piece;
            count += 1;
        }
        let occupied = pos.occ().count_ones() as usize;
        if count != occupied || !(2..=32).contains(&count) || kings != [1, 1] {
            return Err(format!(
                "invalid board: count={count} occupied={occupied} kings={kings:?}"
            ));
        }
        Ok(view)
    }

    fn piece_on(&self, square: u8) -> Result<u8, String> {
        let piece = self.pieces[usize::from(square)];
        if piece == u8::MAX {
            return Err(format!("missing piece on occupied square {square}"));
        }
        Ok(piece)
    }
}

struct AttackIndex {
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
const fn file_of(square: u8) -> usize {
    (square & 7) as usize
}

#[inline]
const fn rank_of(square: u8) -> usize {
    (square >> 3) as usize
}

#[inline]
const fn bit(square: u8) -> u64 {
    1_u64 << square
}

#[inline]
fn mirror_for_king(relative_king: u8) -> bool {
    file_of(relative_king) >= 4
}

#[inline]
fn relative_square(square: u8, ntm: bool, mirror: bool) -> usize {
    let vertical = if ntm { square ^ 56 } else { square };
    usize::from(if mirror { vertical ^ 7 } else { vertical })
}

#[inline]
fn relative_square_no_mirror(square: u8, ntm: bool) -> usize {
    usize::from(if ntm { square ^ 56 } else { square })
}

fn king_bucket16(relative_king: u8) -> usize {
    let file = file_of(relative_king).min(7 - file_of(relative_king));
    let rank_band = rank_of(relative_king).min(3);
    rank_band * 4 + file
}

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
    if !(0..8).contains(&target_rank) {
        return 0;
    }
    let mut attacks = 0_u64;
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
        _ => 0,
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
        _ => 0,
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
    attack_index_table().base[attacker_colour][attacker_type][attacker_square]
        + target_slot * VICTIM_CLASSES
        + victim_class
}

fn pawn_square48(relative_square: usize) -> Option<usize> {
    let rank = relative_square / 8;
    if !(1..=6).contains(&rank) {
        return None;
    }
    Some((rank - 1) * 8 + relative_square % 8)
}

fn pawn_pair_index(a: usize, b: usize) -> usize {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi * (hi - 1) / 2 + lo
}

fn pawn_pair_row(a: usize, b: usize) -> Option<usize> {
    if a == b {
        return None;
    }
    let file_a = (a % PAWN_SQUARES) % 8;
    let file_b = (b % PAWN_SQUARES) % 8;
    if file_a.abs_diff(file_b) > 1 {
        return None;
    }
    Some(pawn_pair_index(a, b))
}

fn output_bucket(pos: &ChessBoard) -> u8 {
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

fn encode_position(pos: &ChessBoard) -> Result<EncodedPosition, String> {
    let view = BoardView::new(pos)?;
    let mut piece = FeaturePair {
        stm: Vec::new(),
        ntm: Vec::new(),
    };
    let mut attack = FeaturePair {
        stm: Vec::new(),
        ntm: Vec::new(),
    };
    let mut pawn_pair = FeaturePair {
        stm: Vec::new(),
        ntm: Vec::new(),
    };

    let mirrors = [
        mirror_for_king(pos.our_ksq()),
        mirror_for_king(pos.opp_ksq()),
    ];
    let buckets = [king_bucket16(pos.our_ksq()), king_bucket16(pos.opp_ksq())];

    for (raw_piece, square) in *pos {
        let piece_type = usize::from(raw_piece & 7);
        let stm_colour = usize::from((raw_piece & 8) != 0);
        for ntm in [false, true] {
            let perspective = usize::from(ntm);
            let relative_colour = stm_colour ^ perspective;
            let relative_sq = relative_square(square, ntm, mirrors[perspective]);
            let feature = piece_feature_index(
                buckets[perspective],
                relative_colour,
                piece_type,
                relative_sq,
            );
            if ntm {
                piece.ntm.push(feature);
            } else {
                piece.stm.push(feature);
            }
        }
    }

    for (attacker, from) in *pos {
        let attacker_type = usize::from(attacker & 7);
        let stm_attacker_colour = usize::from((attacker & 8) != 0);
        let mut targets = occupied_attacks(attacker, from, view.occupied);
        while targets != 0 {
            let to = targets.trailing_zeros() as u8;
            targets &= targets - 1;
            let victim = view.piece_on(to)?;
            let victim_type = usize::from(victim & 7);
            let stm_victim_colour = usize::from((victim & 8) != 0);
            for ntm in [false, true] {
                let perspective = usize::from(ntm);
                let feature = attack_feature_index(
                    stm_attacker_colour ^ perspective,
                    attacker_type,
                    relative_square(from, ntm, mirrors[perspective]),
                    stm_victim_colour ^ perspective,
                    victim_type,
                    relative_square(to, ntm, mirrors[perspective]),
                );
                if ntm {
                    attack.ntm.push(feature);
                } else {
                    attack.stm.push(feature);
                }
            }
        }
    }

    for ntm in [false, true] {
        let perspective = usize::from(ntm);
        let mut tokens = Vec::new();
        for (raw_piece, square) in *pos {
            if raw_piece & 7 != 0 {
                continue;
            }
            let relative_sq = relative_square_no_mirror(square, ntm);
            if let Some(pawn_sq) = pawn_square48(relative_sq) {
                let relative_colour = usize::from((raw_piece & 8) != 0) ^ perspective;
                tokens.push(relative_colour * PAWN_SQUARES + pawn_sq);
            }
        }
        for i in 0..tokens.len() {
            for j in (i + 1)..tokens.len() {
                if let Some(row) = pawn_pair_row(tokens[i], tokens[j]) {
                    if ntm {
                        pawn_pair.ntm.push(row);
                    } else {
                        pawn_pair.stm.push(row);
                    }
                }
            }
        }
    }

    Ok(EncodedPosition {
        piece: FeaturePair::checked(piece.stm, piece.ntm, PIECE_INPUTS)?,
        attack: FeaturePair::checked(attack.stm, attack.ntm, ATTACK_INPUTS)?,
        pawn_pair: FeaturePair::checked(pawn_pair.stm, pawn_pair.ntm, PAWN_PAIR_INPUTS)?,
        bucket: output_bucket(pos),
    })
}

fn fen_side_is_black(fen: &str) -> bool {
    fen.split('|')
        .next()
        .unwrap_or(fen)
        .split_whitespace()
        .nth(1)
        == Some("b")
}

fn dump_indices(label: &str, indices: &[usize]) {
    print!("{label}");
    for index in indices {
        print!(" {index}");
    }
    println!();
}

fn dump_fen(fen: &str) -> Result<(), String> {
    let mut input = fen.to_string();
    if !input.contains('|') {
        input.push_str(" | 0 | 0.5");
    }
    let black_to_move = fen_side_is_black(&input);
    let board: ChessBoard = input
        .parse()
        .map_err(|error| format!("invalid --dump-fen: {error}"))?;
    let encoded = encode_position(&board)?;
    let (white_piece, black_piece) = if black_to_move {
        (&encoded.piece.ntm, &encoded.piece.stm)
    } else {
        (&encoded.piece.stm, &encoded.piece.ntm)
    };
    let (white_attack, black_attack) = if black_to_move {
        (&encoded.attack.ntm, &encoded.attack.stm)
    } else {
        (&encoded.attack.stm, &encoded.attack.ntm)
    };
    let (white_pawn_pair, black_pawn_pair) = if black_to_move {
        (&encoded.pawn_pair.ntm, &encoded.pawn_pair.stm)
    } else {
        (&encoded.pawn_pair.stm, &encoded.pawn_pair.ntm)
    };

    println!("mnue_x2_k16_pawn_q8_a384_features");
    println!("output_bucket {}", encoded.bucket);
    dump_indices("white piece", white_piece);
    dump_indices("black piece", black_piece);
    dump_indices("white attack", white_attack);
    dump_indices("black attack", black_attack);
    dump_indices("white pawn_pair", white_pawn_pair);
    dump_indices("black pawn_pair", black_pawn_pair);
    Ok(())
}

#[derive(Clone, Debug)]
struct ExportedMnue {
    piece_l0w: Vec<i8>,
    attack_l0w: Vec<i8>,
    pawn_pair_l0w: Vec<i8>,
    l0b: Vec<i16>,
    l1w: Vec<i8>,
    l1b: Vec<f32>,
    l2w: Vec<f32>,
    l2b: Vec<f32>,
    l3w: Vec<f32>,
    l3b: Vec<f32>,
}

fn read_u32_le(bytes: &[u8], offset: &mut usize) -> Result<u32, String> {
    let end = *offset + 4;
    let chunk = bytes
        .get(*offset..end)
        .ok_or_else(|| "truncated u32".to_string())?;
    *offset = end;
    Ok(u32::from_le_bytes(chunk.try_into().unwrap()))
}

fn read_i8_vec(bytes: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i8>, String> {
    let end = *offset + count;
    let chunk = bytes
        .get(*offset..end)
        .ok_or_else(|| "truncated i8 tensor".to_string())?;
    *offset = end;
    Ok(chunk.iter().map(|&x| x as i8).collect())
}

fn read_i16_vec(bytes: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i16>, String> {
    let byte_count = count * 2;
    let end = *offset + byte_count;
    let chunk = bytes
        .get(*offset..end)
        .ok_or_else(|| "truncated i16 tensor".to_string())?;
    *offset = end;
    Ok(chunk
        .chunks_exact(2)
        .map(|x| i16::from_le_bytes(x.try_into().unwrap()))
        .collect())
}

fn read_f32_vec(bytes: &[u8], offset: &mut usize, count: usize) -> Result<Vec<f32>, String> {
    let byte_count = count * 4;
    let end = *offset + byte_count;
    let chunk = bytes
        .get(*offset..end)
        .ok_or_else(|| "truncated f32 tensor".to_string())?;
    *offset = end;
    Ok(chunk
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect())
}

fn validate_eval_header(header: &[u32; 28]) -> Result<(), String> {
    let expected = [
        MNUE_MAGIC,
        MNUE_VERSION,
        MNUE_ARCH,
        MNUE_HEADER_BYTES,
        PIECE_INPUTS as u32,
        ATTACK_INPUTS as u32,
        PAWN_PAIR_INPUTS as u32,
        PIECE_HIDDEN as u32,
        ATTACK_HIDDEN as u32,
        PAWN_HIDDEN as u32,
        MERGED_HIDDEN as u32,
        INPUT_BUCKETS as u32,
        OUTPUT_BUCKETS as u32,
        L1_SIZE as u32,
        L2_SIZE as u32,
        SCALE,
        QA,
        PIECE_QUANT as u32,
        PIECE_RESCALE,
        ATTACK_QUANT as u32,
        ATTACK_RESCALE,
        PAWN_PAIR_QUANT as u32,
        PAWN_PAIR_RESCALE,
        L1_QUANT as u32,
        MNUE_FEATURE_VERSION,
        0,
        0,
        0,
    ];
    for (index, (&got, &want)) in header.iter().zip(expected.iter()).enumerate() {
        if got != want {
            return Err(format!(
                "header field {index} mismatch: got {got}, expected {want}"
            ));
        }
    }
    Ok(())
}

fn load_exported_mnue(path: &Path) -> Result<ExportedMnue, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let expected_file = MNUE_HEADER_BYTES as usize + expected_payload_bytes();
    if bytes.len() != expected_file {
        return Err(format!(
            "file size mismatch: got {}, expected {expected_file}",
            bytes.len()
        ));
    }

    let mut offset = 0;
    let mut header = [0_u32; 28];
    for field in &mut header {
        *field = read_u32_le(&bytes, &mut offset)?;
    }
    validate_eval_header(&header)?;

    let network = ExportedMnue {
        piece_l0w: read_i8_vec(&bytes, &mut offset, PIECE_INPUTS * PIECE_HIDDEN)?,
        attack_l0w: read_i8_vec(&bytes, &mut offset, ATTACK_INPUTS * ATTACK_HIDDEN)?,
        pawn_pair_l0w: read_i8_vec(&bytes, &mut offset, PAWN_PAIR_INPUTS * PAWN_HIDDEN)?,
        l0b: read_i16_vec(&bytes, &mut offset, MERGED_HIDDEN)?,
        l1w: read_i8_vec(&bytes, &mut offset, OUTPUT_BUCKETS * L1_SIZE * HEAD_INPUT)?,
        l1b: read_f32_vec(&bytes, &mut offset, OUTPUT_BUCKETS * L1_SIZE)?,
        l2w: read_f32_vec(&bytes, &mut offset, OUTPUT_BUCKETS * L2_SIZE * L1_SIZE)?,
        l2b: read_f32_vec(&bytes, &mut offset, OUTPUT_BUCKETS * L2_SIZE)?,
        l3w: read_f32_vec(&bytes, &mut offset, OUTPUT_BUCKETS * L2_SIZE)?,
        l3b: read_f32_vec(&bytes, &mut offset, OUTPUT_BUCKETS)?,
    };
    if offset != bytes.len() {
        return Err(format!(
            "internal payload cursor mismatch: stopped at {offset}, file has {} bytes",
            bytes.len()
        ));
    }
    Ok(network)
}

fn add_quant_row(accumulator: &mut [i32], weights: &[i8], hidden: usize, row: usize, rescale: u32) {
    let start = row * hidden;
    let weights = &weights[start..start + hidden];
    for (acc, &weight) in accumulator.iter_mut().zip(weights) {
        *acc += i32::from(weight) * rescale as i32;
    }
}

fn pairwise_from_accumulators(piece: &[i32], attack: &[i32], pawn: &[i32], l0b: &[i16]) -> Vec<u8> {
    let mut merged = vec![0_i32; MERGED_HIDDEN];
    for column in 0..MERGED_HIDDEN {
        let mut value = i32::from(l0b[column]) + piece[column] + pawn[column];
        if column < ATTACK_HIDDEN {
            value += attack[column];
        }
        merged[column] = value;
    }

    let mut pairwise = vec![0_u8; PAIRWISE];
    for index in 0..PAIRWISE {
        let left = merged[index].clamp(0, QA as i32);
        let right = merged[index + PAIRWISE].clamp(0, QA as i32);
        pairwise[index] = ((left * right) / QA as i32) as u8;
    }
    pairwise
}

fn crelu(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn forward_exported(network: &ExportedMnue, board: &ChessBoard) -> Result<i32, String> {
    let encoded = encode_position(board)?;
    let mut piece_stm = vec![0_i32; PIECE_HIDDEN];
    let mut piece_ntm = vec![0_i32; PIECE_HIDDEN];
    let mut attack_stm = vec![0_i32; ATTACK_HIDDEN];
    let mut attack_ntm = vec![0_i32; ATTACK_HIDDEN];
    let mut pawn_stm = vec![0_i32; PAWN_HIDDEN];
    let mut pawn_ntm = vec![0_i32; PAWN_HIDDEN];

    for &feature in &encoded.piece.stm {
        add_quant_row(
            &mut piece_stm,
            &network.piece_l0w,
            PIECE_HIDDEN,
            feature,
            PIECE_RESCALE,
        );
    }
    for &feature in &encoded.piece.ntm {
        add_quant_row(
            &mut piece_ntm,
            &network.piece_l0w,
            PIECE_HIDDEN,
            feature,
            PIECE_RESCALE,
        );
    }
    for &feature in &encoded.attack.stm {
        add_quant_row(
            &mut attack_stm,
            &network.attack_l0w,
            ATTACK_HIDDEN,
            feature,
            ATTACK_RESCALE,
        );
    }
    for &feature in &encoded.attack.ntm {
        add_quant_row(
            &mut attack_ntm,
            &network.attack_l0w,
            ATTACK_HIDDEN,
            feature,
            ATTACK_RESCALE,
        );
    }
    for &feature in &encoded.pawn_pair.stm {
        add_quant_row(
            &mut pawn_stm,
            &network.pawn_pair_l0w,
            PAWN_HIDDEN,
            feature,
            PAWN_PAIR_RESCALE,
        );
    }
    for &feature in &encoded.pawn_pair.ntm {
        add_quant_row(
            &mut pawn_ntm,
            &network.pawn_pair_l0w,
            PAWN_HIDDEN,
            feature,
            PAWN_PAIR_RESCALE,
        );
    }

    let stm_pairwise = pairwise_from_accumulators(&piece_stm, &attack_stm, &pawn_stm, &network.l0b);
    let ntm_pairwise = pairwise_from_accumulators(&piece_ntm, &attack_ntm, &pawn_ntm, &network.l0b);
    let mut input = vec![0_u8; HEAD_INPUT];
    input[..PAIRWISE].copy_from_slice(&stm_pairwise);
    input[PAIRWISE..].copy_from_slice(&ntm_pairwise);

    let bucket = usize::from(encoded.bucket);
    let l1_base = bucket * L1_SIZE * HEAD_INPUT;
    let l1b_base = bucket * L1_SIZE;
    let mut hidden1 = [0.0_f32; L1_SIZE];
    let l1_scale = 1.0 / (QA as f32 * L1_QUANT as f32);
    for row in 0..L1_SIZE {
        let mut sum = network.l1b[l1b_base + row];
        let weights = &network.l1w[l1_base + row * HEAD_INPUT..l1_base + (row + 1) * HEAD_INPUT];
        for (&activation, &weight) in input.iter().zip(weights) {
            if activation != 0 {
                sum += f32::from(activation) * f32::from(weight) * l1_scale;
            }
        }
        hidden1[row] = crelu(sum);
    }

    let l2_base = bucket * L2_SIZE * L1_SIZE;
    let l2b_base = bucket * L2_SIZE;
    let mut hidden2 = [0.0_f32; L2_SIZE];
    for row in 0..L2_SIZE {
        let mut sum = network.l2b[l2b_base + row];
        let weights = &network.l2w[l2_base + row * L1_SIZE..l2_base + (row + 1) * L1_SIZE];
        for (&activation, &weight) in hidden1.iter().zip(weights) {
            sum += activation * weight;
        }
        hidden2[row] = crelu(sum);
    }

    let mut output = network.l3b[bucket];
    let l3_base = bucket * L2_SIZE;
    for (&activation, &weight) in hidden2.iter().zip(&network.l3w[l3_base..l3_base + L2_SIZE]) {
        output += activation * weight;
    }
    Ok((f64::from(output) * f64::from(SCALE)).round() as i32)
}

#[derive(Clone, Debug)]
struct FloatNetwork {
    piece_l0w: Vec<f32>,
    attack_l0w: Vec<f32>,
    pawn_pair_l0w: Vec<f32>,
    l0b: Vec<f32>,
    l1w: Vec<f32>,
    l1b: Vec<f32>,
    l2w: Vec<f32>,
    l2b: Vec<f32>,
    l3w: Vec<f32>,
    l3b: Vec<f32>,
}

impl FloatNetwork {
    fn from_store(store: &ModelWeights) -> Self {
        Self::from_store_with_l0_scale(store, 1.0)
    }

    fn from_store_with_l0_scale(store: &ModelWeights, l0_scale: f32) -> Self {
        let scale_l0 = |mut values: Vec<f32>| {
            if l0_scale != 1.0 {
                for value in &mut values {
                    *value *= l0_scale;
                }
            }
            values
        };
        Self {
            piece_l0w: scale_l0(floats(store, "piece_l0w")),
            attack_l0w: scale_l0(floats(store, "attack_l0w")),
            pawn_pair_l0w: scale_l0(floats(store, "pawn_pair_l0w")),
            l0b: floats(store, "l0b"),
            l1w: floats(store, "l1w"),
            l1b: floats(store, "l1b"),
            l2w: floats(store, "l2w"),
            l2b: floats(store, "l2b"),
            l3w: floats(store, "l3w"),
            l3b: floats(store, "l3b"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ForwardMask {
    l0b: bool,
    piece: bool,
    attack: bool,
    pawn_pair: bool,
}

impl ForwardMask {
    const ALL: Self = Self {
        l0b: true,
        piece: true,
        attack: true,
        pawn_pair: true,
    };
    const HEAD_BIAS_ONLY: Self = Self {
        l0b: false,
        piece: false,
        attack: false,
        pawn_pair: false,
    };
    const L0B_ONLY: Self = Self {
        l0b: true,
        piece: false,
        attack: false,
        pawn_pair: false,
    };
    const PIECE_ONLY: Self = Self {
        l0b: true,
        piece: true,
        attack: false,
        pawn_pair: false,
    };
    const ATTACK_ONLY: Self = Self {
        l0b: true,
        piece: false,
        attack: true,
        pawn_pair: false,
    };
    const PAWN_ONLY: Self = Self {
        l0b: true,
        piece: false,
        attack: false,
        pawn_pair: true,
    };
}

#[derive(Clone, Copy, Debug)]
struct LaneStats {
    min: f32,
    max: f32,
    max_abs: f32,
    mean_abs: f32,
    rms: f32,
    nonzero: usize,
    clipped: usize,
    low_clipped: usize,
    high_clipped: usize,
    abs_gt_clamp: usize,
}

fn lane_stats(values: &[f32]) -> LaneStats {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut nonzero = 0;
    let mut clipped = 0;
    let mut low_clipped = 0;
    let mut high_clipped = 0;
    let mut abs_gt_clamp = 0;
    for &value in values {
        min = min.min(value);
        max = max.max(value);
        max_abs = max_abs.max(value.abs());
        sum_abs += f64::from(value.abs());
        sum_sq += f64::from(value) * f64::from(value);
        if value != 0.0 {
            nonzero += 1;
        }
        if value.abs() > 1.0 {
            abs_gt_clamp += 1;
        }
        if value < 0.0 {
            low_clipped += 1;
            clipped += 1;
        } else if value > 1.0 {
            high_clipped += 1;
            clipped += 1;
        }
    }
    let count = values.len().max(1) as f64;
    LaneStats {
        min,
        max,
        max_abs,
        mean_abs: (sum_abs / count) as f32,
        rms: (sum_sq / count).sqrt() as f32,
        nonzero,
        clipped,
        low_clipped,
        high_clipped,
        abs_gt_clamp,
    }
}

#[derive(Clone, Copy, Debug)]
struct PairwiseStats {
    min: f32,
    max: f32,
    max_abs: f32,
    mean_abs: f32,
    rms: f32,
    pair_nonzero: usize,
    pair_clipped: usize,
    low_clipped_lanes: usize,
    high_clipped_lanes: usize,
    abs_gt_clamp_lanes: usize,
}

#[derive(Clone, Copy, Debug)]
struct ActivationStats {
    piece_stm: LaneStats,
    piece_ntm: LaneStats,
    attack_stm: LaneStats,
    attack_ntm: LaneStats,
    pawn_stm: LaneStats,
    pawn_ntm: LaneStats,
    merged_stm: PairwiseStats,
    merged_ntm: PairwiseStats,
}

fn print_pairwise_precrelu(label: &str, name: &str, stats: PairwiseStats) {
    eprintln!(
        "calibration {label} precrelu {name} min {:.3} max {:.3} mean_abs {:.3} rms {:.3} abs_gt1 {}/{} low {}/{} high {}/{} post_crelu_pair_nonzero {} pair_clipped {}/{}",
        stats.min,
        stats.max,
        stats.mean_abs,
        stats.rms,
        stats.abs_gt_clamp_lanes,
        MERGED_HIDDEN,
        stats.low_clipped_lanes,
        MERGED_HIDDEN,
        stats.high_clipped_lanes,
        MERGED_HIDDEN,
        stats.pair_nonzero,
        stats.pair_clipped,
        PAIRWISE
    );
}

fn print_lane_precrelu(label: &str, name: &str, stats: LaneStats, total: usize) {
    eprintln!(
        "calibration {label} precrelu {name} min {:.3} max {:.3} mean_abs {:.3} rms {:.3} abs_gt1 {}/{} low {}/{} high {}/{} nonzero {} clipped {}/{}",
        stats.min,
        stats.max,
        stats.mean_abs,
        stats.rms,
        stats.abs_gt_clamp,
        total,
        stats.low_clipped,
        total,
        stats.high_clipped,
        total,
        stats.nonzero,
        stats.clipped,
        total
    );
}

fn add_float_row(accumulator: &mut [f32], weights: &[f32], hidden: usize, row: usize) {
    let start = row * hidden;
    let weights = &weights[start..start + hidden];
    for (acc, &weight) in accumulator.iter_mut().zip(weights) {
        *acc += weight;
    }
}

fn pairwise_float_from_accumulators(
    piece: &[f32],
    attack: &[f32],
    pawn: &[f32],
    l0b: &[f32],
    mask: ForwardMask,
) -> (Vec<f32>, PairwiseStats) {
    let mut merged = vec![0.0_f32; MERGED_HIDDEN];
    for column in 0..MERGED_HIDDEN {
        let mut value = 0.0;
        if mask.l0b {
            value += l0b[column];
        }
        if mask.piece {
            value += piece[column];
        }
        if mask.pawn_pair {
            value += pawn[column];
        }
        if mask.attack && column < ATTACK_HIDDEN {
            value += attack[column];
        }
        merged[column] = value;
    }

    let mut pairwise = vec![0.0_f32; PAIRWISE];
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut pair_clipped = 0;
    let mut nonzero = 0;
    let mut low_clipped_lanes = 0;
    let mut high_clipped_lanes = 0;
    let mut abs_gt_clamp_lanes = 0;
    for index in 0..PAIRWISE {
        let left = merged[index];
        let right = merged[index + PAIRWISE];
        for value in [left, right] {
            min = min.min(value);
            max = max.max(value);
            max_abs = max_abs.max(value.abs());
            sum_abs += f64::from(value.abs());
            sum_sq += f64::from(value) * f64::from(value);
            if value.abs() > 1.0 {
                abs_gt_clamp_lanes += 1;
            }
            if value < 0.0 {
                low_clipped_lanes += 1;
            } else if value > 1.0 {
                high_clipped_lanes += 1;
            }
        }
        let cl = left.clamp(0.0, 1.0);
        let cr = right.clamp(0.0, 1.0);
        if left != cl || right != cr {
            pair_clipped += 1;
        }
        let value = cl * cr;
        if value != 0.0 {
            nonzero += 1;
        }
        pairwise[index] = value;
    }
    let count = MERGED_HIDDEN as f64;
    (
        pairwise,
        PairwiseStats {
            min,
            max,
            max_abs,
            mean_abs: (sum_abs / count) as f32,
            rms: (sum_sq / count).sqrt() as f32,
            pair_nonzero: nonzero,
            pair_clipped,
            low_clipped_lanes,
            high_clipped_lanes,
            abs_gt_clamp_lanes,
        },
    )
}

fn rebuild_float_accumulators(
    network: &FloatNetwork,
    encoded: &EncodedPosition,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut piece_stm = vec![0.0_f32; PIECE_HIDDEN];
    let mut piece_ntm = vec![0.0_f32; PIECE_HIDDEN];
    let mut attack_stm = vec![0.0_f32; ATTACK_HIDDEN];
    let mut attack_ntm = vec![0.0_f32; ATTACK_HIDDEN];
    let mut pawn_stm = vec![0.0_f32; PAWN_HIDDEN];
    let mut pawn_ntm = vec![0.0_f32; PAWN_HIDDEN];

    for &feature in &encoded.piece.stm {
        add_float_row(&mut piece_stm, &network.piece_l0w, PIECE_HIDDEN, feature);
    }
    for &feature in &encoded.piece.ntm {
        add_float_row(&mut piece_ntm, &network.piece_l0w, PIECE_HIDDEN, feature);
    }
    for &feature in &encoded.attack.stm {
        add_float_row(&mut attack_stm, &network.attack_l0w, ATTACK_HIDDEN, feature);
    }
    for &feature in &encoded.attack.ntm {
        add_float_row(&mut attack_ntm, &network.attack_l0w, ATTACK_HIDDEN, feature);
    }
    for &feature in &encoded.pawn_pair.stm {
        add_float_row(&mut pawn_stm, &network.pawn_pair_l0w, PAWN_HIDDEN, feature);
    }
    for &feature in &encoded.pawn_pair.ntm {
        add_float_row(&mut pawn_ntm, &network.pawn_pair_l0w, PAWN_HIDDEN, feature);
    }

    (
        piece_stm, piece_ntm, attack_stm, attack_ntm, pawn_stm, pawn_ntm,
    )
}

fn forward_float_masked(
    network: &FloatNetwork,
    board: &ChessBoard,
    mask: ForwardMask,
) -> Result<(i32, ActivationStats), String> {
    let encoded = encode_position(board)?;
    let (piece_stm, piece_ntm, attack_stm, attack_ntm, pawn_stm, pawn_ntm) =
        rebuild_float_accumulators(network, &encoded);
    let piece_stm_stats = lane_stats(&piece_stm);
    let piece_ntm_stats = lane_stats(&piece_ntm);
    let attack_stm_stats = lane_stats(&attack_stm);
    let attack_ntm_stats = lane_stats(&attack_ntm);
    let pawn_stm_stats = lane_stats(&pawn_stm);
    let pawn_ntm_stats = lane_stats(&pawn_ntm);

    let (stm_pairwise, stm_stats) =
        pairwise_float_from_accumulators(&piece_stm, &attack_stm, &pawn_stm, &network.l0b, mask);
    let (ntm_pairwise, ntm_stats) =
        pairwise_float_from_accumulators(&piece_ntm, &attack_ntm, &pawn_ntm, &network.l0b, mask);

    let mut input = vec![0.0_f32; HEAD_INPUT];
    input[..PAIRWISE].copy_from_slice(&stm_pairwise);
    input[PAIRWISE..].copy_from_slice(&ntm_pairwise);

    let bucket = usize::from(encoded.bucket);
    let l1_rows = OUTPUT_BUCKETS * L1_SIZE;
    let mut hidden1 = [0.0_f32; L1_SIZE];
    for row in 0..L1_SIZE {
        let row_index = bucket * L1_SIZE + row;
        let mut sum = network.l1b[row_index];
        for (column, &activation) in input.iter().enumerate() {
            sum += activation * network.l1w[column * l1_rows + row_index];
        }
        hidden1[row] = sum.clamp(0.0, 1.0);
    }

    let l2_rows = OUTPUT_BUCKETS * L2_SIZE;
    let mut hidden2 = [0.0_f32; L2_SIZE];
    for row in 0..L2_SIZE {
        let row_index = bucket * L2_SIZE + row;
        let mut sum = network.l2b[row_index];
        for (column, &activation) in hidden1.iter().enumerate() {
            sum += activation * network.l2w[column * l2_rows + row_index];
        }
        hidden2[row] = sum.clamp(0.0, 1.0);
    }

    let mut output = network.l3b[bucket];
    for (column, &activation) in hidden2.iter().enumerate() {
        output += activation * network.l3w[column * OUTPUT_BUCKETS + bucket];
    }

    Ok((
        (f64::from(output) * f64::from(SCALE)).round() as i32,
        ActivationStats {
            piece_stm: piece_stm_stats,
            piece_ntm: piece_ntm_stats,
            attack_stm: attack_stm_stats,
            attack_ntm: attack_ntm_stats,
            pawn_stm: pawn_stm_stats,
            pawn_ntm: pawn_ntm_stats,
            merged_stm: stm_stats,
            merged_ntm: ntm_stats,
        },
    ))
}

fn forward_float(
    network: &FloatNetwork,
    board: &ChessBoard,
) -> Result<(i32, ActivationStats), String> {
    forward_float_masked(network, board, ForwardMask::ALL)
}

fn material_units(board: &ChessBoard) -> i32 {
    let mut total = 0_i32;
    for (raw_piece, _) in *board {
        total += match raw_piece & 7 {
            0 => 1,
            1 | 2 => 3,
            3 => 5,
            4 => 9,
            _ => 0,
        };
    }
    total.clamp(0, 78)
}

fn phase_units(board: &ChessBoard) -> i32 {
    let mut total = 0_i32;
    for (raw_piece, _) in *board {
        total += match raw_piece & 7 {
            1 | 2 => 1,
            3 => 2,
            4 => 4,
            _ => 0,
        };
    }
    total
}

fn raw_to_searchcp(raw: i32, board: &ChessBoard) -> i32 {
    let material = f64::from(material_units(board)) / 78.0;
    let mut denominator = 144.0 + 16.0 * material;
    if phase_units(board) < 10 {
        denominator -= 8.0;
    }
    denominator = denominator.clamp(128.0, 168.0);
    let cp = (f64::from(raw.abs().min(32_768)) * 100.0 / denominator).round() as i32;
    if raw >= 0 { cp } else { -cp }
}

fn eval_mnue(path: &Path, fens: &[String]) -> Result<(), String> {
    let network = load_exported_mnue(path)?;
    println!("mnue_x2_k16_pawn_q8_a384_eval");
    println!("file {}", path.display());
    println!(
        "scale {SCALE} qa {QA} piece_rescale {PIECE_RESCALE} attack_rescale {ATTACK_RESCALE} pawn_pair_rescale {PAWN_PAIR_RESCALE} l1_quant {L1_QUANT}"
    );

    if fens.is_empty() {
        return Err("--eval-mnue requires at least one --eval-fen".to_string());
    }

    for fen in fens {
        let mut input = fen.to_string();
        if !input.contains('|') {
            input.push_str(" | 0 | 0.5");
        }
        let black_to_move = fen_side_is_black(&input);
        let board: ChessBoard = input
            .parse()
            .map_err(|error| format!("invalid --eval-fen: {error}"))?;
        let raw_stm = forward_exported(&network, &board)?;
        let raw_white = if black_to_move { -raw_stm } else { raw_stm };
        let searchcp_stm = raw_to_searchcp(raw_stm, &board);
        let searchcp_white = if black_to_move {
            -searchcp_stm
        } else {
            searchcp_stm
        };
        let bucket = output_bucket(&board);
        println!("fen {fen}");
        println!("rust_raw_stm {raw_stm}");
        println!("rust_raw_white {raw_white}");
        println!("rust_searchcp_stm {searchcp_stm}");
        println!("rust_searchcp_white {searchcp_white}");
        println!("rust_output_bucket {bucket}");
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct Sample {
    encoded: EncodedPosition,
    bucket: u8,
    target: f32,
}

fn sigmoid(score: i16, scale: f32) -> f32 {
    1.0 / (1.0 + (-f32::from(score) / scale).exp())
}

fn target(pos: &ChessBoard) -> f32 {
    0.75 * pos.result() + 0.25 * sigmoid(pos.score(), SCALE as f32)
}

#[derive(Clone, Debug)]
struct RunningStats {
    count: u64,
    sum: f64,
    sum_sq: f64,
    min: f64,
    max: f64,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn push(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_sq += value * value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    fn stddev(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            let mean = self.mean();
            ((self.sum_sq / self.count as f64) - mean * mean)
                .max(0.0)
                .sqrt()
        }
    }
}

#[derive(Clone, Debug)]
struct DataStats {
    records: u64,
    accepted: u64,
    invalid: u64,
    score: RunningStats,
    result: RunningStats,
    target: RunningStats,
    score_positive: u64,
    score_negative: u64,
    score_zero: u64,
    result_loss: u64,
    result_draw: u64,
    result_win: u64,
    target_near_half: u64,
}

impl DataStats {
    fn new() -> Self {
        Self {
            records: 0,
            accepted: 0,
            invalid: 0,
            score: RunningStats::new(),
            result: RunningStats::new(),
            target: RunningStats::new(),
            score_positive: 0,
            score_negative: 0,
            score_zero: 0,
            result_loss: 0,
            result_draw: 0,
            result_win: 0,
            target_near_half: 0,
        }
    }

    fn push(&mut self, board: &ChessBoard, valid_features: bool) {
        self.records += 1;
        if !valid_features {
            self.invalid += 1;
            return;
        }
        self.accepted += 1;
        let score = f64::from(board.score());
        let result = f64::from(board.result());
        let target = f64::from(target(board));
        self.score.push(score);
        self.result.push(result);
        self.target.push(target);
        match board.score().cmp(&0) {
            std::cmp::Ordering::Greater => self.score_positive += 1,
            std::cmp::Ordering::Less => self.score_negative += 1,
            std::cmp::Ordering::Equal => self.score_zero += 1,
        }
        match board.result {
            0 => self.result_loss += 1,
            1 => self.result_draw += 1,
            2 => self.result_win += 1,
            _ => {}
        }
        if (target - 0.5).abs() <= 0.05 {
            self.target_near_half += 1;
        }
    }

    fn print(&self) {
        println!("Data stats records inspected: {}", self.records);
        println!("Data stats accepted: {}", self.accepted);
        println!("Data stats invalid feature positions: {}", self.invalid);
        println!(
            "Data stats side-to-move distribution: unavailable in bulletformat ChessBoard records"
        );
        println!(
            "Data stats score mean/std/min/max: {:.3} {:.3} {:.0} {:.0}",
            self.score.mean(),
            self.score.stddev(),
            self.score.min,
            self.score.max
        );
        println!(
            "Data stats target mean/std/min/max: {:.6} {:.6} {:.6} {:.6}",
            self.target.mean(),
            self.target.stddev(),
            self.target.min,
            self.target.max
        );
        println!(
            "Data stats result mean/std/min/max: {:.6} {:.6} {:.1} {:.1}",
            self.result.mean(),
            self.result.stddev(),
            self.result.min,
            self.result.max
        );
        println!(
            "Data stats score pos/zero/neg: {} {} {}",
            self.score_positive, self.score_zero, self.score_negative
        );
        println!(
            "Data stats result win/draw/loss: {} {} {}",
            self.result_win, self.result_draw, self.result_loss
        );
        println!(
            "Data stats target within 0.05 of 0.5: {}",
            self.target_near_half
        );
    }
}

fn inspect_data_stats(kind: InputDataKind, paths: &[PathBuf], limit: u64) -> io::Result<DataStats> {
    let mut stats = DataStats::new();
    if limit == 0 {
        return Ok(stats);
    }

    match kind {
        InputDataKind::Direct => {
            'files: for path in paths {
                let mut reader = BufReader::with_capacity(4 * 1024 * 1024, fs::File::open(path)?);
                loop {
                    if stats.records >= limit {
                        break 'files;
                    }
                    let mut bytes = [0_u8; 32];
                    match reader.read_exact(&mut bytes) {
                        Ok(()) => {}
                        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
                        Err(error) => return Err(error),
                    }
                    let board = decode_record(&bytes);
                    stats.push(&board, encode_position(&board).is_ok());
                }
            }
        }
        InputDataKind::SfBinpack => {
            let path_strings = path_strings(paths);
            let path_refs = path_strings.iter().map(String::as_str).collect::<Vec<_>>();
            let loader = SfBinpackLoader::new_concat_multiple(
                &path_refs,
                BINPACK_INSPECT_BUFFER_MB,
                DEFAULT_PREP_THREADS,
                accept_all_binpack,
            );
            loader.map_chunks(0, |chunk: &[ChessBoard]| {
                for board in chunk {
                    if stats.records >= limit {
                        return true;
                    }
                    stats.push(board, encode_position(board).is_ok());
                }
                false
            });
        }
    }
    Ok(stats)
}

fn decode_record(bytes: &[u8; 32]) -> ChessBoard {
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

fn read_samples<F>(
    kind: InputDataKind,
    paths: &[PathBuf],
    limit: u64,
    mut accept: F,
) -> io::Result<usize>
where
    F: FnMut(Sample) -> bool,
{
    let mut accepted = 0_usize;

    match kind {
        InputDataKind::Direct => {
            'files: for path in paths {
                let mut reader = BufReader::with_capacity(4 * 1024 * 1024, fs::File::open(path)?);
                loop {
                    if limit > 0 && accepted as u64 >= limit {
                        break 'files;
                    }
                    let mut bytes = [0_u8; 32];
                    match reader.read_exact(&mut bytes) {
                        Ok(()) => {}
                        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
                        Err(error) => return Err(error),
                    }
                    let board = decode_record(&bytes);
                    let encoded = match encode_position(&board) {
                        Ok(value) => value,
                        Err(_) => continue,
                    };
                    let sample = Sample {
                        bucket: encoded.bucket,
                        target: target(&board),
                        encoded,
                    };
                    accepted += 1;
                    if accept(sample) {
                        break 'files;
                    }
                }
            }
        }
        InputDataKind::SfBinpack => {
            let path_strings = path_strings(paths);
            let path_refs = path_strings.iter().map(String::as_str).collect::<Vec<_>>();
            let loader = SfBinpackLoader::new_concat_multiple(
                &path_refs,
                BINPACK_INSPECT_BUFFER_MB,
                DEFAULT_PREP_THREADS,
                accept_all_binpack,
            );
            loader.map_chunks(0, |chunk: &[ChessBoard]| {
                for board in chunk {
                    if limit > 0 && accepted as u64 >= limit {
                        return true;
                    }
                    let encoded = match encode_position(board) {
                        Ok(value) => value,
                        Err(_) => continue,
                    };
                    let sample = Sample {
                        bucket: encoded.bucket,
                        target: target(board),
                        encoded,
                    };
                    accepted += 1;
                    if accept(sample) {
                        return true;
                    }
                }
                false
            });
        }
    }
    Ok(accepted)
}

fn push_sparse(dst: &mut Vec<i32>, indices: &[usize], max_active: usize) {
    assert!(
        indices.len() <= max_active,
        "{} active features exceeds configured maximum {max_active}",
        indices.len()
    );
    dst.extend(indices.iter().map(|&x| i32::try_from(x).unwrap()));
    dst.resize(dst.len() + max_active - indices.len(), -1);
}

fn write_sparse(dst: &mut [i32], indices: &[usize]) {
    assert!(
        indices.len() <= dst.len(),
        "{} active features exceeds configured maximum {}",
        indices.len(),
        dst.len()
    );
    for (out, &index) in dst.iter_mut().zip(indices) {
        *out = i32::try_from(index).unwrap();
    }
}

fn prepare_batch(samples: &[Sample]) -> PreparedBatchHost {
    let mut piece_stm = Vec::with_capacity(samples.len() * PIECE_MAX_ACTIVE);
    let mut piece_ntm = Vec::with_capacity(samples.len() * PIECE_MAX_ACTIVE);
    let mut attack_stm = Vec::with_capacity(samples.len() * ATTACK_MAX_ACTIVE);
    let mut attack_ntm = Vec::with_capacity(samples.len() * ATTACK_MAX_ACTIVE);
    let mut pawn_stm = Vec::with_capacity(samples.len() * PAWN_PAIR_MAX_ACTIVE);
    let mut pawn_ntm = Vec::with_capacity(samples.len() * PAWN_PAIR_MAX_ACTIVE);
    let mut buckets = Vec::with_capacity(samples.len());
    let mut targets = Vec::with_capacity(samples.len());
    for sample in samples {
        push_sparse(&mut piece_stm, &sample.encoded.piece.stm, PIECE_MAX_ACTIVE);
        push_sparse(&mut piece_ntm, &sample.encoded.piece.ntm, PIECE_MAX_ACTIVE);
        push_sparse(
            &mut attack_stm,
            &sample.encoded.attack.stm,
            ATTACK_MAX_ACTIVE,
        );
        push_sparse(
            &mut attack_ntm,
            &sample.encoded.attack.ntm,
            ATTACK_MAX_ACTIVE,
        );
        push_sparse(
            &mut pawn_stm,
            &sample.encoded.pawn_pair.stm,
            PAWN_PAIR_MAX_ACTIVE,
        );
        push_sparse(
            &mut pawn_ntm,
            &sample.encoded.pawn_pair.ntm,
            PAWN_PAIR_MAX_ACTIVE,
        );
        buckets.push(i32::from(sample.bucket));
        targets.push(sample.target);
    }
    PreparedBatchHost {
        batch_size: samples.len(),
        inputs: [
            ("piece_stm".to_string(), TValue::I32(piece_stm)),
            ("piece_ntm".to_string(), TValue::I32(piece_ntm)),
            ("attack_stm".to_string(), TValue::I32(attack_stm)),
            ("attack_ntm".to_string(), TValue::I32(attack_ntm)),
            ("pawn_pair_stm".to_string(), TValue::I32(pawn_stm)),
            ("pawn_pair_ntm".to_string(), TValue::I32(pawn_ntm)),
            ("buckets".to_string(), TValue::I32(buckets)),
            ("targets".to_string(), TValue::F32(targets)),
        ]
        .into(),
    }
}

fn prepare_boards_batch(boards: &[ChessBoard], threads: usize) -> PreparedBatchHost {
    assert!(!boards.is_empty());
    let threads = threads.clamp(1, boards.len());
    let chunk_size = boards.len().div_ceil(threads);
    let piece_chunk = chunk_size * PIECE_MAX_ACTIVE;
    let attack_chunk = chunk_size * ATTACK_MAX_ACTIVE;
    let pawn_chunk = chunk_size * PAWN_PAIR_MAX_ACTIVE;

    let mut piece_stm = vec![-1; boards.len() * PIECE_MAX_ACTIVE];
    let mut piece_ntm = vec![-1; boards.len() * PIECE_MAX_ACTIVE];
    let mut attack_stm = vec![-1; boards.len() * ATTACK_MAX_ACTIVE];
    let mut attack_ntm = vec![-1; boards.len() * ATTACK_MAX_ACTIVE];
    let mut pawn_stm = vec![-1; boards.len() * PAWN_PAIR_MAX_ACTIVE];
    let mut pawn_ntm = vec![-1; boards.len() * PAWN_PAIR_MAX_ACTIVE];
    let mut buckets = vec![0; boards.len()];
    let mut targets = vec![0.0; boards.len()];

    thread::scope(|scope| {
        let chunks = boards
            .chunks(chunk_size)
            .zip(piece_stm.chunks_mut(piece_chunk))
            .zip(piece_ntm.chunks_mut(piece_chunk))
            .zip(attack_stm.chunks_mut(attack_chunk))
            .zip(attack_ntm.chunks_mut(attack_chunk))
            .zip(pawn_stm.chunks_mut(pawn_chunk))
            .zip(pawn_ntm.chunks_mut(pawn_chunk))
            .zip(buckets.chunks_mut(chunk_size))
            .zip(targets.chunks_mut(chunk_size));

        for (
            (
                (
                    (
                        (
                            (((board_chunk, piece_stm_chunk), piece_ntm_chunk), attack_stm_chunk),
                            attack_ntm_chunk,
                        ),
                        pawn_stm_chunk,
                    ),
                    pawn_ntm_chunk,
                ),
                bucket_chunk,
            ),
            target_chunk,
        ) in chunks
        {
            scope.spawn(move || {
                for (i, board) in board_chunk.iter().enumerate() {
                    let encoded = encode_position(board)
                        .unwrap_or_else(|error| panic!("invalid training record: {error}"));
                    write_sparse(
                        &mut piece_stm_chunk[i * PIECE_MAX_ACTIVE..(i + 1) * PIECE_MAX_ACTIVE],
                        &encoded.piece.stm,
                    );
                    write_sparse(
                        &mut piece_ntm_chunk[i * PIECE_MAX_ACTIVE..(i + 1) * PIECE_MAX_ACTIVE],
                        &encoded.piece.ntm,
                    );
                    write_sparse(
                        &mut attack_stm_chunk[i * ATTACK_MAX_ACTIVE..(i + 1) * ATTACK_MAX_ACTIVE],
                        &encoded.attack.stm,
                    );
                    write_sparse(
                        &mut attack_ntm_chunk[i * ATTACK_MAX_ACTIVE..(i + 1) * ATTACK_MAX_ACTIVE],
                        &encoded.attack.ntm,
                    );
                    write_sparse(
                        &mut pawn_stm_chunk
                            [i * PAWN_PAIR_MAX_ACTIVE..(i + 1) * PAWN_PAIR_MAX_ACTIVE],
                        &encoded.pawn_pair.stm,
                    );
                    write_sparse(
                        &mut pawn_ntm_chunk
                            [i * PAWN_PAIR_MAX_ACTIVE..(i + 1) * PAWN_PAIR_MAX_ACTIVE],
                        &encoded.pawn_pair.ntm,
                    );
                    bucket_chunk[i] = i32::from(encoded.bucket);
                    target_chunk[i] = target(board);
                }
            });
        }
    });

    PreparedBatchHost {
        batch_size: boards.len(),
        inputs: [
            ("piece_stm".to_string(), TValue::I32(piece_stm)),
            ("piece_ntm".to_string(), TValue::I32(piece_ntm)),
            ("attack_stm".to_string(), TValue::I32(attack_stm)),
            ("attack_ntm".to_string(), TValue::I32(attack_ntm)),
            ("pawn_pair_stm".to_string(), TValue::I32(pawn_stm)),
            ("pawn_pair_ntm".to_string(), TValue::I32(pawn_ntm)),
            ("buckets".to_string(), TValue::I32(buckets)),
            ("targets".to_string(), TValue::F32(targets)),
        ]
        .into(),
    }
}

#[derive(Clone)]
struct MnueK16DataLoader {
    kind: InputDataKind,
    paths: Vec<PathBuf>,
    chunk_paths: Vec<PathBuf>,
    chunk_resample_on_exhaustion: bool,
    chunk_shuffle_seed: u64,
    chunk_virtual_epochs: usize,
    chunk_sample: Option<usize>,
    limit: u64,
    threads: usize,
}

impl DataLoader for MnueK16DataLoader {
    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(
        self,
        batch_size: usize,
        mut callback: F,
    ) -> Result<(), DataLoadingError> {
        let path_strings = path_strings(&self.paths);
        let path_refs = path_strings.iter().map(String::as_str).collect::<Vec<_>>();
        let mut batch = Vec::with_capacity(batch_size);
        let mut emitted = 0_u64;

        let mut map_chunk = |chunk: &[ChessBoard]| {
            let mut offset = 0_usize;
            while offset < chunk.len() {
                let used = emitted + batch.len() as u64;
                if self.limit > 0 && used >= self.limit {
                    return true;
                }

                let limit_remaining = if self.limit > 0 {
                    usize::try_from(self.limit - used).unwrap_or(usize::MAX)
                } else {
                    usize::MAX
                };
                let take = (batch_size - batch.len())
                    .min(chunk.len() - offset)
                    .min(limit_remaining);
                if take == 0 {
                    return true;
                }

                batch.extend_from_slice(&chunk[offset..offset + take]);
                offset += take;

                if batch.len() == batch_size {
                    let stop = callback(prepare_boards_batch(&batch, self.threads));
                    emitted += batch_size as u64;
                    batch.clear();
                    if stop {
                        return true;
                    }
                }
            }
            false
        };

        match self.kind {
            InputDataKind::Direct => {
                let direct = DirectSequentialDataLoader::new(&path_refs);
                direct.map_chunks(0, |chunk: &[ChessBoard]| map_chunk(chunk));
            }
            InputDataKind::SfBinpack => {
                if self.chunk_resample_on_exhaustion {
                    let chunk_path_strings = self
                        .chunk_paths
                        .iter()
                        .map(|path| path.to_string_lossy().into_owned())
                        .collect::<Vec<_>>();
                    let binpack = ResamplingSfBinpackLoader::new(
                        chunk_path_strings,
                        BINPACK_BUFFER_MB,
                        self.threads,
                        accept_all_binpack,
                        self.chunk_shuffle_seed,
                        self.chunk_virtual_epochs,
                        self.chunk_sample,
                    );
                    binpack.map_chunks(0, |chunk: &[ChessBoard]| map_chunk(chunk));
                } else {
                    let binpack = SfBinpackLoader::new_concat_multiple(
                        &path_refs,
                        BINPACK_BUFFER_MB,
                        self.threads,
                        accept_all_binpack,
                    );
                    binpack.map_chunks(0, |chunk: &[ChessBoard]| map_chunk(chunk));
                }
            }
        }

        Ok(())
    }
}

struct SingleBatchLoader(PreparedBatchHost);

impl DataLoader for SingleBatchLoader {
    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(
        self,
        _batch_size: usize,
        mut callback: F,
    ) -> Result<(), DataLoadingError> {
        callback(self.0);
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct TrainState {
    accepted_positions_seen: u64,
    training_step: u64,
    last_loss: Option<f32>,
}

type MnueTrainer = Trainer<ExecutionContext, AdamWOptimiser, TrainState>;

fn sparse_affine<'a>(
    weights: ModelNode<'a>,
    input: ModelNode<'a>,
    bias: Option<ModelNode<'a>>,
) -> ModelNode<'a> {
    match bias {
        Some(bias) => weights.matmul(input) + bias,
        None => weights.matmul(input),
    }
}

fn dense_affine<'a>(
    weights: ModelNode<'a>,
    input: ModelNode<'a>,
    bias: ModelNode<'a>,
) -> ModelNode<'a> {
    weights.matmul(input) + bias
}

fn merged_branch<'a>(
    piece_w: ModelNode<'a>,
    attack_w: ModelNode<'a>,
    pawn_w: ModelNode<'a>,
    l0b: ModelNode<'a>,
    piece: ModelNode<'a>,
    attack: ModelNode<'a>,
    pawn_pair: ModelNode<'a>,
) -> ModelNode<'a> {
    let piece = sparse_affine(piece_w, piece, None);
    let attack = sparse_affine(attack_w, attack, None).pad(0, MERGED_HIDDEN - ATTACK_HIDDEN, 0.0);
    let pawn = sparse_affine(pawn_w, pawn_pair, None);
    piece + attack + pawn + l0b
}

fn build_trainer(init: InitScales) -> MnueTrainer {
    let builder = ModelBuilder::default();
    let targets = builder.new_dense_input("targets", Shape::new(1, 1));
    let buckets = builder.new_sparse_input("buckets", Shape::new(OUTPUT_BUCKETS, 1), 1);
    let piece_stm =
        builder.new_sparse_input("piece_stm", Shape::new(PIECE_INPUTS, 1), PIECE_MAX_ACTIVE);
    let piece_ntm =
        builder.new_sparse_input("piece_ntm", Shape::new(PIECE_INPUTS, 1), PIECE_MAX_ACTIVE);
    let attack_stm = builder.new_sparse_input(
        "attack_stm",
        Shape::new(ATTACK_INPUTS, 1),
        ATTACK_MAX_ACTIVE,
    );
    let attack_ntm = builder.new_sparse_input(
        "attack_ntm",
        Shape::new(ATTACK_INPUTS, 1),
        ATTACK_MAX_ACTIVE,
    );
    let pawn_pair_stm = builder.new_sparse_input(
        "pawn_pair_stm",
        Shape::new(PAWN_PAIR_INPUTS, 1),
        PAWN_PAIR_MAX_ACTIVE,
    );
    let pawn_pair_ntm = builder.new_sparse_input(
        "pawn_pair_ntm",
        Shape::new(PAWN_PAIR_INPUTS, 1),
        PAWN_PAIR_MAX_ACTIVE,
    );

    let piece_l0w = builder.new_weights(
        "piece_l0w",
        Shape::new(PIECE_HIDDEN, PIECE_INPUTS),
        InitSettings::Normal {
            mean: 0.0,
            stdev: piece_l0_base_std() * init.piece_l0,
        },
    );
    let attack_l0w = builder.new_weights(
        "attack_l0w",
        Shape::new(ATTACK_HIDDEN, ATTACK_INPUTS),
        InitSettings::Normal {
            mean: 0.0,
            stdev: attack_l0_base_std() * init.attack_l0,
        },
    );
    let pawn_pair_l0w = builder.new_weights(
        "pawn_pair_l0w",
        Shape::new(PAWN_HIDDEN, PAWN_PAIR_INPUTS),
        InitSettings::Normal {
            mean: 0.0,
            stdev: pawn_l0_base_std() * init.pawn_l0,
        },
    );
    let l0b = builder.new_weights("l0b", Shape::new(MERGED_HIDDEN, 1), InitSettings::Zeroed);

    let stm = merged_branch(
        piece_l0w,
        attack_l0w,
        pawn_pair_l0w,
        l0b,
        piece_stm,
        attack_stm,
        pawn_pair_stm,
    )
    .crelu()
    .pairwise_mul();
    let ntm = merged_branch(
        piece_l0w,
        attack_l0w,
        pawn_pair_l0w,
        l0b,
        piece_ntm,
        attack_ntm,
        pawn_pair_ntm,
    )
    .crelu()
    .pairwise_mul();
    let representation = stm.concat(ntm);
    assert_eq!(representation.shape(), Shape::new(HEAD_INPUT, 1));

    let l1w = builder.new_weights(
        "l1w",
        Shape::new(OUTPUT_BUCKETS * L1_SIZE, HEAD_INPUT),
        InitSettings::Normal {
            mean: 0.0,
            stdev: l1_base_std() * init.l1,
        },
    );
    let l1b = builder.new_weights(
        "l1b",
        Shape::new(OUTPUT_BUCKETS * L1_SIZE, 1),
        InitSettings::Zeroed,
    );
    let l2w = builder.new_weights(
        "l2w",
        Shape::new(OUTPUT_BUCKETS * L2_SIZE, L1_SIZE),
        InitSettings::Normal {
            mean: 0.0,
            stdev: l2_base_std() * init.l2,
        },
    );
    let l2b = builder.new_weights(
        "l2b",
        Shape::new(OUTPUT_BUCKETS * L2_SIZE, 1),
        InitSettings::Zeroed,
    );
    let l3w = builder.new_weights(
        "l3w",
        Shape::new(OUTPUT_BUCKETS, L2_SIZE),
        InitSettings::Normal {
            mean: 0.0,
            stdev: l3_base_std() * init.l3,
        },
    );
    let l3b = builder.new_weights("l3b", Shape::new(OUTPUT_BUCKETS, 1), InitSettings::Zeroed);
    let hidden = dense_affine(l1w, representation, l1b)
        .select(buckets)
        .crelu();
    let hidden = dense_affine(l2w, hidden, l2b).select(buckets).crelu();
    let score = dense_affine(l3w, hidden, l3b).select(buckets);
    let loss = score.sigmoid().squared_error(targets);

    let device = Device::<ExecutionContext>::new(0).expect("failed to create Bullet device");
    let model = builder.build(device, loss, score);
    let mut optimiser =
        Optimiser::new(model, AdamWParams::default()).expect("failed to create AdamW optimiser");
    let wide = AdamWParams {
        min_weight: -1.98,
        max_weight: 1.98,
        ..Default::default()
    };
    let pawn = AdamWParams {
        min_weight: -0.99,
        max_weight: 0.99,
        ..Default::default()
    };
    for id in ["piece_l0w", "attack_l0w", "l1w"] {
        optimiser.set_params_for_weight(id, wide);
    }
    optimiser.set_params_for_weight("pawn_pair_l0w", pawn);
    Trainer {
        optimiser,
        state: TrainState {
            accepted_positions_seen: 0,
            training_step: 0,
            last_loss: None,
        },
    }
}

#[derive(Clone, Copy, Debug)]
struct TrainingPlan {
    batch_size: usize,
    batches_per_superbatch: usize,
    superbatches: usize,
    actual_batches: usize,
    actual_positions: u64,
}

fn training_plan(cli: &Cli) -> io::Result<TrainingPlan> {
    let batch_size = if cli.positions > 0 {
        cli.batch_size
            .min(usize::try_from(cli.positions).unwrap_or(usize::MAX))
    } else {
        cli.batch_size
    };
    if batch_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "batch size must be positive",
        ));
    }

    let requested_superbatches = cli.superbatches.unwrap_or(if cli.positions == 0 {
        DEFAULT_FULL_SUPERBATCHES
    } else {
        DEFAULT_POSITION_SUPERBATCHES
    });

    let total_batches = if cli.positions > 0 {
        usize::try_from(cli.positions).unwrap_or(usize::MAX) / batch_size
    } else {
        DEFAULT_BATCHES_PER_SUPERBATCH * requested_superbatches
    };
    if total_batches == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "position limit is too small for one batch",
        ));
    }

    let superbatches = requested_superbatches.min(total_batches).max(1);
    let batches_per_superbatch = total_batches / superbatches;
    if batches_per_superbatch == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "not enough batches for requested superbatches",
        ));
    }

    let actual_batches = batches_per_superbatch * superbatches;
    let actual_positions = (actual_batches as u64) * (batch_size as u64);
    Ok(TrainingPlan {
        batch_size,
        batches_per_superbatch,
        superbatches,
        actual_batches,
        actual_positions,
    })
}

fn schedule(
    batch_size: usize,
    batches: usize,
    superbatches: usize,
    lr_start: f32,
    lr_final: f32,
) -> TrainingSchedule<'static> {
    let total_batches = batches * superbatches;
    TrainingSchedule {
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch: batches,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        lr_schedule: Box::new(move |batch, superbatch| {
            if total_batches <= 1 {
                return lr_final;
            }
            let completed = (superbatch.saturating_sub(1) * batches + batch).min(total_batches - 1);
            let progress = completed as f32 / (total_batches - 1) as f32;
            let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            lr_final + (lr_start - lr_final) * cosine
        }),
        log_rate: (batches / 100).max(1),
    }
}

fn floats(weights: &ModelWeights, id: &str) -> Vec<f32> {
    match weights.get(id).values {
        TValue::F32(values) => values,
        _ => panic!("weight {id} is not f32"),
    }
}

fn transpose_internal(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            output[row * cols + col] = values[col * rows + row];
        }
    }
    output
}

fn q_i8(values: &[f32], multiplier: i16) -> Vec<u8> {
    let scale = f32::from(multiplier);
    values
        .iter()
        .map(|&x| (x * scale).round().clamp(i8::MIN as f32, i8::MAX as f32) as i8 as u8)
        .collect()
}

fn q_i8_values(values: &[f32], multiplier: i16) -> Vec<i8> {
    let scale = f32::from(multiplier);
    values
        .iter()
        .map(|&x| (x * scale).round().clamp(i8::MIN as f32, i8::MAX as f32) as i8)
        .collect()
}

fn q_i16(values: &[f32], multiplier: i16) -> Vec<u8> {
    let scale = f32::from(multiplier);
    let mut out = Vec::with_capacity(values.len() * 2);
    for &value in values {
        let q = (value * scale)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        out.extend_from_slice(&q.to_le_bytes());
    }
    out
}

fn q_i16_values(values: &[f32], multiplier: i16) -> Vec<i16> {
    let scale = f32::from(multiplier);
    values
        .iter()
        .map(|&value| {
            (value * scale)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16
        })
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn l0_feature_major_order<T: Copy>(values: &[T], inputs: usize, hidden: usize) -> Vec<T> {
    assert_eq!(values.len(), inputs * hidden);
    values.to_vec()
}

#[derive(Clone, Copy, Debug)]
enum L0ExportLayout {
    FeatureMajor,
    // Invalid legacy layout kept only as a diagnostic negative control.
    // Do not benchmark, release, or load this as a valid MNUE payload.
    LegacyTransposed,
}

fn exported_from_store(store: &ModelWeights, layout: L0ExportLayout) -> ExportedMnue {
    exported_from_store_with_l0_scale(store, layout, 1.0)
}

fn exported_from_store_with_l0_scale(
    store: &ModelWeights,
    layout: L0ExportLayout,
    l0_scale: f32,
) -> ExportedMnue {
    let scale_l0 = |mut values: Vec<f32>| {
        if l0_scale != 1.0 {
            for value in &mut values {
                *value *= l0_scale;
            }
        }
        values
    };
    let piece_l0 = scale_l0(floats(store, "piece_l0w"));
    let attack_l0 = scale_l0(floats(store, "attack_l0w"));
    let pawn_l0 = scale_l0(floats(store, "pawn_pair_l0w"));
    let piece_l0 = match layout {
        L0ExportLayout::FeatureMajor => {
            l0_feature_major_order(&piece_l0, PIECE_INPUTS, PIECE_HIDDEN)
        }
        L0ExportLayout::LegacyTransposed => {
            transpose_internal(&piece_l0, PIECE_HIDDEN, PIECE_INPUTS)
        }
    };
    let attack_l0 = match layout {
        L0ExportLayout::FeatureMajor => {
            l0_feature_major_order(&attack_l0, ATTACK_INPUTS, ATTACK_HIDDEN)
        }
        L0ExportLayout::LegacyTransposed => {
            transpose_internal(&attack_l0, ATTACK_HIDDEN, ATTACK_INPUTS)
        }
    };
    let pawn_l0 = match layout {
        L0ExportLayout::FeatureMajor => {
            l0_feature_major_order(&pawn_l0, PAWN_PAIR_INPUTS, PAWN_HIDDEN)
        }
        L0ExportLayout::LegacyTransposed => {
            transpose_internal(&pawn_l0, PAWN_HIDDEN, PAWN_PAIR_INPUTS)
        }
    };

    ExportedMnue {
        piece_l0w: q_i8_values(&piece_l0, PIECE_QUANT),
        attack_l0w: q_i8_values(&attack_l0, ATTACK_QUANT),
        pawn_pair_l0w: q_i8_values(&pawn_l0, PAWN_PAIR_QUANT),
        l0b: q_i16_values(&floats(store, "l0b"), QA as i16),
        l1w: q_i8_values(
            &transpose_internal(&floats(store, "l1w"), OUTPUT_BUCKETS * L1_SIZE, HEAD_INPUT),
            L1_QUANT,
        ),
        l1b: floats(store, "l1b"),
        l2w: transpose_internal(&floats(store, "l2w"), OUTPUT_BUCKETS * L2_SIZE, L1_SIZE),
        l2b: floats(store, "l2b"),
        l3w: transpose_internal(&floats(store, "l3w"), OUTPUT_BUCKETS, L2_SIZE),
        l3b: floats(store, "l3b"),
    }
}

#[derive(Clone, Debug)]
struct WeightSummary {
    mean: f64,
    stddev: f64,
    min: f32,
    max: f32,
}

fn weight_summary(values: &[f32]) -> WeightSummary {
    let mut stats = RunningStats::new();
    for &value in values {
        stats.push(f64::from(value));
    }
    WeightSummary {
        mean: stats.mean(),
        stddev: stats.stddev(),
        min: stats.min as f32,
        max: stats.max as f32,
    }
}

fn print_weight_summary(store: &ModelWeights) {
    for id in [
        "piece_l0w",
        "attack_l0w",
        "pawn_pair_l0w",
        "l0b",
        "l1w",
        "l1b",
        "l2w",
        "l2b",
        "l3w",
        "l3b",
    ] {
        let values = floats(store, id);
        let summary = weight_summary(&values);
        eprintln!(
            "calibration weight {id} mean {:.6} std {:.6} min {:.6} max {:.6}",
            summary.mean, summary.stddev, summary.min, summary.max
        );
    }
}

#[derive(Clone, Debug)]
struct CalibrationIssue {
    label: String,
    message: String,
}

#[derive(Clone, Debug)]
struct CalibrationSummary {
    warnings: Vec<CalibrationIssue>,
    failures: Vec<CalibrationIssue>,
}

impl CalibrationSummary {
    fn new() -> Self {
        Self {
            warnings: Vec::new(),
            failures: Vec::new(),
        }
    }

    fn warn(&mut self, label: &str, message: impl Into<String>) {
        self.warnings.push(CalibrationIssue {
            label: label.to_string(),
            message: message.into(),
        });
    }

    fn fail(&mut self, label: &str, message: impl Into<String>) {
        self.failures.push(CalibrationIssue {
            label: label.to_string(),
            message: message.into(),
        });
    }

    fn print_issues(&self) {
        for warning in &self.warnings {
            eprintln!("calibration warning {} {}", warning.label, warning.message);
        }
        for failure in &self.failures {
            eprintln!("calibration failure {} {}", failure.label, failure.message);
        }
    }
}

#[derive(Clone, Copy)]
struct CalibrationEvalDetail {
    fen: &'static str,
    float_raw: i32,
    quant_raw: i32,
    quant_cp: i32,
}

#[derive(Default)]
struct MaterialCounts {
    pawns: u32,
    knights: u32,
    bishops: u32,
    rooks: u32,
    queens: u32,
}

impl MaterialCounts {
    fn add(&mut self, piece: char) {
        match piece.to_ascii_lowercase() {
            'p' => self.pawns += 1,
            'n' => self.knights += 1,
            'b' => self.bishops += 1,
            'r' => self.rooks += 1,
            'q' => self.queens += 1,
            _ => {}
        }
    }

    fn value(&self) -> i32 {
        self.pawns as i32
            + 3 * self.knights as i32
            + 3 * self.bishops as i32
            + 5 * self.rooks as i32
            + 9 * self.queens as i32
    }

    fn compact(&self) -> String {
        format!(
            "Q{} R{} B{} N{} P{} material {}",
            self.queens,
            self.rooks,
            self.bishops,
            self.knights,
            self.pawns,
            self.value()
        )
    }
}

fn fen_side_to_move(fen: &str) -> &'static str {
    match fen.split_whitespace().nth(1) {
        Some("w") => "white",
        Some("b") => "black",
        _ => "unknown",
    }
}

fn material_summary_from_fen(fen: &str) -> String {
    let mut white = MaterialCounts::default();
    let mut black = MaterialCounts::default();
    if let Some(board) = fen.split_whitespace().next() {
        for ch in board.chars() {
            if ch.is_ascii_alphabetic() {
                if ch.is_ascii_uppercase() {
                    white.add(ch);
                } else {
                    black.add(ch);
                }
            }
        }
    }
    format!(
        "white[{}] black[{}] diff{}",
        white.compact(),
        black.compact(),
        white.value() - black.value()
    )
}

fn print_calibration_case_metadata(case: CalibrationCase) {
    eprintln!("calibration case {} fen {}", case.name, case.fen);
    eprintln!(
        "calibration case {} side_to_move {}",
        case.name,
        fen_side_to_move(case.fen)
    );
    eprintln!(
        "calibration case {} material {}",
        case.name,
        material_summary_from_fen(case.fen)
    );
    if let Some(description) = case.move_description {
        eprintln!("calibration case {} move {}", case.name, description);
    }
}

fn print_queen_swing_detail(
    label: &str,
    before_case: &str,
    after_case: &str,
    move_applied: &str,
    before: Option<CalibrationEvalDetail>,
    after: Option<CalibrationEvalDetail>,
) {
    match (before, after) {
        (Some(before), Some(after)) => {
            eprintln!("calibration queen_swing {label} before_case {before_case}");
            eprintln!("calibration queen_swing {label} before_fen {}", before.fen);
            eprintln!("calibration queen_swing {label} after_case {after_case}");
            eprintln!("calibration queen_swing {label} after_fen {}", after.fen);
            eprintln!("calibration queen_swing {label} move_applied {move_applied}");
            eprintln!(
                "calibration queen_swing {label} before_float_raw {} before_quant_raw {} after_float_raw {} after_quant_raw {} delta_cp {}",
                before.float_raw,
                before.quant_raw,
                after.float_raw,
                after.quant_raw,
                after.quant_cp - before.quant_cp
            );
        }
        _ => {
            eprintln!(
                "calibration queen_swing {label} unavailable before_case {before_case} after_case {after_case}"
            );
        }
    }
}

fn print_calibration_report(
    stage: &str,
    trainer: &MnueTrainer,
    lr_estimate: f32,
    print_weights: bool,
    init: InitScales,
) -> CalibrationSummary {
    let store = ModelWeights::from(&trainer.optimiser.model);
    let float_net = FloatNetwork::from_store(&store);
    let quant_net = exported_from_store(&store, L0ExportLayout::FeatureMajor);
    let legacy_net = exported_from_store(&store, L0ExportLayout::LegacyTransposed);
    let mut summary = CalibrationSummary::new();

    eprintln!(
        "calibration stage {stage} step {} accepted {} lr_est {:.9}",
        trainer.state.training_step, trainer.state.accepted_positions_seen, lr_estimate
    );
    eprintln!(
        "calibration constants scale {SCALE} qa {QA} piece_q {PIECE_QUANT} piece_rescale {PIECE_RESCALE} attack_q {ATTACK_QUANT} attack_rescale {ATTACK_RESCALE} pawn_q {PAWN_PAIR_QUANT} pawn_rescale {PAWN_PAIR_RESCALE} l1_q {L1_QUANT}"
    );
    eprintln!(
        "calibration init_std piece {:.6} attack {:.6} pawn {:.6} l1 {:.6} l2 {:.6} l3 {:.6}",
        piece_l0_base_std() * init.piece_l0,
        attack_l0_base_std() * init.attack_l0,
        pawn_l0_base_std() * init.pawn_l0,
        l1_base_std() * init.l1,
        l2_base_std() * init.l2,
        l3_base_std() * init.l3
    );
    eprintln!(
        "calibration init_scale piece {:.6} attack {:.6} pawn {:.6} l1 {:.6} l2 {:.6} l3 {:.6}",
        init.piece_l0, init.attack_l0, init.pawn_l0, init.l1, init.l2, init.l3
    );
    if print_weights {
        print_weight_summary(&store);
    }

    let mut startpos_cp = None;
    let mut white_queen_missing_cp = None;
    let mut black_queen_missing_cp = None;
    let mut low_material_pawn_cp = None;
    let mut startpos_crelu_stm = None;
    let mut startpos_crelu_ntm = None;
    let mut startpos_crelu_stm_low = None;
    let mut startpos_crelu_stm_high = None;
    let mut startpos_piece_clip = None;
    let mut startpos_attack_clip = None;
    let mut startpos_pawn_clip = None;
    let mut startpos_merged_max = None;
    let mut startpos_w_detail = None;
    let mut white_queen_missing_detail = None;
    let mut black_queen_missing_detail = None;
    let mut float_quant_max_abs_diff = 0_i32;
    let mut legacy_l0_transpose_diff_startpos = None;

    for case in CALIBRATION_CASES {
        let label = case.name;
        let fen = case.fen;
        print_calibration_case_metadata(case);
        let mut input = fen.to_string();
        input.push_str(" | 0 | 0.5");
        let board: ChessBoard = match input.parse() {
            Ok(board) => board,
            Err(error) => {
                summary.fail(label, format!("invalid calibration FEN: {error}"));
                continue;
            }
        };
        let encoded = match encode_position(&board) {
            Ok(encoded) => encoded,
            Err(error) => {
                summary.fail(label, format!("feature encode failed: {error}"));
                continue;
            }
        };

        let (float_raw, stats) = match forward_float(&float_net, &board) {
            Ok(value) => value,
            Err(error) => {
                summary.fail(label, format!("float forward failed: {error}"));
                continue;
            }
        };
        let (head_bias_only_raw, _) =
            forward_float_masked(&float_net, &board, ForwardMask::HEAD_BIAS_ONLY)
                .unwrap_or((0, stats));
        let (l0b_only_raw, _) =
            forward_float_masked(&float_net, &board, ForwardMask::L0B_ONLY).unwrap_or((0, stats));
        let (piece_only_raw, _) =
            forward_float_masked(&float_net, &board, ForwardMask::PIECE_ONLY).unwrap_or((0, stats));
        let (attack_only_raw, _) =
            forward_float_masked(&float_net, &board, ForwardMask::ATTACK_ONLY)
                .unwrap_or((0, stats));
        let (pawn_only_raw, _) =
            forward_float_masked(&float_net, &board, ForwardMask::PAWN_ONLY).unwrap_or((0, stats));
        let quant_raw = forward_exported(&quant_net, &board).unwrap_or(0);
        let legacy_raw = forward_exported(&legacy_net, &board).unwrap_or(0);
        let float_cp = raw_to_searchcp(float_raw, &board);
        let quant_cp = raw_to_searchcp(quant_raw, &board);
        let legacy_cp = raw_to_searchcp(legacy_raw, &board);
        let float_quant_diff = float_raw - quant_raw;
        let float_legacy_diff = float_raw - legacy_raw;
        float_quant_max_abs_diff = float_quant_max_abs_diff.max(float_quant_diff.abs());
        let detail = CalibrationEvalDetail {
            fen,
            float_raw,
            quant_raw,
            quant_cp,
        };

        match label {
            "startpos_w" => {
                startpos_cp = Some(quant_cp);
                startpos_w_detail = Some(detail);
                startpos_crelu_stm = Some(stats.merged_stm.pair_clipped);
                startpos_crelu_ntm = Some(stats.merged_ntm.pair_clipped);
                startpos_crelu_stm_low = Some(stats.merged_stm.low_clipped_lanes);
                startpos_crelu_stm_high = Some(stats.merged_stm.high_clipped_lanes);
                startpos_piece_clip = Some(stats.piece_stm.clipped);
                startpos_attack_clip = Some(stats.attack_stm.clipped);
                startpos_pawn_clip = Some(stats.pawn_stm.clipped);
                startpos_merged_max = Some(stats.merged_stm.max_abs.max(stats.merged_ntm.max_abs));
                legacy_l0_transpose_diff_startpos = Some(float_legacy_diff);
            }
            "white_queen_missing" => {
                white_queen_missing_cp = Some(quant_cp);
                white_queen_missing_detail = Some(detail);
            }
            "black_queen_missing" => {
                black_queen_missing_cp = Some(quant_cp);
                black_queen_missing_detail = Some(detail);
            }
            "low_material_pawn" => low_material_pawn_cp = Some(quant_cp),
            _ => {}
        }

        eprintln!(
            "calibration {label} float_raw {float_raw} float_cp {float_cp} quant_raw {quant_raw} quant_cp {quant_cp} legacy_l0_transpose_raw {legacy_raw} legacy_l0_transpose_cp {legacy_cp} float_quant_diff {} float_legacy_diff {}",
            float_quant_diff, float_legacy_diff
        );
        eprintln!(
            "calibration {label} branch_raw head_bias_only {head_bias_only_raw} l0b_only {l0b_only_raw} piece_only {piece_only_raw} attack_only {attack_only_raw} pawn_only {pawn_only_raw} piece_delta {} attack_delta {} pawn_delta {}",
            piece_only_raw - l0b_only_raw,
            attack_only_raw - l0b_only_raw,
            pawn_only_raw - l0b_only_raw
        );
        eprintln!(
            "calibration {label} activation merged_stm_max_abs {:.3} merged_ntm_max_abs {:.3} crelu_stm_nonzero {} crelu_ntm_nonzero {} crelu_stm_clipped {} crelu_ntm_clipped {} active_piece_stm {} active_piece_ntm {} active_attack_stm {} active_attack_ntm {} active_pawn_stm {} active_pawn_ntm {}",
            stats.merged_stm.max_abs,
            stats.merged_ntm.max_abs,
            stats.merged_stm.pair_nonzero,
            stats.merged_ntm.pair_nonzero,
            stats.merged_stm.pair_clipped,
            stats.merged_ntm.pair_clipped,
            encoded.piece.stm.len(),
            encoded.piece.ntm.len(),
            encoded.attack.stm.len(),
            encoded.attack.ntm.len(),
            encoded.pawn_pair.stm.len(),
            encoded.pawn_pair.ntm.len()
        );
        eprintln!(
            "calibration {label} crelu_clip_detail merged_stm_low_lanes {}/{} merged_stm_high_lanes {}/{} merged_ntm_low_lanes {}/{} merged_ntm_high_lanes {}/{}",
            stats.merged_stm.low_clipped_lanes,
            MERGED_HIDDEN,
            stats.merged_stm.high_clipped_lanes,
            MERGED_HIDDEN,
            stats.merged_ntm.low_clipped_lanes,
            MERGED_HIDDEN,
            stats.merged_ntm.high_clipped_lanes,
            MERGED_HIDDEN
        );
        print_pairwise_precrelu(label, "merged_stm", stats.merged_stm);
        print_pairwise_precrelu(label, "merged_ntm", stats.merged_ntm);
        print_lane_precrelu(label, "piece_stm", stats.piece_stm, PIECE_HIDDEN);
        print_lane_precrelu(label, "piece_ntm", stats.piece_ntm, PIECE_HIDDEN);
        print_lane_precrelu(label, "attack_stm", stats.attack_stm, ATTACK_HIDDEN);
        print_lane_precrelu(label, "attack_ntm", stats.attack_ntm, ATTACK_HIDDEN);
        print_lane_precrelu(label, "pawn_stm", stats.pawn_stm, PAWN_HIDDEN);
        print_lane_precrelu(label, "pawn_ntm", stats.pawn_ntm, PAWN_HIDDEN);
        eprintln!(
            "calibration {label} branch_activation piece_stm_max_abs {:.3} piece_ntm_max_abs {:.3} piece_stm_nonzero {} piece_ntm_nonzero {} piece_stm_clipped {}/{} piece_stm_low {}/{} piece_stm_high {}/{} piece_ntm_clipped {}/{} piece_ntm_low {}/{} piece_ntm_high {}/{} attack_stm_max_abs {:.3} attack_ntm_max_abs {:.3} attack_stm_nonzero {} attack_ntm_nonzero {} attack_stm_clipped {}/{} attack_stm_low {}/{} attack_stm_high {}/{} attack_ntm_clipped {}/{} attack_ntm_low {}/{} attack_ntm_high {}/{} pawn_stm_max_abs {:.3} pawn_ntm_max_abs {:.3} pawn_stm_nonzero {} pawn_ntm_nonzero {} pawn_stm_clipped {}/{} pawn_stm_low {}/{} pawn_stm_high {}/{} pawn_ntm_clipped {}/{} pawn_ntm_low {}/{} pawn_ntm_high {}/{}",
            stats.piece_stm.max_abs,
            stats.piece_ntm.max_abs,
            stats.piece_stm.nonzero,
            stats.piece_ntm.nonzero,
            stats.piece_stm.clipped,
            PIECE_HIDDEN,
            stats.piece_stm.low_clipped,
            PIECE_HIDDEN,
            stats.piece_stm.high_clipped,
            PIECE_HIDDEN,
            stats.piece_ntm.clipped,
            PIECE_HIDDEN,
            stats.piece_ntm.low_clipped,
            PIECE_HIDDEN,
            stats.piece_ntm.high_clipped,
            PIECE_HIDDEN,
            stats.attack_stm.max_abs,
            stats.attack_ntm.max_abs,
            stats.attack_stm.nonzero,
            stats.attack_ntm.nonzero,
            stats.attack_stm.clipped,
            ATTACK_HIDDEN,
            stats.attack_stm.low_clipped,
            ATTACK_HIDDEN,
            stats.attack_stm.high_clipped,
            ATTACK_HIDDEN,
            stats.attack_ntm.clipped,
            ATTACK_HIDDEN,
            stats.attack_ntm.low_clipped,
            ATTACK_HIDDEN,
            stats.attack_ntm.high_clipped,
            ATTACK_HIDDEN,
            stats.pawn_stm.max_abs,
            stats.pawn_ntm.max_abs,
            stats.pawn_stm.nonzero,
            stats.pawn_ntm.nonzero,
            stats.pawn_stm.clipped,
            PAWN_HIDDEN,
            stats.pawn_stm.low_clipped,
            PAWN_HIDDEN,
            stats.pawn_stm.high_clipped,
            PAWN_HIDDEN,
            stats.pawn_ntm.clipped,
            PAWN_HIDDEN,
            stats.pawn_ntm.low_clipped,
            PAWN_HIDDEN,
            stats.pawn_ntm.high_clipped,
            PAWN_HIDDEN
        );

        if float_quant_diff.abs() > 80 {
            summary.fail(
                label,
                format!(
                    "corrected quantized export differs from training float by {} raw",
                    float_quant_diff
                ),
            );
        }
        if label == "startpos_w" && quant_cp.abs() > 80 {
            summary.warn(
                label,
                format!("abs quant cp {quant_cp} exceeds 80 before export"),
            );
        }
        if label == "startpos_w" && l0b_only_raw.abs() > 80 {
            summary.warn(label, format!("abs l0b_only_raw {l0b_only_raw} exceeds 80"));
        }
        if label == "startpos_w" && stats.merged_stm.pair_clipped > 100 {
            summary.warn(
                label,
                format!(
                    "crelu_stm_clipped {} exceeds 100/{}",
                    stats.merged_stm.pair_clipped, PAIRWISE
                ),
            );
        }
        if label == "white_queen_missing" && quant_cp > 0 {
            summary.warn(
                label,
                format!("side missing queen still has positive quant cp {quant_cp}"),
            );
        }
        if label == "black_queen_missing" && quant_cp < 0 {
            summary.warn(
                label,
                format!("side with extra queen has negative quant cp {quant_cp}"),
            );
        }
        if label == "startpos_w" && float_legacy_diff.abs() > 50 {
            summary.warn(
                label,
                format!(
                    "legacy transposed L0 export differs from training float by {} raw",
                    float_legacy_diff
                ),
            );
        }
    }

    let queen_swing_cp = white_queen_missing_cp
        .zip(black_queen_missing_cp)
        .map(|(white, black)| black - white);
    print_queen_swing_detail(
        "remove_white_queen",
        "startpos_w",
        "white_queen_missing",
        "remove white queen from d1",
        startpos_w_detail,
        white_queen_missing_detail,
    );
    print_queen_swing_detail(
        "remove_black_queen",
        "startpos_w",
        "black_queen_missing",
        "remove black queen from d8",
        startpos_w_detail,
        black_queen_missing_detail,
    );
    if let (Some(white), Some(black)) = (white_queen_missing_detail, black_queen_missing_detail) {
        eprintln!(
            "calibration queen_swing pair white_missing_fen {} black_missing_fen {}",
            white.fen, black.fen
        );
        eprintln!(
            "calibration queen_swing pair white_missing_float_raw {} white_missing_quant_raw {} black_missing_float_raw {} black_missing_quant_raw {} delta_cp {}",
            white.float_raw,
            white.quant_raw,
            black.float_raw,
            black.quant_raw,
            black.quant_cp - white.quant_cp
        );
    }
    let loss = trainer
        .state
        .last_loss
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "na".to_string());
    let fmt_i32 = |value: Option<i32>| {
        value
            .map(|value| value.to_string())
            .unwrap_or_else(|| "na".to_string())
    };
    let fmt_usize = |value: Option<usize>, total: usize| {
        value
            .map(|value| format!("{value}/{total}"))
            .unwrap_or_else(|| format!("na/{total}"))
    };
    let fmt_f32 = |value: Option<f32>| {
        value
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "na".to_string())
    };
    eprintln!(
        "calibration summary stage {stage} step {} loss {loss} startpos_cp {} queen_swing_cp {} low_material_pawn_cp {} crelu_clip_stm {} crelu_clip_ntm {} crelu_low_stm_lanes {} crelu_high_stm_lanes {} piece_clip_stm {} attack_clip_stm {} pawn_clip_stm {} merged_max_abs {} float_quant_max_abs_diff {} legacy_l0_transpose_diff_startpos {}",
        trainer.state.training_step,
        fmt_i32(startpos_cp),
        fmt_i32(queen_swing_cp),
        fmt_i32(low_material_pawn_cp),
        fmt_usize(startpos_crelu_stm, PAIRWISE),
        fmt_usize(startpos_crelu_ntm, PAIRWISE),
        fmt_usize(startpos_crelu_stm_low, MERGED_HIDDEN),
        fmt_usize(startpos_crelu_stm_high, MERGED_HIDDEN),
        fmt_usize(startpos_piece_clip, PIECE_HIDDEN),
        fmt_usize(startpos_attack_clip, ATTACK_HIDDEN),
        fmt_usize(startpos_pawn_clip, PAWN_HIDDEN),
        fmt_f32(startpos_merged_max),
        float_quant_max_abs_diff,
        fmt_i32(legacy_l0_transpose_diff_startpos)
    );

    summary.print_issues();
    summary
}

fn print_l0_scale_sweep(stage: &str, trainer: &MnueTrainer, scales: &[f32]) {
    let store = ModelWeights::from(&trainer.optimiser.model);
    eprintln!(
        "calibration l0_scale_sweep header stage l0_scale startpos_cp queen_swing_cp white_queen_missing_cp black_queen_missing_cp low_material_pawn_cp kiwipete_cp float_quant_max_abs_diff merged_stm_max_abs crelu_clip_stm crelu_clip_ntm crelu_low_stm_lanes crelu_high_stm_lanes piece_clip_stm attack_clip_stm pawn_clip_stm"
    );

    for &scale in scales {
        let float_net = FloatNetwork::from_store_with_l0_scale(&store, scale);
        let quant_net =
            exported_from_store_with_l0_scale(&store, L0ExportLayout::FeatureMajor, scale);
        let mut startpos_cp = None;
        let mut white_queen_missing_cp = None;
        let mut black_queen_missing_cp = None;
        let mut low_material_pawn_cp = None;
        let mut kiwipete_cp = None;
        let mut float_quant_max_abs_diff = 0_i32;
        let mut start_stats = None;

        for case in CALIBRATION_CASES {
            let label = case.name;
            let fen = case.fen;
            let input = format!("{fen} | 0 | 0.5");
            let board: ChessBoard = match input.parse() {
                Ok(board) => board,
                Err(error) => {
                    eprintln!(
                        "calibration l0_scale_sweep stage {stage} l0_scale {scale:.6} failed_to_parse {label}: {error}"
                    );
                    continue;
                }
            };
            let (float_raw, stats) = match forward_float(&float_net, &board) {
                Ok(value) => value,
                Err(error) => {
                    eprintln!(
                        "calibration l0_scale_sweep stage {stage} l0_scale {scale:.6} float_failed {label}: {error}"
                    );
                    continue;
                }
            };
            let quant_raw = match forward_exported(&quant_net, &board) {
                Ok(value) => value,
                Err(error) => {
                    eprintln!(
                        "calibration l0_scale_sweep stage {stage} l0_scale {scale:.6} quant_failed {label}: {error}"
                    );
                    continue;
                }
            };
            float_quant_max_abs_diff = float_quant_max_abs_diff.max((float_raw - quant_raw).abs());
            let quant_cp = raw_to_searchcp(quant_raw, &board);

            match label {
                "startpos_w" => {
                    startpos_cp = Some(quant_cp);
                    start_stats = Some(stats);
                }
                "white_queen_missing" => white_queen_missing_cp = Some(quant_cp),
                "black_queen_missing" => black_queen_missing_cp = Some(quant_cp),
                "low_material_pawn" => low_material_pawn_cp = Some(quant_cp),
                "kiwipete" => kiwipete_cp = Some(quant_cp),
                _ => {}
            }
        }

        let queen_swing_cp = white_queen_missing_cp
            .zip(black_queen_missing_cp)
            .map(|(white, black)| black - white);
        let fmt_i32 = |value: Option<i32>| {
            value
                .map(|value| value.to_string())
                .unwrap_or_else(|| "na".to_string())
        };
        let fmt_pair = |value: Option<usize>, total: usize| {
            value
                .map(|value| format!("{value}/{total}"))
                .unwrap_or_else(|| format!("na/{total}"))
        };
        let fmt_f32 = |value: Option<f32>| {
            value
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "na".to_string())
        };
        eprintln!(
            "calibration l0_scale_sweep row {stage} {scale:.6} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
            fmt_i32(startpos_cp),
            fmt_i32(queen_swing_cp),
            fmt_i32(white_queen_missing_cp),
            fmt_i32(black_queen_missing_cp),
            fmt_i32(low_material_pawn_cp),
            fmt_i32(kiwipete_cp),
            float_quant_max_abs_diff,
            fmt_f32(start_stats.map(|stats| stats.merged_stm.max_abs)),
            fmt_pair(
                start_stats.map(|stats| stats.merged_stm.pair_clipped),
                PAIRWISE
            ),
            fmt_pair(
                start_stats.map(|stats| stats.merged_ntm.pair_clipped),
                PAIRWISE
            ),
            fmt_pair(
                start_stats.map(|stats| stats.merged_stm.low_clipped_lanes),
                MERGED_HIDDEN
            ),
            fmt_pair(
                start_stats.map(|stats| stats.merged_stm.high_clipped_lanes),
                MERGED_HIDDEN
            ),
            fmt_pair(
                start_stats.map(|stats| stats.piece_stm.clipped),
                PIECE_HIDDEN
            ),
            fmt_pair(
                start_stats.map(|stats| stats.attack_stm.clipped),
                ATTACK_HIDDEN
            ),
            fmt_pair(start_stats.map(|stats| stats.pawn_stm.clipped), PAWN_HIDDEN)
        );
    }
}

fn scheduled_lr(plan: TrainingPlan, lr_start: f32, lr_final: f32, completed_batches: usize) -> f32 {
    if plan.actual_batches <= 1 {
        return lr_final;
    }
    let completed = completed_batches.min(plan.actual_batches - 1);
    let progress = completed as f32 / (plan.actual_batches - 1) as f32;
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    lr_final + (lr_start - lr_final) * cosine
}

fn freeze_l0b_if_needed(trainer: &mut MnueTrainer, freeze_steps: u64) {
    if trainer.state.training_step < freeze_steps {
        let ok = trainer
            .optimiser
            .model
            .set_weights("l0b", &TValue::F32(vec![0.0; MERGED_HIDDEN]));
        debug_assert!(ok, "l0b weight tensor missing");
    }
}

fn load_trainer_weights_if_requested(
    trainer: &mut MnueTrainer,
    path: Option<&Path>,
) -> io::Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    let weight_path = if path.is_dir() {
        path.join("weights.bin")
    } else {
        path.to_path_buf()
    };
    trainer
        .optimiser
        .load_weights_from_file(weight_path.to_string_lossy().as_ref())
        .map_err(|error| {
            io::Error::other(format!(
                "failed to load weights from {}: {error:?}",
                weight_path.display()
            ))
        })?;
    println!("Loaded trainer weights: {}", weight_path.display());
    Ok(())
}

fn expected_payload_bytes() -> usize {
    PIECE_INPUTS * PIECE_HIDDEN
        + ATTACK_INPUTS * ATTACK_HIDDEN
        + PAWN_PAIR_INPUTS * PAWN_HIDDEN
        + MERGED_HIDDEN * size_of::<i16>()
        + OUTPUT_BUCKETS * L1_SIZE * HEAD_INPUT
        + OUTPUT_BUCKETS * L1_SIZE * size_of::<f32>()
        + OUTPUT_BUCKETS * L2_SIZE * L1_SIZE * size_of::<f32>()
        + OUTPUT_BUCKETS * L2_SIZE * size_of::<f32>()
        + OUTPUT_BUCKETS * L2_SIZE * size_of::<f32>()
        + OUTPUT_BUCKETS * size_of::<f32>()
}

fn export_network(trainer: &MnueTrainer, output: &Path) -> io::Result<()> {
    let store = ModelWeights::from(&trainer.optimiser.model);
    let mut file = Vec::with_capacity(MNUE_HEADER_BYTES as usize + expected_payload_bytes());
    for field in [
        MNUE_MAGIC,
        MNUE_VERSION,
        MNUE_ARCH,
        MNUE_HEADER_BYTES,
        PIECE_INPUTS as u32,
        ATTACK_INPUTS as u32,
        PAWN_PAIR_INPUTS as u32,
        PIECE_HIDDEN as u32,
        ATTACK_HIDDEN as u32,
        PAWN_HIDDEN as u32,
        MERGED_HIDDEN as u32,
        INPUT_BUCKETS as u32,
        OUTPUT_BUCKETS as u32,
        L1_SIZE as u32,
        L2_SIZE as u32,
        SCALE,
        QA,
        PIECE_QUANT as u32,
        PIECE_RESCALE,
        ATTACK_QUANT as u32,
        ATTACK_RESCALE,
        PAWN_PAIR_QUANT as u32,
        PAWN_PAIR_RESCALE,
        L1_QUANT as u32,
        MNUE_FEATURE_VERSION,
        0,
        0,
        0,
    ] {
        write_u32(&mut file, field);
    }

    // SparseMatmul stores weights feature-major: values[feature * hidden + lane].
    file.extend_from_slice(&q_i8(
        &l0_feature_major_order(&floats(&store, "piece_l0w"), PIECE_INPUTS, PIECE_HIDDEN),
        PIECE_QUANT,
    ));
    file.extend_from_slice(&q_i8(
        &l0_feature_major_order(&floats(&store, "attack_l0w"), ATTACK_INPUTS, ATTACK_HIDDEN),
        ATTACK_QUANT,
    ));
    file.extend_from_slice(&q_i8(
        &l0_feature_major_order(
            &floats(&store, "pawn_pair_l0w"),
            PAWN_PAIR_INPUTS,
            PAWN_HIDDEN,
        ),
        PAWN_PAIR_QUANT,
    ));
    file.extend_from_slice(&q_i16(&floats(&store, "l0b"), QA as i16));
    file.extend_from_slice(&q_i8(
        &transpose_internal(&floats(&store, "l1w"), OUTPUT_BUCKETS * L1_SIZE, HEAD_INPUT),
        L1_QUANT,
    ));
    file.extend_from_slice(&f32_bytes(&floats(&store, "l1b")));
    file.extend_from_slice(&f32_bytes(&transpose_internal(
        &floats(&store, "l2w"),
        OUTPUT_BUCKETS * L2_SIZE,
        L1_SIZE,
    )));
    file.extend_from_slice(&f32_bytes(&floats(&store, "l2b")));
    file.extend_from_slice(&f32_bytes(&transpose_internal(
        &floats(&store, "l3w"),
        OUTPUT_BUCKETS,
        L2_SIZE,
    )));
    file.extend_from_slice(&f32_bytes(&floats(&store, "l3b")));

    let expected_file = MNUE_HEADER_BYTES as usize + expected_payload_bytes();
    if file.len() != expected_file {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "internal export size mismatch: got {}, expected {expected_file}",
                file.len()
            ),
        ));
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, file)?;
    Ok(())
}

fn run_training(cli: Cli) -> io::Result<()> {
    assert_eq!(PIECE_INPUTS, 12_288);
    assert_eq!(attack_index_table().size, ATTACK_INPUTS);
    assert_eq!(PAWN_PAIR_INPUTS, 4_560);
    assert_eq!(expected_payload_bytes(), 47_636_512);

    println!("Architecture: MNUE-X2-K16-pawn-Q8-A384");
    println!("CPU feature prepare threads: {}", cli.threads);
    println!("Expected payload bytes: {}", expected_payload_bytes());
    println!(
        "Expected file bytes: {}",
        expected_payload_bytes() + MNUE_HEADER_BYTES as usize
    );
    let init_scales = cli.init_scales();
    println!(
        "Init scales: piece_l0 {:.6} attack_l0 {:.6} pawn_l0 {:.6} l1 {:.6} l2 {:.6} l3 {:.6}",
        init_scales.piece_l0,
        init_scales.attack_l0,
        init_scales.pawn_l0,
        init_scales.l1,
        init_scales.l2,
        init_scales.l3
    );
    println!("l0b freeze steps: {}", cli.l0b_freeze_steps);
    if (cli.l0b_lr_scale - 1.0).abs() > f32::EPSILON {
        println!(
            "l0b lr scale requested: {:.6}; per-weight LR is not supported by the current Bullet AdamW path, using freeze-only controls",
            cli.l0b_lr_scale
        );
    }

    let mut trainer = build_trainer(init_scales);
    load_trainer_weights_if_requested(&mut trainer, cli.load_weights.as_deref())?;

    if cli.calibration_only {
        let _ = print_calibration_report("calibration_only", &trainer, cli.lr, true, init_scales);
        if cli.calibration_l0_scale_sweep {
            print_l0_scale_sweep("calibration_only", &trainer, &cli.calibration_l0_scales);
        }
        return Ok(());
    }

    let data_schedule = build_data_schedule(cli.data_schedule_options())
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))?;
    let scheduled_paths = data_schedule.scheduled_paths.clone();
    println!(
        "Training data format: {}",
        input_data_kind_name(data_schedule.kind)
    );
    print_startup_log(&data_schedule);

    let inspect_limit = if cli.dry_run {
        cli.batch_size as u64
    } else {
        16
    };
    let mut retained = Vec::new();
    let accepted = read_samples(
        data_schedule.kind,
        &scheduled_paths,
        inspect_limit,
        |sample| {
            if retained.len() < inspect_limit as usize {
                retained.push(sample);
            }
            false
        },
    )?;
    if retained.is_empty() {
        return Err(io::Error::other(format!(
            "no accepted valid positions in {} input file(s)",
            scheduled_paths.len()
        )));
    }
    println!("Accepted inspected positions: {accepted}");
    if cli.data_stats_limit > 0 {
        let stats = inspect_data_stats(data_schedule.kind, &scheduled_paths, cli.data_stats_limit)?;
        stats.print();
    }

    let _ = print_calibration_report("init", &trainer, cli.lr, true, init_scales);
    if cli.calibration_l0_scale_sweep {
        print_l0_scale_sweep("init", &trainer, &cli.calibration_l0_scales);
    }
    if cli.dry_run {
        let batch_len = retained.len().min(cli.batch_size);
        let batch = prepare_batch(&retained[..batch_len]);
        let l0b_freeze_steps = cli.l0b_freeze_steps;
        trainer
            .train_custom(
                schedule(batch_len, 1, 1, cli.lr, cli.lr_final),
                SingleBatchLoader(batch),
                |trainer, _, _, loss| {
                    trainer.state.last_loss = Some(loss);
                    freeze_l0b_if_needed(trainer, l0b_freeze_steps);
                    trainer.state.training_step += 1;
                    trainer.state.accepted_positions_seen += batch_len as u64;
                },
                |_, _| {},
            )
            .map_err(|error| io::Error::other(format!("{error:?}")))?;
        let summary =
            print_calibration_report("dry_run_final", &trainer, cli.lr_final, true, init_scales);
        if cli.calibration_l0_scale_sweep {
            print_l0_scale_sweep("dry_run_final", &trainer, &cli.calibration_l0_scales);
        }
        if cli.strict_calibration_gate && !summary.failures.is_empty() {
            return Err(io::Error::other(format!(
                "strict calibration gate failed with {} failure(s), {} warning(s)",
                summary.failures.len(),
                summary.warnings.len()
            )));
        }
        if cli.strict_calibration_gate {
            println!(
                "Strict calibration gate passed with {} warning(s)",
                summary.warnings.len()
            );
        }
        export_network(&trainer, &cli.export)?;
        println!("Dry-run export: {}", cli.export.display());
        return Ok(());
    }

    let plan = training_plan(&cli)?;
    println!("Training batch size: {}", plan.batch_size);
    println!("Training superbatches: {}", plan.superbatches);
    println!(
        "Training batches/superbatch: {}",
        plan.batches_per_superbatch
    );
    println!("Training total batches: {}", plan.actual_batches);
    println!("Training actual positions: {}", plan.actual_positions);
    println!("Learning rate: {} -> {}", cli.lr, cli.lr_final);
    fs::create_dir_all(&cli.output_dir)?;
    let l0b_freeze_steps = cli.l0b_freeze_steps;
    trainer
        .train_custom(
            schedule(
                plan.batch_size,
                plan.batches_per_superbatch,
                plan.superbatches,
                cli.lr,
                cli.lr_final,
            ),
            MnueK16DataLoader {
                kind: data_schedule.kind,
                paths: scheduled_paths,
                chunk_paths: data_schedule.chunk_paths.clone(),
                chunk_resample_on_exhaustion: data_schedule.resample_on_exhaustion,
                chunk_shuffle_seed: data_schedule.seed,
                chunk_virtual_epochs: data_schedule.virtual_epochs,
                chunk_sample: data_schedule.chunk_sample,
                limit: if cli.positions == 0 {
                    0
                } else {
                    plan.actual_positions
                },
                threads: cli.threads,
            },
            |trainer, _, _, loss| {
                trainer.state.last_loss = Some(loss);
                freeze_l0b_if_needed(trainer, l0b_freeze_steps);
                trainer.state.training_step += 1;
                trainer.state.accepted_positions_seen += plan.batch_size as u64;
            },
            {
                let output_dir = cli.output_dir.clone();
                let plan_for_calibration = plan;
                let lr_start = cli.lr;
                let lr_final = cli.lr_final;
                let init_scales = init_scales;
                move |trainer, superbatch| {
                    let checkpoint =
                        output_dir.join(format!("mnue-x2-k16-pawn-q8-a384-{superbatch}"));
                    let optimiser_state = checkpoint.join("optimiser_state");
                    fs::create_dir_all(&optimiser_state)
                        .expect("failed to create optimiser checkpoint directory");
                    trainer
                        .optimiser
                        .write_to_checkpoint(optimiser_state.to_string_lossy().as_ref())
                        .expect("failed to write optimiser checkpoint");
                    let completed_batches =
                        superbatch * plan_for_calibration.batches_per_superbatch;
                    let lr_estimate =
                        scheduled_lr(plan_for_calibration, lr_start, lr_final, completed_batches);
                    let _ = print_calibration_report(
                        &format!("superbatch_{superbatch}"),
                        trainer,
                        lr_estimate,
                        false,
                        init_scales,
                    );
                }
            },
        )
        .map_err(|error| io::Error::other(format!("{error:?}")))?;
    let summary = print_calibration_report(
        "final_pre_export",
        &trainer,
        cli.lr_final,
        true,
        init_scales,
    );
    if cli.calibration_l0_scale_sweep {
        print_l0_scale_sweep("final_pre_export", &trainer, &cli.calibration_l0_scales);
    }
    if cli.strict_calibration_gate && !summary.failures.is_empty() {
        return Err(io::Error::other(format!(
            "strict calibration gate failed with {} failure(s), {} warning(s)",
            summary.failures.len(),
            summary.warnings.len()
        )));
    }
    if cli.strict_calibration_gate {
        println!(
            "Strict calibration gate passed with {} warning(s)",
            summary.warnings.len()
        );
    }
    export_network(&trainer, &cli.export)?;
    println!("Export: {}", cli.export.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(fen: &str) -> ChessBoard {
        format!("{fen} | 0 | 0.5").parse().unwrap()
    }

    #[test]
    fn attack_table_has_expected_size() {
        assert_eq!(attack_index_table().size, ATTACK_INPUTS);
    }

    #[test]
    fn k16_bucket_bounds_and_examples() {
        assert_eq!(king_bucket16(0), 0);
        assert_eq!(king_bucket16(7), 0);
        assert_eq!(king_bucket16(8), 4);
        assert_eq!(king_bucket16(16), 8);
        assert_eq!(king_bucket16(24), 12);
        assert_eq!(king_bucket16(63), 12);
        for square in 0..64 {
            assert!(king_bucket16(square) < INPUT_BUCKETS);
        }
    }

    #[test]
    fn pawn_pair_index_is_triangular() {
        assert_eq!(pawn_pair_index(1, 0), 0);
        assert_eq!(pawn_pair_index(2, 0), 1);
        assert_eq!(pawn_pair_index(2, 1), 2);
        assert_eq!(pawn_pair_index(95, 94), PAWN_PAIR_INPUTS - 1);
    }

    #[test]
    fn start_position_features_are_bounded_and_deterministic() {
        let position = parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let first = encode_position(&position).unwrap();
        let second = encode_position(&position).unwrap();
        assert_eq!(first.piece, second.piece);
        assert_eq!(first.attack, second.attack);
        assert_eq!(first.pawn_pair, second.pawn_pair);
        assert!(first.piece.stm.len() <= PIECE_MAX_ACTIVE);
        assert!(first.attack.stm.len() <= ATTACK_MAX_ACTIVE);
        assert!(first.pawn_pair.stm.len() <= PAWN_PAIR_MAX_ACTIVE);
    }

    #[test]
    fn side_to_move_flip_swaps_perspectives() {
        let white = parse("r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1");
        let black = parse("r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R b KQkq - 0 1");
        let w = encode_position(&white).unwrap();
        let b = encode_position(&black).unwrap();
        assert_eq!(w.piece.stm, b.piece.ntm);
        assert_eq!(w.piece.ntm, b.piece.stm);
        assert_eq!(w.attack.stm, b.attack.ntm);
        assert_eq!(w.attack.ntm, b.attack.stm);
        assert_eq!(w.pawn_pair.stm, b.pawn_pair.ntm);
        assert_eq!(w.pawn_pair.ntm, b.pawn_pair.stm);
    }

    #[test]
    fn expected_sizes_are_exact() {
        assert_eq!(PIECE_INPUTS, 12_288);
        assert_eq!(ATTACK_INPUTS, 90_048);
        assert_eq!(PAWN_PAIR_INPUTS, 4_560);
        assert_eq!(expected_payload_bytes(), 47_636_512);
        assert_eq!(
            expected_payload_bytes() + MNUE_HEADER_BYTES as usize,
            47_636_624
        );
    }

    #[test]
    fn init_scale_cli_inherits_and_overrides() {
        let cli = Cli::parse([
            "trainer",
            "--arch",
            ARCH_NAME,
            "--l0-init-scale",
            "0.5",
            "--pawn-l0-init-scale",
            "0.25",
            "--head-init-scale",
            "0.75",
            "--l2-init-scale",
            "0.6",
            "--l0b-freeze-steps",
            "512",
        ])
        .unwrap()
        .unwrap();
        let scales = cli.init_scales();
        assert_eq!(scales.piece_l0, 0.5);
        assert_eq!(scales.attack_l0, 0.5);
        assert_eq!(scales.pawn_l0, 0.25);
        assert_eq!(scales.l1, 0.75);
        assert_eq!(scales.l2, 0.6);
        assert_eq!(scales.l3, 0.75);
        assert_eq!(cli.l0b_freeze_steps, 512);
    }

    #[test]
    fn l0_export_order_is_feature_major() {
        const INPUTS: usize = 4;
        const HIDDEN: usize = 5;
        let sentinel: Vec<i64> = (0..INPUTS)
            .flat_map(|feature| {
                (0..HIDDEN).map(move |hidden| feature as i64 * 100_000 + hidden as i64)
            })
            .collect();

        let packed = l0_feature_major_order(&sentinel, INPUTS, HIDDEN);
        assert_eq!(packed, sentinel);
        assert_eq!(&packed[0..HIDDEN], &[0, 1, 2, 3, 4]);
        assert_eq!(
            &packed[HIDDEN..2 * HIDDEN],
            &[100_000, 100_001, 100_002, 100_003, 100_004]
        );

        let mut legacy_transposed = Vec::with_capacity(INPUTS * HIDDEN);
        for hidden in 0..HIDDEN {
            for feature in 0..INPUTS {
                legacy_transposed.push(sentinel[feature * HIDDEN + hidden]);
            }
        }
        assert_ne!(packed, legacy_transposed);
    }

    #[test]
    fn positioned_training_defaults_to_multiple_superbatches() {
        let mut cli = Cli::default();
        cli.arch = ARCH_NAME.to_string();
        cli.positions = 268_435_456;
        cli.batch_size = 65_536;
        let plan = training_plan(&cli).unwrap();
        assert_eq!(plan.batch_size, 65_536);
        assert_eq!(plan.superbatches, DEFAULT_POSITION_SUPERBATCHES);
        assert_eq!(plan.batches_per_superbatch, 1024);
        assert_eq!(plan.actual_batches, 4096);
        assert_eq!(plan.actual_positions, 268_435_456);
    }

    #[test]
    fn learning_rate_decays_across_all_superbatches() {
        let schedule = schedule(1, 2, 2, 0.001, 0.0001);
        let start = (schedule.lr_schedule)(0, 1);
        let end = (schedule.lr_schedule)(1, 2);
        assert!((start - 0.001).abs() < 1.0e-8);
        assert!((end - 0.0001).abs() < 1.0e-8);
    }
}
