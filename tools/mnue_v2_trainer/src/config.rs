pub const ARCHITECTURE_NAME: &str = "MNUEv2-PositionAttackStructure";
pub const FORMAT_VERSION: u32 = 4;
pub const ARCHITECTURE_ID: u32 = 7;
pub const BUCKET_MAPPING_VERSION: u32 = 1;
pub const FEATURE_MAPPING_VERSION: u32 = 1;

pub const POSITION_WIDTH: usize = 320;
pub const ATTACK_WIDTH: usize = 288;
pub const STRUCTURE_WIDTH: usize = 160;
pub const CONCAT_WIDTH: usize = 768;
pub const HEAD_HIDDEN_1: usize = 32;
pub const HEAD_HIDDEN_2: usize = 32;
pub const OUTPUT_BUCKETS: usize = 12;

pub const POSITION_MAX_ACTIVE: usize = 32;
pub const ATTACK_MAX_ACTIVE: usize = 384;
pub const STRUCTURE_MAX_ACTIVE: usize = 64;

pub const SCORE_SCALE: i32 = 400;
pub const FEATURE_QUANT: i16 = 255;
pub const HEAD_QUANT: i16 = 64;
pub const DEFAULT_ATTACK_QUANT_SCALE: f32 = 1.0 / 127.0;

pub const DEFAULT_DATA: &str = "D:/NNUE/data/train_3b_shuffled.data";
pub const DEFAULT_OUTPUT_DIR: &str = "D:/NNUE/runs/mnue_v2_pas";
pub const DEFAULT_NET: &str = "D:/NNUE/nets/mnue/mnue_v2_pas.mnue";
pub const DEFAULT_BATCH_SIZE: usize = 16_384;

const _: () = assert!(POSITION_WIDTH + ATTACK_WIDTH + STRUCTURE_WIDTH == CONCAT_WIDTH);
const _: () = assert!(POSITION_WIDTH & 1 == 0);
const _: () = assert!(ATTACK_WIDTH & 1 == 0);
const _: () = assert!(STRUCTURE_WIDTH & 1 == 0);
