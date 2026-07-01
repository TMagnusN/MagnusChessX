use std::{
    fs::File,
    io::{self, BufReader, Read},
    path::{Path, PathBuf},
};

use bullet_compiler::tensor::TValue;
use bullet_trainer::run::dataloader::{DataLoader, DataLoadingError, PreparedBatchHost};
use bulletformat::{BulletFormat, ChessBoard};

use crate::{
    board::{decode_record, validate_position},
    buckets::material_bucket,
    config::{ATTACK_MAX_ACTIVE, POSITION_MAX_ACTIVE, STRUCTURE_MAX_ACTIVE},
    features::{EncodedPosition, encode_all},
    stats::DatasetStats,
};

#[derive(Clone, Debug)]
pub struct Sample {
    pub board: ChessBoard,
    pub encoded: EncodedPosition,
    pub bucket: u8,
    pub target: f32,
}

#[derive(Clone, Debug)]
pub struct DatasetConfig {
    pub paths: Vec<PathBuf>,
    pub accepted_limit: u64,
    pub wdl_blend: f32,
    pub score_scale: f32,
}

fn sigmoid(score: i16, scale: f32) -> f32 {
    1.0 / (1.0 + (-f32::from(score) / scale).exp())
}

pub fn target(pos: &ChessBoard, blend: f32, scale: f32) -> f32 {
    blend * pos.result() + (1.0 - blend) * sigmoid(pos.score(), scale)
}

pub fn read_samples<F>(config: &DatasetConfig, mut accept: F) -> io::Result<DatasetStats>
where
    F: FnMut(Sample) -> bool,
{
    let mut stats = DatasetStats::default();
    'files: for path in &config.paths {
        let mut reader = BufReader::with_capacity(4 * 1024 * 1024, File::open(path)?);
        loop {
            if config.accepted_limit > 0 && stats.accepted >= config.accepted_limit {
                break 'files;
            }
            let mut bytes = [0_u8; 32];
            match reader.read_exact(&mut bytes) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(error),
            }
            let board = decode_record(&bytes);
            if validate_position(&board).is_err() {
                stats.rejected += 1;
                continue;
            }
            let encoded = match encode_all(&board) {
                Ok(value) => value,
                Err(_) => {
                    stats.rejected += 1;
                    continue;
                }
            };
            let bucket = material_bucket(&board);
            let sample = Sample {
                target: target(&board, config.wdl_blend, config.score_scale),
                board,
                encoded,
                bucket,
            };
            stats.accepted += 1;
            stats.position.push(sample.encoded.position.stm.len());
            stats.attack.push(sample.encoded.attack.stm.len());
            stats.structure.push(sample.encoded.structure.stm.len());
            stats.buckets[usize::from(bucket)] += 1;
            if accept(sample) {
                break 'files;
            }
        }
    }
    Ok(stats)
}

pub fn count_valid(paths: &[PathBuf], limit: u64) -> io::Result<DatasetStats> {
    read_samples(
        &DatasetConfig {
            paths: paths.to_vec(),
            accepted_limit: limit,
            wdl_blend: 0.75,
            score_scale: 400.0,
        },
        |_| false,
    )
}

pub fn path_label(path: &Path) -> String {
    path.display().to_string()
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

pub fn prepare_batch(samples: &[Sample]) -> PreparedBatchHost {
    let mut p_stm = Vec::with_capacity(samples.len() * POSITION_MAX_ACTIVE);
    let mut p_ntm = Vec::with_capacity(samples.len() * POSITION_MAX_ACTIVE);
    let mut a_stm = Vec::with_capacity(samples.len() * ATTACK_MAX_ACTIVE);
    let mut a_ntm = Vec::with_capacity(samples.len() * ATTACK_MAX_ACTIVE);
    let mut s_stm = Vec::with_capacity(samples.len() * STRUCTURE_MAX_ACTIVE);
    let mut s_ntm = Vec::with_capacity(samples.len() * STRUCTURE_MAX_ACTIVE);
    let mut buckets = Vec::with_capacity(samples.len());
    let mut targets = Vec::with_capacity(samples.len());
    for sample in samples {
        push_sparse(
            &mut p_stm,
            &sample.encoded.position.stm,
            POSITION_MAX_ACTIVE,
        );
        push_sparse(
            &mut p_ntm,
            &sample.encoded.position.ntm,
            POSITION_MAX_ACTIVE,
        );
        push_sparse(&mut a_stm, &sample.encoded.attack.stm, ATTACK_MAX_ACTIVE);
        push_sparse(&mut a_ntm, &sample.encoded.attack.ntm, ATTACK_MAX_ACTIVE);
        push_sparse(
            &mut s_stm,
            &sample.encoded.structure.stm,
            STRUCTURE_MAX_ACTIVE,
        );
        push_sparse(
            &mut s_ntm,
            &sample.encoded.structure.ntm,
            STRUCTURE_MAX_ACTIVE,
        );
        buckets.push(i32::from(sample.bucket));
        targets.push(sample.target);
    }
    PreparedBatchHost {
        batch_size: samples.len(),
        inputs: [
            ("position_stm".to_string(), TValue::I32(p_stm)),
            ("position_ntm".to_string(), TValue::I32(p_ntm)),
            ("attack_stm".to_string(), TValue::I32(a_stm)),
            ("attack_ntm".to_string(), TValue::I32(a_ntm)),
            ("structure_stm".to_string(), TValue::I32(s_stm)),
            ("structure_ntm".to_string(), TValue::I32(s_ntm)),
            ("buckets".to_string(), TValue::I32(buckets)),
            ("targets".to_string(), TValue::F32(targets)),
        ]
        .into(),
    }
}

#[derive(Clone)]
pub struct MnueDataLoader {
    pub config: DatasetConfig,
}

impl DataLoader for MnueDataLoader {
    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(
        self,
        batch_size: usize,
        mut callback: F,
    ) -> Result<(), DataLoadingError> {
        let mut batch = Vec::with_capacity(batch_size);
        read_samples(&self.config, |sample| {
            batch.push(sample);
            if batch.len() == batch_size {
                let stop = callback(prepare_batch(&batch));
                batch.clear();
                stop
            } else {
                false
            }
        })
        .map_err(|e| DataLoadingError::Message(e.to_string()))?;
        Ok(())
    }
}

pub struct SingleBatchLoader(pub PreparedBatchHost);

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

#[cfg(test)]
mod tests {
    use super::*;

    fn write_records(path: &Path, records: &[ChessBoard]) {
        let bytes = ChessBoard::as_bytes_slice(records);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn malformed_records_do_not_consume_limit_across_files() {
        let dir = std::env::temp_dir().join(format!("mnue-v2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let valid: ChessBoard = "8/8/8/8/8/8/4k3/4K3 w - - 0 1 | 0 | 0.5".parse().unwrap();
        let mut bad = valid;
        bad.result = 9;
        let first = dir.join("a.data");
        let second = dir.join("b.data");
        write_records(&first, &[bad, valid]);
        write_records(&second, &[valid, valid]);
        let stats = count_valid(&[first, second], 2).unwrap();
        assert_eq!(stats.accepted, 2);
        assert_eq!(stats.rejected, 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn zero_limit_is_unlimited_for_single_file() {
        let dir = std::env::temp_dir().join(format!("mnue-v2-unlimited-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let valid: ChessBoard = "8/8/8/8/8/8/4k3/4K3 w - - 0 1 | 0 | 0.5".parse().unwrap();
        let path = dir.join("all.data");
        write_records(&path, &[valid, valid, valid]);
        let stats = count_valid(&[path], 0).unwrap();
        assert_eq!(stats.accepted, 3);
        let _ = std::fs::remove_dir_all(dir);
    }
}
