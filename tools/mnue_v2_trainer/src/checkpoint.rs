use std::{fs, io, path::Path};

use serde::{Deserialize, Serialize};

use crate::{
    config::{
        ARCHITECTURE_NAME, ATTACK_WIDTH, BUCKET_MAPPING_VERSION, CONCAT_WIDTH, FORMAT_VERSION,
        HEAD_HIDDEN_1, HEAD_HIDDEN_2, OUTPUT_BUCKETS, POSITION_WIDTH, SCORE_SCALE, STRUCTURE_WIDTH,
    },
    features::{ATTACK_FEATURE_COUNT, POSITION_FEATURE_COUNT, STRUCTURE_FEATURE_COUNT},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FakeQuantConfiguration {
    pub attack_enabled: bool,
    pub attack_scale_mode: String,
    pub attack_scale: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CheckpointMetadata {
    pub format_version: u32,
    pub architecture_name: String,
    pub position_feature_count: usize,
    pub attack_feature_count: usize,
    pub structure_feature_count: usize,
    pub position_width: usize,
    pub attack_width: usize,
    pub structure_width: usize,
    pub concat_width: usize,
    pub head_hidden_1: usize,
    pub head_hidden_2: usize,
    pub output_buckets: usize,
    pub bucket_mapping_version: u32,
    pub score_scale: i32,
    pub fake_quant_configuration: FakeQuantConfiguration,
    pub training_step: u64,
    pub accepted_positions_seen: u64,
}

impl CheckpointMetadata {
    pub fn new(fake_quant: FakeQuantConfiguration, training_step: u64, accepted: u64) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            architecture_name: ARCHITECTURE_NAME.to_string(),
            position_feature_count: POSITION_FEATURE_COUNT,
            attack_feature_count: ATTACK_FEATURE_COUNT,
            structure_feature_count: STRUCTURE_FEATURE_COUNT,
            position_width: POSITION_WIDTH,
            attack_width: ATTACK_WIDTH,
            structure_width: STRUCTURE_WIDTH,
            concat_width: CONCAT_WIDTH,
            head_hidden_1: HEAD_HIDDEN_1,
            head_hidden_2: HEAD_HIDDEN_2,
            output_buckets: OUTPUT_BUCKETS,
            bucket_mapping_version: BUCKET_MAPPING_VERSION,
            score_scale: SCORE_SCALE,
            fake_quant_configuration: fake_quant,
            training_step,
            accepted_positions_seen: accepted,
        }
    }

    pub fn validate_compatible(&self) -> Result<(), String> {
        let expected = Self::new(
            self.fake_quant_configuration.clone(),
            self.training_step,
            self.accepted_positions_seen,
        );
        let compatible = self.format_version == expected.format_version
            && self.architecture_name == expected.architecture_name
            && self.position_feature_count == expected.position_feature_count
            && self.attack_feature_count == expected.attack_feature_count
            && self.structure_feature_count == expected.structure_feature_count
            && self.position_width == expected.position_width
            && self.attack_width == expected.attack_width
            && self.structure_width == expected.structure_width
            && self.concat_width == expected.concat_width
            && self.head_hidden_1 == expected.head_hidden_1
            && self.head_hidden_2 == expected.head_hidden_2
            && self.output_buckets == expected.output_buckets
            && self.bucket_mapping_version == expected.bucket_mapping_version
            && self.score_scale == expected.score_scale;
        compatible.then_some(()).ok_or_else(|| {
            "checkpoint is incompatible with MNUEv2-PositionAttackStructure; legacy monolithic checkpoints cannot be loaded directly".to_string()
        })
    }
}

pub fn write_metadata(path: &Path, metadata: &CheckpointMetadata) -> io::Result<()> {
    fs::create_dir_all(path)?;
    let json = serde_json::to_vec_pretty(metadata).map_err(io::Error::other)?;
    fs::write(path.join("metadata.json"), json)
}

pub fn read_metadata(path: &Path) -> Result<CheckpointMetadata, String> {
    let file = path.join("metadata.json");
    let data = fs::read(&file).map_err(|_| {
        format!(
            "missing {}; refusing legacy/incompatible checkpoint",
            file.display()
        )
    })?;
    let metadata: CheckpointMetadata = serde_json::from_slice(&data).map_err(|e| e.to_string())?;
    metadata.validate_compatible()?;
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_or_wrong_architecture_is_rejected() {
        let mut metadata = CheckpointMetadata::new(
            FakeQuantConfiguration {
                attack_enabled: false,
                attack_scale_mode: "per-table-symmetric".to_string(),
                attack_scale: 1.0 / 127.0,
            },
            0,
            0,
        );
        assert!(metadata.validate_compatible().is_ok());
        metadata.format_version -= 1;
        assert!(metadata.validate_compatible().is_err());
    }
}
