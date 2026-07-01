use bullet_gpu::runtime::Device;
use bullet_lib::nn::{
    Affine, ExecutionContext, ModelBuilder, Shape,
    optimiser::{AdamWOptimiser, AdamWParams, Optimiser},
};
use bullet_trainer::Trainer;

use crate::{
    config::{
        ATTACK_MAX_ACTIVE, ATTACK_WIDTH, CONCAT_WIDTH, HEAD_HIDDEN_1, HEAD_HIDDEN_2,
        OUTPUT_BUCKETS, POSITION_MAX_ACTIVE, POSITION_WIDTH, STRUCTURE_MAX_ACTIVE, STRUCTURE_WIDTH,
    },
    features::{ATTACK_FEATURE_COUNT, POSITION_FEATURE_COUNT, STRUCTURE_FEATURE_COUNT},
};

#[derive(Clone, Debug)]
pub struct TrainState {
    pub attack_fake_quant: bool,
    pub attack_quant_scale: f32,
    pub accepted_positions_seen: u64,
    pub training_step: u64,
}

pub type MnueTrainer = Trainer<ExecutionContext, AdamWOptimiser, TrainState>;

fn branch<'a>(
    affine: Affine<'a>,
    stm: bullet_lib::nn::ModelNode<'a>,
    ntm: bullet_lib::nn::ModelNode<'a>,
) -> bullet_lib::nn::ModelNode<'a> {
    let stm = affine.forward(stm).crelu().pairwise_mul();
    let ntm = affine.forward(ntm).crelu().pairwise_mul();
    stm.concat(ntm)
}

pub fn build_trainer(attack_fake_quant: bool, attack_quant_scale: f32) -> MnueTrainer {
    let builder = ModelBuilder::default();
    let targets = builder.new_dense_input("targets", Shape::new(1, 1));
    let buckets = builder.new_sparse_input("buckets", Shape::new(OUTPUT_BUCKETS, 1), 1);
    let p_stm = builder.new_sparse_input(
        "position_stm",
        Shape::new(POSITION_FEATURE_COUNT, 1),
        POSITION_MAX_ACTIVE,
    );
    let p_ntm = builder.new_sparse_input(
        "position_ntm",
        Shape::new(POSITION_FEATURE_COUNT, 1),
        POSITION_MAX_ACTIVE,
    );
    let a_stm = builder.new_sparse_input(
        "attack_stm",
        Shape::new(ATTACK_FEATURE_COUNT, 1),
        ATTACK_MAX_ACTIVE,
    );
    let a_ntm = builder.new_sparse_input(
        "attack_ntm",
        Shape::new(ATTACK_FEATURE_COUNT, 1),
        ATTACK_MAX_ACTIVE,
    );
    let s_stm = builder.new_sparse_input(
        "structure_stm",
        Shape::new(STRUCTURE_FEATURE_COUNT, 1),
        STRUCTURE_MAX_ACTIVE,
    );
    let s_ntm = builder.new_sparse_input(
        "structure_ntm",
        Shape::new(STRUCTURE_FEATURE_COUNT, 1),
        STRUCTURE_MAX_ACTIVE,
    );

    let position = builder.new_affine("position", POSITION_FEATURE_COUNT, POSITION_WIDTH);
    position.init_with_effective_input_size(32);
    let mut attack = builder.new_affine("attack", ATTACK_FEATURE_COUNT, ATTACK_WIDTH);
    attack.init_with_effective_input_size(96);
    if attack_fake_quant {
        let bound = 127.0 * attack_quant_scale;
        attack.weights = attack
            .weights
            .clip_pass_through_grad(-bound, bound)
            .faux_quantise(1.0 / attack_quant_scale, true);
    }
    let structure = builder.new_affine("structure", STRUCTURE_FEATURE_COUNT, STRUCTURE_WIDTH);
    structure.init_with_effective_input_size(32);

    let p = branch(position, p_stm, p_ntm);
    let a = branch(attack, a_stm, a_ntm);
    let s = branch(structure, s_stm, s_ntm);
    let representation = p.concat(a).concat(s);
    assert_eq!(representation.shape(), Shape::new(CONCAT_WIDTH, 1));

    let head1 = builder.new_affine("head1", CONCAT_WIDTH, OUTPUT_BUCKETS * HEAD_HIDDEN_1);
    let head2 = builder.new_affine("head2", HEAD_HIDDEN_1, OUTPUT_BUCKETS * HEAD_HIDDEN_2);
    let output = builder.new_affine("output", HEAD_HIDDEN_2, OUTPUT_BUCKETS);
    let hidden = head1.forward(representation).select(buckets).screlu();
    let hidden = head2.forward(hidden).select(buckets).screlu();
    let score = output.forward(hidden).select(buckets);
    let loss = score.sigmoid().squared_error(targets);

    let device = Device::<ExecutionContext>::new(0).expect("failed to create Bullet device");
    let model = builder.build(device, loss, score);
    let mut optimiser =
        Optimiser::new(model, AdamWParams::default()).expect("failed to create AdamW optimiser");
    let bounded = AdamWParams {
        min_weight: -1.98,
        max_weight: 1.98,
        ..Default::default()
    };
    for id in [
        "positionw",
        "attackw",
        "structurew",
        "head1w",
        "head2w",
        "outputw",
    ] {
        optimiser.set_params_for_weight(id, bounded);
    }
    Trainer {
        optimiser,
        state: TrainState {
            attack_fake_quant,
            attack_quant_scale,
            accepted_positions_seen: 0,
            training_step: 0,
        },
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ParameterSummary {
    pub position_table: usize,
    pub attack_table: usize,
    pub structure_table: usize,
    pub table_biases: usize,
    pub heads: usize,
    pub total: usize,
}

pub fn parameter_summary() -> ParameterSummary {
    let position_table = POSITION_FEATURE_COUNT * POSITION_WIDTH;
    let attack_table = ATTACK_FEATURE_COUNT * ATTACK_WIDTH;
    let structure_table = STRUCTURE_FEATURE_COUNT * STRUCTURE_WIDTH;
    let table_biases = POSITION_WIDTH + ATTACK_WIDTH + STRUCTURE_WIDTH;
    let heads = OUTPUT_BUCKETS
        * (CONCAT_WIDTH * HEAD_HIDDEN_1
            + HEAD_HIDDEN_1
            + HEAD_HIDDEN_1 * HEAD_HIDDEN_2
            + HEAD_HIDDEN_2
            + HEAD_HIDDEN_2
            + 1);
    ParameterSummary {
        position_table,
        attack_table,
        structure_table,
        table_biases,
        heads,
        total: position_table + attack_table + structure_table + table_biases + heads,
    }
}

pub fn print_model_summary() {
    let p = parameter_summary();
    println!("Architecture: {}", crate::config::ARCHITECTURE_NAME);
    println!(
        "Dimensions: Position {POSITION_WIDTH}, Attack {ATTACK_WIDTH}, Structure {STRUCTURE_WIDTH}, concat {CONCAT_WIDTH}"
    );
    println!(
        "Feature vocabularies: Position {POSITION_FEATURE_COUNT}, Attack {ATTACK_FEATURE_COUNT}, Structure {STRUCTURE_FEATURE_COUNT}"
    );
    println!("Position table params: {}", p.position_table);
    println!("Attack table params: {}", p.attack_table);
    println!("Structure table params: {}", p.structure_table);
    println!("Bias params: {}", p.table_biases);
    println!("All 12 output heads: {}", p.heads);
    println!("Total params: {}", p.total);

    let mib = |bytes: usize| bytes as f64 / 1_048_576.0;
    let pos = p.position_table * 2 + POSITION_WIDTH * 4;
    let atk16 = p.attack_table * 2 + ATTACK_WIDTH * 4;
    let atk8 = p.attack_table + ATTACK_WIDTH * 4;
    let struc = p.structure_table * 2 + STRUCTURE_WIDTH * 4;
    let head_weights = OUTPUT_BUCKETS
        * (CONCAT_WIDTH * HEAD_HIDDEN_1 + HEAD_HIDDEN_1 * HEAD_HIDDEN_2 + HEAD_HIDDEN_2);
    let head_biases = OUTPUT_BUCKETS * (HEAD_HIDDEN_1 + HEAD_HIDDEN_2 + 1);
    let heads = head_weights * 2 + head_biases * 4;
    println!(
        "All-int16 deployment: Position {:.2} MiB, Attack {:.2} MiB, Structure {:.2} MiB, Heads {:.2} MiB, Total {:.2} MiB",
        mib(pos),
        mib(atk16),
        mib(struc),
        mib(heads),
        mib(pos + atk16 + struc + heads)
    );
    println!(
        "Mixed Attack-int8 deployment: Position {:.2} MiB, Attack {:.2} MiB, Structure {:.2} MiB, Heads {:.2} MiB, Total {:.2} MiB",
        mib(pos),
        mib(atk8),
        mib(struc),
        mib(heads),
        mib(pos + atk8 + struc + heads)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representation_and_head_shapes_are_exact() {
        assert_eq!(
            POSITION_WIDTH + ATTACK_WIDTH + STRUCTURE_WIDTH,
            CONCAT_WIDTH
        );
        assert_eq!(CONCAT_WIDTH, 768);
        assert_eq!(HEAD_HIDDEN_1, 32);
        assert_eq!(HEAD_HIDDEN_2, 32);
        assert_eq!(OUTPUT_BUCKETS, 12);
        assert!(parameter_summary().total > parameter_summary().attack_table);
    }
}
