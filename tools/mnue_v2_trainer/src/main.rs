use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
};

use bullet_trainer::run::schedule::{TrainingSchedule, TrainingSteps};
use bulletformat::ChessBoard;
use mnue_v2_trainer::{
    checkpoint::{CheckpointMetadata, FakeQuantConfiguration, read_metadata, write_metadata},
    cli::Cli,
    config::{ARCHITECTURE_NAME, SCORE_SCALE},
    dataset::{DatasetConfig, MnueDataLoader, SingleBatchLoader, prepare_batch, read_samples},
    export::{ExportedNetwork, export_network},
    features::{
        ATTACK_FEATURE_COUNT, POSITION_FEATURE_COUNT, STRUCTURE_FEATURE_COUNT,
        decode_attack_feature, decode_position_feature, decode_structure_feature,
    },
    model::{build_trainer, print_model_summary},
};

fn main() {
    let cli = match Cli::parse_from(std::env::args()) {
        Ok(cli) => cli,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    if let Some((family, index)) = &cli.decode_feature {
        let decoded = match family.as_str() {
            "position" => decode_position_feature(*index),
            "attack" => decode_attack_feature(*index),
            "structure" => decode_structure_feature(*index),
            _ => None,
        };
        println!(
            "{}",
            decoded.unwrap_or_else(|| "feature index/family is invalid".to_string())
        );
        return;
    }
    if let Some(fen) = &cli.dump_fen {
        let input = if fen.contains('|') {
            fen.clone()
        } else {
            format!("{fen} | 0 | 0.5")
        };
        let board: ChessBoard = input.parse().unwrap_or_else(|error| {
            eprintln!("invalid --dump-fen: {error}");
            std::process::exit(2);
        });
        let encoded = mnue_v2_trainer::features::encode_all(&board).unwrap_or_else(|error| {
            eprintln!("feature encoding failed: {error}");
            std::process::exit(1);
        });
        println!(
            "material_units {}",
            mnue_v2_trainer::buckets::material_units(&board)
        );
        println!(
            "material_bucket {}",
            mnue_v2_trainer::buckets::material_bucket(&board)
        );
        for (name, pair) in [
            ("Position", &encoded.position),
            ("Attack", &encoded.attack),
            ("Structure", &encoded.structure),
        ] {
            println!("{name} stm {:?}", pair.stm);
            println!("{name} ntm {:?}", pair.ntm);
        }
        return;
    }
    if let Some(output) = &cli.golden_vectors {
        let network = cli.golden_network.as_deref().unwrap_or_else(|| {
            eprintln!("--golden-vectors requires --golden-network");
            std::process::exit(2);
        });
        write_golden_vectors(output, network).unwrap_or_else(|error| {
            eprintln!("failed to write golden vectors: {error}");
            std::process::exit(1);
        });
        println!("Wrote {}", output.display());
        return;
    }

    println!("Architecture: {ARCHITECTURE_NAME}");
    if cli.date == 0 {
        println!("Position limit: unlimited");
    } else {
        println!("Position limit: {}", format_number(cli.date));
    }
    println!(
        "Attack fake quant: {}",
        if cli.attack_fake_quant {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!("Seed: {}", cli.seed);
    if cli.summary {
        print_model_summary();
    }

    let mut samples = Vec::new();
    let retained_samples = if cli.dry_run {
        if cli.date == 0 {
            cli.batch_size
        } else {
            cli.batch_size.min(cli.date as usize)
        }
    } else {
        16
    };
    let inspect_config = DatasetConfig {
        paths: cli.data.clone(),
        accepted_limit: retained_samples as u64,
        wdl_blend: 0.75,
        score_scale: SCORE_SCALE as f32,
    };
    let stats = read_samples(&inspect_config, |sample| {
        if samples.len() < retained_samples {
            samples.push(sample);
        }
        false
    })
    .unwrap_or_else(|e| {
        eprintln!("dataset error: {e}");
        std::process::exit(1);
    });
    if cli.dry_run {
        stats.print((
            POSITION_FEATURE_COUNT,
            ATTACK_FEATURE_COUNT,
            STRUCTURE_FEATURE_COUNT,
        ));
    }
    if samples.is_empty() {
        eprintln!("no accepted valid positions");
        std::process::exit(1);
    }

    let mut trainer = build_trainer(cli.attack_fake_quant, cli.attack_quant_scale);
    if let Some(checkpoint) = &cli.resume {
        let metadata = read_metadata(checkpoint).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        trainer
            .optimiser
            .load_from_checkpoint(
                checkpoint
                    .join("optimiser_state")
                    .to_string_lossy()
                    .as_ref(),
            )
            .expect("failed to load optimiser state");
        trainer.state.training_step = metadata.training_step;
        trainer.state.accepted_positions_seen = metadata.accepted_positions_seen;
    }

    if cli.dry_run {
        let batch_len = samples.len().min(cli.batch_size);
        let batch = prepare_batch(&samples[..batch_len]);
        let schedule = schedule(batch_len, 1, 1);
        trainer
            .train_custom(
                schedule,
                SingleBatchLoader(batch),
                |trainer, _, _, _| {
                    trainer.state.training_step += 1;
                    trainer.state.accepted_positions_seen += batch_len as u64;
                },
                |_, _| {},
            )
            .expect("dry-run forward/backward step failed");
        println!("Dry-run forward/backward step: passed ({batch_len} positions)");
        let dry_export = cli.output_dir.join("dry-run.mnue");
        export_network(
            &trainer,
            &dry_export,
            cli.attack_fake_quant,
            cli.attack_quant_scale,
            &samples
                .iter()
                .take(16)
                .map(|x| (x.encoded.clone(), x.bucket))
                .collect::<Vec<_>>(),
        )
        .expect("dry-run export failed");
        println!("Dry-run export: {}", dry_export.display());
        return;
    }

    let effective_batch = if cli.date > 0 {
        cli.batch_size
            .min(usize::try_from(cli.date).unwrap_or(usize::MAX))
    } else {
        cli.batch_size
    };
    let batches = if cli.date > 0 {
        usize::try_from(cli.date).unwrap_or(usize::MAX) / effective_batch
    } else {
        6104
    };
    if batches == 0 {
        eprintln!("position limit is too small for one batch");
        std::process::exit(2);
    }
    let usable_limit = if cli.date > 0 {
        (batches * effective_batch) as u64
    } else {
        0
    };
    let loader = MnueDataLoader {
        config: DatasetConfig {
            paths: cli.data.clone(),
            accepted_limit: usable_limit,
            wdl_blend: 0.75,
            score_scale: SCORE_SCALE as f32,
        },
    };
    let output_dir = cli.output_dir.clone();
    let fake = FakeQuantConfiguration {
        attack_enabled: cli.attack_fake_quant,
        attack_scale_mode: "per-table-symmetric".to_string(),
        attack_scale: cli.attack_quant_scale,
    };
    fs::create_dir_all(&output_dir).expect("failed to create output directory");
    let superbatches = if cli.date == 0 { 3 } else { 1 };
    trainer
        .train_custom(
            schedule(effective_batch, batches, superbatches),
            loader,
            |trainer, _, _, _| {
                trainer.state.training_step += 1;
                trainer.state.accepted_positions_seen += effective_batch as u64;
            },
            move |trainer, superbatch| {
                let checkpoint = output_dir.join(format!("mnue-v2-{superbatch}"));
                let optimiser_state = checkpoint.join("optimiser_state");
                fs::create_dir_all(&optimiser_state)
                    .expect("failed to create checkpoint directory");
                trainer
                    .optimiser
                    .write_to_checkpoint(optimiser_state.to_string_lossy().as_ref())
                    .expect("failed to write optimiser checkpoint");
                let metadata = CheckpointMetadata::new(
                    fake.clone(),
                    trainer.state.training_step,
                    trainer.state.accepted_positions_seen,
                );
                write_metadata(&checkpoint, &metadata)
                    .expect("failed to write checkpoint metadata");
            },
        )
        .expect("training failed");

    println!(
        "Actual accepted positions trained: {}",
        trainer.state.accepted_positions_seen
    );
    export_network(
        &trainer,
        &cli.export,
        cli.attack_fake_quant,
        cli.attack_quant_scale,
        &samples
            .iter()
            .take(16)
            .map(|x| (x.encoded.clone(), x.bucket))
            .collect::<Vec<_>>(),
    )
    .expect("export failed");
    println!("Wrote {}", cli.export.display());
}

fn schedule(batch_size: usize, batches: usize, superbatches: usize) -> TrainingSchedule<'static> {
    TrainingSchedule {
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch: batches,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        lr_schedule: Box::new(|_, _| 0.001),
        log_rate: (batches / 100).max(1),
    }
}

fn format_number(value: u64) -> String {
    let chars: Vec<_> = value.to_string().chars().rev().collect();
    let mut out = String::new();
    for (i, ch) in chars.iter().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(*ch);
    }
    out.chars().rev().collect()
}

fn write_golden_vectors(output: &Path, network_path: &Path) -> std::io::Result<()> {
    const FENS: [&str; 20] = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "8/8/8/8/8/8/4k3/4K3 w - - 0 1",
        "r4rk1/ppp2ppp/2n5/3pp3/3PP3/2N2N2/PPP2PPP/R4RK1 w - - 0 1",
        "r3k2r/ppp2ppp/2n5/3pp3/3PP3/2N2N2/PPP2PPP/R3K2R w KQkq - 0 1",
        "4k3/4r3/8/8/4N3/8/4K3/8 w - - 0 1",
        "4k3/4r3/8/4b3/4N3/8/4K3/8 w - - 0 1",
        "4k3/8/8/3b4/4N3/8/4R3/4K3 w - - 0 1",
        "4k3/8/8/8/3q4/4N3/8/4K3 w - - 0 1",
        "4k3/8/8/8/3q4/2N1N3/8/4K3 w - - 0 1",
        "4k3/8/3P4/8/8/8/8/4K3 w - - 0 1",
        "4k3/3p4/2P5/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/2P5/2P5/4K3 w - - 0 1",
        "4k3/8/8/8/8/P1P5/P7/4K3 w - - 0 1",
        "4k3/pp6/8/8/8/8/PP6/4K2R w - - 0 1",
        "Q3k3/8/8/8/8/8/8/4K3 b - - 0 1",
        "7k/P7/8/8/8/8/8/4K3 w - - 0 1",
        "7k/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r4rk1/8/8/8/8/8/8/R3K2R b KQ - 1 1",
        "r3k2r/8/8/8/3Q4/8/8/R3K2R b KQkq - 0 1",
    ];

    let network = ExportedNetwork::load(network_path)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(fs::File::create(output)?);
    writeln!(writer, "MNUEV2_GOLDEN 1")?;
    writeln!(writer, "NETWORK {}", network_path.display())?;
    for fen in FENS {
        let board: ChessBoard = format!("{fen} | 0 | 0.5")
            .parse()
            .map_err(std::io::Error::other)?;
        let encoded =
            mnue_v2_trainer::features::encode_all(&board).map_err(std::io::Error::other)?;
        let bucket = mnue_v2_trainer::buckets::material_bucket(&board);
        let result = network.golden_evaluate(&encoded, bucket);
        writeln!(writer, "CASE")?;
        writeln!(writer, "FEN {fen}")?;
        writeln!(
            writer,
            "SIDE {}",
            fen.split_whitespace().nth(1).unwrap_or("w")
        )?;
        writeln!(
            writer,
            "MATERIAL {}",
            mnue_v2_trainer::buckets::material_units(&board)
        )?;
        writeln!(writer, "BUCKET {bucket}")?;
        write_indices(&mut writer, "POSITION_STM", &encoded.position.stm)?;
        write_indices(&mut writer, "POSITION_NTM", &encoded.position.ntm)?;
        write_indices(&mut writer, "ATTACK_STM", &encoded.attack.stm)?;
        write_indices(&mut writer, "ATTACK_NTM", &encoded.attack.ntm)?;
        write_indices(&mut writer, "STRUCTURE_STM", &encoded.structure.stm)?;
        write_indices(&mut writer, "STRUCTURE_NTM", &encoded.structure.ntm)?;
        writeln!(
            writer,
            "POSITION_HASH {:016x} {:016x}",
            result.position_hash[0], result.position_hash[1]
        )?;
        writeln!(
            writer,
            "ATTACK_HASH {:016x} {:016x}",
            result.attack_hash[0], result.attack_hash[1]
        )?;
        writeln!(
            writer,
            "STRUCTURE_HASH {:016x} {:016x}",
            result.structure_hash[0], result.structure_hash[1]
        )?;
        writeln!(writer, "OUTPUT {:.9}", result.output)?;
        writeln!(writer, "SCORE {}", result.engine_score)?;
        writeln!(writer, "END")?;
    }
    writer.flush()
}

fn write_indices(writer: &mut impl Write, name: &str, indices: &[usize]) -> std::io::Result<()> {
    write!(writer, "{name} {}", indices.len())?;
    for index in indices {
        write!(writer, " {index}")?;
    }
    writeln!(writer)
}
