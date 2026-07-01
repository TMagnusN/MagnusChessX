use std::path::PathBuf;

use crate::config::{DEFAULT_BATCH_SIZE, DEFAULT_DATA, DEFAULT_NET, DEFAULT_OUTPUT_DIR};

#[derive(Clone, Debug, PartialEq)]
pub struct Cli {
    pub data: Vec<PathBuf>,
    pub output_dir: PathBuf,
    pub export: PathBuf,
    pub date: u64,
    pub dry_run: bool,
    pub summary: bool,
    pub attack_fake_quant: bool,
    pub attack_quant_scale: f32,
    pub batch_size: usize,
    pub seed: u64,
    pub resume: Option<PathBuf>,
    pub decode_feature: Option<(String, usize)>,
    pub dump_fen: Option<String>,
    pub golden_vectors: Option<PathBuf>,
    pub golden_network: Option<PathBuf>,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            data: vec![PathBuf::from(DEFAULT_DATA)],
            output_dir: PathBuf::from(DEFAULT_OUTPUT_DIR),
            export: PathBuf::from(DEFAULT_NET),
            date: 0,
            dry_run: false,
            summary: false,
            attack_fake_quant: false,
            attack_quant_scale: crate::config::DEFAULT_ATTACK_QUANT_SCALE,
            batch_size: DEFAULT_BATCH_SIZE,
            seed: 1,
            resume: None,
            decode_feature: None,
            dump_fen: None,
            golden_vectors: None,
            golden_network: None,
        }
    }
}

impl Cli {
    pub fn parse_from<I, S>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut cli = Self::default();
        let mut args = args.into_iter().map(Into::into);
        let _program = args.next();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--date" | "--positions" => {
                    let text = next_value(&mut args, &arg)?;
                    let signed: i128 = text
                        .parse()
                        .map_err(|_| format!("invalid position limit: {text}"))?;
                    if signed < 0 {
                        return Err("--date must be zero or a positive integer".to_string());
                    }
                    cli.date =
                        u64::try_from(signed).map_err(|_| "--date is too large".to_string())?;
                }
                "--data" => {
                    let path = PathBuf::from(next_value(&mut args, &arg)?);
                    if cli.data == [PathBuf::from(DEFAULT_DATA)] {
                        cli.data.clear();
                    }
                    cli.data.push(path);
                }
                "--output-dir" => cli.output_dir = PathBuf::from(next_value(&mut args, &arg)?),
                "--export" => cli.export = PathBuf::from(next_value(&mut args, &arg)?),
                "--batch-size" => {
                    cli.batch_size = next_value(&mut args, &arg)?
                        .parse()
                        .map_err(|_| "invalid --batch-size".to_string())?;
                    if cli.batch_size == 0 {
                        return Err("--batch-size must be positive".to_string());
                    }
                }
                "--seed" => {
                    cli.seed = next_value(&mut args, &arg)?
                        .parse()
                        .map_err(|_| "invalid --seed".to_string())?
                }
                "--resume" => cli.resume = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                "--dry-run" => cli.dry_run = true,
                "--model-summary" | "--summary" => cli.summary = true,
                "--attack-fake-quant" => cli.attack_fake_quant = true,
                "--attack-quant-scale" => {
                    cli.attack_quant_scale = next_value(&mut args, &arg)?
                        .parse()
                        .map_err(|_| "invalid attack quant scale".to_string())?;
                    if !(cli.attack_quant_scale.is_finite() && cli.attack_quant_scale > 0.0) {
                        return Err("--attack-quant-scale must be finite and positive".to_string());
                    }
                }
                "--decode-feature" => {
                    let family = next_value(&mut args, &arg)?;
                    let index = next_value(&mut args, &arg)?
                        .parse()
                        .map_err(|_| "invalid feature index".to_string())?;
                    cli.decode_feature = Some((family, index));
                }
                "--dump-fen" => cli.dump_fen = Some(next_value(&mut args, &arg)?),
                "--golden-vectors" => {
                    cli.golden_vectors = Some(PathBuf::from(next_value(&mut args, &arg)?))
                }
                "--golden-network" => {
                    cli.golden_network = Some(PathBuf::from(next_value(&mut args, &arg)?))
                }
                "-h" | "--help" => return Err(Self::help().to_string()),
                _ => return Err(format!("unknown argument: {arg}\n{}", Self::help())),
            }
        }
        Ok(cli)
    }

    pub const fn help() -> &'static str {
        "MNUEv2 trainer\n\
         --date N                  accepted valid-position limit; 0 is unlimited\n\
         --positions N             alias for --date\n\
         --data PATH               input file; repeat for multiple files\n\
         --dry-run                 inspect data and run one train step\n\
         --model-summary           print model/size summary\n\
         --attack-fake-quant       enable Attack-table STE fake quantisation\n\
         --attack-quant-scale S    symmetric Attack scale (default 1/127)\n\
         --decode-feature FAMILY INDEX\n\
         --dump-fen FEN            print all three perspective index sets\n\
         --golden-vectors PATH     write deterministic engine fixture\n\
         --golden-network PATH     exported v4 network used by the fixture"
    }
}

fn next_value<I: Iterator<Item = String>>(args: &mut I, option: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value after {option}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_semantics() {
        assert_eq!(
            Cli::parse_from(["x", "--date", "10000"]).unwrap().date,
            10_000
        );
        assert_eq!(Cli::parse_from(["x", "--date", "0"]).unwrap().date, 0);
        assert!(Cli::parse_from(["x", "--date", "-1"]).is_err());
    }
}
