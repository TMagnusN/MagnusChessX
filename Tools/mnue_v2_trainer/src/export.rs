use std::{
    fs,
    io::{self, Cursor, Read},
    path::Path,
};

use bullet_compiler::tensor::TValue;
use bullet_trainer::model::save::ModelWeights;

use crate::{
    config::{
        ARCHITECTURE_ID, ATTACK_WIDTH, CONCAT_WIDTH, FEATURE_QUANT, FORMAT_VERSION, HEAD_HIDDEN_1,
        HEAD_HIDDEN_2, HEAD_QUANT, OUTPUT_BUCKETS, POSITION_WIDTH, SCORE_SCALE, STRUCTURE_WIDTH,
    },
    features::{
        ATTACK_FEATURE_COUNT, EncodedPosition, POSITION_FEATURE_COUNT, STRUCTURE_FEATURE_COUNT,
    },
    model::MnueTrainer,
};

const MAGIC: [u8; 8] = *b"MNUEV2\0\0";
const DTYPE_I8: u32 = 1;
const DTYPE_I16: u32 = 2;
const DTYPE_I32: u32 = 3;
const SECTION_DESC_BYTES: usize = 48;

#[derive(Clone)]
struct Section {
    name: &'static str,
    dtype: u32,
    data: Vec<u8>,
    offset: u64,
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

fn q_i16(values: &[f32], scale: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &value in values {
        let q = (value / scale)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        out.extend_from_slice(&q.to_le_bytes());
    }
    out
}

fn q_i32(values: &[f32], scale: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &value in values {
        let q = (value / scale)
            .round()
            .clamp(i32::MIN as f32, i32::MAX as f32) as i32;
        out.extend_from_slice(&q.to_le_bytes());
    }
    out
}

fn q_i8(values: &[f32], scale: f32) -> Vec<u8> {
    values
        .iter()
        .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8 as u8)
        .collect()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn write_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn write_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub fn attack_quant_diagnostics(values: &[f32], scale: f32) -> (f64, f64, f64, f64) {
    let mut saturation = 0_usize;
    let mut zero = 0_usize;
    let mut abs_error = 0.0_f64;
    let mut max_error = 0.0_f64;
    for &value in values {
        let raw = (value / scale).round();
        if raw <= -127.0 || raw >= 127.0 {
            saturation += 1;
        }
        let q = raw.clamp(-127.0, 127.0);
        if q == 0.0 {
            zero += 1;
        }
        let error = f64::from((q * scale - value).abs());
        abs_error += error;
        max_error = max_error.max(error);
    }
    let n = values.len().max(1) as f64;
    (
        saturation as f64 / n,
        zero as f64 / n,
        abs_error / n,
        max_error,
    )
}

pub fn export_network(
    trainer: &MnueTrainer,
    output: &Path,
    attack_int8: bool,
    attack_scale: f32,
    test_positions: &[(EncodedPosition, u8)],
) -> io::Result<()> {
    let store = ModelWeights::from(&trainer.optimiser.model);
    let float_reference = FloatReference::from_store(&store);
    let p_scale = 1.0 / f32::from(FEATURE_QUANT);
    let h_scale = 1.0 / f32::from(HEAD_QUANT);
    let attack_w = transpose_internal(
        &floats(&store, "attackw"),
        ATTACK_WIDTH,
        ATTACK_FEATURE_COUNT,
    );
    let (sat, zero, mean, max) = attack_quant_diagnostics(&attack_w, attack_scale);
    println!("Attack quant scale: {attack_scale:.9}");
    println!("Attack saturation rate: {:.6}%", sat * 100.0);
    println!("Attack zero rate: {:.6}%", zero * 100.0);
    println!("Attack mean absolute quantisation error: {mean:.9}");
    println!("Attack maximum quantisation error: {max:.9}");

    let mut sections = vec![
        Section {
            name: "position.weights",
            dtype: DTYPE_I16,
            data: q_i16(
                &transpose_internal(
                    &floats(&store, "positionw"),
                    POSITION_WIDTH,
                    POSITION_FEATURE_COUNT,
                ),
                p_scale,
            ),
            offset: 0,
        },
        Section {
            name: "position.bias",
            dtype: DTYPE_I32,
            data: q_i32(&floats(&store, "positionb"), p_scale),
            offset: 0,
        },
        Section {
            name: "attack.weights",
            dtype: if attack_int8 { DTYPE_I8 } else { DTYPE_I16 },
            data: if attack_int8 {
                q_i8(&attack_w, attack_scale)
            } else {
                q_i16(&attack_w, p_scale)
            },
            offset: 0,
        },
        Section {
            name: "attack.bias",
            dtype: DTYPE_I32,
            data: q_i32(
                &floats(&store, "attackb"),
                if attack_int8 { attack_scale } else { p_scale },
            ),
            offset: 0,
        },
        Section {
            name: "structure.weights",
            dtype: DTYPE_I16,
            data: q_i16(
                &transpose_internal(
                    &floats(&store, "structurew"),
                    STRUCTURE_WIDTH,
                    STRUCTURE_FEATURE_COUNT,
                ),
                p_scale,
            ),
            offset: 0,
        },
        Section {
            name: "structure.bias",
            dtype: DTYPE_I32,
            data: q_i32(&floats(&store, "structureb"), p_scale),
            offset: 0,
        },
        Section {
            name: "head1.weights",
            dtype: DTYPE_I16,
            data: q_i16(
                &transpose_internal(
                    &floats(&store, "head1w"),
                    OUTPUT_BUCKETS * HEAD_HIDDEN_1,
                    CONCAT_WIDTH,
                ),
                h_scale,
            ),
            offset: 0,
        },
        Section {
            name: "head1.bias",
            dtype: DTYPE_I32,
            data: q_i32(&floats(&store, "head1b"), h_scale),
            offset: 0,
        },
        Section {
            name: "head2.weights",
            dtype: DTYPE_I16,
            data: q_i16(
                &transpose_internal(
                    &floats(&store, "head2w"),
                    OUTPUT_BUCKETS * HEAD_HIDDEN_2,
                    HEAD_HIDDEN_1,
                ),
                h_scale,
            ),
            offset: 0,
        },
        Section {
            name: "head2.bias",
            dtype: DTYPE_I32,
            data: q_i32(&floats(&store, "head2b"), h_scale),
            offset: 0,
        },
        Section {
            name: "output.weights",
            dtype: DTYPE_I16,
            data: q_i16(
                &transpose_internal(&floats(&store, "outputw"), OUTPUT_BUCKETS, HEAD_HIDDEN_2),
                h_scale,
            ),
            offset: 0,
        },
        Section {
            name: "output.bias",
            dtype: DTYPE_I32,
            data: q_i32(&floats(&store, "outputb"), h_scale),
            offset: 0,
        },
    ];
    let fixed_header = 8 + 15 * 4 + 4 * 4 + 8;
    let header_bytes = fixed_header + sections.len() * SECTION_DESC_BYTES;
    let mut offset = header_bytes as u64;
    for section in &mut sections {
        section.offset = offset;
        offset += section.data.len() as u64;
    }
    let payload: Vec<u8> = sections
        .iter()
        .flat_map(|s| s.data.iter().copied())
        .collect();
    let checksum = fnv1a64(&payload);
    let mut file = Vec::with_capacity(offset as usize);
    file.extend_from_slice(&MAGIC);
    for value in [
        FORMAT_VERSION,
        ARCHITECTURE_ID,
        header_bytes as u32,
        sections.len() as u32,
        POSITION_FEATURE_COUNT as u32,
        ATTACK_FEATURE_COUNT as u32,
        STRUCTURE_FEATURE_COUNT as u32,
        POSITION_WIDTH as u32,
        ATTACK_WIDTH as u32,
        STRUCTURE_WIDTH as u32,
        CONCAT_WIDTH as u32,
        HEAD_HIDDEN_1 as u32,
        HEAD_HIDDEN_2 as u32,
        OUTPUT_BUCKETS as u32,
        SCORE_SCALE as u32,
    ] {
        write_u32(&mut file, value);
    }
    write_f32(&mut file, p_scale);
    write_f32(&mut file, if attack_int8 { attack_scale } else { p_scale });
    write_f32(&mut file, h_scale);
    write_f32(&mut file, h_scale);
    write_u64(&mut file, checksum);
    for section in &sections {
        let mut name = [0_u8; 24];
        let bytes = section.name.as_bytes();
        name[..bytes.len().min(24)].copy_from_slice(&bytes[..bytes.len().min(24)]);
        file.extend_from_slice(&name);
        write_u32(&mut file, section.dtype);
        write_u32(&mut file, 0);
        write_u64(&mut file, section.offset);
        write_u64(&mut file, section.data.len() as u64);
    }
    file.extend_from_slice(&payload);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, &file)?;

    let loaded = ExportedNetwork::load(output)?;
    let mut maximum_difference = 0.0_f32;
    for (encoded, bucket) in test_positions.iter().take(16) {
        let quantised = loaded.evaluate(encoded, *bucket);
        let reference = float_reference.evaluate(encoded, *bucket);
        maximum_difference = maximum_difference.max((quantised - reference).abs());
        if !quantised.is_finite() {
            return Err(io::Error::other("non-finite score in export round-trip"));
        }
        if maximum_difference > 0.25 {
            return Err(io::Error::other(format!(
                "export inference difference {maximum_difference} exceeds tolerance 0.25"
            )));
        }
    }
    println!(
        "Export round-trip verified: offsets, lengths, shapes, checksum, inference max |error|={maximum_difference:.6}"
    );
    Ok(())
}

#[derive(Clone)]
struct LoadedSection {
    dtype: u32,
    bytes: Vec<u8>,
}

pub struct ExportedNetwork {
    sections: std::collections::BTreeMap<String, LoadedSection>,
    p_scale: f32,
    attack_scale: f32,
    head_scale: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct GoldenNetworkResult {
    pub position_hash: [u64; 2],
    pub attack_hash: [u64; 2],
    pub structure_hash: [u64; 2],
    pub output: f32,
    pub engine_score: i32,
}

impl ExportedNetwork {
    pub fn load(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let mut c = Cursor::new(&bytes);
        let mut magic = [0_u8; 8];
        c.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(io::Error::other("bad MNUEv2 magic"));
        }
        let read_u32 = |c: &mut Cursor<&Vec<u8>>| -> io::Result<u32> {
            let mut b = [0; 4];
            c.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        };
        let read_f32 = |c: &mut Cursor<&Vec<u8>>| -> io::Result<f32> {
            let mut b = [0; 4];
            c.read_exact(&mut b)?;
            Ok(f32::from_le_bytes(b))
        };
        let version = read_u32(&mut c)?;
        let arch = read_u32(&mut c)?;
        let header = read_u32(&mut c)? as usize;
        let section_count = read_u32(&mut c)? as usize;
        if version != FORMAT_VERSION || arch != ARCHITECTURE_ID {
            return Err(io::Error::other("incompatible MNUEv2 file"));
        }
        for _ in 0..11 {
            let _ = read_u32(&mut c)?;
        }
        let p_scale = read_f32(&mut c)?;
        let attack_scale = read_f32(&mut c)?;
        let head_scale = read_f32(&mut c)?;
        let _output_scale = read_f32(&mut c)?;
        let mut hb = [0; 8];
        c.read_exact(&mut hb)?;
        let expected_hash = u64::from_le_bytes(hb);
        let mut descriptions = Vec::new();
        for _ in 0..section_count {
            let mut name = [0; 24];
            c.read_exact(&mut name)?;
            let name = String::from_utf8_lossy(&name)
                .trim_end_matches('\0')
                .to_string();
            let dtype = read_u32(&mut c)?;
            let _reserved = read_u32(&mut c)?;
            let mut b = [0; 8];
            c.read_exact(&mut b)?;
            let offset = u64::from_le_bytes(b) as usize;
            c.read_exact(&mut b)?;
            let len = u64::from_le_bytes(b) as usize;
            descriptions.push((name, dtype, offset, len));
        }
        if c.position() as usize != header || header > bytes.len() {
            return Err(io::Error::other("invalid header size"));
        }
        if fnv1a64(&bytes[header..]) != expected_hash {
            return Err(io::Error::other("checksum mismatch"));
        }
        let mut sections = std::collections::BTreeMap::new();
        let mut expected_offset = header;
        for (name, dtype, offset, len) in descriptions {
            let end = offset
                .checked_add(len)
                .ok_or_else(|| io::Error::other("section overflow"))?;
            if offset != expected_offset || end > bytes.len() {
                return Err(io::Error::other("non-contiguous or out-of-range section"));
            }
            expected_offset = end;
            sections.insert(
                name,
                LoadedSection {
                    dtype,
                    bytes: bytes[offset..end].to_vec(),
                },
            );
        }
        if expected_offset != bytes.len() {
            return Err(io::Error::other("trailing or missing section bytes"));
        }
        let expected = [
            ("position.weights", POSITION_FEATURE_COUNT * POSITION_WIDTH),
            ("position.bias", POSITION_WIDTH),
            ("attack.weights", ATTACK_FEATURE_COUNT * ATTACK_WIDTH),
            ("attack.bias", ATTACK_WIDTH),
            (
                "structure.weights",
                STRUCTURE_FEATURE_COUNT * STRUCTURE_WIDTH,
            ),
            ("structure.bias", STRUCTURE_WIDTH),
            (
                "head1.weights",
                OUTPUT_BUCKETS * HEAD_HIDDEN_1 * CONCAT_WIDTH,
            ),
            ("head1.bias", OUTPUT_BUCKETS * HEAD_HIDDEN_1),
            (
                "head2.weights",
                OUTPUT_BUCKETS * HEAD_HIDDEN_2 * HEAD_HIDDEN_1,
            ),
            ("head2.bias", OUTPUT_BUCKETS * HEAD_HIDDEN_2),
            ("output.weights", OUTPUT_BUCKETS * HEAD_HIDDEN_2),
            ("output.bias", OUTPUT_BUCKETS),
        ];
        for (name, elements) in expected {
            let section = sections
                .get(name)
                .ok_or_else(|| io::Error::other(format!("missing section {name}")))?;
            let element_bytes = match section.dtype {
                DTYPE_I8 => 1,
                DTYPE_I16 => 2,
                DTYPE_I32 => 4,
                _ => 0,
            };
            if element_bytes == 0 || section.bytes.len() != elements * element_bytes {
                return Err(io::Error::other(format!(
                    "invalid shape/length for section {name}"
                )));
            }
        }
        Ok(Self {
            sections,
            p_scale,
            attack_scale,
            head_scale,
        })
    }

    fn values(&self, name: &str, scale: f32) -> Vec<f32> {
        let section = &self.sections[name];
        match section.dtype {
            DTYPE_I8 => section
                .bytes
                .iter()
                .map(|&x| f32::from(x as i8) * scale)
                .collect(),
            DTYPE_I16 => section
                .bytes
                .chunks_exact(2)
                .map(|x| f32::from(i16::from_le_bytes(x.try_into().unwrap())) * scale)
                .collect(),
            DTYPE_I32 => section
                .bytes
                .chunks_exact(4)
                .map(|x| i32::from_le_bytes(x.try_into().unwrap()) as f32 * scale)
                .collect(),
            _ => panic!("unknown dtype"),
        }
    }

    fn raw_values(&self, name: &str) -> Vec<i32> {
        let section = &self.sections[name];
        match section.dtype {
            DTYPE_I8 => section.bytes.iter().map(|&x| i32::from(x as i8)).collect(),
            DTYPE_I16 => section
                .bytes
                .chunks_exact(2)
                .map(|x| i32::from(i16::from_le_bytes(x.try_into().unwrap())))
                .collect(),
            DTYPE_I32 => section
                .bytes
                .chunks_exact(4)
                .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
                .collect(),
            _ => panic!("unknown dtype"),
        }
    }

    fn accumulator_hashes(
        &self,
        prefix: &str,
        width: usize,
        features: &crate::features::FeaturePair,
    ) -> [u64; 2] {
        let weights = self.raw_values(&format!("{prefix}.weights"));
        let bias = self.raw_values(&format!("{prefix}.bias"));
        let hash = |indices: &[usize]| {
            let mut accumulator = bias.clone();
            for &index in indices {
                for (dst, value) in accumulator
                    .iter_mut()
                    .zip(&weights[index * width..(index + 1) * width])
                {
                    *dst += value;
                }
            }
            let mut bytes = Vec::with_capacity(accumulator.len() * 4);
            for value in accumulator {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            fnv1a64(&bytes)
        };
        [hash(&features.stm), hash(&features.ntm)]
    }

    pub fn golden_evaluate(&self, encoded: &EncodedPosition, bucket: u8) -> GoldenNetworkResult {
        let output = self.evaluate(encoded, bucket);
        GoldenNetworkResult {
            position_hash: self.accumulator_hashes("position", POSITION_WIDTH, &encoded.position),
            attack_hash: self.accumulator_hashes("attack", ATTACK_WIDTH, &encoded.attack),
            structure_hash: self.accumulator_hashes(
                "structure",
                STRUCTURE_WIDTH,
                &encoded.structure,
            ),
            output,
            engine_score: (output * SCORE_SCALE as f32).round() as i32,
        }
    }

    fn branch(
        &self,
        prefix: &str,
        width: usize,
        features: &crate::features::FeaturePair,
    ) -> Vec<f32> {
        let scale = if prefix == "attack" {
            self.attack_scale
        } else {
            self.p_scale
        };
        let weights = self.values(&format!("{prefix}.weights"), scale);
        let bias = self.values(&format!("{prefix}.bias"), scale);
        let accumulate = |indices: &[usize]| {
            let mut x = bias.clone();
            for &index in indices {
                let row = &weights[index * width..(index + 1) * width];
                for (dst, value) in x.iter_mut().zip(row) {
                    *dst += value;
                }
            }
            let half = width / 2;
            (0..half)
                .map(|i| x[i].clamp(0.0, 1.0) * x[i + half].clamp(0.0, 1.0))
                .collect::<Vec<_>>()
        };
        let mut output = accumulate(&features.stm);
        output.extend(accumulate(&features.ntm));
        output
    }

    pub fn evaluate(&self, encoded: &EncodedPosition, bucket: u8) -> f32 {
        let mut x = self.branch("position", POSITION_WIDTH, &encoded.position);
        x.extend(self.branch("attack", ATTACK_WIDTH, &encoded.attack));
        x.extend(self.branch("structure", STRUCTURE_WIDTH, &encoded.structure));
        let affine = |input: &[f32],
                      wname: &str,
                      bname: &str,
                      in_size: usize,
                      out_size: usize,
                      bucket: usize| {
            let w = self.values(wname, self.head_scale);
            let b = self.values(bname, self.head_scale);
            let row_base = bucket * out_size;
            (0..out_size)
                .map(|o| {
                    let row = row_base + o;
                    b[row]
                        + input
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| w[row * in_size + i] * v)
                            .sum::<f32>()
                })
                .collect::<Vec<_>>()
        };
        let x = affine(
            &x,
            "head1.weights",
            "head1.bias",
            CONCAT_WIDTH,
            HEAD_HIDDEN_1,
            bucket as usize,
        )
        .into_iter()
        .map(|v| v.clamp(0.0, 1.0).powi(2))
        .collect::<Vec<_>>();
        let x = affine(
            &x,
            "head2.weights",
            "head2.bias",
            HEAD_HIDDEN_1,
            HEAD_HIDDEN_2,
            bucket as usize,
        )
        .into_iter()
        .map(|v| v.clamp(0.0, 1.0).powi(2))
        .collect::<Vec<_>>();
        affine(
            &x,
            "output.weights",
            "output.bias",
            HEAD_HIDDEN_2,
            1,
            bucket as usize,
        )[0]
    }
}

struct FloatReference {
    position_w: Vec<f32>,
    position_b: Vec<f32>,
    attack_w: Vec<f32>,
    attack_b: Vec<f32>,
    structure_w: Vec<f32>,
    structure_b: Vec<f32>,
    head1_w: Vec<f32>,
    head1_b: Vec<f32>,
    head2_w: Vec<f32>,
    head2_b: Vec<f32>,
    output_w: Vec<f32>,
    output_b: Vec<f32>,
}

impl FloatReference {
    fn from_store(store: &ModelWeights) -> Self {
        Self {
            position_w: transpose_internal(
                &floats(store, "positionw"),
                POSITION_WIDTH,
                POSITION_FEATURE_COUNT,
            ),
            position_b: floats(store, "positionb"),
            attack_w: transpose_internal(
                &floats(store, "attackw"),
                ATTACK_WIDTH,
                ATTACK_FEATURE_COUNT,
            ),
            attack_b: floats(store, "attackb"),
            structure_w: transpose_internal(
                &floats(store, "structurew"),
                STRUCTURE_WIDTH,
                STRUCTURE_FEATURE_COUNT,
            ),
            structure_b: floats(store, "structureb"),
            head1_w: transpose_internal(
                &floats(store, "head1w"),
                OUTPUT_BUCKETS * HEAD_HIDDEN_1,
                CONCAT_WIDTH,
            ),
            head1_b: floats(store, "head1b"),
            head2_w: transpose_internal(
                &floats(store, "head2w"),
                OUTPUT_BUCKETS * HEAD_HIDDEN_2,
                HEAD_HIDDEN_1,
            ),
            head2_b: floats(store, "head2b"),
            output_w: transpose_internal(&floats(store, "outputw"), OUTPUT_BUCKETS, HEAD_HIDDEN_2),
            output_b: floats(store, "outputb"),
        }
    }

    fn branch(
        weights: &[f32],
        bias: &[f32],
        width: usize,
        features: &crate::features::FeaturePair,
    ) -> Vec<f32> {
        let accumulate = |indices: &[usize]| {
            let mut x = bias.to_vec();
            for &index in indices {
                for (dst, value) in x
                    .iter_mut()
                    .zip(&weights[index * width..(index + 1) * width])
                {
                    *dst += value;
                }
            }
            let half = width / 2;
            (0..half)
                .map(|i| x[i].clamp(0.0, 1.0) * x[i + half].clamp(0.0, 1.0))
                .collect::<Vec<_>>()
        };
        let mut x = accumulate(&features.stm);
        x.extend(accumulate(&features.ntm));
        x
    }

    fn affine(
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        in_size: usize,
        out_size: usize,
        bucket: usize,
    ) -> Vec<f32> {
        let base = bucket * out_size;
        (0..out_size)
            .map(|o| {
                let row = base + o;
                bias[row]
                    + input
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| weights[row * in_size + i] * v)
                        .sum::<f32>()
            })
            .collect()
    }

    fn evaluate(&self, encoded: &EncodedPosition, bucket: u8) -> f32 {
        let mut x = Self::branch(
            &self.position_w,
            &self.position_b,
            POSITION_WIDTH,
            &encoded.position,
        );
        x.extend(Self::branch(
            &self.attack_w,
            &self.attack_b,
            ATTACK_WIDTH,
            &encoded.attack,
        ));
        x.extend(Self::branch(
            &self.structure_w,
            &self.structure_b,
            STRUCTURE_WIDTH,
            &encoded.structure,
        ));
        let x = Self::affine(
            &x,
            &self.head1_w,
            &self.head1_b,
            CONCAT_WIDTH,
            HEAD_HIDDEN_1,
            bucket as usize,
        )
        .into_iter()
        .map(|v| v.clamp(0.0, 1.0).powi(2))
        .collect::<Vec<_>>();
        let x = Self::affine(
            &x,
            &self.head2_w,
            &self.head2_b,
            HEAD_HIDDEN_1,
            HEAD_HIDDEN_2,
            bucket as usize,
        )
        .into_iter()
        .map(|v| v.clamp(0.0, 1.0).powi(2))
        .collect::<Vec<_>>();
        Self::affine(
            &x,
            &self.output_w,
            &self.output_b,
            HEAD_HIDDEN_2,
            1,
            bucket as usize,
        )[0]
    }
}
