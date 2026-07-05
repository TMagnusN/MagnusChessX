# MNUEv2 Position / Attack / Structure

Architecture name: `MNUEv2-PositionAttackStructure`

The network uses three independent sparse feature tables. Their accumulator
outputs are never summed:

```text
Position  12,288 -> 320
Attack   153,216 -> 288
Structure 34,685 -> 160

pairwise CReLU per perspective
STM/NTM concatenation inside each branch
Position || Attack || Structure = 768

12 material heads: 768 -> 32 -> 32 -> 1
```

Each table produces one accumulator for the side-to-move perspective and one
for the opposite perspective. A width-`W` accumulator is clamped, split into
two `W/2` halves, and pairwise multiplied. Concatenating both perspectives
restores width `W`; concatenating the three branch representations gives 768.

## Feature Mappings

All mappings are deterministic and perspective-normalised. Sparse indices are
range checked and duplicate emissions are deduplicated because the features
are binary states.

- Position: 16 king zones x 2 relative colours x 6 piece types x 64 squares.
  It contains only piece identity, colour, square, and king/input bucket.
- Attack: attacked/defended state, occupied attacker-target edges,
  per-piece hanging/overload/pin/lower-value-attacker flags, and king-zone
  pressure counts.
- Structure: pawn status masks, file state, pawn islands, centre pawn
  occupancy, king shelter, structural knight/bishop outposts, pawn colour
  complexes, and passed-pawn blockers.

Structure features use pawn and piece topology only. They do not consume the
tactical attack map.

## Material Buckets

Material units are `P=1`, `N/B=3`, `R=5`, `Q=9`, and `K=0`. Bucket version 1
uses inclusive upper bounds:

```text
7, 11, 13, 17, 21, 27, 33, 41, 50, 59, 69, infinity
```

Routing is hard to one bucket. The model and format keep the bucket ID
separate so interpolation can be added later.

## `--date`

`--date N` and its alias `--positions N` limit accepted valid positions:

- positive: stop after at most `N` valid decoded positions;
- zero: no limit;
- negative: CLI error.

Malformed records increment the rejected count and do not consume the budget.
The reader applies one global counter while streaming files in order, before
feature preparation; there is no worker-count multiplication.

## Checkpoints

Every checkpoint directory contains `metadata.json` next to
`optimiser_state/`. Metadata records the format and architecture versions,
three vocabulary sizes and widths, head dimensions, bucket mapping, score
scale, fake-quant configuration, step, and accepted-position count. A missing
or incompatible metadata file is rejected. Monolithic X1 checkpoints are not
shape- or semantics-compatible.

## Binary Export

The file is explicitly little-endian and does not serialize a C/Rust struct.
It contains:

1. `MNUEV2` magic and fixed architecture metadata;
2. per-section dtype, offset, and byte length descriptors;
3. position weights/bias;
4. attack weights/bias;
5. structure weights/bias;
6. all three head layers;
7. an FNV-1a-64 payload checksum.

The fixed header is 668 bytes:

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 8 | `MNUEV2\0\0` |
| 8 | 60 | fifteen little-endian u32 architecture fields |
| 68 | 16 | Position, Attack, head, and output f32 scales |
| 84 | 8 | payload FNV-1a-64 |
| 92 | 576 | twelve 48-byte section descriptors |

The fifteen u32 values are format version, architecture ID, header bytes,
section count, three vocabulary sizes, three raw accumulator widths, concat
width, two hidden widths, output bucket count, and score scale.

Each descriptor is written explicitly as:

```text
name[24], dtype u32, reserved u32, offset u64, byte_length u64
```

`reserved` is zero. Sections are contiguous, ordered, non-overlapping, and
extend exactly to end of file. The payload checksum starts at byte 668.

Section order and byte sizes are:

| Section | Dtype | Bytes |
|---|---|---:|
| `position.weights` | i16 | 7,864,320 |
| `position.bias` | i32 | 1,280 |
| `attack.weights` | i16 or i8 | 88,252,416 or 44,126,208 |
| `attack.bias` | i32 | 1,152 |
| `structure.weights` | i16 | 11,099,200 |
| `structure.bias` | i32 | 640 |
| `head1.weights` | i16 | 589,824 |
| `head1.bias` | i32 | 1,536 |
| `head2.weights` | i16 | 24,576 |
| `head2.bias` | i32 | 1,536 |
| `output.weights` | i16 | 768 |
| `output.bias` | i32 | 48 |

Format-v4 scales are bit-exact f32 values: Position/Structure `1/255`,
Attack `1/255` for i16 or `1/127` for i8, head/output `1/64`, and score scale
400.

Position and Structure weights are int16. Attack is int16 by default and int8
when Attack fake quant/export is enabled. Accumulator and head biases use
int32. The exporter immediately reloads the file, validates section shapes,
contiguous offsets and checksum, then compares quantised inference against the
float reference on fixed positions.

The exported reference inference gathers only the selected material head.
Bullet's training graph retains its existing bucketed-affine implementation,
which materialises bucket outputs before `select`; this is training-only and
is not the required engine inference path.

MagnusChessX Thinking integration and runtime diagnostics are documented in
[MNUE_V2_ENGINE.md](MNUE_V2_ENGINE.md).

## Commands

```powershell
cd tools\mnue_v2_trainer
cargo test --offline
cargo run --features cuda --offline -- --dry-run --date 10000 --model-summary
cargo run --features cuda --offline -- --date 1000000
cargo run --features cuda --offline -- --date 0
```
