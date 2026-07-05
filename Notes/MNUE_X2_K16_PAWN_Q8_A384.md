# MNUE-X2-K16-pawn-Q8-A384

This is an experimental, non-default MNUE-X2 layout. It can be selected with
`MNUEfile` as a Stage 0 search evaluator, using the scalar full-rebuild oracle.
The embedded/default evaluator remains P2/P2Pro.

## Format

Header fields are 28 little-endian `u32` values:

```text
magic, version, arch, header_bytes,
piece_input_size, attack_input_size, pawn_pair_input_size,
piece_hidden_size, attack_hidden_size, pawn_pair_hidden_size,
merged_hidden_size,
input_buckets, output_buckets, l1_size, l2_size,
scale, qa,
piece_quant, piece_rescale,
attack_quant, attack_rescale,
pawn_pair_quant, pawn_pair_rescale,
l1_quant,
feature_version, flags, reserved0, reserved1
```

The current values are `version=3`, `arch=8`, `feature_version=2`, and
`header_bytes=112`.

Payload order:

```text
piece_l0w       i8  [12288][768]
attack_l0w      i8  [90048][384]
pawn_pair_l0w   i8  [4560][768]
l0b             i16 [768]
l1w             i8  [8][16][768]
l1b             f32 [8][16]
l2w             f32 [8][32][16]
l2b             f32 [8][32]
l3w             f32 [8][32]
l3b             f32 [8]
```

Expected payload bytes:

```text
12288*768 + 90048*384 + 4560*768 + 768*2
+ 8*16*768 + 8*16*4
+ 8*32*16*4 + 8*32*4
+ 8*32*4 + 8*4
= 47,636,512
```

Expected file bytes are `47,636,624`.

## Commands

Rust feature dump:

```powershell
cargo run --manifest-path Tools/mnue_x2_trainer/Cargo.toml -- --arch x2-k16-pawn-q8-a384 --dump-fen "<fen>"
```

Rust train/export, using the CUDA feature as with the existing trainer:

```powershell
cargo run --manifest-path Tools/mnue_x2_trainer/Cargo.toml --features cuda -- --arch x2-k16-pawn-q8-a384 --data <train.data> --export nets/mnue/mnue_x2_k16_pawn_q8_a384.mnue --batch-size 65536 --threads 12 --superbatches 4 --lr 0.001 --lr-final 0.0001
```

C++ diagnostics:

```text
mnuex2k16load <path>
mnuex2k16info
mnuex2k16eval
mnuex2k16dump
```

All C++ diagnostic output is emitted as `info string`.

Stage 0 search opt-in:

```text
setoption name MNUEfile value <path-to-mnue>
position startpos
eval
go depth 1
setoption name MNUEfile value <embedded>
```

When `MNUEfile` points at an arch-8 file, the engine loads
`MNUE-X2-K16-pawn-Q8-A384` and search dispatch uses the scalar full-rebuild
reference evaluator. Invalid arch-8 headers are rejected without falling back
to P2/P2Pro. The diagnostic `mnuex2k16load` command remains available, but
formal search selection should use `MNUEfile`.

## Feature Parity Workflow

For each FEN, dump Rust and C++ features, strip the C++ `info string ` prefix,
and diff the resulting `output_bucket`, `white/black piece`, `white/black
attack`, and `white/black pawn_pair` lines.

Representative FENs:

```text
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1
r4rk1/8/8/8/8/8/8/R3K2R b KQ - 1 1
7k/P7/8/8/8/8/8/4K3 w - - 0 1
7k/8/8/3pP3/8/8/8/4K3 w - d6 0 1
4k3/4r3/8/8/4N3/8/4K3/8 w - - 0 1
4k3/4r3/8/4b3/4N3/8/4K3/8 w - - 0 1
4k3/8/8/8/3q4/2N1N3/8/4K3 w - - 0 1
4k3/8/3P4/8/8/8/8/4K3 w - - 0 1
4k3/3p4/2P5/8/8/8/8/4K3 w - - 0 1
4k3/8/8/8/8/2P5/2P5/4K3 w - - 0 1
4k3/8/8/3pP3/3P4/8/8/4K3 w - - 0 1
```

Keep random legal FEN batches in the same diff loop before treating feature
parity as complete.
