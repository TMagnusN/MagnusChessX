# MNUE-P2 Trainer

This repository keeps the MagnusChessX-side P2 trainer in
`Tools/mnue_p2_trainer`. The trainer depends on
[bullet](https://github.com/jw1912/bullet) through Cargo's Git dependency
support; local Bullet checkouts can be passed through Cargo `--config patch`
overrides when needed. The architecture, export header, command-line defaults,
and network workflow live next to the engine code.

For a local Bullet checkout, create a machine-local Cargo config file outside
the repository:

```toml
[patch."https://github.com/jw1912/bullet"]
bullet_lib = { path = "<bullet-root>/crates/bullet_lib" }
"bullet-trainer" = { path = "<bullet-root>/crates/trainer" }
"bullet-gpu" = { path = "<bullet-root>/crates/gpu" }
"bullet-compiler" = { path = "<bullet-root>/crates/compiler" }
```

Pass it to Cargo with `--config <local-bullet-config.toml>`.

## Supported Layouts

Select the layout with `--arch p2` or `--arch p2-k32`. The default is `p2`.

### P2

```text
shape          = 1 x 32 x 16 x 1024 x 10240
output buckets = 32
input buckets  = 16
hidden size    = 1024
input features = 16 x 2 x 5 x 64 = 10240
```

Input features are:

```text
input_bucket16 x relative_color2 x non_king_piece_type5 x relative_square64
```

The current king bucket is `king_zone16`:

```text
file_group = file / 2      // 0..3
rank_group = rank / 2      // 0..3
bucket     = rank_group * 4 + file_group
```

Output buckets are:

```text
phase_bucket2 x stm_king_zone16 = 32
```

### P2-K32

P2-K32 keeps the P2 hidden size and feature semantics, but doubles king bucket
resolution:

```text
shape          = 1 x 64 x 32 x 1024 x 20480
output buckets = 64
input buckets  = 32
hidden size    = 1024
input features = 32 x 2 x 5 x 64 = 20480
```

The K32 king bucket is:

```text
file_group = min(file, 7 - file)  // 0..3, horizontal mirror
rank       = rank                 // 0..7
bucket     = rank * 4 + file_group
```

Output buckets are:

```text
phase_bucket2 x stm_king_zone32 = 64
```

The exported `.mnue` file has the standard 40-byte Magnus MNUE header followed
by the Bullet `quantised.bin` payload.

## Training

Example 8-hour-style run at roughly 2M positions/s:

```powershell
cargo run --manifest-path Tools/mnue_p2_trainer/Cargo.toml --release --features cuda -- --name mnue_p2_8h --data <train.data> --positions 57600000000 --batch-size 65536 --threads 12 --superbatches 500 --lr 0.001 --lr-final 0.0001 --l0-init-scale 0.25 --save-sb 5
```

Example P2-K32 100M / 5-superbatch smoke run:

```powershell
cargo run --manifest-path Tools/mnue_p2_trainer/Cargo.toml --release --features cuda -- --arch p2-k32 --name mnue_p2k32_100m_5sb --data <train.data> --positions 100000000 --batch-size 65536 --threads 12 --superbatches 5 --lr 0.001 --lr-final 0.0001 --l0-init-scale 0.25 --save-sb 1
```

Important parameters:

```text
--arch p2|p2-k32  trainer architecture, default p2
--name NAME          net id, output directory name, checkpoint prefix
--data PATH          training .data or .binpack file; repeatable
--data-dir DIR       directory of .binpack chunks; repeatable
--positions N        total training positions
--batch-size N       positions per GPU batch
--threads N          CPU data-prep threads
--superbatches N     split total training into N superbatches
--lr X               initial learning rate
--lr-final X         final exponential-decay learning rate
--l0-init-scale X    multiplier for l0 weight init stdev
--save-sb N          save one checkpoint every N superbatches
--save-rate N        alias for --save-sb
--output-dir PATH    checkpoint root, default runs/<name>
--net-dir PATH       exported .mnue directory, default nets/mnue
```

Training input is required. Use either one or more `--data` paths, or one or
more `--data-dir` directories containing `.binpack` chunks.

Outputs:

```text
runs/<name>/<name>-<superbatch>/quantised.bin
nets/mnue/<name>-<superbatch>.mnue
```

## Embedded Network Sync

Network filenames use the payload SHA-512, not the full headered file hash.
Strip the 40-byte MNUE header, hash the remaining payload, and use the first
nine hex digits:

```text
mm-<payload-sha512-prefix9>.MNUE
```

Example:

```text
source       = nets/mnue/mnue_p2_8h-500.mnue
payload hash = 4ac81907ebaea42872fb89e1823ffb6add6b5dba01384f5d11e29759f1a70a6d...
embedded     = mm-4ac81907e.MNUE
```

After selecting a network:

1. Copy it into `Src/build/<embedded-name>.MNUE` for local builds.
2. Copy the same file into the `TMagnusN/MagnusMNUE` repository root.
3. Commit and push the network repository.
4. Update `Src/Makefile`:

```make
MNUE_EMBEDDED_FILE := mm-4ac81907e.MNUE
```

5. Update the fallback in `Src/mnue/Mnue.h`:

```cpp
#define MNUE_EMBEDDED_FILENAME "mm-4ac81907e.MNUE"
```

The Makefile downloads from:

```text
https://raw.githubusercontent.com/TMagnusN/MagnusMNUE/main/$(MNUE_EMBEDDED_FILE)
```

## Future K64 Experiment

If K32 is stable, a more aggressive P2-K64 experiment can follow.


```text
input buckets  = 64
output buckets = 128
input features = 64 x 2 x 5 x 64 = 40960
```

K64 doubles the first-layer size again and should be treated as a separate
capacity/data experiment, not the first upgrade step.
