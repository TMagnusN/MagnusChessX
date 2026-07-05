# MNUE-X1 Design Specification

MNUE-X1 is a tactical-feature network design for MagnusChessX Thinking. It is
not a blind widening of the existing P2 hidden layer; it adds information that
the P2 feature set cannot represent while keeping piece-position state
incremental.

## Goals

- Trainer and engine must produce identical active feature indices.
- The engine implementation must support deterministic scalar/reference
  inference before any SIMD path is considered authoritative.
- The network file must carry enough metadata to reject incompatible shape or
  feature-mapping changes.
- Incremental update rules must preserve exact feature parity with a
  full-refresh reference path.

## Network Structure

```text
59,392 sparse inputs
        |
        v
shared affine 59,392 -> 768, CReLU, pairwise multiply
        |
        +-- STM 384
        +-- NTM 384
                |
                v
             concat 768
                |
       material-bucket affine 768 -> 16, SCReLU
                |
       material-bucket affine 16 -> 32, SCReLU
                |
       material-bucket affine 32 -> 1
```

The input space is semantically split into piece-position and tactical-state
features, but the model is one jointly trained network.

Fixed dimensions:

| Item | Value |
|---|---:|
| Input king buckets | 16 |
| Piece inputs | 10,240 |
| Tactical inputs | 49,152 |
| Total inputs | 59,392 |
| Transformer width | 768 |
| Output buckets | 8 |
| Backend | 768 -> 16 -> 32 -> 1 |
| Parameters | 45,715,712 |
| Quantised payload | 91,433,760 bytes |
| File size with header | 91,433,824 bytes |

The maximum active feature count is 62: up to 30 non-king piece-position
features plus up to 32 tactical-state features.

## Piece-Position Features

Piece-position features reuse the P2-style non-king piece mapping:

```text
index =
    (((king_bucket16 * 2 + relative_colour)
       * 5 + non_king_piece_type)
      * 64 + relative_square)
```

The valid range is `[0, 10240)`. Kings are represented by the input bucket and
are not active features in this group.

## Tactical-State Features

Each piece, including kings, enables exactly one tactical feature:

```text
relative_piece_class = relative_colour * 6 + piece_type

index =
    10240
    + ((relative_piece_class * 64 + relative_square) * 64)
    + tactical_status
```

`tactical_status` is a 6-bit mask:

| Bit | Meaning |
|---:|---|
| 0 | attacked by enemy pawn |
| 1 | attacked by enemy knight |
| 2 | attacked by enemy bishop or queen diagonal ray |
| 3 | attacked by enemy rook or queen orthogonal ray |
| 4 | attacked by enemy king |
| 5 | defended by at least one friendly piece |

Attacks are pseudo-attacks: pinned pieces still count as attackers. Sliders
attack only through the first occupied square, matching the engine's attack
tables. The valid range is `[10240, 59392)`.

## Output Buckets

Output buckets are based on total piece count:

| Bucket | Total piece count |
|---:|---|
| 0 | 2-5 |
| 1 | 6-8 |
| 2 | 9-11 |
| 3 | 12-14 |
| 4 | 15-17 |
| 5 | 18-20 |
| 6 | 21-24 |
| 7 | 25-32 |

The king location is already represented by the input bucket, so the output
layer does not repeat the P2 king-zone split.

## Quantisation And File Format

MNUE-X1 uses MNUE file version 2 and architecture id 5. The header is fixed at
64 bytes, followed by tensors in this order:

| Tensor | Shape | Type | Quantisation |
|---|---|---|---:|
| `l0w` | 59,392 x 768 | i16 | QA=255 |
| `l0b` | 768 | i16 | QA=255 |
| `l1w` | 8 x 16 x 768 | i16 | QB=64 |
| `l1b` | 8 x 16 | i16 | QA=255 |
| `l2w` | 8 x 32 x 16 | i16 | QB=64 |
| `l2b` | 8 x 32 | i16 | QA=255 |
| `l3w` | 8 x 1 x 32 | i16 | QB=64 |
| `l3b` | 8 | i32 | QA*QB=16,320 |

Header fields are 16 little-endian 32-bit values:

```text
magic, version, arch, header_bytes,
input_size, hidden_size, input_buckets, output_buckets,
l1_size, l2_size, scale, qa,
qb, feature_version, flags, reserved
```

## Memory And Update Model

- Network weights are shared by all search workers.
- Each worker owns its accumulator state.
- Tactical-state updates are the primary complexity risk because one move can
  change attack status for pieces other than the moved piece.

The first engine reference path may rebuild tactical features in full. The
incremental path should update only:

1. pieces directly moved, captured, promoted, or castled;
2. pieces attacked from the moved piece's old and new squares;
3. the first piece on each ray from every changed occupancy square;
4. accumulator rows whose old and new tactical status differ.

## Implementation Order

1. Build trainer and C++ feature-index modules.
2. Test STM/NTM feature-index symmetry on fixed FEN positions.
3. Train a smoke network for loader and inference validation.
4. Implement version-2 loader and scalar/reference forward pass.
5. Connect the reference evaluator to search behind an explicit selection path.
6. Replace full tactical rebuilds with delta-based updates.
7. Keep parity tests as the acceptance gate for every optimisation.

## X2 Direction

X2 keeps the MagnusChessX Thinking feature definitions while adopting a more
separated accumulator layout:

- piece transformer and attack-edge transformer are stored separately and
  combined before activation;
- piece rows use i16 and attack-edge rows use i8;
- piece features include kings and are mirrored by own-king half;
- tactical features become explicit attacker-to-target occupied edges;
- pairwise CReLU produces compact activation values;
- the first head layer may consume sparse activation chunks directly.

X2 dimensions:

| Item | Value |
|---|---:|
| Piece inputs | 7,680 |
| Attack-edge inputs | 90,048 |
| Transformer width | 768 |
| Quantised transformer | piece i16 + edge i8 |
| Approximate network size | 77.3 MiB |

Attack-edge indices are independently defined for MagnusChessX Thinking by
relative attacker colour, attacker piece type, source square, legal empty-board
target slot, and relative target piece class.
