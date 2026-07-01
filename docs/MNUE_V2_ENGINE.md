# MagnusChessX Thinking MNUEv2 Engine Support

The canonical training and binary-format description is
[MNUE_V2_FORMAT.md](MNUE_V2_FORMAT.md). Engine constants must match
`tools/mnue_v2_trainer/src/model.rs`, `features/`, `buckets.rs`, and
`export.rs`; this document describes the engine integration.

## Architecture Dispatch

MagnusChessX Thinking keeps the legacy MNUE P2/P2Pro loader and evaluator. The
`MNUEfile` option probes the first bytes of the selected file:

- legacy files are passed to the P2/P2Pro loader;
- `MNUEV2\0\0` files are passed to the strict MNUEv2 loader;
- a v2 file is accepted only when its format and architecture fields match the
  engine layout.

Search dispatch remains `X1`, MNUEv2, legacy P2/P2Pro, NNUE, then HCE. Loading
one MNUE architecture unloads the other MNUE architecture, while immutable
network storage remains process-global and accumulator state remains local to
each search worker.

## Perspective And Dimension Contract

Each sparse table produces one raw i32 accumulator per colour perspective.
Pairwise CReLU clamps each raw accumulator, multiplies its two equal halves,
and produces:

| Branch | Vocabulary | Raw width per perspective | Pairwise output per perspective | Final STM/NTM branch |
|---|---:|---:|---:|---:|
| Position | 12,288 | 320 | 160 | 320 |
| Attack | 153,216 | 288 | 144 | 288 |
| Structure | 34,685 | 160 | 80 | 160 |

For the current side to move, each branch is ordered
`pairwise(STM) || pairwise(NTM)`. The head input is:

```text
Position[320] || Attack[288] || Structure[160] = 768
```

Compile-time assertions enforce every width.

## Feature Parity

`MnueV2Features.cpp` is a semantic port of the trainer encoders:

- Position: perspective-relative piece identity, square, colour, and own-king
  input bucket.
- Attack: tactical status, occupied attack edges, hanging/overload/pin/
  lower-value-attacker relations, and king-zone pressure.
- Structure: pawn states, files, islands, centre openness, king shelter,
  outposts, colour complexes, and passed-pawn blockers.

All feature lists are sorted and deduplicated because the trainer treats them
as binary sparse states. Debug builds range-check every emitted index.
`mnuev2debug` prints indices and decoded feature-family meanings for the
current FEN.

Attack encoding computes every piece attack bitboard once per position and
reuses it for both perspectives. This preserves trainer semantics while
avoiding repeated attacker-map reconstruction.

## Scalar Inference

The scalar full-refresh evaluator is the reference implementation:

1. copy each i32 table bias;
2. add every active sparse row without saturation;
3. apply the exported scale and pairwise CReLU;
4. concatenate STM/NTM Position, Attack, and Structure branches;
5. evaluate only the selected material head;
6. multiply by score scale and round with `llround`.

`mnuev2scalar on` forces scalar row updates and dense inference. The selected
AVX2 head uses runtime-transposed 32-row weight blocks; its branch-separated
accumulation order is covered by golden-vector tolerance checks.

## Incremental State

Each search worker owns one semantic state and one materialised accumulator.
The semantic state contains:

- square-addressed Position slots;
- a persistent `attacks_from[64]` / `attackers_to[64]` AttackGraph;
- addressable Attack status, relation, occupied-edge, and pressure slots;
- addressable Structure pawn, file, shelter, outpost, complex, and blocker
  slots;
- direct persistent binary-feature refcounts;
- fixed per-thread slot-undo and packed-row-delta arenas;
- compact per-ply frame offsets and dirty-branch masks.

Position updates come directly from changed move squares. Structure updates
are restricted to dirty files, affected pawns, changed blockers, shelter, and
relevant outposts. Attack source discovery uses changed squares plus old/new
slider rays; stable edges from unchanged sliders are not refreshed. Pin/x-ray
summaries inspect only changed king rays.

Semantic work is lazy: `after_make` records a pending frame. The frame is
updated only if the position is evaluated or another move is pushed below it.
Pending frames discarded by an immediate unmake do no feature work.

Row deltas remain lazy until evaluation. The AVX2 materialiser dispatches once
per frame, prefetches sparse rows, and fuses adjacent remove/add pairs as
`new_row - old_row`. Unmake applies exact inverse integer deltas and then
rewinds semantic and row arenas.

The selected head caches separate 32-element Position, Attack, and Structure
first-layer partials by branch epoch, bucket, and STM orientation. A material
bucket change invalidates all three partials.

Reloading a network increments a generation number, invalidating existing
state safely. Hash clearing does not own or mutate MNUEv2 state. Accumulators are
mathematically reversible i32 values; no saturating integer operation is used.

## Attack Int8 And AVX2

Position and Structure rows are int16. Attack rows may be:

- int16 with scale `1/255`;
- int8 with scale `1/127`.

Both accumulate into i32. The scalar int8 kernel sign-extends each byte. The
AVX2 kernel loads 16 bytes, sign-extends them to two groups of eight i32
lanes, then adds or subtracts without saturation. GCC/Clang builds place this
kernel in an AVX2-targeted function and dispatch with
`__builtin_cpu_supports("avx2")`; portable64 builds retain a scalar fallback.

## Material Buckets

Bucket mapping version 1 uses material units `P=1`, `N/B=3`, `R=5`, `Q=9`,
and `K=0`, with inclusive upper bounds:

```text
7, 11, 13, 17, 21, 27, 33, 41, 50, 59, 69, infinity
```

Routing uses the first bound greater than or equal to the current material
value. Only that bucket's `768 -> 32 -> 32 -> 1` head is evaluated.

## Diagnostics

Hot-path telemetry is disabled by default:

```powershell
mingw32-make MNUEV2_TELEMETRY=1
```

When enabled, `mnuev2telemetry 100` reports per-branch row additions,
removals, cache activity, rebuilds, move-type distributions, row-delta
percentiles, Attack accumulator ranges, clipping, slider-edge changes,
tactical-summary transitions, and Structure rebuilds on ordinary non-pawn
moves. Disabled builds omit the collection work.

Useful commands:

```text
mnuev2info
mnuev2debug
mnuev2eval
mnuev2loadcheck
mnuev2bucketcheck
mnuev2specialcheck
mnuev2goldencheck <fixture>
mnuev2check 1000 100
mnuev2simdcheck
mnuev2scalar on|off
```
