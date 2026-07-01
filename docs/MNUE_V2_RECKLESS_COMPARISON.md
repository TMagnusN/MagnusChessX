# MNUEv2 / Reckless Architecture Reference

This document records design-level observations from a read-only inspection of
Reckless and compares them with the MNUEv2 integration in MagnusChessX
Thinking. It intentionally excludes run results, strength claims, throughput
figures, and promotional build notes.

## Purpose

The comparison is useful only as an architecture reference. Reckless and
MagnusChessX Thinking use different feature semantics, file formats, and
evaluation layouts, so implementation details are not directly interchangeable.
Any transferred idea must be re-derived inside the MagnusChessX Thinking code
base and validated against existing parity checks.

## License Boundary

- Reckless is licensed under AGPL terms.
- MagnusChessX Thinking is distributed under the MIT License.

Copying or adapting Reckless implementation code would not be appropriate for
an MIT-only distribution. Architectural ideas, public interfaces, dimensions,
and independently derived algorithms can be studied, but source expression,
comments, generated tables, and intrinsic sequences must not be copied.

This is an engineering note, not legal advice.

## Reckless Network Shape

Reckless uses two sparse feature families:

```text
Piece/king-bucket table
    7,680 x 768, int16
          |
          +--> PST accumulator[pov], int16[768]

Directed threat-edge table
    66,864 x 768, int8
          |
          +--> Threat accumulator[pov], int16[768]

For each POV:
    combined[768] = PST[768] + Threat[768]
    pairwise activation:
        combined[0..384] * combined[384..768]
        clipped/shifted/packed -> u8[384]

Final input:
    activated STM[384] || activated NTM[384] = u8[768]
```

The two sparse families are accumulated separately and summed before
activation. They are not separate final networks.

## MagnusChessX Thinking MNUEv2 Shape

MNUEv2 uses three sparse families with independent branch representations:

```text
Position  -> pairwise STM/NTM branch
Attack    -> pairwise STM/NTM branch
Structure -> pairwise STM/NTM branch

Position[320] || Attack[288] || Structure[160] = 768
```

The branch split is semantic: Position, Attack, and Structure retain separate
accumulators and separate first-head cache keys. This layout is stricter than
the Reckless combined-accumulator model and cannot be replaced without changing
the exported network contract.

## Move-To-Evaluation Flow

Reckless exposes a useful high-level pattern:

1. derive sparse deltas directly from board mutations;
2. keep perspective-local accumulators;
3. delay neural row work until an evaluation is required;
4. make unmake cheap by rewinding owned state instead of recomputing the full
   feature set.

MagnusChessX Thinking already follows the same broad direction for MNUEv2:

- changed squares and slider rays determine the Attack frontier;
- semantic updates are lazy;
- row deltas are materialised only when evaluation requires them;
- selected-head partials can be reused by branch epoch and bucket.

The remaining design question is how much state to store at each ply and how
to represent Attack/Structure summaries so that unmake and selected-head work
stay local.

## Feature And Format Compatibility

Directly reusable at the idea level:

- lazy semantic frames;
- event-local attack frontier discovery;
- persistent counters for attack summaries;
- branch-level invalidation keys;
- scalar reference paths for correctness.

Requires a format or exporter change:

- quantised sparse first-head weights;
- alternative activation packing;
- different output-bucket semantics;
- replacing branch-separated MNUEv2 tensors with a combined sparse family.

Incompatible without redesign:

- copying Reckless threat-edge index tables;
- adopting generated geometry tables verbatim;
- treating Reckless's combined accumulator as equivalent to the MNUEv2
  Position/Attack/Structure branch contract.

## Implementation Guidance

For MagnusChessX Thinking, improvements should preserve these constraints:

- feature encoders must match trainer output exactly;
- scalar and SIMD paths must remain parity-checkable;
- loader compatibility checks must reject mismatched dimensions or metadata;
- tablebase, search, and time-management behavior must stay independent of
  MNUE storage layout;
- any optional fast path must fall back to the reference path when unavailable.

Suggested engineering order:

1. keep scalar parity tests passing for every feature-family change;
2. isolate state-layout changes behind MNUEv2-owned structures;
3. add counters or signatures only when they are derivable from existing board
   mutation events;
4. prefer one small correctness-preserving change per patch;
5. keep format-changing work separate from runtime implementation changes.

## Reference Map

Areas to inspect in Reckless when re-deriving an idea:

| Area | What To Learn |
|---|---|
| sparse accumulator ownership | how per-thread network state is separated |
| threat-edge frontier | how board mutations identify affected edges |
| activation packing | how sparse first-head input is represented |
| selected head | how only the routed bucket is evaluated |
| search integration | where evaluation state is refreshed and rewound |

Areas to inspect in MagnusChessX Thinking before implementing:

| Area | What To Preserve |
|---|---|
| `MnueV2Features.*` | trainer-compatible feature semantics |
| `MnueV2Network.*` | loader, accumulator, and head contracts |
| `Search.*` | evaluation call sites and worker ownership |
| `Uci.cpp` | diagnostics and explicit network selection |
| `tools/mnue_v2_trainer` | exporter metadata and golden-vector generation |

The comparison should remain a design aid, not a source-transplant plan.
