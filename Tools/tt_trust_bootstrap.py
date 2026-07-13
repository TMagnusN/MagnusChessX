#!/usr/bin/env python3
"""Deterministic block-bootstrap acceptance check for TT move trust telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

BUCKETS = 8
REPLICATES = 10_000


@dataclass
class Counts:
    searched: int = 0
    tt_best: int = 0
    updates: int = 0
    saturated: int = 0

    def add(self, other: "Counts") -> None:
        self.searched += other.searched
        self.tt_best += other.tt_best
        self.updates += other.updates
        self.saturated += other.saturated


def block_key(row: dict[str, str], mode: str) -> tuple[str, ...]:
    run = row.get("run_id", "")
    if mode == "pair":
        value = row.get("opening_pair_id", "")
        if not value:
            raise ValueError("pair mode requires opening_pair_id on every row")
        return (run, value)
    if mode == "game":
        value = row.get("game_id", "") or row.get("game_epoch", "")
        if not value:
            raise ValueError("game mode requires game_id or game_epoch on every row")
        return (run, value)
    value = row.get("search_id", "")
    if value:
        return (run, value)
    epoch = row.get("game_epoch", "")
    sequence = row.get("search_sequence", "")
    if not sequence:
        raise ValueError("standalone mode requires search_id or search_sequence")
    return (run, epoch, sequence)


def choose_mode(rows: list[dict[str, str]], requested: str) -> str:
    if requested != "auto":
        return requested
    if rows and all(row.get("opening_pair_id", "") for row in rows):
        return "pair"
    if rows and all(row.get("game_id", "") for row in rows):
        return "game"
    return "standalone"


def load_blocks(path: Path, requested_mode: str) -> tuple[str, list[list[Counts]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("telemetry CSV is empty")

    required = {"bucket", "searched", "tt_best", "updates", "saturated"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"missing CSV fields: {', '.join(sorted(missing))}")

    mode = choose_mode(rows, requested_mode)
    grouped: dict[tuple[str, ...], list[Counts]] = defaultdict(
        lambda: [Counts() for _ in range(BUCKETS)]
    )
    for row in rows:
        bucket = int(row["bucket"])
        if not 0 <= bucket < BUCKETS:
            raise ValueError(f"bucket outside 0..7: {bucket}")
        grouped[block_key(row, mode)][bucket].add(
            Counts(
                searched=int(row["searched"]),
                tt_best=int(row["tt_best"]),
                updates=int(row["updates"]),
                saturated=int(row["saturated"]),
            )
        )
    return mode, list(grouped.values())


def aggregate(blocks: list[list[Counts]], indices: list[int] | None = None) -> list[Counts]:
    total = [Counts() for _ in range(BUCKETS)]
    source = range(len(blocks)) if indices is None else indices
    for index in source:
        for bucket in range(BUCKETS):
            total[bucket].add(blocks[index][bucket])
    return total


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    fraction = position - lower
    if lower + 1 == len(ordered):
        return ordered[lower]
    return ordered[lower] * (1.0 - fraction) + ordered[lower + 1] * fraction


def analyze(
    mode: str,
    blocks: list[list[Counts]],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    total = aggregate(blocks)
    block_presence = [
        sum(block[bucket].searched > 0 for block in blocks)
        for bucket in range(BUCKETS)
    ]
    minimum_blocks = 1_000 if mode == "standalone" else 200
    reasons: list[str] = []
    if len(blocks) < minimum_blocks:
        reasons.append(f"need {minimum_blocks} blocks, found {len(blocks)}")
    for bucket in range(BUCKETS):
        if block_presence[bucket] < 50:
            reasons.append(
                f"bucket {bucket} appears in {block_presence[bucket]} blocks; need 50"
            )
        if total[bucket].searched < 10_000:
            reasons.append(
                f"bucket {bucket} has {total[bucket].searched} searched; need 10000"
            )

    point_rates = [
        count.tt_best / count.searched if count.searched else None
        for count in total
    ]
    result: dict[str, object] = {
        "status": "inconclusive" if reasons else "pending",
        "mode": mode,
        "blocks": len(blocks),
        "replicates": replicates,
        "seed": seed,
        "bucket_block_presence": block_presence,
        "bucket_searched": [count.searched for count in total],
        "point_tt_best_rates": point_rates,
        "inconclusive_reasons": reasons,
    }
    if reasons:
        return result

    rng = random.Random(seed)
    monotonic = 0
    spreads: list[float] = []
    saturation_rates: list[float] = []
    block_count = len(blocks)
    for _ in range(replicates):
        sample = [rng.randrange(block_count) for _ in range(block_count)]
        sampled = aggregate(blocks, sample)
        rates = [count.tt_best / count.searched for count in sampled]
        monotonic += all(rates[i] <= rates[i + 1] for i in range(BUCKETS - 1))
        spreads.append(rates[-1] - rates[0])
        update_count = sum(count.updates for count in sampled)
        saturated = sum(count.saturated for count in sampled)
        saturation_rates.append(saturated / update_count if update_count else 0.0)

    monotonic_fraction = monotonic / replicates
    spread_lower = percentile(spreads, 0.025)
    saturation_upper = percentile(saturation_rates, 0.975)
    passed = (
        monotonic_fraction >= 0.95
        and spread_lower > 0.0
        and saturation_upper < 0.01
    )
    result.update(
        status="pass" if passed else "fail",
        monotonic_replicate_fraction=monotonic_fraction,
        spread_95_lower=spread_lower,
        saturation_95_upper=saturation_upper,
        criteria={
            "monotonic_at_least_0.95": monotonic_fraction >= 0.95,
            "spread_lower_above_0": spread_lower > 0.0,
            "saturation_upper_below_0.01": saturation_upper < 0.01,
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, nargs="?")
    parser.add_argument(
        "--mode", choices=("auto", "pair", "game", "standalone"), default="auto"
    )
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--seed", type=int, default=0x53463138)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.replicates <= 0:
        parser.error("--replicates must be positive")
    if args.self_test:
        blocks = []
        for _ in range(200):
            blocks.append([
                Counts(searched=100, tt_best=50 + 5 * bucket, updates=100)
                for bucket in range(BUCKETS)
            ])
        result = analyze("game", blocks, min(args.replicates, 100), args.seed)
        if result["status"] != "pass":
            print(json.dumps(result, indent=2), file=sys.stderr)
            return 1
        print("tt trust block bootstrap self-test ok")
        return 0
    if args.csv is None:
        parser.error("csv is required unless --self-test is used")
    try:
        mode, blocks = load_blocks(args.csv, args.mode)
        result = analyze(mode, blocks, args.replicates, args.seed)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "error", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"pass", "inconclusive"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
