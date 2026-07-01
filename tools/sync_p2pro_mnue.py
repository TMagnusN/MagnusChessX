#!/usr/bin/env python3
"""
Convert a Bullet P2Pro quantised.bin into a MagnusChessX Thinking .MNUE file.

Typical use:
    python tools/sync_p2pro_mnue.py ^
        D:/NNUE/runs/mnue_p2pro_40b_replay3b/mnue_p2pro_40b_replay3b-10/quantised.bin

This writes:
    src/build/<checkpoint-directory>.MNUE

By default the engine's embedded network stays unchanged. Add --set-embedded
only when the P2Pro file should become the compile-time default.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


MNUE_MAGIC = 0x4555_4E4D  # file bytes: "MNUE"
MNUE_VERSION = 1
MNUE_HEADER_BYTES = 40


@dataclass(frozen=True)
class P2ProLayout:
    arch_name: str = "P2Pro"
    arch_id: int = 2
    input_size: int = 10_240
    hidden_size: int = 1_408
    input_buckets: int = 16
    output_buckets: int = 32
    scale: int = 400
    qa: int = 255
    qb: int = 64

    @property
    def payload_bytes(self) -> int:
        values = (
            self.input_size * self.hidden_size
            + self.hidden_size
            + self.output_buckets * 2 * self.hidden_size
            + self.output_buckets
        )
        return values * 2

    @property
    def file_bytes(self) -> int:
        return MNUE_HEADER_BYTES + self.payload_bytes

    @property
    def display_mib(self) -> str:
        return f"{self.file_bytes / (1024.0 * 1024.0):.1f}"

    @property
    def shape_text(self) -> str:
        return (
            f"(1,{self.output_buckets},{self.input_buckets},"
            f"{self.hidden_size},{self.input_size})"
        )

    def header(self) -> bytes:
        return struct.pack(
            "<7I3i",
            MNUE_MAGIC,
            MNUE_VERSION,
            self.arch_id,
            self.input_size,
            self.hidden_size,
            self.input_buckets,
            self.output_buckets,
            self.scale,
            self.qa,
            self.qb,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Bullet P2Pro quantised.bin to headered .MNUE. "
            "The embedded default remains unchanged unless --set-embedded is used."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to Bullet quantised.bin, or an already-headered P2Pro .MNUE.",
    )
    parser.add_argument(
        "--name",
        help=(
            "Output .MNUE filename. Defaults to '<checkpoint-dir>.MNUE' when "
            "input is quantised.bin, otherwise '<input-stem>.MNUE'."
        ),
    )
    parser.add_argument(
        "--engine-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="MagnusChessX Thinking repository root. Default: this script's repository.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .MNUE path. Default: <engine-root>/src/build/<name>.",
    )
    parser.add_argument(
        "--hidden-size",
        type=positive_int,
        default=1_408,
        help="P2Pro hidden size to write into the header and engine layout.",
    )
    parser.add_argument(
        "--set-embedded",
        action="store_true",
        help=(
            "Also update Makefile, fallback macro, and README so this P2Pro "
            "file becomes the embedded default. Omit to keep the default P2."
        ),
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Only write the .MNUE file; do not update code/docs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print planned writes without changing files.",
    )
    return parser.parse_args()


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected integer, got {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected positive integer, got {value}")
    return value


def default_name(source: Path) -> str:
    if source.name.lower() == "quantised.bin":
        return f"{source.parent.name}.MNUE"
    return f"{source.stem}.MNUE"


def read_payload(source: Path, layout: P2ProLayout) -> bytes:
    data = source.read_bytes()
    if len(data) == layout.payload_bytes:
        return data

    if len(data) == layout.file_bytes and data[:4] == struct.pack("<I", MNUE_MAGIC):
        validate_existing_header(source, data[:MNUE_HEADER_BYTES], layout)
        return data[MNUE_HEADER_BYTES:]

    raise SystemExit(
        "bad payload size for P2Pro MNUE\n"
        f"  input: {source}\n"
        f"  got: {len(data):,} bytes\n"
        f"  expected raw quantised.bin: {layout.payload_bytes:,} bytes\n"
        f"  expected headered .MNUE: {layout.file_bytes:,} bytes"
    )


def validate_existing_header(source: Path, header: bytes, layout: P2ProLayout) -> None:
    fields = struct.unpack("<7I3i", header)
    expected = (
        MNUE_MAGIC,
        MNUE_VERSION,
        layout.arch_id,
        layout.input_size,
        layout.hidden_size,
        layout.input_buckets,
        layout.output_buckets,
        layout.scale,
        layout.qa,
        layout.qb,
    )
    if fields != expected:
        raise SystemExit(
            f"{source} has an MNUE header, but it does not match the requested P2Pro layout\n"
            f"  got:      {fields}\n"
            f"  expected: {expected}"
        )


def write_mnue(output: Path, payload: bytes, layout: P2ProLayout, dry_run: bool) -> None:
    if dry_run:
        print(f"would write {output} ({layout.file_bytes:,} bytes)")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f"{output.name}.tmp")
    tmp.write_bytes(layout.header() + payload)
    tmp.replace(output)
    print(f"wrote {output} ({output.stat().st_size:,} bytes)")


def sync_text_files(
    root: Path,
    mnue_name: str,
    layout: P2ProLayout,
    dry_run: bool,
    set_embedded: bool,
) -> None:
    replacements = {
        root / "src" / "mnue" / "Mnue.h": [
            (
                r"(?m)^(// MNUE-P2Pro:.*\n)//   1 x \d+ x \d+ x \d+ x \d+$",
                (
                    rf"\g<1>//   1 x {layout.output_buckets} x "
                    rf"{layout.input_buckets} x {layout.hidden_size} x {layout.input_size}"
                ),
                "P2Pro layout comment",
            ),
            (
                r"(?s)(struct P2ProLayout \{.*?static constexpr int HiddenSize = )\d+(;)",
                rf"\g<1>{layout.hidden_size}\g<2>",
                "P2Pro hidden size",
            ),
        ],
    }

    if set_embedded:
        replacements[root / "src" / "Makefile"] = [
            (
                r"(?m)^MNUE_EMBEDDED_FILE := .+$",
                f"MNUE_EMBEDDED_FILE := {mnue_name}",
                "embedded filename",
            ),
            (
                r"(?m)^# the (?:[0-9.]+ MiB|[0-9]+ MB) incbin isolated from instrumentation and whole-program analysis\.$",
                f"# the {layout.display_mib} MiB incbin isolated from instrumentation and whole-program analysis.",
                "embedded object size comment",
            ),
            (
                r"(?m)^# The incbin source is a raw (?:[0-9.]+ MiB|[0-9]+ MB) MNUE file\. We compile it as a standalone$",
                f"# The incbin source is a raw {layout.display_mib} MiB MNUE file. We compile it as a standalone",
                "incbin size comment",
            ),
        ]
        replacements[root / "src" / "mnue" / "Mnue.h"].append(
            (
                r'(?m)^#define MNUE_EMBEDDED_FILENAME ".+"$',
                f'#define MNUE_EMBEDDED_FILENAME "{mnue_name}"',
                "fallback embedded filename",
            )
        )
        replacements[root / "README.MD"] = [
            (
                r"The standard build embeds a default MNUE P2(?:Pro)? network\.",
                "The standard build embeds a default MNUE P2Pro network.",
                "README embedded net family text",
            ),
            (
                r"(?m)^MNUE_EMBEDDED_FILE := .+\.MNUE$",
                f"MNUE_EMBEDDED_FILE := {mnue_name}",
                "README embedded filename snippet",
            ),
            (
                r"(?m)^\| MNUEfile \| string \| `[^`]+` \| External MNUE file path or `<embedded>`\. \|$",
                f"| MNUEfile | string | `{mnue_name}` | External MNUE file path or `<embedded>`. |",
                "README UCI option default",
            ),
        ]

    for path, path_replacements in replacements.items():
        replace_in_file(path, path_replacements, dry_run)


def replace_in_file(path: Path, replacements: list[tuple[str, str, str]], dry_run: bool) -> None:
    text = path.read_text(encoding="utf-8")
    updated = text

    for pattern, replacement, label in replacements:
        updated, count = re.subn(pattern, replacement, updated, count=1)
        if count != 1:
            raise SystemExit(f"could not update {label} in {path}")

    if updated == text:
        print(f"unchanged {path}")
        return

    if dry_run:
        print(f"would update {path}")
        return

    path.write_text(updated, encoding="utf-8", newline="\n")
    print(f"updated {path}")


def main() -> int:
    args = parse_args()
    root = args.engine_root.resolve()
    source = args.input.resolve()
    if not source.is_file():
        raise SystemExit(f"input file not found: {source}")

    mnue_name = args.name or default_name(source)
    if not mnue_name.lower().endswith(".mnue"):
        mnue_name += ".MNUE"

    layout = P2ProLayout(hidden_size=args.hidden_size)
    payload = read_payload(source, layout)
    output = (args.output or (root / "src" / "build" / mnue_name)).resolve()

    write_mnue(output, payload, layout, args.dry_run)
    if not args.no_sync:
        sync_text_files(root, output.name, layout, args.dry_run, args.set_embedded)

    sync_mode = (
        "embedded default updated"
        if args.set_embedded and not args.no_sync
        else "embedded default unchanged"
    )
    print(
        "P2Pro MNUE sync complete: "
        f"{output.name}, {layout.file_bytes:,} bytes, "
        f"{layout.display_mib} MiB, {layout.shape_text}, {sync_mode}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
