#!/usr/bin/env python3
"""Regenerate the tables at the bottom of json_escape.simd.

Everything above the marker line is hand-written; everything below is produced
here so the tables stay in sync with their definitions.
"""
import pathlib

MARKER = "-- === GENERATED TABLES: edit gen_tables.py, not below this line ==="
PATH = pathlib.Path(__file__).resolve().parent / "json_escape.simd"

SHORT = {8: "b", 9: "t", 10: "n", 12: "f", 13: "r", 34: '"', 92: "\\"}


def expand_entry(mask):
    """tbl indices taking 8 input lanes to their expansion. Table lane 8 holds '\\'."""
    out = []
    for lane in range(8):
        if mask >> lane & 1:
            out.append(8)
        out.append(lane)
    return out + [255] * (16 - len(out))


def escape_text(c):
    """What byte c becomes in the output."""
    if c in SHORT:
        return f"\\{SHORT[c]}".encode()
    if c < 32:
        return f"\\u{c:04x}".encode()
    return bytes([c])


def table(name, rows):
    body = ",\n".join("    " + ", ".join(str(v) for v in row) for row in rows)
    return f"const {name} = [u8:\n{body}\n]\n"


expand = table("EXPAND", [expand_entry(m) for m in range(256)])
counts = table("POPCOUNT8", [[bin(m).count("1") for m in range(i, i + 16)] for i in range(0, 256, 16)])
escapes = table("ESCAPE", [list(escape_text(c).ljust(8, b"\0")) for c in range(256)])
escape_lens = table("ESCAPE_LEN", [[len(escape_text(c)) for c in range(i, i + 16)] for i in range(0, 256, 16)])
lanes = table("LANE_MASK", [[255] * 16, [0] * 16])

head = PATH.read_text().split(MARKER)[0]
PATH.write_text(f"{head}{MARKER}\n\n{expand}\n{counts}\n{escapes}\n{escape_lens}\n{lanes}")
