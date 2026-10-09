"""Show the structure of one decoded hit from a SAMPIC binary file."""

import argparse
from pathlib import Path

from sampiclyser import SAMPIC_Run_Decoder


DEFAULT_FILE = Path(
    "/GPT6/shared1/sbashiri/TestBeam/2026-08/SAMPIC/"
    "sampic_20260805_013030_run1/sampic_20260805_013030_run1.bin"
)


def describe(value):
    """Summarize arrays instead of dumping a whole waveform."""
    if isinstance(value, (list, tuple)):
        preview = list(value[:5])
        suffix = ", ..." if len(value) > 5 else ""
        return f"{type(value).__name__}(length={len(value)}, first_values={preview}{suffix})"
    return repr(value)


def main():
    parser = argparse.ArgumentParser(
        description="Print file information and the structure of its first SAMPIC hit."
    )
    parser.add_argument("bin_file", nargs="?", type=Path, default=DEFAULT_FILE)
    args = parser.parse_args()

    path = args.bin_file
    if not path.is_file():
        parser.error(f"file does not exist: {path}")

    print(f"File: {path}")
    print(f"Size: {path.stat().st_size:,} bytes")

    decoder = SAMPIC_Run_Decoder(path.parent)
    decoder.run_files = [path]
    hit = next(decoder.parse_hit_records(limit_hits=1), None)
    if hit is None:
        print("No hit records found.")
        return

    print("\nFirst hit record structure:")
    for name, value in hit.items():
        print(f"  {name}: {describe(value)}")


if __name__ == "__main__":
    main()
