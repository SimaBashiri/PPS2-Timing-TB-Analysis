"""Print the column names and a small sample of a Parquet file."""

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def preview(value, max_chars=120):
    """Keep long waveform arrays readable in the printed row preview."""
    text = repr(value)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Display Parquet column names and the first few rows."
    )
    parser.add_argument("parquet", type=Path, help="Input .parquet or .pq file")
    parser.add_argument(
        "-n", "--rows", type=int, default=5,
        help="Number of rows to display (default: 5)",
    )
    args = parser.parse_args()

    if args.rows < 0:
        parser.error("--rows must be zero or greater")
    if not args.parquet.is_file():
        parser.error(f"file does not exist: {args.parquet}")

    parquet = pq.ParquetFile(args.parquet)
    columns = parquet.schema_arrow.names
    print(f"File: {args.parquet}")
    print(f"Rows: {parquet.metadata.num_rows:,}")
    print("Columns:")
    for name in columns:
        print(f"  {name}")

    print(f"\nFirst {min(args.rows, parquet.metadata.num_rows)} row(s):")
    if args.rows == 0:
        return
    batch = next(parquet.iter_batches(batch_size=args.rows))
    for row_number, row in enumerate(batch.to_pylist(), start=1):
        print(f"Row {row_number}:")
        for name in columns:
            print(f"  {name}: {preview(row[name])}")


if __name__ == "__main__":
    main()
