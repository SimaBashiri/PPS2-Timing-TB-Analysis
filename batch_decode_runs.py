"""Run decode_parquet_awkward.py once for each SAMPIC capture directory.

Example:
    python batch_decode_runs.py -- \
        --baseline-window 1 20 --signal-window 22 50 --timing-only
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import subprocess
import sys


DEFAULT_SOURCE = Path("/GPT6/shared1/sbashiri/TestBeam/2026-08/SAMPIC")


def capture_directories(root: Path):
    """Yield directories containing hit binaries, excluding trigger-data files."""
    dirs = set()
    for binary in root.rglob("*.bin"):
        if binary.name.endswith("_trigger_data.bin"):
            continue
        dirs.add(binary.parent)
    return sorted(dirs, key=lambda path: path.relative_to(root).as_posix())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=Path("/GPT6/shared1/sbashiri/TestBeam/output/testbeam_2026-08"))
    parser.add_argument("--plots-root", type=Path, default=Path("/GPT6/shared1/sbashiri/TestBeam/output/testbeam_2026-08/plots/testbeam_2026-08"))
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Keep processing later runs after a failure")
    parser.add_argument("--max-parallel", type=int, default=20, metavar="N",
                        help="Maximum number of decoder processes to run at once (default: 20)")
    args, decoder_args = parser.parse_known_args()

    if args.max_parallel < 1:
        parser.error("--max-parallel must be at least 1")

    root = args.source_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Source root is not a directory: {root}")

    decoder = Path(__file__).with_name("decode_parquet_awkward.py").resolve()
    captures = capture_directories(root)
    if not captures:
        parser.error(f"No SAMPIC hit binaries found under {root}")

    extra = decoder_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    failed = []
    print(f"Found {len(captures)} capture directories under {root}", flush=True)
    jobs = []
    for index, capture in enumerate(captures, 1):
        # Relative paths keep archived captures such as run31 distinct.
        rel = capture.relative_to(root)
        parquet = args.output_root / rel.parent / f"{capture.name}.parquet"
        plots = args.plots_root / rel
        if parquet.is_file():
            # Reuse a completed decode when resuming after a later analysis error.
            source = parquet
            output_args = []
            print(f"  Reusing Parquet: {parquet}", flush=True)
        else:
            source = capture
            output_args = ["--output", str(parquet)]
        command = [sys.executable, str(decoder), str(source),
                   *output_args, "--plots-dir", str(plots), *extra]
        jobs.append((index, capture, command))

    next_job = 0
    stop_submitting = False
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        running = {}
        while next_job < len(jobs) or running:
            while (not stop_submitting and next_job < len(jobs)
                   and len(running) < args.max_parallel):
                index, capture, command = jobs[next_job]
                next_job += 1
                print(f"[{index}/{len(captures)}] {capture}", flush=True)
                future = executor.submit(subprocess.run, command, check=False)
                running[future] = capture

            if not running:
                break
            completed, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in completed:
                capture = running.pop(future)
                result = future.result()
                if result.returncode:
                    failed.append((capture, result.returncode))
                    print(f"FAILED ({result.returncode}): {capture}", flush=True)
                    if not args.continue_on_error:
                        stop_submitting = True

    if failed:
        print(f"\n{len(failed)} run(s) failed:", flush=True)
        for capture, code in failed:
            print(f"  exit {code}: {capture}", flush=True)
        return 1
    print("\nAll runs completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
