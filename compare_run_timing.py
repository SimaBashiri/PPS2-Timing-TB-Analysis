#!/usr/bin/env python3
"""Compare per-run CFD timing summaries across test-beam runs.

Reads ``timing/timing_summary.json`` files produced by timing_analysis.py.
Optionally accepts a CSV with columns ``run,angle_deg`` to use detector angle
on the x axis. Without it, plots use run number and write an angle template.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run_number(path):
    match = re.search(r"_run(\d+)$", path.parent.parent.name)
    return int(match.group(1)) if match else None


def read_angles(path):
    if not path:
        return {}
    with open(path, newline="", encoding="utf-8") as stream:
        result = {}
        for row in csv.DictReader(stream):
            if not row.get("angle_deg", "").strip():
                continue
            key = row.get("run_id", "").strip()
            if key:
                result[key] = float(row["angle_deg"])
            elif row.get("run", "").strip():
                result[int(row["run"])] = float(row["angle_deg"])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plots_dir", type=Path, help="Parent directory containing run folders")
    parser.add_argument("--angles-csv", type=Path, help="CSV mapping run,angle_deg")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: <plots_dir>/run_comparison)")
    args = parser.parse_args()

    summaries = sorted(args.plots_dir.glob("*/timing/timing_summary.json"),
                       key=lambda p: run_number(p) or 0)
    if not summaries:
        parser.error(f"No */timing/timing_summary.json files found under {args.plots_dir}")
    output = args.output_dir or args.plots_dir / "run_comparison"
    output.mkdir(parents=True, exist_ok=True)
    angles = read_angles(args.angles_csv)

    rows = []
    pairs = set()
    channels = set()
    for path in summaries:
        run = run_number(path)
        if run is None:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        accepted = [p for p in data["pairs"] if p.get("pair_resolution_ps") is not None]
        pair_sigmas = [p["pair_resolution_ps"] for p in accepted]
        for pair in data["pairs"]:
            pairs.add((pair["first"], pair["second"]))
        delays = data.get("calibration", {}).get("delays_ns", {})
        errors = data.get("calibration", {}).get("errors_ns", {})
        channels.update(int(ch) for ch in delays)
        run_id = path.parent.parent.name
        rows.append({"run": run, "run_id": run_id,
                     "angle_deg": angles.get(run_id, angles.get(run)),
                     "summary": data, "median_pair_resolution_ps":
                     float(np.median(pair_sigmas)) if pair_sigmas else None,
                     "accepted_pair_fits": len(accepted)})

    use_angle = bool(args.angles_csv)
    xkey = "angle_deg" if use_angle else "run"
    xlabel = "Detector angle (degrees)" if use_angle else "Run number"
    points = [row for row in rows if row[xkey] is not None]
    points.sort(key=lambda row: row[xkey])

    with (output / "run_timing_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run_id", "run", "angle_deg", "accepted_pair_fits", "median_pair_resolution_ps",
                         "pair", "pair_resolution_ps", "pair_resolution_error_ps",
                         "channel", "delay_ns", "delay_error_ns"])
        for row in rows:
            d = row["summary"]
            accepted = [p for p in d["pairs"] if p.get("pair_resolution_ps") is not None]
            delay_map = d.get("calibration", {}).get("delays_ns", {})
            err_map = d.get("calibration", {}).get("errors_ns", {})
            if accepted:
                for p in accepted:
                    writer.writerow([row["run_id"], row["run"], row["angle_deg"], row["accepted_pair_fits"],
                                     row["median_pair_resolution_ps"],
                                     f"{p['first']}-{p['second']}", p["pair_resolution_ps"],
                                     p.get("pair_resolution_error_ps"), "", "", ""])
            for ch, value in delay_map.items():
                writer.writerow([row["run_id"], row["run"], row["angle_deg"], row["accepted_pair_fits"],
                                 row["median_pair_resolution_ps"], "", "", "", ch, value,
                                 err_map.get(ch)])

    if not args.angles_csv:
        with (output / "run_angle_template.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["run_id", "run", "angle_deg"])
            writer.writerows((r["run_id"], r["run"], "") for r in rows)

    # Pair-wise resolution summary: color encodes fitted pair sigma; missing
    # fits stay blank so they cannot be mistaken for zero resolution.
    pair_list = sorted(pairs)
    matrix = np.full((len(pair_list), len(points)), np.nan)
    for col, row in enumerate(points):
        values = {(p["first"], p["second"]): p.get("pair_resolution_ps")
                  for p in row["summary"]["pairs"]}
        for index, pair in enumerate(pair_list):
            if values.get(pair) is not None:
                matrix[index, col] = values[pair]
    fig, ax = plt.subplots(figsize=(max(12, len(points) * .24), max(7, len(pair_list) * .20)))
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", interpolation="nearest",
                       cmap="viridis", origin="lower")
    ax.set(xticks=range(len(points)), xticklabels=[f"{r[xkey]:g}" for r in points],
           yticks=range(len(pair_list)), yticklabels=[f"{a}-{b}" for a, b in pair_list],
           xlabel=xlabel, ylabel="Channel pair", title="Pair timing resolution by run")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.colorbar(image, ax=ax, label="Pair resolution σ (ps)")
    fig.tight_layout()
    fig.savefig(output / "pair_resolution_by_run.png", dpi=180)
    plt.close(fig)

    # Show individual channel delays against the same run/angle coordinate.
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.get_cmap("tab10")
    for index, channel in enumerate(sorted(channels)):
        xs, ys, es = [], [], []
        for row in points:
            delay = row["summary"].get("calibration", {}).get("delays_ns", {}).get(str(channel))
            if delay is not None:
                xs.append(row[xkey]); ys.append(delay)
                err = row["summary"].get("calibration", {}).get("errors_ns", {}).get(str(channel))
                es.append(err or 0)
        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt="o-", ms=3, lw=1, capsize=2,
                        color=colors(index % 10), label=f"Ch {channel}")
    ax.axhline(0, color="0.5", lw=.8)
    ax.set(xlabel=xlabel, ylabel="Relative channel delay (ns)",
           title="Fitted channel delays by run")
    ax.grid(alpha=.25)
    if channels:
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "channel_delay_by_run.png", dpi=180)
    plt.close(fig)

    # A compact run-level trend; this median is descriptive and can combine
    # different pairs from run to run, so the pair heatmap remains primary.
    fig, ax = plt.subplots(figsize=(12, 5))
    valid = [r for r in points if r["median_pair_resolution_ps"] is not None]
    ax.plot([r[xkey] for r in valid], [r["median_pair_resolution_ps"] for r in valid], "o")
    ax.set(xlabel=xlabel, ylabel="Median accepted pair σ (ps)",
           title="Median accepted pair timing resolution by run")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "median_pair_resolution_by_run.png", dpi=180)
    plt.close(fig)
    print(f"Read {len(rows)} timing summaries; {len(points)} plotted. Wrote outputs to {output}")
    if not args.angles_csv:
        print(f"Fill {output / 'run_angle_template.csv'} and rerun with --angles-csv to plot against angle.")


if __name__ == "__main__":
    main()
