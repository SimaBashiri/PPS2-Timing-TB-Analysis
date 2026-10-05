"""Build globally exclusive multi-channel hit groups and summarize timing resolution.

Run the regular timing analysis first; this script reuses its timing_summary.json
for channel selection, CFD settings, pair widths, and delay corrections.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import nnls

from decode_parquet_awkward import waveform_features
from pulse_measurements import sample_period_ns, threshold_tag


def collect_corrected_hits(parquet_path, summary, step_size=100_000):
    """Recompute valid CFD hit times and subtract saved channel delays."""
    settings = summary["settings"]
    channels = sorted(int(value) for value in summary["valid_hits_per_channel"])
    delays = summary["calibration"]["delays_ns"]
    uncalibrated = [channel for channel in channels
                    if delays.get(str(channel)) is None]
    if len(channels) < 2:
        raise ValueError("At least two selected channels need fitted delays.")

    baseline_window = tuple(settings["baseline_window"])
    signal_window = (tuple(settings["signal_window"])
                     if settings.get("signal_window") is not None else None)
    polarity = settings["polarity"]
    fraction = float(settings["cfd_percent"])
    cfd_key = f"LeadingTime_{threshold_tag(fraction, 'pct')}"
    min_amplitude = float(settings.get("min_amplitude_V", 0))
    min_snr = float(settings.get("min_snr", 0))
    period_override = settings.get("sample_period_ns")

    times_by_channel = {channel: [] for channel in channels}
    with pq.ParquetFile(parquet_path) as source:
        names = source.schema_arrow.names
        period_ns = sample_period_ns(source.schema_arrow.metadata, period_override)
        precise = "FirstSampleTime_in_ps" in names
        time_column = "FirstSampleTime_in_ps" if precise else "OrderedCell0Time"
        required = {"Channel", time_column, "DataSize", "DataSample", "Baseline"}
        missing = required - set(names)
        if missing:
            raise ValueError(f"Missing timing columns: {sorted(missing)}")
        fine_column = "FirstSampleTime_in_ps_fine"
        columns = sorted(required | ({fine_column} if precise and fine_column in names else set()))
        origin = None
        for batch in source.iter_batches(columns=columns, batch_size=step_size):
            channel_values = batch.column("Channel").to_numpy(zero_copy_only=False)
            stored = batch.column(time_column).to_numpy(zero_copy_only=False)
            if precise and not np.issubdtype(stored.dtype, np.integer):
                raise ValueError("FirstSampleTime_in_ps must contain integer picoseconds.")
            raw = stored.astype(np.longdouble)
            if origin is None and np.any(np.isfinite(raw)):
                origin = raw[np.flatnonzero(np.isfinite(raw))[0]]
            raw -= origin if origin is not None else 0
            if precise:
                if fine_column in columns:
                    raw += batch.column(fine_column).to_numpy(
                        zero_copy_only=False).astype(np.longdouble)
                raw /= 1000
            features = waveform_features(
                batch, baseline_window, signal_window, polarity, period_ns, (fraction,))
            offsets = features[cfd_key]
            valid = np.isfinite(raw) & np.isfinite(offsets) & np.isfinite(channel_values)
            valid &= features["PulseAmplitude"] > min_amplitude
            if min_snr > 0:
                valid &= features["SNR"] >= min_snr
            for channel in channels:
                selected = valid & (channel_values == channel)
                if np.any(selected):
                    corrected = raw[selected] + offsets[selected].astype(np.longdouble)
                    delay = delays.get(str(channel))
                    if delay is not None:
                        corrected -= np.longdouble(delay)
                    times_by_channel[channel].append(corrected)

    result = {}
    for channel, chunks in times_by_channel.items():
        values = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.longdouble)
        result[channel] = np.sort(values)
    return result, uncalibrated


def build_exclusive_events(times_by_channel, window_ns):
    """Seed each event with the earliest unused hit; use each hit once globally."""
    channels = sorted(times_by_channel)
    sorted_times = {channel: np.asarray(times_by_channel[channel], dtype=np.longdouble)
                    for channel in channels}
    used = {channel: np.zeros(len(sorted_times[channel]), dtype=bool) for channel in channels}
    cursors = {channel: 0 for channel in channels}
    events = []

    while True:
        seeds = []
        for channel, values in sorted_times.items():
            while cursors[channel] < len(values) and used[channel][cursors[channel]]:
                cursors[channel] += 1
            if cursors[channel] < len(values):
                index = cursors[channel]
                seeds.append((values[index], channel, index))
        if not seeds:
            break
        _, anchor_channel, anchor_index = min(seeds)
        anchor_time = sorted_times[anchor_channel][anchor_index]
        members = {anchor_channel: (anchor_index, anchor_time)}
        used[anchor_channel][anchor_index] = True

        for channel in channels:
            if channel == anchor_channel:
                continue
            values = sorted_times[channel]
            lo = np.searchsorted(values, anchor_time - window_ns, side="left")
            hi = np.searchsorted(values, anchor_time + window_ns, side="right")
            candidates = np.arange(lo, hi, dtype=int)
            candidates = candidates[~used[channel][candidates]]
            if len(candidates):
                nearest = candidates[np.argmin(np.abs(values[candidates] - anchor_time))]
                members[channel] = (int(nearest), values[nearest])
                used[channel][nearest] = True

        if len(members) >= 2:
            member_times = np.array([float(value[1]) for value in members.values()])
            center = float(np.median(member_times))
            residuals = member_times - center
            events.append({
                "anchor_channel": anchor_channel,
                "anchor_time_ns": float(anchor_time),
                "hits": {str(channel): float(value[1])
                         for channel, value in members.items()},
                "n_channels": len(members),
                "span_ns": float(member_times.max() - member_times.min()),
                "rms_from_median_ns": float(np.sqrt(np.mean(residuals**2))),
            })

    events.sort(key=lambda event: (event["rms_from_median_ns"], event["span_ns"]))
    for index, event in enumerate(events):
        event["event_id"] = index
    matched_hits = sum(event["n_channels"] for event in events)
    total_hits = sum(len(values) for values in sorted_times.values())
    return events, {
        "total_hits": total_hits,
        "matched_hits": matched_hits,
        "unmatched_hits": total_hits - matched_hits,
        "event_count": len(events),
        "matching": (
            "Global greedy event grouping: earliest unused hit seeds an event; "
            "nearest unused hit within the window is selected from each other channel. "
            "Each hit is consumed at most once across all events."
        ),
    }


def rank_pair_resolutions(summary):
    ranked = []
    for pair in summary["pairs"]:
        fit = pair.get("fit")
        if fit is None:
            continue
        ranked.append({
            "first": pair["first"],
            "second": pair["second"],
            "sigma_ps": float(pair["pair_resolution_ps"]),
            "error_ps": float(pair["pair_resolution_error_ps"]),
            "mean_ns": float(fit["mean_ns"]),
            "entries": int(pair["entries"]),
        })
    return sorted(ranked, key=lambda item: item["sigma_ps"])


def estimate_single_channel_resolutions(pair_results, channels):
    """Solve sigma_ij^2 = sigma_i^2 + sigma_j^2 with nonnegative variances."""
    usable = [item for item in pair_results
              if item["first"] in channels and item["second"] in channels]
    design = np.zeros((len(usable), len(channels)), dtype=float)
    variances = np.zeros(len(usable), dtype=float)
    for row, pair in enumerate(usable):
        design[row, channels.index(pair["first"])] = 1
        design[row, channels.index(pair["second"])] = 1
        variances[row] = pair["sigma_ps"] ** 2
    if len(usable) < len(channels) or np.linalg.matrix_rank(design) < len(channels):
        return {str(channel): None for channel in channels}, None, {
            "identifiable": False,
            "note": "Accepted pair widths do not constrain a unique per-channel solution."
        }
    solution, residual_norm = nnls(design, variances)
    sigmas = np.sqrt(solution)
    by_channel = {str(channel): float(value)
                  for channel, value in zip(channels, sigmas)}
    overall_rms = float(np.sqrt(np.mean(sigmas**2)))
    return by_channel, overall_rms, {
        "identifiable": True,
        "model": "pair_sigma_squared = sigma_first_squared + sigma_second_squared",
        "fit_residual_norm_ps_squared": float(residual_norm),
        "overall_rms_single_channel_ps": overall_rms,
    }


def save_outputs(output_dir, events, stats, pairs, single, overall, single_fit, channels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    event_path = output_dir / "all_channel_events.csv"
    with event_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            ["event_id", "anchor_channel", "anchor_time_ns", "n_channels",
             "span_ns", "rms_from_median_ns"] + [f"ch{channel}_time_ns" for channel in channels]))
        writer.writeheader()
        for event in events:
            row = {key: event[key] for key in writer.fieldnames if key in event}
            for channel in channels:
                row[f"ch{channel}_time_ns"] = event["hits"].get(str(channel), "")
            writer.writerow(row)

    if events:
        matrix = np.full((len(events), len(channels)), np.nan)
        for row, event in enumerate(events):
            available = [float(value) for value in event["hits"].values()]
            center = float(np.median(available))
            for column, channel in enumerate(channels):
                if str(channel) in event["hits"]:
                    matrix[row, column] = event["hits"][str(channel)] - center
        fig, ax = plt.subplots(figsize=(max(8, len(channels) * 0.75), 7))
        image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="coolwarm",
                          vmin=-max(0.05, np.nanpercentile(abs(matrix), 98)),
                          vmax=max(0.05, np.nanpercentile(abs(matrix), 98)))
        fig.colorbar(image, ax=ax, label="Hit time − event median (ns)")
        ax.set(xticks=range(len(channels)), xticklabels=channels,
               xlabel="Channel", ylabel="Events sorted by timing spread",
               title="Globally exclusive multi-channel hit groups")
        fig.tight_layout()
        fig.savefig(output_dir / "all_channel_event_timing.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist([event["span_ns"] * 1000 for event in events], bins="auto",
                color="tab:blue", alpha=0.8)
        ax.set(xlabel="Within-event time span (ps)", ylabel="Events",
               title="Spread of exclusive multi-channel hit groups")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "all_channel_event_spread.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 0.35), 5))
    if pairs:
        x = np.arange(len(pairs))
        ax.errorbar(x, [item["sigma_ps"] for item in pairs],
                    yerr=[item["error_ps"] for item in pairs],
                    fmt="o", capsize=3, color="tab:blue")
        ax.set(xticks=x,
               xticklabels=[f'{item["second"]}−{item["first"]}' for item in pairs])
        ax.tick_params(axis="x", labelrotation=90)
    ax.set(xlabel="Channel pair, ranked best to worst",
           ylabel="Pair timing resolution σ (ps)",
           title="Ranked channel-pair timing resolutions")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ranked_pair_timing_resolution.png", dpi=160)
    plt.close(fig)

    values = [single[str(channel)] for channel in channels]
    fig, ax = plt.subplots(figsize=(max(7, len(channels) * 0.7), 5))
    valid = [(channel, value) for channel, value in zip(channels, values)
             if value is not None]
    if valid:
        ax.bar([str(item[0]) for item in valid], [item[1] for item in valid],
               color="tab:green", alpha=0.8)
        if overall is not None:
            ax.axhline(overall, color="tab:red", linestyle="--",
                       label=f"All-channel RMS = {overall:.1f} ps")
            ax.legend()
    ax.set(xlabel="Channel", ylabel="Estimated single-channel σ (ps)",
           title="Single-channel resolutions inferred from pair widths")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "single_channel_timing_resolution.png", dpi=160)
    plt.close(fig)

    result = dict(event_matching=stats, channel_pair_resolution_ranked=pairs,
                  single_channel_resolution_ps=single,
                  all_channel_rms_single_resolution_ps=overall,
                  single_resolution_fit=single_fit,
                  events=events)
    (output_dir / "all_channel_timing_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, help="Decoded run Parquet file")
    parser.add_argument("timing_summary", type=Path,
                        help="timing_summary.json from the regular timing analysis")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory (default: beside timing_summary.json)")
    parser.add_argument("--step-size", type=int, default=100_000)
    parser.add_argument("--event-window-ns", type=float,
                        help="Override the coincidence window saved in timing_summary.json")
    args = parser.parse_args()
    if args.step_size <= 0:
        parser.error("--step-size must be positive")
    if not args.parquet.is_file() or not args.timing_summary.is_file():
        parser.error("Both the Parquet input and timing summary must exist.")

    summary = json.loads(args.timing_summary.read_text())
    window = (float(summary["settings"]["window_ns"])
              if args.event_window_ns is None else args.event_window_ns)
    if not np.isfinite(window) or window <= 0:
        parser.error("--event-window-ns must be finite and positive")
    channels = sorted(int(value) for value in summary["valid_hits_per_channel"])
    times, uncalibrated = collect_corrected_hits(args.parquet, summary, args.step_size)
    events, stats = build_exclusive_events(times, window)
    stats["channels_without_fitted_delay"] = uncalibrated
    pairs = rank_pair_resolutions(summary)
    single, overall, single_fit = estimate_single_channel_resolutions(pairs, channels)
    output_dir = args.output_dir or args.timing_summary.parent / "all_channel"
    save_outputs(output_dir, events, stats, pairs, single, overall, single_fit, channels)
    print(f"Matched {stats['matched_hits']}/{stats['total_hits']} hits into "
          f"{stats['event_count']} events; outputs: {output_dir}")


if __name__ == "__main__":
    main()
