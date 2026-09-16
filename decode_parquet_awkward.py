"""Decode SAMPIC binaries, inspect Parquet contents, and save histograms using PyArrow and NumPy."""

import argparse
from pathlib import Path

import json

from pulse_measurements import (DEFAULT_THRESHOLDS, sample_period_ns, pulse_features,
                                histogram_mpv, feature_unit)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


DEFAULT_INPUT = Path(
    "./TestBeam/2025-05-14/SAMPIC_data/"
    "Run018_SAMPIC_5_17_2025_23h_13min_Binary"
)


def decode_to_parquet(source, output):
    from sampiclyser import SAMPIC_Run_Decoder

    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Pass that Parquet file as input to inspect it."
        )
    decoder = SAMPIC_Run_Decoder(source if source.is_dir() else source.parent)
    if source.is_dir():
        decoder.run_files = sorted(path for path in source.glob("*.bin*") if path.is_file())
    elif source.is_file():
        if source not in decoder.run_files:
            raise ValueError(f"Not a SAMPIC hit binary: {source}")
        decoder.run_files = [source]
    if not decoder.run_files:
        raise ValueError(f"No SAMPIC hit binaries found in {source}")
    for binary in decoder.run_files:
        print(f"Decoding: {binary}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    decoder.decode_data(parquet_path=output)
    if not output.is_file():
        raise RuntimeError("The decoder did not produce a Parquet file.")
    return output


def as_numpy(array):
    return array.to_numpy(zero_copy_only=False)


def waveform_features(batch, baseline_window=None, signal_window=None, polarity="positive",
                      period_ns=None, thresholds=DEFAULT_THRESHOLDS, voltage_thresholds=()):
    """Reduce finite samples within DataSize without loading the full run."""
    samples = batch.column("DataSample")
    sizes = as_numpy(batch.column("DataSize"))
    lengths = as_numpy(pc.fill_null(pc.list_value_length(samples), 0))
    present = as_numpy(samples.is_valid()) & np.isfinite(sizes)
    if np.any(present & ((sizes < 0) | (sizes > lengths))):
        raise ValueError("DataSize is outside the stored waveform length.")
    # flatten() excludes null lists, including those backed by nonempty buffers.
    values = as_numpy(samples.flatten()).astype(np.float64, copy=False)
    parents = np.repeat(np.arange(len(batch)), lengths)
    starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
    positions = np.arange(len(values)) - starts
    keep = np.isfinite(values) & present[parents] & (positions < sizes[parents])
    values, parents, positions = values[keep], parents[keep], positions[keep]
    counts = np.bincount(parents, minlength=len(batch))
    sums = np.bincount(parents, weights=values, minlength=len(batch))
    low = np.full(len(batch), np.inf)
    high = np.full(len(batch), -np.inf)
    np.minimum.at(low, parents, values)
    np.maximum.at(high, parents, values)
    valid = counts > 0
    mean = np.full(len(batch), np.nan)
    np.divide(sums, counts, out=mean, where=valid)
    baseline = as_numpy(batch.column("Baseline"))
    result = {
        "SampleCount": counts,
        "SampleMean": mean,
        "PeakToPeak": np.where(valid, high - low, np.nan),
        "MaxAbsBaselineDeviation": np.where(
            valid, np.maximum(abs(high - baseline), abs(low - baseline)), np.nan
        ),
    }
    if baseline_window is not None:
        start, stop = baseline_window
        quiet = (positions >= start) & (positions < stop)
        qparents, qvalues = parents[quiet], values[quiet]
        n = np.bincount(qparents, minlength=len(batch))
        total = np.bincount(qparents, weights=qvalues, minlength=len(batch))
        pedestal = np.full(len(batch), np.nan)
        # Require every sample in the requested baseline interval to be valid.
        complete = n == stop - start
        np.divide(total, n, out=pedestal, where=complete)
        residuals = qvalues - pedestal[qparents]
        sumsq = np.bincount(qparents, weights=residuals**2, minlength=len(batch))
        noise = np.full(len(batch), np.nan)
        np.divide(sumsq, n - 1, out=noise, where=complete & (n > 1))
        noise = np.sqrt(noise)
        signal = (positions >= stop) if signal_window is None else (
            (positions >= signal_window[0]) & (positions < signal_window[1])
        )
        signed = values - pedestal[parents]
        if polarity == "negative":
            signed = -signed
        signal &= np.isfinite(signed)
        amplitude = np.full(len(batch), -np.inf)
        np.maximum.at(amplitude, parents[signal], signed[signal])
        amplitude[~np.isfinite(amplitude)] = np.nan
        snr = np.full(len(batch), np.nan)
        np.divide(amplitude, noise, out=snr,
                  where=np.isfinite(amplitude) & (amplitude > 0) & (noise > 0))
        result.update({
            "BaselineEstimate": pedestal,
            "BaselineResidual": baseline - pedestal,
            "NoiseRMS": noise,
            "PulseAmplitude": np.where(amplitude > 0, amplitude, np.nan),
            "SNR": snr,
            "InvalidBaseline": (~complete).astype(int),
        })
        if period_ns is not None:
            result.update(pulse_features(
                parents[signal], positions[signal], signed[signal], amplitude, sizes,
                signal_window if signal_window is not None else (stop, None),
                period_ns, thresholds, voltage_thresholds))
    return result


CHANNEL_FIELDS = (
    "Baseline", "Amplitude", "BaselineEstimate", "BaselineResidual",
    "NoiseRMS", "PulseAmplitude", "SNR",
)


def batch_values(batch, scalar_fields, baseline_window=None, signal_window=None,
                 polarity="positive", period_ns=None, thresholds=DEFAULT_THRESHOLDS,
                 voltage_thresholds=()):

    values = {name: as_numpy(batch.column(name)) for name in scalar_fields}
    values.update(waveform_features(batch, baseline_window, signal_window, polarity,
                                   period_ns, thresholds, voltage_thresholds))
    return values


def inspect_and_analyze(parquet_path, inspect_only=False, step_size=100_000,
                        plots_dir=None, bins=100, baseline_window=None, signal_window=None,
                        polarity="positive", period_ns=None, thresholds=DEFAULT_THRESHOLDS,
                        voltage_thresholds=()):
    """Two bounded-memory passes: determine full ranges, then count histogram bins."""
    with pq.ParquetFile(parquet_path) as parquet_file:
        print(f"\nParquet file: {parquet_path}")
        print(f"Entries: {parquet_file.metadata.num_rows}; row groups: {parquet_file.num_row_groups}")
        print("\nSchema:")
        print(parquet_file.schema_arrow.remove_metadata())
        if inspect_only:
            return
        period_ns = sample_period_ns(parquet_file.schema_arrow.metadata, period_ns)
        print(f"Uniform sample period: {period_ns:g} ns; waveform voltage: V")
        if baseline_window is None:
            print("Derived pulse measurements disabled: choose --baseline-window START STOP after inspecting previews.")
        required = {"Channel", "DataSize", "DataSample", "Baseline"}
        missing = required - set(parquet_file.schema_arrow.names)
        if missing:
            raise ValueError(f"Waveform analysis requires missing fields: {sorted(missing)}")
        scalar_fields = [field.name for field in parquet_file.schema_arrow
                         if pa.types.is_integer(field.type) or pa.types.is_floating(field.type)]
        columns = scalar_fields + ["DataSample"]
        ranges, channel_counts, examples = {}, {}, {}
        hit_count = sample_count = waveform_count = 0
        peak_sum = 0.0
        print("Pass 1/2: measuring histogram ranges and waveform summaries", flush=True)
        for batch in parquet_file.iter_batches(columns=columns, batch_size=step_size):
            values = batch_values(batch, scalar_fields, baseline_window, signal_window, polarity,
                                  period_ns, thresholds, voltage_thresholds)
            for name, data in values.items():
                finite = data[np.isfinite(data)]
                if finite.size:
                    lo, hi = finite.min().item(), finite.max().item()
                    old = ranges.get(name, (lo, hi))
                    ranges[name] = (min(lo, old[0]), max(hi, old[1]))
            channel = values["Channel"]
            keys, counts = np.unique(channel[np.isfinite(channel)], return_counts=True)
            for key, count in zip(keys, counts):
                channel_counts[int(key)] = channel_counts.get(int(key), 0) + int(count)
            for key in keys:
                channel_id = int(key)
                saved = examples.setdefault(channel_id, [])
                for index in np.flatnonzero(channel == key)[:max(0, 3 - len(saved))]:
                    raw = batch.column("DataSample")[int(index)].as_py()
                    size = values["DataSize"][index]
                    if raw is not None and np.isfinite(size):
                        saved.append(np.asarray(raw[:int(size)], dtype=float))
            hit_count += len(batch)
            sample_count += int(values["SampleCount"].sum())
            peaks = values["PeakToPeak"]
            finite = peaks[np.isfinite(peaks)]
            waveform_count += finite.size
            peak_sum += float(finite.sum())
            print(f"\rScanned {hit_count:,} hits", end="", flush=True)
        print()
        # Shift integer fields before float conversion to preserve large timestamp differences.
        origins = {
            name: lo if isinstance(lo, int) and max(abs(lo), abs(hi)) > 2**53 else 0
            for name, (lo, hi) in ranges.items()
        }
        channel_fields = list(CHANNEL_FIELDS) + [
            name for name in ranges if name not in scalar_fields and name not in CHANNEL_FIELDS
        ]
        edges = {}
        for name, (lo, hi) in ranges.items():
            origin = origins[name]
            low, high = float(lo - origin), float(hi - origin)
            if name == "Channel":
                edges[name] = np.arange(low - 0.5, high + 1.5)
            else:
                if low == high:
                    width = max(abs(low) * 0.01, 1e-12 if feature_unit(name) else 0.5)
                    low, high = low - width, high + width
                edges[name] = np.linspace(low, high, bins + 1)
        histograms = {name: np.zeros(len(edge) - 1, dtype=np.int64)
                      for name, edge in edges.items()}
        channel_histograms = {
            (channel, name): np.zeros(len(edges[name]) - 1, dtype=np.int64)
            for channel in channel_counts for name in channel_fields if name in edges
        }
        print("Pass 2/2: filling histograms", flush=True)
        processed = 0
        for batch in parquet_file.iter_batches(columns=columns, batch_size=step_size):
            values = batch_values(batch, scalar_fields, baseline_window, signal_window, polarity,
                                  period_ns, thresholds, voltage_thresholds)
            for name, data in values.items():
                if name not in edges:
                    continue
                finite = data[np.isfinite(data)]
                shifted = (finite - origins[name]).astype(np.float64)
                histograms[name] += np.histogram(shifted, bins=edges[name])[0]
            for (channel, name), counts in channel_histograms.items():
                data = values[name][values["Channel"] == channel]
                finite = data[np.isfinite(data)]
                counts += np.histogram((finite - origins[name]).astype(float), edges[name])[0]
            processed += len(batch)
            print(f"\rHistogrammed {processed:,} hits", end="", flush=True)
        print()

    plots_dir = Path(plots_dir) if plots_dir else Path.cwd() / "plots" / parquet_path.stem
    plots_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, counts in histograms.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        origin = origins[name]
        ax.stairs(counts, edges[name], fill=True, alpha=0.75)
        unit = feature_unit(name)
        ax.set_xlabel((f"{name} - {origin}" if origin else name) + (f" ({unit})" if unit else ""))
        ax.set_ylabel("Hits / bin")
        ax.set_title(f"{name} ({int(counts.sum()):,} finite entries)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{name}.png", dpi=160)
        plt.close(fig)
        np.savetxt(plots_dir / f"{name}.csv",
                   np.column_stack((edges[name][:-1], edges[name][1:], counts)),
                   delimiter=",", header=f"left_edge,right_edge,count; x_origin={origin}; unit={feature_unit(name)}",
                   fmt=["%.17g", "%.17g", "%d"])
    for name in channel_fields:
        if name not in edges:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for channel in sorted(channel_counts):
            counts = channel_histograms[channel, name]
            total = int(counts.sum())
            if total:
                ax.stairs(counts / total, edges[name], label=f"Ch {channel} (n={total:,})")
            np.savetxt(plots_dir / f"{name}_channel_{channel}.csv",
                       np.column_stack((edges[name][:-1], edges[name][1:], counts)),
                       delimiter=",", header=f"left_edge,right_edge,count; x_origin={origins[name]}; unit={feature_unit(name)}",
                       fmt=["%.17g", "%.17g", "%d"])
        ax.set_xlabel(name + (f" ({feature_unit(name)})" if feature_unit(name) else ""))
        ax.set_ylabel("Fraction of valid hits / bin")
        ax.set_title(f"{name} by channel")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{name}_by_channel.png", dpi=160)
        plt.close(fig)
    for channel, waveforms in sorted(examples.items()):
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, waveform in enumerate(waveforms):
            ax.plot(np.arange(len(waveform)) * period_ns, waveform, ".-", label=f"Example {i + 1}")
        if baseline_window is not None:
            ax.axvspan((baseline_window[0] - 0.5) * period_ns, (baseline_window[1] - 0.5) * period_ns,
                       alpha=0.15, color="green", label="Baseline window")
        if signal_window is not None:
            ax.axvspan((signal_window[0] - 0.5) * period_ns, (signal_window[1] - 0.5) * period_ns,
                       alpha=0.12, color="orange", label="Signal window")
        ax.set(xlabel="Time from first stored sample (ns)", ylabel="DataSample (V)",
               title=f"Channel {channel}: first available waveforms")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / f"waveforms_channel_{channel}.png", dpi=160)
        plt.close(fig)
    summary = {
        "input": str(parquet_path), "hits": hit_count,
        "valid_finite_waveform_samples": sample_count,
        "hits_per_channel": dict(sorted(channel_counts.items())),
        "mean_waveform_peak_to_peak": peak_sum / waveform_count if waveform_count else None,
        "histogram_origins": origins,
        "histogram_ranges": ranges,
        "baseline_window": baseline_window, "signal_window": signal_window,
        "polarity": polarity,
        "sample_period_ns": period_ns,
        "time_axis": "uniform chronological samples; relative to first stored sample; no cell timing correction applied",
        "thresholds_percent": list(thresholds),
        "thresholds_volts": list(voltage_thresholds),
        "units": {name: feature_unit(name) for name in ranges},
        "amplitude_mpv_by_channel": {
            str(channel): histogram_mpv(counts, edges[name])
            for (channel, name), counts in channel_histograms.items() if name == "PulseAmplitude"
        },
        "snr_definition": "positive pulse height / within-waveform baseline sample std (ddof=1)",
        "channel_histogram_valid_entries": {
            f"channel_{channel}/{name}": int(counts.sum())
            for (channel, name), counts in channel_histograms.items()
        },
    }
    save_amplitude_mpv(plots_dir, summary["amplitude_mpv_by_channel"])
    (plots_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Analyzed hits: {hit_count}; valid finite waveform samples: {sample_count}")
    print(f"Hits per channel: {summary['hits_per_channel']}")
    print(f"Mean waveform peak-to-peak: {summary['mean_waveform_peak_to_peak']}")
    print(f"Saved {len(histograms)} histograms (PNG + CSV) and summary.json to {plots_dir}")


def save_amplitude_mpv(plots_dir, mpvs):
    import matplotlib.pyplot as plt
    if mpvs:
        import csv
        with (plots_dir / "AmplitudeMPV_by_channel.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["channel", "mpv_V", "bootstrap_std_V", "bin_low_V", "bin_high_V", "entries", "tied_peak_bins"])
            for channel, estimate in mpvs.items():
                writer.writerow([channel] + [estimate.get(key) for key in
                    ("value_V", "bootstrap_std_V", "bin_low_V", "bin_high_V", "entries", "tied_peak_bins")])
        valid_mpvs = [(int(ch), estimate) for ch, estimate in mpvs.items() if estimate["value_V"] is not None]
        if valid_mpvs:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.errorbar([ch for ch, _ in valid_mpvs], [m["value_V"] for _, m in valid_mpvs],
                        yerr=[m["bootstrap_std_V"] for _, m in valid_mpvs], fmt="o", capsize=3)
            ax.set(xlabel="Channel", ylabel="Amplitude MPV (V)",
                   title="Histogram mode; errors: binned bootstrap standard deviation")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(plots_dir / "AmplitudeMPV_by_channel.png", dpi=160)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT,
                        help="SAMPIC run directory, single .bin file, or existing Parquet file")
    parser.add_argument("--output", type=Path, help="New Parquet output path (never overwritten)")
    parser.add_argument("--inspect-only", action="store_true", help="Print structure without analysis")
    parser.add_argument("--step-size", type=int, default=100_000, help="Hits per analysis batch")
    parser.add_argument("--plots-dir", type=Path, help="Plot directory (default: ./plots/<run>)")
    parser.add_argument("--bins", type=int, default=100, help="Bins per continuous histogram")
    parser.add_argument("--baseline-window", nargs=2, type=int, metavar=("START", "STOP"),
                        help="Quiet sample interval [START, STOP); enables noise and SNR")
    parser.add_argument("--signal-window", nargs=2, type=int, metavar=("START", "STOP"),
                        help="Pulse interval [START, STOP); default: after baseline window through DataSize")
    parser.add_argument("--polarity", choices=("positive", "negative"), default="positive")
    parser.add_argument("--sample-period-ns", type=float,
                        help="Override uniform sample spacing; otherwise read sampling_frequency metadata")
    parser.add_argument("--thresholds-percent", nargs="+", type=float, default=DEFAULT_THRESHOLDS,
                        help="Fractions of baseline-subtracted peak, in percent (default: 10 through 90)")
    parser.add_argument("--thresholds-volts", nargs="+", type=float, default=(),
                        help="Optional fixed thresholds above baseline, in polarity-corrected V")
    args = parser.parse_args()
    if args.sample_period_ns is not None and (not np.isfinite(args.sample_period_ns) or args.sample_period_ns <= 0):
        parser.error("--sample-period-ns must be finite and positive")
    if any(not np.isfinite(v) or not 0 < v < 100 for v in args.thresholds_percent):
        parser.error("--thresholds-percent must be finite and between 0 and 100 (exclusive)")
    if any(not np.isfinite(v) or v <= 0 for v in args.thresholds_volts):
        parser.error("--thresholds-volts must be finite and positive")
    if args.baseline_window is not None:
        start, stop = args.baseline_window
        if start < 0 or stop - start < 2:
            parser.error("--baseline-window requires START >= 0 and at least two samples")
    if args.signal_window is not None:
        start, stop = args.signal_window
        if args.baseline_window is None:
            parser.error("--signal-window requires --baseline-window")
        if start < 0 or stop <= start:
            parser.error("--signal-window requires 0 <= START < STOP")
        if max(start, args.baseline_window[0]) < min(stop, args.baseline_window[1]):
            parser.error("Signal and baseline windows must not overlap")
    if args.bins <= 0:
        parser.error("--bins must be positive")
    if args.step_size <= 0:
        parser.error("--step-size must be positive")
    source = args.input.resolve()
    if not source.exists():
        parser.error(f"Input does not exist: {source}")
    if source.is_file() and source.suffix.lower() in {".parquet", ".pq"}:
        if args.output:
            parser.error("--output is only used when decoding binaries")
        parquet_path = source
    else:
        default_output = (source / f"{source.name}.parquet" if source.is_dir()
                          else source.with_suffix(".parquet"))
        if args.output is None and default_output.is_file():
            parquet_path = default_output
            print(f"Using existing Parquet: {parquet_path}")
        else:
            output = args.output or default_output
            parquet_path = decode_to_parquet(source, output.resolve())
    inspect_and_analyze(parquet_path, args.inspect_only, args.step_size,
                        args.plots_dir, args.bins, args.baseline_window,
                        args.signal_window, args.polarity, args.sample_period_ns,
                        args.thresholds_percent, args.thresholds_volts)


if __name__ == "__main__":
    main()
