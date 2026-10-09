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
    # "/GPT6/shared1/sbashiri/TestBeam/2025-05-14/SAMPIC_data/"
    # "Run018_SAMPIC_5_17_2025_23h_13min_Binary"
    "/GPT6/shared1/sbashiri/TestBeam/2026-08/SAMPIC/sampic_20260805_013030_run1"
)


def decode_to_parquet(source, output):
    from sampiclyser import SAMPIC_Run_Decoder

    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Pass that Parquet file as input to inspect it."
        )
    decoder = SAMPIC_Run_Decoder(source if source.is_dir() else source.parent)
    # Preserve the decoder's natural ordering and exclusion of trigger-data
    # binaries, which do not use the hit-record header format.
    decoder.run_files = [path for path in decoder.run_files if path.is_file()]
    if source.is_file():
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


def parquet_output_path(source, output):
    """Resolve an output file or directory to the Parquet file for this run."""
    output = output.expanduser()
    # Accept a directory (including a not-yet-created directory such as ./data)
    # as the output destination, as in DecodeFiles.ipynb. A .parquet/.pq path
    # remains an explicit filename.
    if output.is_dir() or output.suffix.lower() not in {".parquet", ".pq"}:
        run_name = source.name if source.is_dir() else source.stem
        output = output / f"{run_name}.parquet"
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
        first_dropped = np.zeros(len(batch), dtype=bool)
        if start == 0 and stop >= 3:
            # Judge sample zero independently so it cannot inflate its own
            # reference noise. Only use complete, finite reference windows.
            reference = quiet & (positions > 0)
            rp, rv = parents[reference], values[reference]
            rn = np.bincount(rp, minlength=len(batch))
            reference_mean = np.zeros(len(batch))
            np.divide(np.bincount(rp, weights=rv, minlength=len(batch)), rn,
                      out=reference_mean, where=rn > 0)
            reference_variance = np.zeros(len(batch))
            np.divide(np.bincount(rp, weights=(rv - reference_mean[rp])**2,
                                  minlength=len(batch)), rn - 1,
                      out=reference_variance, where=rn > 1)
            first = positions == 0
            fp, fv = parents[first], values[first]
            # A numerical floor avoids rejecting roundoff in flat baselines.
            tolerance = 16 * np.finfo(float).eps * np.maximum(
                np.abs(fv), np.abs(reference_mean[fp]))
            first_dropped[fp] = (rn[fp] == stop - 1) & (
                np.abs(fv - reference_mean[fp]) > np.maximum(
                    5 * np.sqrt(reference_variance[fp]), tolerance))
            quiet &= ~((positions == 0) & first_dropped[parents])
        qparents, qvalues = parents[quiet], values[quiet]
        n = np.bincount(qparents, minlength=len(batch))
        total = np.bincount(qparents, weights=qvalues, minlength=len(batch))
        pedestal = np.full(len(batch), np.nan)
        # All requested samples except a deliberately rejected first sample
        # must still exist and be finite.
        complete = (n == stop - start - first_dropped.astype(int)) & (n >= 2)
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
            "FirstBaselineSampleDropped": first_dropped.astype(int),
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
        span = edges[name][-1] - edges[name][0]
        ax.set_xlim(edges[name][0] - 0.03 * span, edges[name][-1] + 0.03 * span)
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
    parser.add_argument("--output", type=Path,
                        help="Output directory (writes <run>.parquet) or explicit .parquet filename")
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
    parser.add_argument("--timing", action="store_true", help="Also fit coincidence timing and calibrate channel delays")
    parser.add_argument("--timing-only", action="store_true", help="Run timing analysis without waveform histograms")
    parser.add_argument("--timing-channels", nargs="+", type=int, help="Channels to compare (default: all)")
    parser.add_argument("--reference-channel", type=int, help="Zero-delay channel (default: lowest selected)")
    parser.add_argument("--cfd-percent", type=float, default=50)
    parser.add_argument("--coincidence-window-ns", type=float, default=30)
    parser.add_argument("--timing-bin-width-ns", type=float, default=0.02)
    parser.add_argument("--timing-fit-window-ns", type=float, default=1)
    parser.add_argument("--timing-sigma-max-ns", type=float, default=0.5)
    parser.add_argument("--timing-min-counts", type=int, default=50)
    parser.add_argument("--timing-min-amplitude", type=float, default=0, help="Minimum pulse amplitude in V")
    parser.add_argument("--timing-min-snr", type=float, default=0)
    args = parser.parse_args()
    if args.timing or args.timing_only:
        if args.inspect_only or args.baseline_window is None:
            parser.error("Timing requires --baseline-window and cannot be combined with --inspect-only")
        if not np.isfinite(args.cfd_percent) or not 0 < args.cfd_percent < 100:
            parser.error("--cfd-percent must be between 0 and 100")
        for name in ("coincidence_window_ns", "timing_bin_width_ns", "timing_fit_window_ns", "timing_sigma_max_ns"):
            if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                parser.error(f"--{name.replace('_', '-')} must be finite and positive")
        for name in ("timing_min_amplitude", "timing_min_snr"):
            if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
                parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
        if args.timing_min_counts < 5:
            parser.error("--timing-min-counts must be at least 5")
        if args.timing_bin_width_ns >= args.timing_sigma_max_ns:
            parser.error("Timing bin width must be smaller than the maximum fitted sigma")
        if args.timing_fit_window_ns >= args.coincidence_window_ns:
            parser.error("Timing fit window must be smaller than the coincidence window")
        if args.timing_channels is not None:
            if len(set(args.timing_channels)) < 2:
                parser.error("Select at least two timing channels")
            if args.reference_channel is not None and args.reference_channel not in args.timing_channels:
                parser.error("Reference channel must be in --timing-channels")
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
        default_output = (Path("./data") / f"{source.name}.parquet" if source.is_dir()
                          else source.with_suffix(".parquet"))
        if args.output is None and default_output.is_file():
            parquet_path = default_output
            print(f"Using existing Parquet: {parquet_path}")
        else:
            output = parquet_output_path(source, args.output) if args.output else default_output
            parquet_path = decode_to_parquet(source, output.resolve())
    schema_names = set(pq.ParquetFile(parquet_path).schema_arrow.names)
    has_waveforms = {"DataSample", "DataSize", "Baseline"}.issubset(schema_names)
    if not args.timing_only:
        if has_waveforms or args.inspect_only:
            inspect_and_analyze(parquet_path, args.inspect_only, args.step_size,
                                args.plots_dir, args.bins, args.baseline_window,
                                args.signal_window, args.polarity, args.sample_period_ns,
                                args.thresholds_percent, args.thresholds_volts)
        else:
            missing = sorted({"DataSample", "DataSize", "Baseline"} - schema_names)
            print("Skipping waveform plots; this compact-mode Parquet lacks "
                  f"waveform fields: {missing}")
    if args.timing or args.timing_only:
        if not has_waveforms:
            print("Skipping CFD timing analysis; it requires waveform samples to "
                  "calculate constant-fraction times.")
            return
        from timing_analysis import analyze_timing
        output_dir = args.plots_dir or Path.cwd() / "plots" / parquet_path.stem
        analyze_timing(
            parquet_path, output_dir / "timing", waveform_features,
            baseline_window=args.baseline_window, signal_window=args.signal_window,
            polarity=args.polarity, period_ns=args.sample_period_ns,
            channels=args.timing_channels, reference=args.reference_channel,
            fraction=args.cfd_percent, window=args.coincidence_window_ns,
            bin_width=args.timing_bin_width_ns, fit_window=args.timing_fit_window_ns,
            sigma_max=args.timing_sigma_max_ns, min_counts=args.timing_min_counts,
            min_amplitude=args.timing_min_amplitude, min_snr=args.timing_min_snr,
            step_size=args.step_size)


if __name__ == "__main__":
    main()
