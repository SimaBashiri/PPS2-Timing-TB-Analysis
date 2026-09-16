"""Batch pulse measurements for chronological samples in volts, times in ns."""

import re
import numpy as np

DEFAULT_THRESHOLDS = (10, 20, 30, 40, 50, 60, 70, 80, 90)


def sample_period_ns(metadata, override=None):
    if override is not None:
        period = float(override)
    else:
        raw = (metadata or {}).get(b"sampling_frequency", b"").decode("ascii")
        match = re.fullmatch(r"\s*([\d.eE+-]+)\s*([GMk]?)S/s\s*", raw)
        if not match:
            raise ValueError("Unknown sampling frequency; supply --sample-period-ns.")
        frequency = float(match[1]) * {"G": 1e9, "M": 1e6, "k": 1e3, "": 1}[match[2]]
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("Sampling frequency must be finite and positive.")
        period = 1e9 / frequency
    if not np.isfinite(period) or period <= 0:
        raise ValueError("Sample period must be finite and positive.")
    return period


def threshold_tag(value, suffix):
    return f"{float(value):g}".replace(".", "p") + suffix


def pulse_features(parents, positions, signed, amplitude, sizes, signal_window,
                   period, thresholds=DEFAULT_THRESHOLDS, voltage_thresholds=()):
    """Inputs contain only finite signal samples. Never bridge missing samples.

    Crossings bound the connected excursion containing the selected peak.
    Any incomplete/nonfinite signal window invalidates area and timing.
    """
    n = len(amplitude)
    start, stop = signal_window
    expected = np.full(n, stop - start) if stop is not None else sizes - start
    count = np.bincount(parents, minlength=n)
    complete = (count == expected) & (expected >= 2)
    positive = np.isfinite(amplitude) & (amplitude > 0)
    valid = complete & positive
    peak = np.full(n, np.inf)
    at_peak = signed == amplitude[parents]
    np.minimum.at(peak, parents[at_peak], positions[at_peak])
    result = {
        "InvalidSignalWindow": (~complete).astype(int),
        "NonpositivePulse": (~positive).astype(int),
        "PeakTime": np.where(valid, peak * period, np.nan),
    }
    adjacent = (parents[1:] == parents[:-1]) & (positions[1:] == positions[:-1] + 1)
    left = np.flatnonzero(adjacent)
    rows = parents[left]
    x = positions[left]
    y0, y1 = signed[left], signed[left + 1]
    area = np.bincount(rows, weights=(y0 + y1) * 0.5 * period, minlength=n)
    result["SignalArea"] = np.where(complete, area, np.nan)

    def crossings(level):
        level = np.broadcast_to(level, (n,))
        h = level[rows]
        up = (y0 < h) & (y1 >= h) & (x < peak[rows])
        down = (y0 >= h) & (y1 < h) & (x >= peak[rows])
        rising, falling = np.full(n, -np.inf), np.full(n, np.inf)
        for mask, out, reduce in ((up, rising, np.maximum.at), (down, falling, np.minimum.at)):
            times = (x[mask] + (h[mask] - y0[mask]) / (y1[mask] - y0[mask])) * period
            reduce(out, rows[mask], times)
        for out in (rising, falling):
            out[~valid | ~np.isfinite(out) | (level >= amplitude)] = np.nan
        return rising, falling

    fractions = sorted(set(thresholds) | {10, 20, 50, 80, 90})
    times = {}
    for percent in fractions:
        rising, falling = crossings(amplitude * percent / 100)
        times[percent] = rising, falling
        if percent in thresholds:
            tag = threshold_tag(percent, "pct")
            result[f"LeadingTime_{tag}"] = rising
            result[f"TrailingTime_{tag}"] = falling
            result[f"ToT_{tag}"] = falling - rising
    for voltage in voltage_thresholds:
        rising, falling = crossings(voltage)
        tag = threshold_tag(voltage, "V")
        result[f"LeadingTime_{tag}"] = rising
        result[f"TrailingTime_{tag}"] = falling
        result[f"ToT_{tag}"] = falling - rising
    result["RiseTime10_90"] = times[90][0] - times[10][0]
    result["RiseTime20_80"] = times[80][0] - times[20][0]
    result["FallTime90_10"] = times[10][1] - times[90][1]
    result["FWHM"] = times[50][1] - times[50][0]
    result["Missing10pctCrossing"] = (~np.isfinite(times[10][0]) | ~np.isfinite(times[10][1])).astype(int)
    up50 = (y0 < amplitude[rows] * 0.5) & (y1 >= amplitude[rows] * 0.5)
    result["Multiple50pctExcursions"] = np.where(
        valid, np.bincount(rows[up50], minlength=n) > 1, False).astype(int)
    return result


def histogram_mpv(counts, edges):
    """Binned mode, with bin bounds; not a Landau fit or a fit uncertainty."""
    counts, edges = np.asarray(counts), np.asarray(edges)
    total = int(np.sum(counts))
    if total < 2:
        return {"value_V": None, "entries": total, "method": "insufficient entries"}
    winners = np.flatnonzero(counts == np.max(counts))
    index = int(winners[0])
    rng = np.random.default_rng(1729)
    draws = rng.multinomial(total, np.asarray(counts) / total, size=300)
    centers = (edges[:-1] + edges[1:]) / 2
    uncertainty = float(np.std(centers[np.argmax(draws, axis=1)], ddof=1))
    return {"value_V": float((edges[index] + edges[index + 1]) / 2),
            "bin_low_V": float(edges[index]), "bin_high_V": float(edges[index + 1]),
            "bootstrap_std_V": uncertainty,
            "entries": total, "tied_peak_bins": len(winners),
            "method": "histogram mode (first bin on ties); 300 multinomial bootstrap replicas; bin bounds describe resolution"}


def feature_unit(name):
    if name == "SignalArea":
        return "V·ns"
    if name.startswith(("LeadingTime_", "TrailingTime_", "ToT_", "RiseTime", "FallTime")) or name in {"PeakTime", "FWHM", "Time", "TOTValue", "PhysicalCell0Time", "OrderedCell0Time"}:
        return "ns"
    if name in {"Baseline", "Amplitude", "RawPeak", "BaselineEstimate", "BaselineResidual", "NoiseRMS", "PulseAmplitude", "SampleMean", "PeakToPeak", "MaxAbsBaselineDeviation"}:
        return "V"
    if name == "SNR":
        return "dimensionless"
    if name.startswith("FirstSampleTime_in_ps"):
        return "ps (stored)"
    if name == "UnixTime":
        return "s (stored)"
    return ""
