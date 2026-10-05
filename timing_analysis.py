"""Coincidence timing and relative channel delays for chronological waveforms."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from pulse_measurements import sample_period_ns, threshold_tag


def gaussian_background(times, amplitude, mean, sigma, background):
    return amplitude * np.exp(-0.5 * ((times - mean) / sigma)**2) + background


def fit_peak(counts, edges, fit_window, sigma_max, min_counts):
    """Fit isolated peak candidates, retaining zero-count bins and fit failures."""
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    if counts.sum() < min_counts:
        return None
    peaks, _ = find_peaks(counts, prominence=max(3, counts.max() * 0.1),
                          distance=max(1, int(sigma_max / width)))
    fits = []
    for peak in sorted(peaks, key=lambda index: counts[index], reverse=True)[:4]:
        seed = centers[peak]
        selected = abs(centers - seed) <= fit_window
        times, entries = centers[selected], counts[selected]
        if len(times) <= 4 or entries.sum() < min_counts:
            continue
        initial = [float(counts[peak]), seed,
                   min(max(width, 0.05), sigma_max / 2), float(np.median(entries))]
        try:
            parameters, covariance = curve_fit(
                gaussian_background, times, entries, p0=initial,
                sigma=np.sqrt(np.maximum(entries, 1)), absolute_sigma=True,
                bounds=([0, seed - fit_window, width / 2, 0],
                        [np.inf, seed + fit_window, sigma_max, np.inf]), maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        errors = np.sqrt(np.diag(covariance))
        amplitude, mean, sigma, background = parameters
        if (not np.all(np.isfinite(errors)) or errors[1] <= 0
                or errors[1] >= sigma or errors[2] >= sigma
                or sigma <= width * 0.51 or sigma >= sigma_max * 0.99
                or amplitude < 3 * errors[0]
                or mean - 3 * sigma < times[0] or mean + 3 * sigma > times[-1]):
            continue
        residual = (entries - gaussian_background(times, *parameters)) / np.sqrt(np.maximum(entries, 1))
        fits.append(dict(amplitude=float(amplitude), mean_ns=float(mean),
                         mean_error_ns=float(errors[1]), sigma_ns=float(sigma),
                         sigma_error_ns=float(errors[2]), background=float(background),
                         chi2=float(residual @ residual), ndof=len(times) - 4,
                         fit_low_ns=float(times[0]), fit_high_ns=float(times[-1])))
    return max(fits, key=lambda fit: fit['amplitude']) if fits else None


def pair_deltas(raw_first, offset_first, raw_second, offset_second, window):
    """Greedily match closest CFD hits one-to-one within the coincidence window."""
    if not len(raw_first) or not len(raw_second):
        return np.empty(0)
    first = raw_first + offset_first.astype(np.longdouble)
    second = raw_second + offset_second.astype(np.longdouble)

    # Generate only in-window candidates, then take the globally closest pairs
    # first. Each selected hit is excluded from further matches.
    first_order = np.argsort(first, kind="stable")
    first_sorted = first[first_order]
    candidates = []
    for second_index, second_time in enumerate(second):
        lo = np.searchsorted(first_sorted, second_time - window, side="left")
        hi = np.searchsorted(first_sorted, second_time + window, side="right")
        for position in range(lo, hi):
            first_index = int(first_order[position])
            delta = (raw_second[second_index] - raw_first[first_index]
                     + offset_second[second_index] - offset_first[first_index])
            candidates.append((abs(float(delta)), second_index, first_index, float(delta)))
    candidates.sort(key=lambda item: item[0])
    used_first, used_second, deltas = set(), set(), []
    for _, second_index, first_index, delta in candidates:
        if second_index not in used_second and first_index not in used_first:
            used_second.add(second_index)
            used_first.add(first_index)
            deltas.append(delta)
    return np.asarray(deltas, dtype=float)


def global_pair_deltas(hits, channels, window):
    """Pair hits globally in time order; consume each hit at most once."""
    times = {
        channel: np.asarray(raw + offsets.astype(np.longdouble), dtype=np.longdouble)
        for channel, (raw, offsets) in hits.items()
    }
    sorted_hits = {}
    for channel in channels:
        order = np.argsort(times[channel], kind="stable")
        sorted_hits[channel] = (times[channel][order], order)
    used = {channel: np.zeros(len(times[channel]), dtype=bool) for channel in channels}
    stream = [(values[position], channel, int(original[position]))
              for channel, (values, original) in sorted_hits.items()
              for position in range(len(values))]
    stream.sort(key=lambda hit: hit[0])
    result = {(first, second): [] for first, second in combinations(channels, 2)}

    for anchor_time, anchor_channel, anchor_index in stream:
        if used[anchor_channel][anchor_index]:
            continue
        best = None
        for channel in channels:
            if channel == anchor_channel:
                continue
            values, original = sorted_hits[channel]
            left = np.searchsorted(values, anchor_time - window, side="left")
            right = np.searchsorted(values, anchor_time + window, side="right")
            for position in range(left, right):
                candidate_index = int(original[position])
                if used[channel][candidate_index]:
                    continue
                candidate_time = values[position]
                difference = candidate_time - anchor_time
                candidate = (abs(difference), candidate_time, channel,
                             candidate_index, difference)
                if best is None or candidate[:4] < best[:4]:
                    best = candidate

        used[anchor_channel][anchor_index] = True
        if best is None:
            continue
        _, _, partner_channel, partner_index, difference = best
        used[partner_channel][partner_index] = True
        first, second = sorted((anchor_channel, partner_channel))
        signed_difference = difference if anchor_channel == first else -difference
        result[(first, second)].append(float(signed_difference))

    return {pair: np.asarray(deltas, dtype=float) for pair, deltas in result.items()}


def solve_delays(pairs, channels, reference):
    """Solve delay[b] - delay[a] = mean(t[b] - t[a]), reference fixed at zero."""
    edges = [pair for pair in pairs if pair['fit'] is not None]
    connected = {reference}
    while True:
        previous = len(connected)
        for pair in edges:
            if pair['first'] in connected or pair['second'] in connected:
                connected.update((pair['first'], pair['second']))
        if len(connected) == previous:
            break
    free = sorted(connected - {reference})
    edges = [pair for pair in edges if pair['first'] in connected]
    delays = {str(channel): None for channel in channels}
    errors = delays.copy()
    delays[str(reference)] = errors[str(reference)] = 0.0
    if not free:
        return dict(delays_ns=delays, errors_ns=errors, reference_channel=reference,
                    chi2=None, ndof=0, residuals=[])
    design = np.array([[int(pair['second'] == channel) - int(pair['first'] == channel)
                        for channel in free] for pair in edges], dtype=float)
    means = np.array([pair['fit']['mean_ns'] for pair in edges])
    uncertainty = np.array([pair['fit']['mean_error_ns'] for pair in edges])
    weighted = design / uncertainty[:, None]
    solution = np.linalg.lstsq(weighted, means / uncertainty, rcond=None)[0]
    covariance = np.linalg.inv(weighted.T @ weighted)
    for channel, value, error in zip(free, solution, np.sqrt(np.diag(covariance))):
        delays[str(channel)], errors[str(channel)] = float(value), float(error)
    residuals = design @ solution - means
    return dict(delays_ns=delays, errors_ns=errors, reference_channel=reference,
                chi2=float(np.sum((residuals / uncertainty)**2)), ndof=len(edges) - len(free),
                residuals=[dict(first=pair['first'], second=pair['second'],
                                residual_ns=float(residual), pull=float(residual / error))
                           for pair, residual, error in zip(edges, residuals, uncertainty)])


def plot_timing_summary(output_dir, report):
    """Render summary figures from saved fit results and per-pair CSV histograms."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    channels = sorted(int(channel) for channel in report['valid_hits_per_channel'])
    pairs = report['pairs']
    accepted = [pair for pair in pairs if pair['fit'] is not None]
    positions = {channel: index for index, channel in enumerate(channels)}
    resolutions = np.full((len(channels), len(channels)), np.nan)
    for pair in accepted:
        first, second = positions[pair['first']], positions[pair['second']]
        resolutions[first, second] = resolutions[second, first] = pair['pair_resolution_ps']
    figure, axis = plt.subplots(figsize=(8, 7))
    colormap = plt.get_cmap('viridis').with_extremes(bad='lightgray')
    heatmap = axis.imshow(np.ma.masked_invalid(resolutions), cmap=colormap,
                          **({} if accepted else dict(vmin=0, vmax=1)))
    if accepted:
        figure.colorbar(heatmap, ax=axis, label='Pair sigma (ps)')
    for row in range(len(channels)):
        for column in range(len(channels)):
            value = resolutions[row, column]
            axis.text(column, row, f'{value:.1f}' if np.isfinite(value) else '—',
                      ha='center', va='center', fontsize=8,
                      bbox=dict(facecolor='white', edgecolor='none', alpha=0.65))
    axis.set(xticks=range(len(channels)), xticklabels=channels,
             yticks=range(len(channels)), yticklabels=channels,
             xlabel='Channel', ylabel='Channel',
             title='CFD pair timing resolution (gray = unavailable)')
    figure.tight_layout()
    figure.savefig(output_dir / 'timing_resolution_heatmap.png', dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(max(8, len(pairs) * 0.28), 5))
    for index, pair in enumerate(pairs):
        if pair['fit'] is not None:
            axis.errorbar(index, pair['pair_resolution_ps'],
                          yerr=pair['pair_resolution_error_ps'], fmt='o',
                          color='tab:blue', capsize=3)
        else:
            axis.text(index, 0.02, 'N/A', transform=axis.get_xaxis_transform(),
                      rotation=90, ha='center', va='bottom', fontsize=8)
    axis.set(xticks=range(len(pairs)),
             xticklabels=[f"{pair['second']}−{pair['first']}" for pair in pairs],
             xlabel='Channel pair (B−A)', ylabel='Pair sigma (ps)',
             title='CFD pair timing resolution')
    axis.tick_params(axis='x', labelrotation=90)
    axis.set_ylim(bottom=0)
    axis.grid(axis='y', alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / 'timing_resolution_by_pair.png', dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    if accepted:
        pair = max(accepted, key=lambda item: item['fit']['amplitude'])
        fit = pair['fit']
        histogram = np.loadtxt(output_dir / f"delta_t_ch{pair['second']}_minus_ch{pair['first']}.csv",
                               delimiter=',', ndmin=2)
        edges = np.r_[histogram[:, 0], histogram[-1, 1]]
        axis.stairs(histogram[:, 2], edges, label='Coincidences')
        grid = np.linspace(fit['fit_low_ns'], fit['fit_high_ns'], 500)
        axis.plot(grid, gaussian_background(grid, fit['amplitude'], fit['mean_ns'],
                                            fit['sigma_ns'], fit['background']),
                  label=f"Gaussian + background; pair sigma={pair['pair_resolution_ps']:.1f} ps")
        axis.set(xlim=(fit['fit_low_ns'], fit['fit_high_ns']),
                 xlabel=f"t(ch {pair['second']}) - t(ch {pair['first']}) [ns]",
                 title=f"Strongest accepted CFD peak: ch {pair['second']} − ch {pair['first']}\n"
                       f"offset={fit['mean_ns']:.4f} ns")
        axis.legend()
    else:
        axis.text(0.5, 0.5, 'No accepted coincidence peak', transform=axis.transAxes,
                  ha='center', va='center')
        axis.set(xlabel='Time difference [ns]', title='CFD coincidence peak')
    axis.set_ylabel('Coincidences / bin')
    figure.tight_layout()
    figure.savefig(output_dir / 'timing_coincidence_peak.png', dpi=160)
    plt.close(figure)

    calibration = report['calibration']
    delay_channels = [channel for channel in channels
                      if calibration['delays_ns'][str(channel)] is not None]
    delay_values = [calibration['delays_ns'][str(channel)] for channel in delay_channels]
    delay_errors = [calibration['errors_ns'][str(channel)] or 0.0 for channel in delay_channels]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(delay_channels, delay_values, yerr=delay_errors, fmt='o', capsize=4)
    axis.axhline(0, color='black', linewidth=0.8, alpha=0.6)
    axis.set(xticks=channels, xlabel='Channel', ylabel='Relative delay (ns)',
             title=f"Channel delays relative to channel {calibration['reference_channel']}")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / 'timing_channel_delays.png', dpi=160)
    plt.close(figure)


def analyze_timing(parquet_path, output_dir, feature_function, *, baseline_window,
                   signal_window=None, polarity='positive', period_ns=None,
                   channels=None, reference=None, fraction=50, window=30,
                   bin_width=0.02, fit_window=1, sigma_max=0.5,
                   min_counts=50, min_amplitude=0, min_snr=0, step_size=100000):
    """Stream waveforms once; retain only compact timing arrays across batches."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = {}
    diagnostics = dict(total_hits=0, selected_hits=0, backward_steps=0,
                       largest_backward_step_ns=0.0, max_stored_float_spacing_ps=0.0)
    previous = {}
    origin = None
    with pq.ParquetFile(parquet_path) as source:
        names = source.schema_arrow.names
        period_ns = sample_period_ns(source.schema_arrow.metadata, period_ns)
        precise = 'FirstSampleTime_in_ps' in names
        time_column = 'FirstSampleTime_in_ps' if precise else 'OrderedCell0Time'
        required = {'Channel', time_column, 'DataSize', 'DataSample', 'Baseline'}
        if required - set(names):
            raise ValueError(f"Missing timing columns: {sorted(required - set(names))}")
        fine_column = 'FirstSampleTime_in_ps_fine'
        columns = sorted(required | ({fine_column} if precise and fine_column in names else set()))
        if np.finfo(np.longdouble).nmant < 63:
            raise ValueError('Timing analysis requires extended precision longdouble on this platform.')
        for batch in source.iter_batches(columns=columns, batch_size=step_size):
            diagnostics['total_hits'] += len(batch)
            channel_values = batch.column('Channel').to_numpy(zero_copy_only=False)
            stored = batch.column(time_column).to_numpy(zero_copy_only=False)
            if precise and not np.issubdtype(stored.dtype, np.integer):
                raise ValueError('FirstSampleTime_in_ps must be non-null integer picoseconds.')
            raw = stored.astype(np.longdouble)
            if origin is None and np.any(np.isfinite(raw)):
                origin = raw[np.flatnonzero(np.isfinite(raw))[0]]
            raw -= origin if origin is not None else 0
            if precise:
                if fine_column in columns:
                    raw += batch.column(fine_column).to_numpy(zero_copy_only=False).astype(np.longdouble)
                raw /= 1000
            elif np.any(np.isfinite(stored)):
                spacing = np.max(abs(np.spacing(stored[np.isfinite(stored)]))) * 1000
                diagnostics['max_stored_float_spacing_ps'] = max(
                    diagnostics['max_stored_float_spacing_ps'], float(spacing))
            features = feature_function(batch, baseline_window, signal_window, polarity,
                                        period_ns, (fraction,))
            offsets = features[f'LeadingTime_{threshold_tag(fraction, "pct")}']
            valid = np.isfinite(raw) & np.isfinite(offsets) & np.isfinite(channel_values)
            valid &= features['PulseAmplitude'] > min_amplitude
            if min_snr > 0:
                valid &= features['SNR'] >= min_snr
            for channel in np.unique(channel_values[np.isfinite(channel_values)]):
                channel = int(channel)
                if channels is not None and channel not in channels:
                    continue
                times = raw[(channel_values == channel) & np.isfinite(raw)]
                if len(times):
                    steps = np.diff(np.r_[previous.get(channel, times[0]), times])
                    diagnostics['backward_steps'] += int(np.sum(steps < 0))
                    diagnostics['largest_backward_step_ns'] = max(
                        diagnostics['largest_backward_step_ns'], float(max(0, -steps.min())))
                    previous[channel] = times[-1]
                selected = valid & (channel_values == channel)
                chunks.setdefault(channel, []).append((raw[selected], offsets[selected]))
                diagnostics['selected_hits'] += int(selected.sum())
    if not precise and diagnostics['largest_backward_step_ns'] > 1e9:
        raise ValueError('Possible counter rollover/reset: use decoded FirstSampleTime_in_ps before coincidence analysis.')
    channels = sorted(chunks) if channels is None else sorted(set(channels))
    if len(channels) < 2:
        raise ValueError('Timing analysis needs at least two channels.')
    reference = channels[0] if reference is None else reference
    if reference not in channels:
        raise ValueError('Reference channel is not in the selected channels.')
    hits = {channel: (np.concatenate([chunk[0] for chunk in chunks[channel]]),
                      np.concatenate([chunk[1] for chunk in chunks[channel]]))
            if channel in chunks else (np.array([], dtype=np.longdouble), np.array([]))
            for channel in channels}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    edges = np.linspace(-window, window, int(np.ceil(2 * window / bin_width)) + 1)
    globally_matched = global_pair_deltas(hits, channels, window)
    diagnostics['globally_matched_pairs'] = int(sum(len(values) for values in globally_matched.values()))
    diagnostics['globally_unmatched_hits'] = int(
        sum(len(hits[channel][0]) for channel in channels)
        - 2 * diagnostics['globally_matched_pairs'])
    pairs = []
    for first, second in combinations(channels, 2):
        deltas = globally_matched[(first, second)]
        counts = np.histogram(deltas, edges)[0]
        fit = fit_peak(counts, edges, fit_window, sigma_max, min_counts)
        pair = dict(first=first, second=second, entries=len(deltas), fit=fit,
                    pair_resolution_ps=fit['sigma_ns'] * 1000 if fit else None,
                    pair_resolution_error_ps=fit['sigma_error_ns'] * 1000 if fit else None)
        pairs.append(pair)
        stem = output_dir / f'delta_t_ch{second}_minus_ch{first}'
        np.savetxt(stem.with_suffix('.csv'), np.column_stack((edges[:-1], edges[1:], counts)),
                   delimiter=',', header='left_ns,right_ns,count', fmt=['%.12g', '%.12g', '%d'])
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.stairs(counts, edges)
        if fit:
            grid = np.linspace(fit['fit_low_ns'], fit['fit_high_ns'], 500)
            axis.plot(grid, gaussian_background(grid, fit['amplitude'], fit['mean_ns'],
                                                fit['sigma_ns'], fit['background']),
                      label=f"offset={fit['mean_ns']:.4f} ns; pair sigma={pair['pair_resolution_ps']:.1f} ps")
            axis.legend()
        axis.set(xlabel=f't(ch {second}) - t(ch {first}) [ns]', ylabel='Coincidences / bin',
                 title='CFD coincidence timing' if fit else 'CFD coincidence timing: no accepted fit')
        figure.tight_layout()
        figure.savefig(stem.with_suffix('.png'), dpi=160)
        plt.close(figure)
    calibration = solve_delays(pairs, channels, reference)
    for pair in pairs:
        first_delay = calibration['delays_ns'][str(pair['first'])]
        second_delay = calibration['delays_ns'][str(pair['second'])]
        pair['corrected_mean_ns'] = (pair['fit']['mean_ns'] - (second_delay - first_delay)
                                     if pair['fit'] and first_delay is not None and second_delay is not None else None)
    report = dict(input=str(parquet_path), time_column=time_column, diagnostics=diagnostics,
                  settings=dict(baseline_window=baseline_window, signal_window=signal_window,
                                polarity=polarity, sample_period_ns=period_ns, cfd_percent=fraction,
                                window_ns=window, bin_width_ns=float(edges[1] - edges[0]),
                                fit_window_ns=fit_window, sigma_max_ns=sigma_max,
                                min_counts=min_counts, min_amplitude_V=min_amplitude, min_snr=min_snr),
                  valid_hits_per_channel={str(channel): len(hits[channel][0]) for channel in channels},
                  pairs=pairs, calibration=calibration,
                  correction='corrected_time = first_sample_time + CFD_offset - delay_ns',
                  matching='Global greedy matching in timestamp order: each hit takes its nearest available hit from any other channel within the window; both are then removed from all candidates.',
                  resolution='Reported sigma is the pair width. Divide by sqrt(2) only for equal independent detectors.',
                  validation='Corrected means are in-sample closure diagnostics; offsets do not change individual pair widths.')
    (output_dir / 'timing_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    plot_timing_summary(output_dir, report)
    print(f"Timing: {sum(pair['fit'] is not None for pair in pairs)}/{len(pairs)} accepted pair fits; saved to {output_dir}")
    return report
