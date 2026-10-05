import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from decode_parquet_awkward import waveform_features
from timing_analysis import analyze_timing, fit_peak, pair_deltas, solve_delays

PLOT_OUTPUT_DIR = Path(__file__).parent / 'plots' / 'run018' / 'timing' / 'test_outputs'


class TimingAnalysisTest(unittest.TestCase):
    def assert_summary_plots(self, output_dir):
        for name in ('timing_resolution_heatmap', 'timing_resolution_by_pair',
                     'timing_coincidence_peak', 'timing_channel_delays'):
            with self.subTest(plot=name):
                path = output_dir / f'{name}.png'
                self.assertTrue(path.is_file())
                self.assertEqual(path.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')

    def test_precision_and_nearest_assignment(self):
        first = np.array([10**18, 10**18 + 100000], dtype=np.int64).astype(np.longdouble)
        second = first + 7
        result = pair_deltas(first / 1000, np.array([0.123, 0.123]),
                             second / 1000, np.array([0.125, 0.125]), 1)
        np.testing.assert_allclose(result, [0.009, 0.009], atol=0.0001)
        result = pair_deltas(np.array([0, 10], dtype=np.longdouble), np.zeros(2),
                             np.array([5, 9, 30], dtype=np.longdouble), np.zeros(3), 6)
        np.testing.assert_array_equal(result, [5, -1])

    def test_delay_sign_and_disconnected_channel(self):
        pairs = [dict(first=first, second=second,
                      fit=dict(mean_ns=mean, mean_error_ns=0.01))
                 for first, second, mean in [(0, 1, 2), (1, 2, -3), (0, 2, -1)]]
        result = solve_delays(pairs, [0, 1, 2, 3], 1)
        self.assertAlmostEqual(result['delays_ns']['0'], -2)
        self.assertAlmostEqual(result['delays_ns']['2'], -3)
        self.assertIsNone(result['delays_ns']['3'])
        self.assertAlmostEqual(result['chi2'], 0)
        self.assertEqual(result['ndof'], 1)

    def test_fit_known_width_and_empty_histogram(self):
        generator = np.random.default_rng(42)
        edges = np.linspace(-5, 5, 1001)
        counts = np.histogram(generator.normal(1.25, 0.08, 15000), edges)[0]
        fit = fit_peak(counts, edges, 0.6, 0.3, 50)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit['mean_ns'], 1.25, delta=0.003)
        self.assertAlmostEqual(fit['sigma_ns'], 0.08, delta=0.003)
        self.assertIsNone(fit_peak(np.zeros(1000), edges, 0.6, 0.3, 50))

    def test_parquet_pipeline_across_batches(self):
        generator = np.random.default_rng(123)
        events = 2000
        timestamps = np.repeat(np.arange(events, dtype=np.int64) * 100000 + 10**18, 2)
        timestamps[1::2] += np.rint(generator.normal(1500, 80, events)).astype(np.int64)
        waveform = [0.99, 1, 1.01, 1, 2, 3, 4, 3, 2, 1]
        table = pa.table(dict(Channel=np.tile([0, 1], events),
                              FirstSampleTime_in_ps=timestamps,
                              FirstSampleTime_in_ps_fine=np.zeros(events * 2),
                              DataSize=np.full(events * 2, 10), Baseline=np.ones(events * 2),
                              DataSample=[waveform] * (events * 2)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'hits.parquet'
            pq.write_table(table, path, row_group_size=311)
            output_dir = PLOT_OUTPUT_DIR / 'parquet_pipeline'
            result = analyze_timing(path, output_dir, waveform_features,
                                    baseline_window=(0, 3), signal_window=(3, 10),
                                    period_ns=0.5, step_size=117, window=5, fit_window=0.6)
            self.assertEqual(result['diagnostics']['selected_hits'], events * 2)
            self.assertAlmostEqual(result['calibration']['delays_ns']['1'], 1.5, delta=0.01)
            self.assertAlmostEqual(result['pairs'][0]['pair_resolution_ps'], 80, delta=6)
            self.assertAlmostEqual(result['pairs'][0]['corrected_mean_ns'], 0)
            self.assertTrue((output_dir / 'timing_summary.json').is_file())
            self.assert_summary_plots(output_dir)

    def test_fallback_rollover_is_rejected_across_batches(self):
        waveform = [0.99, 1, 1.01, 1, 2, 3, 4, 3, 2, 1]
        table = pa.table(dict(Channel=[0, 1, 0, 1],
                              OrderedCell0Time=[3e9, 3e9 + 1, 0., 1.],
                              DataSize=[10] * 4, Baseline=[1.] * 4,
                              DataSample=[waveform] * 4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'hits.parquet'
            pq.write_table(table, path)
            with self.assertRaisesRegex(ValueError, 'rollover/reset'):
                analyze_timing(path, Path(directory) / 'results', waveform_features,
                               baseline_window=(0, 3), signal_window=(3, 10),
                               period_ns=0.5, step_size=2)

    def test_quality_cut_leaves_no_fabricated_calibration(self):
        waveform = [0.99, 1, 1.01, 1, 2, 3, 4, 3, 2, 1]
        table = pa.table(dict(Channel=[0, 1], OrderedCell0Time=[100., 101.],
                              DataSize=[10, 10], Baseline=[1., 1.],
                              DataSample=[waveform, waveform]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'hits.parquet'
            pq.write_table(table, path)
            output_dir = PLOT_OUTPUT_DIR / 'quality_cut'
            result = analyze_timing(path, output_dir, waveform_features,
                                    baseline_window=(0, 3), signal_window=(3, 10),
                                    period_ns=0.5, min_amplitude=5, channels=[0, 1, 2])
            self.assertEqual(result['diagnostics']['selected_hits'], 0)
            self.assertTrue(all(pair['fit'] is None for pair in result['pairs']))
            self.assertIsNone(result['calibration']['delays_ns']['1'])
            self.assertIsNone(result['calibration']['delays_ns']['2'])
            self.assert_summary_plots(output_dir)


if __name__ == '__main__':
    unittest.main()
