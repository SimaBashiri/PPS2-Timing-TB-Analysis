import unittest

import numpy as np
import pyarrow as pa

from decode_parquet_awkward import waveform_features
from pulse_measurements import sample_period_ns, histogram_mpv


def batch(waves, sizes=None):
    return pa.record_batch({
        "DataSample": pa.array(waves, type=pa.list_(pa.float64())),
        "DataSize": sizes if sizes is not None else [len(w) if w is not None else 0 for w in waves],
        "Baseline": [1.] * len(waves),
    })


class PulseMeasurementsTest(unittest.TestCase):
    def features(self, waves, **kwargs):
        return waveform_features(batch(waves), (0, 3), (3, 10),
                                 period_ns=0.5, **kwargs)

    def test_triangle_and_polarity(self):
        wave = [0.9, 1., 1.1, 1., 2., 3., 4., 3., 2., 1.]
        for sign in (1, -1):
            f = self.features([1 + sign * (np.array(wave) - 1)],
                              polarity="positive" if sign == 1 else "negative",
                              voltage_thresholds=[1.5])
            expected = {"BaselineEstimate": 1, "NoiseRMS": 0.1,
                        "PulseAmplitude": 3, "SNR": 30, "SignalArea": 4.5,
                        "PeakTime": 3, "RiseTime10_90": 1.2,
                        "RiseTime20_80": 0.9, "FallTime90_10": 1.2,
                        "FWHM": 1.5, "ToT_10pct": 2.7,
                        "ToT_20pct": 2.4, "ToT_60pct": 1.2,
                        "ToT_1p5V": 1.5, "LeadingTime_50pct": 2.25}
            for key, value in expected.items():
                self.assertAlmostEqual(f[key][0], value, msg=key)

    def test_invalid_and_truncated(self):
        base = [1., 1., 1., 1., 2., 3., 4., 3., 2., 1.]
        missing = base.copy()
        missing[7] = np.nan
        bad_baseline = base.copy()
        bad_baseline[1] = np.nan
        clipped_tail = base.copy()
        clipped_tail[-2:] = [3., 3.]
        f = self.features([base, missing, bad_baseline, clipped_tail, None, base[:8]])
        self.assertTrue(np.isnan(f["SNR"][0]))  # zero noise
        for i in (1, 2, 4, 5):
            self.assertTrue(np.isnan(f["SignalArea"][i]))
            self.assertTrue(np.isnan(f["FWHM"][i]))
        self.assertTrue(np.isnan(f["ToT_10pct"][3]))
        self.assertTrue(np.isfinite(f["LeadingTime_10pct"][3]))

    def test_nearest_peak_excursion(self):
        wave = [1., 1., 1., 1., 3., 1., 5., 1., 1., 1.]
        f = self.features([wave])
        self.assertAlmostEqual(f["FWHM"][0], 0.5)
        self.assertEqual(f["Multiple50pctExcursions"][0], 1)

    def test_metadata_and_mpv(self):
        self.assertEqual(sample_period_ns({b"sampling_frequency": b"6400 MS/s"}), 0.15625)
        self.assertEqual(sample_period_ns({}, 0.25), 0.25)
        for metadata in ({}, {b"sampling_frequency": b"0 MS/s"}):
            with self.assertRaises(ValueError):
                sample_period_ns(metadata)
        self.assertIsNone(histogram_mpv([1, 0], [0, 1, 2])["value_V"])
        self.assertEqual(histogram_mpv([2, 10], [0, 1, 2])["value_V"], 1.5)


if __name__ == "__main__":
    unittest.main()
