"""CPU-only checks for the GPU result gate; no GPU or PyTorch needed."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import report
from run_suite import PERFORMANCE_BYTES, PERFORMANCE_SHAPE, metric, performance_summary


class PerformanceGateTests(unittest.TestCase):
    def test_invalid_timings_are_rejected(self):
        for timing in ("nan", "inf", "0", "-1"):
            with self.subTest(timing=timing), self.assertRaises(ValueError):
                metric(
                    f"CAESAR_BENCHMARK compression {timing} {PERFORMANCE_BYTES}",
                    "compression",
                    PERFORMANCE_BYTES,
                )
        with self.assertRaises(ValueError):
            metric("CAESAR_BENCHMARK compression 1 4", "compression", PERFORMANCE_BYTES)
        with self.assertRaises(ValueError):
            metric("", "compression", PERFORMANCE_BYTES)

    def test_fastest_of_five_and_thresholds(self):
        samples = {
            "compression": [{"gb_per_second": value} for value in [1, 1, 1.3, 1, 1]],
            "decompression": [{"gb_per_second": value} for value in [3, 3, 3, 3, 3.5]],
        }
        self.assertTrue(
            all(value["passed"] for value in performance_summary(samples).values())
        )
        samples["compression"][2]["gb_per_second"] = 1.29
        self.assertFalse(performance_summary(samples)["compression"]["passed"])
        samples["compression"].pop()
        with self.assertRaises(ValueError):
            performance_summary(samples)

    def test_report_recalculates_rates_instead_of_trusting_summary(self):
        result = {
            "sha": "a" * 40,
            "backend": "nvidia",
            "mode": "performance",
            "passed": True,
            "tests": [{"passed": True}],
            "shape": PERFORMANCE_SHAPE,
            "error_bound": 0.1,
            "performance": {
                operation: {"runs": [{"seconds": 100, "gb_per_second": 999}] * 5}
                for operation in ("compression", "decompression")
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(result))
            argv = ["report.py", str(path), "--repo", "owner/repo", "--dry-run"]
            with patch("sys.argv", argv), self.assertRaises(ValueError):
                report.main()

    def test_report_uses_tested_sha_and_dry_run_does_not_publish(self):
        result = {
            "sha": "b" * 40,
            "backend": "amd",
            "mode": "correctness",
            "passed": True,
            "tests": [{"passed": True}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(result))
            output = io.StringIO()
            with patch(
                "sys.argv",
                ["report.py", str(path), "--repo", "owner/repo", "--dry-run"],
            ), patch("subprocess.run") as publish, contextlib.redirect_stdout(output):
                report.main()
            publish.assert_not_called()
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["sha"], result["sha"])
            self.assertEqual(payload["context"], "gpu-tests/amd/correctness")


if __name__ == "__main__":
    unittest.main()
