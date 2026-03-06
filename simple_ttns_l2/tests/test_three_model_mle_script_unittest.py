from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simple_ttns_l2.experiments.fit_limit_balanced_extreme_four_models import _config
from simple_ttns_l2.experiments.fit_limit_chain_extreme_tt_vs_ttns_two_topologies_mle_8d import (
    _get_target_topology_from_env,
    _write_report,
)


class ThreeModelMLEScriptTests(unittest.TestCase):
    def test_get_target_topology_from_env(self):
        with patch.dict(os.environ, {}, clear=False):
            self.assertEqual(_get_target_topology_from_env(), "chain")

        with patch.dict(os.environ, {"TARGET_TOPOLOGY": "balanced"}, clear=False):
            self.assertEqual(_get_target_topology_from_env(), "balanced")

        with patch.dict(os.environ, {"TARGET_TOPOLOGY": "invalid"}, clear=False):
            with self.assertRaises(ValueError):
                _get_target_topology_from_env()

    def test_write_report_accepts_none_metrics(self):
        cfg = _config()
        tt_summary = {"final_val_nll": None, "finite_val_ll": -1.23, "total_time_sec": 10.0}
        ttns_bal_summary = {"final_val_nll": 1.5, "finite_val_ll": -1.1, "total_time_sec": 11.0}
        ttns_chain_summary = {"final_val_nll": None, "finite_val_ll": None, "total_time_sec": 12.0}
        rows = [
            {
                "dim_i": 0,
                "dim_j": 1,
                "noise_floor": 0.01,
                "iae_ttde_tt": 0.11,
                "iae_ttnsde_balanced": 0.12,
                "iae_ttnsde_chain": 0.13,
                "ratio_ttde_tt": 11.0,
                "ratio_ttnsde_balanced": 12.0,
                "ratio_ttnsde_chain": 13.0,
            }
        ]

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report.md"
            _write_report(
                path=out,
                cfg=cfg,
                tt_summary=tt_summary,
                ttns_bal_summary=ttns_bal_summary,
                ttns_chain_summary=ttns_chain_summary,
                rows=rows,
                svg_name="dummy.svg",
                target_topology="balanced",
            )
            text = out.read_text(encoding="utf-8")
            self.assertIn("Complex Balanced Target", text)
            self.assertIn("NA", text)


if __name__ == "__main__":
    unittest.main()
