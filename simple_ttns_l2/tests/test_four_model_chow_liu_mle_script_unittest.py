from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simple_ttns_l2.experiments.fit_limit_balanced_extreme_four_models import _config
from simple_ttns_l2.experiments.fit_limit_extreme_tt_vs_ttns_three_topologies_mle_8d import _write_report


class FourModelChowLiuMLEScriptTests(unittest.TestCase):
    def test_write_report_accepts_none_metrics(self):
        cfg = _config()
        summaries = {
            "ttde_tt_summary": {"final_val_nll": None, "finite_val_ll": -1.23, "total_time_sec": 10.0},
            "ttnsde_ttns_balanced_summary": {"final_val_nll": 1.5, "finite_val_ll": -1.1, "total_time_sec": 11.0},
            "ttnsde_ttns_chain_summary": {"final_val_nll": None, "finite_val_ll": None, "total_time_sec": 12.0},
            "ttnsde_ttns_chow_liu_summary": {"final_val_nll": 1.2, "finite_val_ll": -1.0, "total_time_sec": 9.5},
        }
        rows = [
            {
                "dim_i": 0,
                "dim_j": 1,
                "noise_floor": 0.01,
                "iae_ttde_tt": 0.11,
                "iae_ttnsde_balanced": 0.12,
                "iae_ttnsde_chain": 0.13,
                "iae_ttnsde_chow_liu": 0.10,
                "ratio_ttde_tt": 11.0,
                "ratio_ttnsde_balanced": 12.0,
                "ratio_ttnsde_chain": 13.0,
                "ratio_ttnsde_chow_liu": 10.0,
            }
        ]

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report.md"
            _write_report(
                path=out,
                cfg=cfg,
                summaries=summaries,
                rows=rows,
                svg_name="dummy.svg",
                target_topology="balanced",
                target_parent=[0, 0, 0, 1, 1, 2],
                chow_liu_parent=[0, 0, 0, 1, 1, 2],
                chow_liu_edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5]],
                target_spec=None,
            )
            text = out.read_text(encoding="utf-8")
            self.assertIn("chow_liu_parent", text)
            self.assertIn("ttnsde_chow_liu", text)
            self.assertIn("NA", text)


if __name__ == "__main__":
    unittest.main()
