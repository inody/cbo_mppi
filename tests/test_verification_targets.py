import json
from pathlib import Path

from cbo_mppi.reproduction.verify import _check

ROOT = Path(__file__).resolve().parents[1]


def test_verification_targets_are_predeclared_and_complete():
    targets = json.loads((ROOT / "data/reference/verification_targets.json").read_text())
    assert targets["policy"]["frozen_before_fresh_recomputation"] is True
    assert targets["policy"]["timing_excluded"] is True
    assert len(targets["nominal"]) == 32
    assert len(targets["sensitivity"]) == 88
    assert len(targets["horizon"]) == 66


def test_tolerance_boundary_is_not_rejected_by_float_roundoff():
    report = []
    _check(report, "boundary", 0.5, 0.35, 0.15)
    assert report[0]["pass"] is True
