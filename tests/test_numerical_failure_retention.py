import json

import numpy as np

from cbo_mppi.reproduction.evaluate import evaluate_external


def test_external_evaluation_retains_nonfinite_trial(tmp_path):
    source = tmp_path / "trajectory.npz"
    states = np.zeros((81, 4), dtype=np.float32)
    states[79, 0] = np.nan
    controls = np.zeros((80, 2), dtype=np.float32)
    np.savez_compressed(source, states=states, controls=controls)

    destination = evaluate_external(
        "sensitivity", "navigation", "test-cell", 71000, 15.0,
        source, tmp_path, cell_metadata={"test": True},
    )
    payload = json.loads(destination.read_text())
    assert payload["metric_status"] == "completed_with_numerical_failure"
    assert payload["first_nonfinite_state_step"] == 79
    assert payload["F_CL"] is None
    assert payload["J_CL"] is None
