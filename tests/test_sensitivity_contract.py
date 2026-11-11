from cbo_mppi.reproduction.nominal import BUDGET, TASKS
from cbo_mppi.reproduction.sensitivity import SEEDS, cells


def test_sensitivity_inventory():
    assert len(SEEDS) == 20
    assert len(cells()) == 88
    for task in TASKS:
        task_cells = cells(task)
        assert len(task_cells) == 22
        assert len({cell.cell_id for cell in task_cells}) == 22
        assert all(cell.rollout_budget == BUDGET[task] for cell in task_cells)
