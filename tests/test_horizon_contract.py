from cbo_mppi.reproduction.horizon import HORIZONS, SEEDS, TASKS, cells
from cbo_mppi.reproduction.nominal import BUDGET


def test_horizon_inventory_and_budgets():
    assert len(SEEDS) == 20
    assert len(cells()) == 66
    for task in TASKS:
        task_cells = cells(task)
        assert len(task_cells) == 3 * len(HORIZONS[task])
        assert len({cell.cell_id for cell in task_cells}) == len(task_cells)
        assert all(cell.rollout_budget == BUDGET[task] for cell in task_cells)
