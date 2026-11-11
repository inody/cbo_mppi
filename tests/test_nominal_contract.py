import math

from cbo_mppi.reproduction.nominal import (
    ALPHA,
    BUDGET,
    HORIZON,
    SEEDS,
    STEPS,
    TASKS,
    cells,
    controller_config,
)


def test_nominal_campaign_inventory_and_equal_budgets():
    all_cells = cells()
    assert len(all_cells) == 28
    for task in TASKS:
        task_cells = cells(task)
        assert len(task_cells) == 7
        assert len(SEEDS[task]) == 100
        assert all(cell.rollout_budget == BUDGET[task] for cell in task_cells)


def test_canonical_cbo_mpc_occurs_once_per_task():
    dimensions = {
        "navigation": 120,
        "many_obstacles": 100,
        "pendulum": 80,
        "planar_arm": 60,
    }
    for task in TASKS:
        cbo = [cell for cell in cells(task) if cell.method == "cbo_mpc"]
        assert len(cbo) == 1
        assert math.isclose(cbo[0].sigma_cbo, math.sqrt(1.0 / dimensions[task]))
        assert controller_config(cbo[0]).alpha == ALPHA[task]


def test_final_planar_arm_contract_is_integrated_directly():
    assert HORIZON["planar_arm"] == 20
    assert STEPS["planar_arm"] == 35
    cbo = next(cell for cell in cells("planar_arm") if cell.method == "cbo_mpc")
    assert controller_config(cbo).alpha == 1.0
