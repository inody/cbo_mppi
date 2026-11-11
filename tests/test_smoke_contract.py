from cbo_mppi.reproduction.smoke import jobs


def test_smoke_covers_all_task_method_pairs():
    pairs = {(cell.task, cell.method) for cell, _ in jobs()}
    assert len(jobs()) == 16
    assert len(pairs) == 16
