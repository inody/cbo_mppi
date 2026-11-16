# Frozen public inputs

These compact inputs support the fast figure-replay workflow. They are outputs
of the final numerical campaigns, not substitutes for the full simulation
workflow.

- `figures/closed_loop_figure_trajectories.npz` contains all 100 realized state
  trajectories used in the closed-loop trajectory and posture figures.
- `figures/navigation/route_heights.npz` contains the route-height trajectory
  family and task geometry for the Navigation open-loop figure.
  `free_energy_landscape.npz` contains the corresponding nominal-cost and
  free-energy curves from the independently verified full reproduction.
- `figures/pendulum/landscape_tau10.npz` contains the evaluated Pendulum
  landscape grid.
- `figures/planar_arm/landscape_arrays.npz` contains the evaluated Planar Arm
  posture family and landscape arrays. `frozen_design.json` contains only the
  task geometry required by the plotter.
- `figures/horizon/figure_input.csv` contains the aggregated horizon-sweep
  values shown in the manuscript figure.

`manifest.json` records the size and SHA-256 digest of every frozen input. The
full workflow rebuilds equivalent numerical inputs under
`artifacts/full-run/assembled/figures/` from an empty result directory.
