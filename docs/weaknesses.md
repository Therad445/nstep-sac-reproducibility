# Weaknesses and Limitations

## Method-level limitations

N-step returns can improve credit assignment, but in off-policy learning they can also create distribution mismatch. The replay buffer contains trajectories from older policies, while the current actor and critic are updated using those stored trajectories.

A larger horizon is therefore not automatically better. It may propagate useful rewards faster, but it may also increase variance and instability.

## Experiment-level limitations

The project uses a limited compute budget. This means:

- fewer environments than in full papers;
- fewer seeds;
- shorter training runs;
- weaker statistical claims;
- possible sensitivity to hyperparameters.

## Reproduction limitations

This project is a bounded reproduction. It does not attempt to fully reproduce the original paper-scale T-SAC experiments. The main experiment focuses on the n-step target mechanism and compares several horizons in a compact setting.

## What would strengthen the project later

- 5 to 10 seeds per method;
- more MuJoCo environments;
- comparison with an official T-SAC run;
- SACn-style correction;
- confidence intervals and statistical tests;
- Docker image or locked environment file.
