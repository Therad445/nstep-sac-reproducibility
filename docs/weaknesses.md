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

## Experimental limitations observed so far

The HalfCheetah ablation gives a useful signal, but it is still a single-seed experiment. The current results should not be used to claim that 3-step SAC is generally better than 1-step SAC across all tasks.

The 5-step run performed poorly. There are several possible explanations:

- the longer target may increase off-policy bias;
- accumulated rewards may increase target variance;
- the same learning rate and entropy settings may not be equally suitable for all horizons;
- one seed is not enough to separate a robust failure from an unlucky run.

For this reason, the strongest conclusion is not "n=3 is always best", but rather:

> In this limited-compute setting, moderate n-step targets improved learning, while longer naive n-step targets were much less stable.

## Reproduction limitations

This project is a bounded reproduction. It does not attempt to fully reproduce the original paper-scale T-SAC experiments. The main experiment focuses on the n-step target mechanism and compares several horizons in a compact setting.

## What would strengthen the project later

- 5 to 10 seeds per method;
- more MuJoCo environments;
- comparison with an official T-SAC run;
- SACn-style correction;
- confidence intervals and statistical tests;
- Docker image or locked environment file.
