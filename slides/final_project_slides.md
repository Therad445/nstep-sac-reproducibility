# Final Project Slides Draft

## Slide 1 — Title

# Stress-Testing n-step Soft Actor-Critic under Limited Compute

**Bounded reproduction and ablation study of n-step SAC ideas**

Course final project  
Reinforcement Learning  
Radmir I.

Main message:

> I test whether n-step returns help SAC in a small-compute continuous-control setting, and where the horizon becomes unstable.

---

## Slide 2 — Motivation

# Why look at n-step SAC?

Many RL tasks have delayed credit assignment.

A useful action may affect reward several steps later. Standard 1-step TD targets are stable, but reward information moves backward slowly.

N-step returns are attractive because they can propagate reward information faster.

But SAC is off-policy, so this is not a free trick.

Key tension:

```text
short target  -> stable, but slower credit assignment
longer target -> faster reward propagation, but more bias/variance
```

---

## Slide 3 — SAC baseline

# Soft Actor-Critic in one slide

SAC is an off-policy actor-critic method for continuous control.

Core components:

- stochastic actor;
- twin Q-critics;
- target Q-networks;
- replay buffer;
- entropy regularization;
- optional automatic entropy tuning.

Standard critic target:

```text
r_t + gamma * (min Q_target(s_{t+1}, a_{t+1}) - alpha log pi(a_{t+1}|s_{t+1}))
```

Why SAC is a good baseline:

> It is already strong and stable, so improvements or failures are meaningful.

---

## Slide 4 — N-step returns

# From 1-step to n-step targets

Instead of bootstrapping after one reward, n-step targets accumulate several rewards:

```text
G_t^(n) = r_t + gamma r_{t+1} + ... + gamma^(n-1) r_{t+n-1}
          + gamma^n V(s_{t+n})
```

Expected benefit:

- faster reward propagation;
- better early learning;
- improved sample efficiency.

Possible cost:

- more off-policy bias;
- higher target variance;
- more sensitivity to horizon and hyperparameters.

---

## Slide 5 — Place in the literature

# SAC → SACn → T-SAC

This project is motivated by recent work on n-step SAC variants.

**SAC**  
Stable off-policy actor-critic baseline.

**SACn**  
Studies the difficulty of adding n-step returns to SAC and the off-policy correction problem.

**T-SAC / Chunking the Critic**  
Uses trajectory chunks, a sequence-aware critic, and aggregated n-step targets.

**This project**  
A smaller implementation-level study:

```text
SAC n=1 vs SAC n=3 vs SAC n=5
```

---

## Slide 6 — Project question

# Research question

> Do n-step returns improve SAC under limited compute, and how sensitive is the result to the horizon?

Compared variants:

| Method | Target |
|---|---|
| SAC n=1 | standard 1-step target |
| SAC n=3 | moderate n-step target |
| SAC n=5 | longer n-step target |

This is a bounded reproduction and ablation study.

Not a state-of-the-art claim.

---

## Slide 7 — Implementation

# What was implemented

The project contains a compact SAC training pipeline:

```text
src/train.py
src/rl/sac.py
src/rl/nstep_buffer.py
src/rl/replay_buffer.py
src/rl/actor.py
src/rl/critic.py
```

Training flow:

```text
environment step
-> n-step buffer
-> replay buffer
-> critic update
-> actor update
-> target update
-> deterministic evaluation
-> CSV log
```

For `n=1`, the same pipeline reduces to standard SAC.

---

## Slide 8 — Code explanation: n-step buffer

# Main algorithmic change

The n-step buffer stores a short queue of transitions.

For each ready transition, it builds:

```text
state      = s_t
action     = a_t
reward     = r_t + gamma r_{t+1} + ... + gamma^(n-1) r_{t+n-1}
next_state = s_{t+n}
discount   = gamma^n
done       = terminal flag inside the window
```

This aggregated transition is then sampled from the replay buffer like usual.

The critic target becomes:

```text
reward_n + discount_n * target_value(next_state)
```

---

## Slide 9 — Experimental setup

# Setup

Environments:

- `Pendulum-v1` — cheap sanity check;
- `HalfCheetah-v5` — main MuJoCo benchmark.

Training budget:

| Environment | Steps | Seed |
|---|---:|---:|
| Pendulum-v1 | 30,000 | 0 |
| HalfCheetah-v5 | 100,000 | 0 |

Machine:

- Ryzen 5700 laptop;
- WSL Ubuntu;
- CPU training.

Evaluation:

- deterministic policy;
- 5 evaluation episodes;
- periodic CSV logging.

---

## Slide 10 — Pendulum sanity check

# Pendulum-v1 results

Main observation:

- all variants train successfully;
- all variants reach similar final quality;
- `n=3` improves early, but the task is too simple for a strong conclusion.

Interpretation:

> Pendulum confirms that the implementation works and that n-step target construction does not immediately break SAC.

Use this slide with:

```text
results/plots/pendulum_learning_curves.png
```

---

## Slide 11 — HalfCheetah main result

# HalfCheetah-v5 ablation

| Method | Final eval return | Best eval return |
|---|---:|---:|
| SAC n=1 | 4354.49 | 4354.49 |
| SAC n=3 | 4578.91 | 4578.91 |
| SAC n=5 | 764.49 | 833.37 |

Main observation:

- `n=3` gives the best final return;
- `n=3` reaches strong performance earlier than `n=1`;
- `n=5` fails to learn a strong running policy.

Use this slide with:

```text
results/plots/halfcheetah_learning_curves.png
```

---

## Slide 12 — What the result means

# The horizon matters

The HalfCheetah result supports the expected trade-off.

Moderate horizon:

```text
n=3 -> better sample efficiency and better final return
```

Too long naive horizon:

```text
n=5 -> unstable or ineffective under same hyperparameters
```

Interpretation:

> N-step credit assignment can help, but longer is not automatically better.

This matches the motivation of SACn and T-SAC: n-step SAC needs careful handling.

---

## Slide 13 — Weaknesses

# Limitations

Main limitations:

- single seed;
- only two environments;
- no full T-SAC reproduction;
- no SACn-style correction;
- no separate hyperparameter tuning for each `n`;
- limited compute.

What I do not claim:

- not state of the art;
- not a general proof that `n=3` is always best;
- not a full reproduction of the original papers.

Correct claim:

> In this limited-compute setting, 3-step SAC improved HalfCheetah learning, while 5-step SAC failed under the same hyperparameters.

---

## Slide 14 — Conclusion

# Conclusion

What was done:

- implemented SAC with configurable n-step targets;
- ran sanity check on Pendulum;
- ran main ablation on HalfCheetah;
- analyzed sample efficiency and failure mode;
- documented reproducibility and limitations.

Main result:

```text
n=3 helped
n=1 remained strong
n=5 failed
```

Future work:

- more seeds;
- more MuJoCo environments;
- SACn-style correction;
- sequence-aware critic;
- longer training and confidence intervals.

Final message:

> N-step SAC is useful, but the horizon is a stability parameter, not just a bigger-is-better knob.
