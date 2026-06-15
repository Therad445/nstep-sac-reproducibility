# Literature Map

## Project position

This project sits between standard Soft Actor-Critic and recent work on n-step or sequence-aware SAC variants.

The main question is practical:

> If n-step returns are useful for credit assignment, how much can they help SAC in a limited-compute continuous-control setting, and where does the method become unstable?

This is not a new algorithm paper. It is a small reproduction and ablation study focused on understanding the mechanism.

## 1. Soft Actor-Critic as the baseline

Soft Actor-Critic (SAC) is an off-policy actor-critic algorithm for continuous control. It learns:

- an actor policy;
- two Q-functions;
- a target critic;
- optionally an automatically tuned entropy coefficient.

The important property of SAC is entropy regularization. The policy is optimized not only to maximize return, but also to keep enough entropy. This makes exploration more stable and often improves robustness.

In the standard implementation, the critic is usually trained with a 1-step target:

```text
target = r_t + gamma * (min Q_target(s_{t+1}, a_{t+1}) - alpha * log pi(a_{t+1}|s_{t+1}))
```

This target is stable and simple, but it propagates reward information one step at a time.

## 2. Why n-step returns are attractive

N-step returns replace the 1-step target with a target that accumulates several rewards before bootstrapping:

```text
G_t^(n) = r_t + gamma r_{t+1} + ... + gamma^(n-1) r_{t+n-1}
          + gamma^n V(s_{t+n})
```

This can improve credit assignment. If the reward is delayed or the task has long-horizon structure, a longer target can move useful information backward faster.

In practice, this means that n-step targets may improve sample efficiency: the agent may learn a good policy with fewer environment steps.

## 3. Why n-step SAC is not trivial

The difficulty is that SAC is off-policy. Samples in the replay buffer were collected by older policies, while the current actor may be different.

With longer n-step targets, the algorithm uses longer trajectory fragments from the replay buffer. This can increase:

- off-policy bias;
- target variance;
- sensitivity to the chosen horizon;
- sensitivity to learning rate, entropy temperature and replay distribution.

A short n-step horizon may help, but a longer one is not automatically better. This is the key trade-off tested in this project.

## 4. SACn

SACn focuses directly on the problem of using n-step returns in SAC. The main motivation is that naive n-step returns can be biased in an off-policy setting. Importance sampling can correct part of this mismatch, but it may also become unstable.

For this project, SACn is important because it gives a theoretical and practical reason to be careful:

> n-step SAC can help, but the off-policy nature of SAC makes the horizon choice and correction mechanism important.

The current implementation does not reproduce full SACn. Instead, it tests the naive n-step mechanism as a compact ablation.

## 5. T-SAC / Chunking the Critic

The T-SAC paper proposes a more structured way to use temporal context. Instead of evaluating only a single state-action pair, the critic is conditioned on trajectory chunks. A lightweight Transformer critic can use a short history of transitions, and the method combines this with aggregated n-step targets.

The idea is relevant because it treats credit assignment as a critic-side temporal modeling problem. This is stronger than simply increasing the n-step horizon.

For this project, T-SAC is used as motivation and literature context. A full transformer-critic reproduction would require more compute and more careful engineering. The implemented experiment focuses on the smaller mechanism that both SACn and T-SAC care about: the effect of n-step targets.

## 6. How this project fits the literature

The project can be summarized as:

```text
SAC baseline
    -> stable 1-step off-policy actor-critic

n-step returns
    -> faster reward propagation, but more bias/variance

SACn
    -> studies how to make n-step SAC more correct and stable

T-SAC
    -> uses sequence-aware critic modeling and n-step targets

This project
    -> small-compute ablation of naive n-step SAC: n=1 vs n=3 vs n=5
```

The main value is not a new method, but an implementation-level check of the trade-off.

## 7. Current experimental reading

The experiments match the expected trade-off.

On `Pendulum-v1`, all variants converge, which suggests that the implementation works and the task is not hard enough to separate the methods strongly.

On `HalfCheetah-v5`, the difference is clear:

- `n=3` gives the best final return and reaches high performance earlier;
- `n=1` is a strong baseline but learns more slowly;
- `n=5` performs poorly and does not learn a strong running policy.

This result supports the literature motivation: n-step credit assignment can be useful, but longer naive targets can be unstable in off-policy SAC.
