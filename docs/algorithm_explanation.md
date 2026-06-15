# Algorithm Explanation

## Standard SAC update

SAC trains a stochastic actor and two Q-critics. The critic estimates the soft value of a state-action pair. The actor is trained to choose actions with high Q-value while keeping enough entropy.

For a one-step target, the critic target has the form:

```text
y = r + gamma * (min(Q1_target, Q2_target) - alpha * log_prob)
```

The twin critics are used to reduce overestimation. The target networks are updated slowly through Polyak averaging.

## N-step target

For an n-step target, instead of using only one reward before bootstrapping, we accumulate several rewards:

```text
y = r_t + gamma*r_{t+1} + ... + gamma^(n-1)*r_{t+n-1}
    + gamma^n * bootstrap_value(s_{t+n})
```

If the episode terminates earlier, the bootstrap part is removed.

## Why it matters

The idea is attractive because rewards can affect earlier decisions faster. But SAC is off-policy, and the replay buffer contains transitions from older policies. Because of this, a larger n-step horizon may add bias or variance. This is the main reason why the ablation is interesting.

## What the code should show

The implementation should make the following path clear:

```text
environment step
-> store transition
-> n-step buffer builds aggregated transition
-> replay buffer samples batch
-> critic target uses n-step reward and n-step discount
-> actor and critic are updated
```

This path is also the main code explanation part for the presentation.
