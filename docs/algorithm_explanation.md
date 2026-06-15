# Algorithm Explanation

## Goal of the implementation

The implementation is a compact Soft Actor-Critic training pipeline with a configurable n-step return horizon.

The main purpose is not to provide a production-level RL library. The goal is to make the algorithm clear enough for a course project and flexible enough for the ablation:

```text
SAC n=1
SAC n=3
SAC n=5
```

The core files are:

```text
src/train.py
src/rl/sac.py
src/rl/replay_buffer.py
src/rl/nstep_buffer.py
src/rl/actor.py
src/rl/critic.py
```

## SAC components

The implementation uses the standard SAC structure:

1. **Actor**  
   The actor outputs a squashed Gaussian policy. Actions are sampled through a tanh transformation so that they fit the environment action bounds.

2. **Twin critics**  
   Two Q-functions are trained in parallel. The target uses the minimum of the two critics to reduce overestimation.

3. **Target critics**  
   Target Q-networks are updated with Polyak averaging.

4. **Entropy coefficient**  
   The implementation supports automatic entropy tuning. The coefficient `alpha` controls how strongly the policy is encouraged to keep entropy.

5. **Replay buffer**  
   Transitions are stored and sampled randomly, which makes the method off-policy.

## Training loop

At a high level, the training loop is:

```text
for each environment step:
    select action
    step environment
    store transition through n-step buffer
    add ready n-step transition to replay buffer

    if enough data is collected:
        sample batch from replay buffer
        update critic
        update actor
        update alpha
        update target critic

    periodically:
        evaluate deterministic policy
        write metrics to CSV
```

The important part for this project is that the replay buffer does not always receive a raw one-step transition. It receives a transition after n-step return aggregation.

## N-step transition construction

The n-step buffer keeps a short queue of recent transitions:

```text
(s_t, a_t, r_t, s_{t+1}, done_t)
```

When the queue contains enough transitions, it builds an aggregated transition:

```text
state      = s_t
action     = a_t
reward     = r_t + gamma r_{t+1} + ... + gamma^(n-1) r_{t+n-1}
next_state = s_{t+n}
done       = whether the episode ended inside the n-step window
discount   = gamma^k
```

Here `k` is the number of actually used steps. If an episode ends earlier than `n`, the target stops at the terminal transition.

For `n=1`, this reduces to the standard SAC transition.

## Critic update

The critic target is computed from the aggregated n-step reward:

```text
target = reward_n + discount_n * (min Q_target(next_state, next_action) - alpha * log_prob(next_action))
```

If the transition is terminal, the bootstrap part is removed.

The critic loss is the mean squared error between the current Q-values and the target:

```text
critic_loss = MSE(Q1(state, action), target) + MSE(Q2(state, action), target)
```

This is the part where n-step targets directly affect learning. A larger `n` changes the reward scale and the amount of information included before bootstrapping.

## Actor update

The actor is updated using the current critics:

```text
actor_loss = alpha * log_prob(action) - min Q(state, action)
```

The actor tries to choose actions with high Q-values while still keeping enough entropy.

The actor update itself does not explicitly depend on `n`, but it depends indirectly because the critics were trained with n-step targets.

## Alpha update

When automatic entropy tuning is enabled, `alpha` is optimized so that the policy entropy stays close to a target entropy. This is useful because too little entropy may reduce exploration, while too much entropy may slow down convergence.

In the experiments, `alpha` is logged together with return and losses.

## Why n-step can help

A 1-step target propagates reward information slowly. A 3-step target can move reward information backward faster and may improve sample efficiency.

This is visible in the HalfCheetah experiment: the 3-step variant reaches strong returns earlier than the 1-step baseline.

## Why n-step can fail

Longer n-step targets are not automatically better.

In off-policy SAC, the sampled trajectory fragment was produced by an older policy. The longer the fragment, the more the target may reflect outdated behavior. The accumulated reward also increases target variance.

This is visible in the 5-step HalfCheetah run: it does not learn a strong running policy under the same hyperparameters. The most likely interpretation is that `n=5` is too aggressive for this simple naive n-step implementation and this compute setting.

## Implementation choice

The project intentionally keeps the implementation simple:

- no transformer critic;
- no full SACn correction mechanism;
- same hyperparameters for `n=1`, `n=3`, `n=5`;
- single-seed runs.

This makes the ablation easier to interpret. If one horizon works better than another, the difference is not hidden behind many extra changes.

At the same time, this is also a limitation. A stronger follow-up study should tune each horizon separately and run more seeds.
