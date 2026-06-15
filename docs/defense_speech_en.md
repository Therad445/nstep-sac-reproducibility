# Defense Speech (English, 5-7 minutes)

Hello everyone. Today I will present my project **Stress-Testing n-step Soft Actor-Critic under Limited Compute**.

The goal of this project was not to propose a new state-of-the-art reinforcement learning algorithm. Instead, I wanted to take one concrete idea from recent n-step SAC work and test it in a small but reproducible experiment.

The main question was: **do n-step returns help Soft Actor-Critic under a limited compute budget, and where does increasing the horizon start to hurt?**

Soft Actor-Critic is a strong off-policy actor-critic algorithm for continuous-control tasks. It uses a replay buffer, a stochastic actor, twin critics, target networks, and entropy regularization. In the standard version, the critic is usually trained with a one-step target. We take one reward and then bootstrap from the next state.

This is stable, but it has a credit-assignment limitation: reward information moves backward only one step at a time. If a useful action affects reward several steps later, a one-step target may learn less sample-efficiently.

N-step returns propose a simple idea: instead of bootstrapping after one reward, we accumulate several rewards first. For example, with `n=3`, the target includes `r_t`, `gamma r_{t+1}`, `gamma^2 r_{t+2}`, and only then bootstraps from `s_{t+3}`. Intuitively, this should propagate reward information backward faster.

However, SAC is off-policy. The trajectory fragments in the replay buffer may have been collected by older policies. So the longer the n-step fragment is, the higher the risk of off-policy bias and target variance. This means that the n-step horizon is not simply a bigger-is-better parameter. It is also a stability parameter.

This is related to recent work such as SACn and T-SAC. SACn is motivated by the difficulty of using n-step returns in off-policy SAC. T-SAC goes further and uses trajectory chunks and a sequence-aware critic. I did not reproduce the full T-SAC model because it would require more engineering and more compute. My project is a bounded reproduction and ablation study of the central mechanism: what happens if we change the SAC critic target horizon?

I compared three variants.

`SAC n=1` is the standard baseline.  
`SAC n=3` is a moderate n-step target.  
`SAC n=5` is a longer and potentially riskier target.

The implementation keeps the standard SAC pipeline. The main change happens before the replay buffer. I added an n-step buffer that stores a short queue of transitions and builds an aggregated transition. It computes the accumulated reward, the discount, the next state, and the done flag. If the episode terminates early, the n-step window is cut early. Then this aggregated transition is stored in the replay buffer, and the critic update uses the target:

`reward_n + discount_n * target_value(next_state)`.

For `n=1`, this reduces to standard SAC.

I used two environments. `Pendulum-v1` was used as a cheap sanity check. `HalfCheetah-v5` was used as the main continuous-control benchmark.

On `Pendulum-v1`, I ran `n=1`, `n=3`, and `n=5` for 30 thousand environment steps. All three variants trained successfully and reached similar final quality. I do not use this as the main evidence. It mainly shows that the implementation works and that n-step target construction does not immediately break SAC on a simple environment.

The main result is on `HalfCheetah-v5`. I ran all three variants for 100 thousand environment steps with seed 0.

The final evaluation return for `SAC n=1` was about **4354**.  
For `SAC n=3`, it was about **4579**.  
For `SAC n=5`, it was only about **764**, with a best return of about **833**.

So in this experiment, `n=3` performed best. It reached strong performance earlier and had the best final return. `n=1` remained a strong baseline and also learned a good policy. But `n=5` almost failed to learn a proper running behavior.

For me, the most interesting result is not only that `n=3` was better. The most interesting result is the failure of `n=5`. It shows that increasing the n-step horizon can make off-policy SAC unstable if we do it naively and without additional correction mechanisms.

I intentionally kept the same hyperparameters for `n=1`, `n=3`, and `n=5`. This makes the comparison cleaner because the main changed variable is the critic target horizon. Of course, this may be unfair to `n=5`, because it might need a different learning rate or entropy settings. But this is also part of the point: the horizon is sensitive and not a free improvement.

The main conclusion is:

**Moderate n-step credit assignment can help SAC sample efficiency, but a longer naive n-step target can become unstable or ineffective in off-policy SAC.**

Or shorter: **the n-step horizon is a stability parameter, not a bigger-is-better knob.**

The main limitation is that this is a single-seed experiment. I do not claim that `n=3` is always better than `n=1`, and I do not claim that `n=5` always fails. The correct claim is narrower: in this limited-compute single-seed ablation, `n=3` improved HalfCheetah learning compared with the 1-step baseline, while `n=5` failed under the same hyperparameters.

There are other limitations as well: only two environments, no full SACn correction mechanism, no transformer critic as in T-SAC, and no separate hyperparameter tuning for each horizon. This is not a paper-scale benchmark. It is a reproducible course project with a focused ablation.

The repository contains the implementation, configs, raw CSV logs, plots, a summary table, documentation, limitations, and presentation materials. So the result is not only described in slides; it can also be inspected through the code and logs.

If I continued the project, the first step would be to run 3 to 5 seeds, add confidence intervals, and test more MuJoCo environments such as Hopper or Walker2d. After that, I would compare the naive n-step version with SACn-style correction or a sequence-aware critic inspired by T-SAC.

To conclude, I did not try to prove a new algorithm. I tested a specific mechanism, obtained a clear trade-off, and analyzed a useful failure mode. On a simple environment all variants train, but on HalfCheetah the moderate 3-step target helps while the longer naive 5-step target fails.

Thank you. I am ready to answer questions.
