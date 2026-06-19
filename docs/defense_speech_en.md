# Defense Speech — final natural version, 5–7 minutes

Hello everyone. Today I will present my project **Stress-Testing n-step Soft Actor-Critic under Limited Compute**.

I deliberately limited the scope of the project. I did not try to propose a new state-of-the-art RL algorithm. I tested one concrete mechanism: what happens to SAC if the usual 1-step critic target is replaced with an n-step target.

The main question is: **do n-step returns help SAC under limited compute, and where does increasing the horizon start to hurt?**

SAC is a strong off-policy actor-critic algorithm for continuous control. It uses a replay buffer, a stochastic actor, twin critics, target networks, and entropy regularization. In standard SAC, the critic is usually trained with a 1-step target: one reward, then bootstrap from the next state.

This is stable, but reward information moves backward slowly. If a useful action affects reward several steps later, a 1-step target may be less sample-efficient.

An n-step return extends the target. For example, with `n=3`, the target includes `r_t`, `gamma r_{t+1}`, `gamma^2 r_{t+2}`, and only then bootstraps from `s_{t+3}`. The intuition is that reward information can move backward faster.

However, SAC is off-policy. Replay fragments may have been collected by older policies. The longer the n-step fragment is, the higher the risk of off-policy bias and target variance. So the horizon is not simply a bigger-is-better knob. It is also a stability parameter.

The project is related to SACn and T-SAC. SACn is motivated by the difficulty of using n-step returns in off-policy SAC. T-SAC goes further and uses trajectory chunks and a sequence-aware critic. I did not reproduce the full T-SAC model because that would require more engineering and more compute. I isolated the smaller shared mechanism: the critic target horizon.

I compared three variants: `SAC n=1` as the baseline, `SAC n=3` as a moderate horizon, and `SAC n=5` as a riskier target.

The implementation keeps the standard SAC pipeline. The main change happens before the replay buffer. I added an n-step buffer that stores a short queue of transitions and builds an aggregated transition: accumulated reward, discount, next state, and done flag. If the episode terminates early, the window is cut early. The critic target then becomes: `reward_n + discount_n * target_value(next_state)`. For `n=1`, this reduces to standard SAC.

I used two environments. `Pendulum-v1` is a sanity check: it verifies that the code works and that the n-step target does not immediately break SAC on a simple task. `HalfCheetah-v5` is the main benchmark.

On Pendulum, all three variants trained successfully and reached similar final quality. I do not use this as strong evidence. It is mainly a check that the implementation is alive.

The main result is on HalfCheetah. With 100k environment steps and seed 0, `SAC n=1` reached a final return of about **4354**, `SAC n=3` reached about **4579**, and `SAC n=5` reached only about **764**, with a best return of about **833**.

In this single-seed run, `n=3` performed best. It reached strong behavior earlier and had the best final return. But I do not claim that `n=3` is always better. The correct claim is narrower: **in this setting, a moderate horizon helped, while naive `n=5` failed under the same hyperparameters**.

The most useful result is not only the best `n=3` curve. It is also the failure mode of `n=5`. It shows that increasing the target horizon can hurt off-policy SAC if it is done naively and without correction mechanisms.

I intentionally kept the same hyperparameters for all horizons. This makes the ablation cleaner because only the critic target horizon changes. At the same time, it may be unfair to `n=5`. A longer horizon may need different learning rates, entropy settings, or replay settings. So the result is better interpreted as evidence of horizon sensitivity, not as evidence against 5-step SAC in general.

The main limitations are: one seed, two environments, no SACn-style correction, no sequence-aware critic, and no separate tuning for each horizon. In this version I evaluate stability mostly through return curves and the failure mode, not through full Q-value, entropy, or critic-loss diagnostics. That would be the next thing to add.

If I continued the project, I would first add 3–5 seeds, confidence intervals, and more MuJoCo environments such as Hopper or Walker2d. Then I would compare the naive n-step version with SACn-style correction or a sequence-aware critic.

To conclude, I tested a specific mechanism, observed a clear trade-off, and kept the claim narrow. **In SAC, the n-step horizon is a stability parameter, not a bigger-is-better knob.**

Thank you. I am ready for questions.
