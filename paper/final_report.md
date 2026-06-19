---
title: "Stress-Testing n-step Soft Actor-Critic under Limited Compute"
subtitle: "Расширенный отчет по RL final project"
author: "Radmir I."
date: "2026"
lang: ru-RU
---

\begin{keybox}
\textbf{Коротко.} Проект проверяет одну центральную идею из n-step SAC: помогает ли замена стандартного 1-step critic target на n-step target при ограниченном compute. На HalfCheetah-v5 вариант $n=3$ улучшил обучение относительно baseline, а $n=5$ провалился при тех же hyperparameters. Главный вывод намеренно узкий: n-step horizon - это параметр стабильности, а не ручка "чем больше, тем лучше".
\end{keybox}

# Abstract

This project studies a compact n-step modification of Soft Actor-Critic (SAC) under a limited compute budget. The main question is whether replacing the standard 1-step critic target with n-step targets improves sample efficiency in continuous-control tasks, and where the horizon becomes unstable. I implement a configurable n-step buffer before the replay buffer and compare SAC with $n=1$, $n=3$, and $n=5$. The experiments include Pendulum-v1 as a sanity check and HalfCheetah-v5 as the main benchmark. On HalfCheetah-v5, $n=3$ reaches the best final return and reaches high-return behavior earlier than the baseline, while $n=5$ fails under the same hyperparameters. The result suggests that n-step credit assignment can help SAC, but the horizon is a stability parameter rather than a simple bigger-is-better knob.

# Введение

Soft Actor-Critic (SAC) - сильный off-policy actor-critic алгоритм для задач continuous control. Он использует replay buffer, stochastic actor, twin critics, target networks и maximum entropy objective: policy оптимизирует не только expected return, но и entropy [@haarnoja2018sac; @haarnoja2018applications]. Поэтому SAC удобен как baseline: он уже достаточно силен, и если небольшое изменение в critic target помогает или ломает обучение, это интересно анализировать.

В стандартном SAC critic обычно обучается по 1-step temporal-difference target. Такой target устойчивый и простой, но информация о reward распространяется назад только на один шаг. N-step returns предлагают более агрессивную идею: накопить несколько rewards перед bootstrap. Это может ускорить credit assignment, особенно если полезное действие влияет на reward не сразу [@sutton2018rl].

Но SAC является off-policy алгоритмом. Transitions в replay buffer могли быть собраны более старой policy. Чем длиннее n-step fragment, тем выше риск off-policy bias и target variance. Именно поэтому n-step SAC - не тривиальная модификация, а вопрос stability/sensitivity [@lyskawa2025sacn; @tian2025chunking].

Проект не заявляет новый SOTA-алгоритм. Это bounded reproduction и ablation study: я изолирую один механизм - critic target horizon - и проверяю, проявляется ли ожидаемый trade-off в компактной воспроизводимой постановке.

# Research question and contribution

Главный вопрос проекта:

\begin{keybox}
Помогают ли n-step returns алгоритму SAC при ограниченном compute, и насколько чувствителен результат к выбранному horizon?
\end{keybox}

Сравнивались три варианта:

| Method | Description | Expected behavior |
|---|---|---|
| SAC $n=1$ | standard 1-step SAC baseline | stable baseline |
| SAC $n=3$ | moderate 3-step target | may improve sample efficiency |
| SAC $n=5$ | longer 5-step target | riskier target |

Вклад проекта небольшой, но конкретный:

1. configurable n-step transition buffer, встроенный перед replay buffer;
2. ablation $n=1/3/5$ на Pendulum-v1 и HalfCheetah-v5;
3. raw logs, plots, summary table и reproduction notes в репозитории;
4. failure analysis для $n=5$ при фиксированных hyperparameters.

# Контекст литературы

## Soft Actor-Critic

SAC - off-policy maximum-entropy actor-critic algorithm. Actor оптимизируется не только на expected return, но и на entropy, что помогает exploration и устойчивости. Практическая версия SAC использует twin Q-critics, target networks, replay buffer и automatic entropy temperature tuning [@haarnoja2018sac; @haarnoja2018applications].

Для этого проекта SAC подходит как baseline, потому что это сильный continuous-control метод. Если n-step modification помогает или ломает SAC, это не выглядит тривиальным эффектом.

## N-step returns and credit assignment

N-step returns - классическая идея temporal-difference learning. Вместо bootstrap после одного reward алгоритм накапливает несколько rewards и только потом bootstraps:

$$
G_t^{(n)} = r_t + \gamma r_{t+1} + \dots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n}).
$$

Это может быстрее распространять delayed reward назад. Но target содержит больше sampled rewards и меняет bias-variance trade-off [@sutton2018rl; @sharma2017mixing].

## Off-policy multi-step correction

Главная сложность - off-policy setting. В replay-based learning более длинные trajectory fragments могли быть собраны behavior policy, отличающейся от текущей policy. Поэтому многие multi-step методы явно вводят correction или trace truncation. Retrace($\lambda$) был предложен как safe and efficient off-policy return-based method [@munos2016retrace]. IMPALA использует V-trace, чтобы стабилизировать off-policy correction в distributed actor-learner setting [@espeholt2018impala].

В этом проекте Retrace, V-trace и importance sampling не реализуются. Это сделано намеренно: я тестирую naive n-step SAC, чтобы увидеть чувствительность именно к horizon.

## SACn and T-SAC

SACn напрямую изучает, как соединить Soft Actor-Critic с n-step returns. Его мотивация близка к этому проекту: n-step returns могут ускорить обучение, но обычная комбинация с SAC вносит bias из-за меняющегося action distribution, а importance sampling может быть численно нестабильным [@lyskawa2025sacn].

T-SAC / Chunking the Critic идет дальше: он использует trajectory chunks, sequence-aware critic и aggregated n-step targets. Там temporal context становится частью critic learning, а не только scalar target [@tian2025chunking].

Этот проект не воспроизводит полный SACn или T-SAC. Он изолирует меньший общий механизм: изменение critic target horizon. Это делает эксперимент выполнимым на limited compute и проще для интерпретации.

# Метод

## Standard SAC components

Реализация сохраняет стандартную структуру SAC:

- squashed Gaussian actor;
- twin Q-critics;
- target critic networks;
- replay buffer;
- automatic entropy tuning;
- periodic deterministic evaluation.

Critic update использует минимум из двух target critics и entropy term:

$$
V_{\text{target}}(s') =
\min_i Q_i^{\text{target}}(s', a') - \alpha \log \pi(a'|s').
$$

Для обычного 1-step варианта target имеет вид:

$$
y_t = r_t + \gamma (1-d_t) V_{\text{target}}(s_{t+1}).
$$

## N-step transition buffer

Единственный алгоритмический компонент, который меняется в основной ablation, - aggregation перед replay buffer. N-step buffer хранит короткую очередь recent transitions. Когда данных хватает, он строит aggregated transition:

```text
state      = s_t
action     = a_t
reward_n   = r_t + gamma r_{t+1} + ... + gamma^(k-1) r_{t+k-1}
next_state = s_{t+k}
discount_n = gamma^k
done       = terminal flag inside the n-step window
```

Здесь $k \leq n$. Если episode заканчивается раньше, window обрывается. При $n=1$ это сводится к обычному SAC.

Critic target становится таким:

$$
y_t^{(n)} = R_t^{(n)} + (1-d_t) \gamma^k V_{\text{target}}(s_{t+k}).
$$

Replay buffer поэтому хранит одинаковый тип объектов для всех runs, но reward и discount кодируют разный target horizon.

## Что фиксируется в ablation

Architecture и hyperparameters одинаковые для $n=1$, $n=3$, $n=5$. Это намеренный выбор: эксперимент проверяет именно sensitivity к target horizon.

\begin{notegray}
Это также означает, что сравнение не является полностью tuned для каждого horizon. Возможно, $n=5$ мог бы стать лучше после отдельного hyperparameter search. Поэтому результат нужно читать как fixed-configuration stability test, а не как доказательство, что $n=5$ никогда не работает.
\end{notegray}

# Experimental setup

## Environments

Использовались две среды:

| Environment | Role | Steps | Seed |
|---|---|---:|---:|
| Pendulum-v1 | sanity check | 30,000 | 0 |
| HalfCheetah-v5 | main benchmark | 100,000 | 0 |

Pendulum-v1 - простая continuous-action classic-control среда с action space `Box(-2.0, 2.0, (1,), float32)` и observation shape `(3,)` [@gymnasium_pendulum]. HalfCheetah-v5 - MuJoCo locomotion benchmark с action space `Box(-1.0, 1.0, (6,), float32)` и observation shape `(17,)`; задача - заставить 2D-робота бежать вперед как можно быстрее [@gymnasium_halfcheetah].

## Hyperparameters

| Parameter | Value |
|---|---:|
| Discount factor `gamma` | 0.99 |
| Target update coefficient `tau` | 0.005 |
| Initial entropy coefficient `alpha` | 0.2 |
| Automatic entropy tuning | yes |
| Actor learning rate | 3e-4 |
| Critic learning rate | 3e-4 |
| Alpha learning rate | 3e-4 |
| Batch size | 256 |
| Replay buffer size | 1,000,000 |
| Hidden sizes | [256, 256] |
| Start steps | 5,000 |
| Update after | 5,000 |
| Evaluation interval | 5,000 steps |
| Evaluation episodes | 5 |
| Compared horizons | $n=1,3,5$ |

Все completed runs используют deterministic evaluation по 5 episodes на checkpoint. Raw CSV logs и plots включены в repository.

## Hardware

Обучение выполнялось на Ryzen 5700 laptop через WSL Ubuntu, CPU training. Это intentionally limited-compute setup. Проект задуман как reproducible student experiment, а не large-scale benchmark.

# Results

## Pendulum-v1 sanity check

На Pendulum-v1 все три варианта успешно обучаются и приходят к похожему final quality. Главная роль этого эксперимента - проверить, что реализация работает и n-step target construction не ломает SAC на простой среде.

![Pendulum-v1 learning curves](results/plots/pendulum_learning_curves.png){#fig:pendulum-ru width=90%}

Кривые показывают, что $n=3$ быстро выходит к хорошему return, но Pendulum-v1 слишком простая среда, чтобы делать сильный вывод о лучшем horizon. Это sanity check, а не основное доказательство.

## HalfCheetah-v5 main ablation

Главный результат получился на HalfCheetah-v5.

![HalfCheetah-v5 learning curves](results/plots/halfcheetah_learning_curves.png){#fig:halfcheetah-ru width=90%}

| Method | Final eval return | Best eval return |
|---|---:|---:|
| SAC $n=1$ | 4354.49 | 4354.49 |
| SAC $n=3$ | 4578.91 | 4578.91 |
| SAC $n=5$ | 764.49 | 833.37 |

\begin{keybox}
\textbf{Главный результат.} В этом single-seed run $n=3$ дал лучший final return и быстрее вышел к сильному поведению. Вариант $n=5$ не научился устойчивой running policy при тех же settings.
\end{keybox}

## Sample-efficiency markers

Final return - не единственная полезная метрика. На HalfCheetah-v5 видно, как быстро варианты доходят до сильного поведения.

| Method | Steps to return $\geq 3000$ | Steps to return $\geq 4000$ |
|---|---:|---:|
| SAC $n=1$ | 65,000 | 100,000 |
| SAC $n=3$ | 45,000 | 70,000 |
| SAC $n=5$ | not reached | not reached |

Эти markers поддерживают чтение графика: $n=3$ учится быстрее в этом run, а $n=5$ вообще не достигает high-return behavior.

# Discussion

Результаты поддерживают ожидаемый trade-off. Moderate n-step horizon может улучшить sample efficiency, потому что reward information быстрее распространяется назад. Вероятно, это объясняет, почему $n=3$ лучше 1-step baseline на HalfCheetah-v5.

Одновременно более длинный naive target может увеличить off-policy bias и target variance. Это вероятное объяснение провала $n=5$ при тех же hyperparameters. Самый полезный результат здесь не только то, что $n=3$ дал лучший final return. Более информативен failure mode $n=5$: он показывает, что увеличение target horizon может изменить стабильность off-policy SAC.

\begin{warningbox}
Самый безопасный вывод узкий: в этом limited-compute single-seed ablation 3-step SAC улучшил HalfCheetah learning по сравнению с 1-step baseline, а 5-step SAC провалился при тех же hyperparameters. Проект не утверждает универсальное превосходство $n=3$.
\end{warningbox}

Этот failure mode совпадает с мотивацией свежих работ по n-step SAC: n-step targets могут помогать credit assignment, но horizon взаимодействует с off-policy bias, target variance, critic stability и entropy dynamics.

# Threats to validity

## Statistical validity

Главное ограничение - один seed. Текущий эксперимент полезен как bounded reproduction и ablation evidence, но его недостаточно, чтобы статистически утверждать, что $n=3$ в целом лучше $n=1$, или что $n=5$ в целом проваливается. Более сильная работа должна использовать хотя бы 3-5 seeds и confidence intervals.

## Algorithmic validity

Реализация использует naive n-step targets без off-policy correction. Это сделано намеренно, потому что проект тестирует прямой эффект critic target horizon. Но это также означает, что провал $n=5$ не нужно считать провалом всех n-step SAC methods. SACn-style correction или sequence-aware critic могли бы стабилизировать longer horizons.

## Hyperparameter fairness

Все horizons используют одинаковые hyperparameters. Это делает ablation чистой, потому что меняется только target horizon. Но это может быть не полностью справедливо к $n=5$. Более длинный horizon может требовать других learning rates, target smoothing, entropy settings или replay settings.

## Environment coverage

Проект использует Pendulum-v1 и HalfCheetah-v5. Pendulum-v1 - sanity check; она слишком простая для сильных выводов о лучшем horizon. HalfCheetah-v5 - основной benchmark, но одной MuJoCo задачи недостаточно для общего вывода. В future work стоит добавить Hopper-v5, Walker2d-v5 и Ant-v5.

## Compute budget

Все эксперименты запускались на laptop CPU. Это определило дизайн: короткие training budgets, single seed и compact architecture. Проект нужно читать как small reproducible ablation study, а не paper-scale benchmark.

# Reproducibility checklist

| Item | Status |
|---|---|
| Code committed | yes |
| Configs committed | yes |
| Raw CSV logs committed | yes |
| Plots committed | yes |
| Summary table committed | yes |
| Random seed recorded | yes, seed 0 |
| Hardware documented | yes |
| Limitations documented | yes |
| Full multi-seed statistics | no |

Ключевые repository artifacts:

- implementation: `src/rl/nstep_buffer.py`, `src/rl/sac.py`, `src/train.py`;
- configs: `configs/pendulum_sac_n*.yaml`, `configs/halfcheetah_sac_n*.yaml`;
- raw logs: `results/raw/*.csv`;
- plots: `results/plots/*.png`;
- summary: `results/halfcheetah_summary.csv`;
- slides and notes: `slides/`, `docs/`.

# Future work

Наиболее полезные продолжения:

1. запустить 3-5 seeds;
2. добавить Hopper-v5, Walker2d-v5 и Ant-v5;
3. посчитать confidence intervals и failure rates;
4. сравнить naive n-step SAC с SACn-style corrections;
5. отдельно tune hyperparameters для каждого horizon;
6. протестировать sequence-aware critics в духе T-SAC.

# Conclusion

Проект реализует configurable n-step targets в SAC и тестирует их в небольшом, но воспроизводимом ablation. Pendulum-v1 подтверждает, что реализация работает. HalfCheetah-v5 показывает основной trade-off: $n=3$ помогает, $n=1$ остается сильным baseline, а $n=5$ проваливается при тех же hyperparameters.

\begin{keybox}
\textbf{Итоговая мысль.} N-step horizon is a stability parameter, not a bigger-is-better knob.
\end{keybox}

# References
