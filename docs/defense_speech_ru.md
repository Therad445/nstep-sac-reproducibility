# Текст защиты на 5–7 минут — финальная живая версия

Здравствуйте. Я расскажу про проект **Stress-Testing n-step Soft Actor-Critic under Limited Compute**.

Я сразу ограничил проект: я не делал новый алгоритм и не пытался заявить SOTA. Я проверял один конкретный механизм — что происходит с SAC, если заменить обычный 1-step critic target на n-step target.

Главный вопрос простой: **помогают ли n-step returns алгоритму SAC при ограниченном compute, и где увеличение горизонта начинает вредить?**

SAC — сильный off-policy actor-critic алгоритм для continuous control. В нём есть replay buffer, stochastic actor, twin critics, target networks и entropy regularization. В стандартной версии critic обычно обучается по 1-step target: один reward, потом bootstrap от следующего состояния.

Это устойчиво, но reward information идёт назад медленно. Если полезное действие влияет на награду через несколько шагов, 1-step target может быть менее sample-efficient.

N-step return делает target более длинным. Например, при `n=3` мы берём `r_t`, `gamma r_{t+1}`, `gamma^2 r_{t+2}` и только потом bootstrap от `s_{t+3}`. Интуиция такая: reward signal быстрее доходит до более ранних действий.

Но в SAC есть важная проблема: это off-policy алгоритм. Фрагменты в replay buffer могли быть собраны старой policy. Поэтому чем длиннее n-step fragment, тем выше риск off-policy bias и target variance. То есть horizon — не ручка “сделать больше”. Это параметр стабильности.

По литературе проект стоит рядом с SACn и T-SAC. SACn как раз обсуждает, почему n-step returns в off-policy SAC не так просто использовать наивно. T-SAC идёт дальше и использует trajectory chunks и sequence-aware critic. Я не воспроизводил полный T-SAC: для этого нужно больше инженерии и compute. Я изолировал меньший общий механизм — horizon critic target.

Я сравнивал три варианта: `SAC n=1` как baseline, `SAC n=3` как умеренный horizon и `SAC n=5` как более рискованный target.

В коде основной SAC pipeline остаётся прежним. Главное изменение происходит перед replay buffer. Я добавил n-step buffer: он хранит короткую очередь transitions и строит aggregated transition — накопленный reward, discount, next state и done flag. Если episode закончился раньше, окно обрывается. После этого critic target имеет вид: `reward_n + discount_n * target_value(next_state)`. При `n=1` это превращается в обычный SAC.

Эксперименты были на двух средах. `Pendulum-v1` — проверочный эксперимент: убедиться, что код живой и n-step target не ломает SAC на простой задаче. `HalfCheetah-v5` — основной benchmark.

На Pendulum все три варианта обучились и пришли к близкому качеству. Я не делаю из этого сильного вывода. Это именно sanity check.

Главный результат — HalfCheetah. При 100 тысячах environment steps и seed 0 получилось так: `SAC n=1` дал final return около **4354**, `SAC n=3` — около **4579**, а `SAC n=5` — около **764**, best около **833**.

В этом конкретном single-seed run `n=3` оказался лучше: он раньше дошёл до сильного поведения и дал лучший final return. Но я не утверждаю, что `n=3` всегда лучше. Корректный вывод уже: **в этой постановке moderate horizon помог, а naive `n=5` провалился при тех же hyperparameters**.

Самое полезное здесь даже не победа `n=3`, а failure mode `n=5`. Он показывает, что длинный n-step target может не улучшить SAC, а сделать обучение хуже, если использовать его наивно и без correction mechanisms.

Я специально держал одинаковые hyperparameters для `n=1`, `n=3` и `n=5`, чтобы менялся только horizon critic target. Это делает ablation чище, но может быть не полностью справедливо к `n=5`. Возможно, ему нужны другие learning rates, entropy settings или replay настройки. Поэтому результат лучше читать как evidence of horizon sensitivity, а не как доказательство против 5-step SAC вообще.

Главные ограничения: один seed, две среды, нет SACn-style correction, нет sequence-aware critic как в T-SAC, нет отдельного tuning для каждого horizon. В текущей версии stability я оцениваю по return curves и failure mode, а не по полной диагностике Q-values, entropy и critic loss. Это я бы добавил в следующей версии.

Если продолжать проект, я бы сначала добил статистику: 3–5 seeds, confidence intervals, ещё Hopper или Walker2d. Потом — сравнение naive n-step SAC с SACn-style correction и, возможно, sequence-aware critic.

Итог: я проверил конкретный механизм, получил понятный trade-off и честно ограничил вывод. **N-step horizon в SAC — это stability parameter, а не bigger-is-better knob.**

Спасибо, готов ответить на вопросы.
