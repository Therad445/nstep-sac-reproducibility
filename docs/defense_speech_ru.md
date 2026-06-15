# Текст защиты на 5-7 минут

Здравствуйте. Я расскажу про проект **Stress-Testing n-step Soft Actor-Critic under Limited Compute**.

Идея проекта была не в том, чтобы предложить новый state-of-the-art алгоритм, а в том, чтобы взять одну конкретную идею из свежих работ про n-step SAC и проверить её на небольшом, но воспроизводимом эксперименте.

Главный вопрос был такой: **помогают ли n-step returns алгоритму Soft Actor-Critic при ограниченном compute, и где увеличение горизонта начинает вредить?**

Soft Actor-Critic - это сильный off-policy actor-critic алгоритм для continuous control. У него есть replay buffer, stochastic actor, twin critics, target networks и entropy regularization. В стандартном SAC critic обычно обучается по 1-step target: мы берём один reward, потом bootstrap от следующего состояния.

Это достаточно стабильно, но у такого подхода есть проблема: reward information распространяется назад довольно медленно. Если полезное действие влияет на награду через несколько шагов, 1-step target может обучаться не очень sample-efficient.

N-step returns предлагают простую идею: вместо одного reward перед bootstrap мы накапливаем несколько rewards. Например, для `n=3` target содержит `r_t`, `gamma r_{t+1}`, `gamma^2 r_{t+2}`, и только потом bootstrap от состояния `s_{t+3}`. Интуитивно это должно быстрее передавать информацию о награде назад.

Но в SAC есть важная сложность: SAC - off-policy алгоритм. Траектории лежат в replay buffer и могли быть собраны старой политикой. Поэтому чем длиннее n-step fragment, тем больше риск off-policy bias и target variance. То есть n-step horizon - это не просто “чем больше, тем лучше”. Это параметр стабильности.

В литературе это связано с работами по SACn и T-SAC. SACn как раз мотивирован тем, что n-step returns в off-policy SAC не так просто использовать наивно. T-SAC идёт дальше и использует trajectory chunks и sequence-aware critic. Я не воспроизводил полный T-SAC, потому что это уже более тяжёлая архитектура и больше compute. Мой проект - это bounded reproduction и ablation study центрального механизма: что будет, если в SAC поменять critic target horizon.

Я сравнивал три варианта:

- `SAC n=1` - стандартный baseline.
- `SAC n=3` - умеренный n-step horizon.
- `SAC n=5` - более длинный, потенциально рискованный horizon.

Код устроен довольно компактно. Основной SAC pipeline остаётся тем же: environment step, replay buffer, critic update, actor update, alpha update, target network update, periodic evaluation. Главное изменение находится перед replay buffer: я добавил n-step buffer, который хранит короткую очередь переходов и строит aggregated transition.

Для каждого такого перехода он считает накопленный reward, discount, next state и done flag. Если эпизод закончился раньше, n-step window обрывается. После этого aggregated transition кладётся в replay buffer, и critic update использует target вида:

`reward_n + discount_n * target_value(next_state)`.

При `n=1` это сводится к обычному SAC.

Экспериментально я использовал две среды. `Pendulum-v1` - как дешёвый sanity check, чтобы проверить, что реализация вообще работает. И `HalfCheetah-v5` - как основной continuous-control benchmark.

На `Pendulum-v1` я запускал `n=1`, `n=3`, `n=5` на 30 тысяч environment steps. Все три варианта обучились и пришли к похожему final return. Для меня это не главный результат, а именно sanity check: он показывает, что n-step target construction не ломает SAC на простой среде.

Главный результат получился на `HalfCheetah-v5`. Там я запускал все три варианта на 100 тысяч environment steps, seed 0. Результаты такие:

- `SAC n=1` получил final eval return около **4354**.
- `SAC n=3` получил около **4579**.
- `SAC n=5` получил только около **764**, best return около **833**.

То есть `n=3` оказался лучшим в этом эксперименте: он быстрее перешёл к сильному поведению и дал лучший final return. `n=1` остался сильным baseline, он тоже хорошо обучился, но медленнее. А `n=5` фактически не научился нормально бежать.

Для меня самый интересный результат здесь - не просто то, что `n=3` лучше. Самый интересный результат - это провал `n=5`. Он показывает, что увеличение n-step horizon действительно может ломать обучение в off-policy SAC, если делать это наивно и без дополнительных correction mechanisms.

Я специально держал одинаковые hyperparameters для `n=1`, `n=3` и `n=5`, чтобы сравнение было чистым: менялся только horizon critic target. Это, конечно, может быть не идеально справедливо к `n=5`, потому что ему, возможно, нужны другие learning rates или entropy settings. Но именно это и показывает чувствительность метода: длинный horizon не является бесплатным улучшением.

Главный вывод проекта такой:

**Moderate n-step credit assignment can help SAC sample efficiency, but a longer naive n-step target can become unstable or ineffective in off-policy SAC.**

Или проще: **n-step horizon - это stability parameter, а не “bigger is better knob”.**

Теперь про ограничения. Самое важное ограничение - это single seed. Я не утверждаю, что `n=3` всегда лучше `n=1`, и не утверждаю, что `n=5` всегда проваливается. Корректная формулировка уже: в моём limited-compute single-seed ablation на HalfCheetah `n=3` сработал лучше baseline, а `n=5` провалился при тех же hyperparameters.

Также у меня только две среды, нет полного SACn correction mechanism, нет transformer critic как в T-SAC, и нет отдельного hyperparameter tuning для каждого horizon. Это не paper-scale benchmark, а учебный reproducible ablation project.

Но зато в проекте есть полный воспроизводимый pipeline: configs, raw CSV logs, plots, summary table, README, final summary, limitations, презентация и код. То есть можно посмотреть не только финальные цифры, но и learning curves, исходные логи и реализацию n-step buffer.

Если бы я продолжал проект, я бы в первую очередь добавил 3-5 seeds, confidence intervals и ещё несколько MuJoCo environments, например Hopper или Walker2d. Следующий шаг после этого - сравнить naive n-step SAC с SACn-style correction или попробовать sequence-aware critic, ближе к идее T-SAC.

Итог: я не пытался доказать новый алгоритм. Я проверил конкретный механизм, получил понятный trade-off и показал failure mode. На простой среде все варианты сходятся, а на HalfCheetah видно, что умеренный `n=3` помогает, а более длинный naive `n=5` становится нестабильным.

Спасибо, готов ответить на вопросы.
