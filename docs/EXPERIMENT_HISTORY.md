# Experiment history and decisions

Этот ledger отделяет измеренные результаты от гипотез и объясняет состав
финальных A/C. Числа разных типов не смешиваются: public leaderboard, grouped
OOF и optimistic threshold-fit отвечают на разные вопросы.

## 1. Метрика

Финальная интерпретация конкурсной метрики:

```text
0.5 * (F1_positive(БАД) + F1_positive(Легковоспламеняющиеся))
```

То есть обе товарные категории весят одинаково, хотя FLAM positives редки.
Ранние pooled расчёты недооценивали цену одной FLAM ошибки. В репозитории
`evaluate_competition_metric.py` явно отделяет конкурсную метрику от
диагностического binary-macro F1.

## 2. Базовый classifier

Выбран прямой verdict-token objective: Qwen отвечает одним токеном `да/нет`, а
вероятность читается из logits. Генерация rationale внутри decision path не
используется. Первые text-only варианты давали около 0.90 public score.

Ключевые инженерные изменения:

- description-only truncation при `max_len=1024`, сохраняющая название;
- group split по product-family proxy вместо случайного row split;
- exact-text deduplication и majority label для повторов;
- category-local class weighting по эффективной массе строк;
- pooled OOF threshold fitting вместо одного редкого fold;
- один измеренный узкий override вместо широкого rule router.

## 3. Curated contrast data

В frozen training lineage добавлено 664 контрастных FLAM карточки: 317 label 0
и 347 label 1. Они использовались как train-only строки, если не совпадали с
official family. Этот шаг дал основной переход от примерно 0.90 к линии C.

Поздний независимый аудит выявил неоднородность качества и systematic family
effects. Поэтому frozen lineage сохраняется как исторический артефакт, но
новые данные должны получать собственный manifest и matched OOF ablation.

## 4. Ансамбли

Пять fold-адаптеров объединены точным mean-delta soup:

- C: rank 80 / alpha 160, public `0.9255540166204985`;
- A: те же пять компонентов плюс один fixed-512 full-data компонент, rank 96 /
  alpha 192, public `0.9261642156862745`.

Среднее строится алгебраически, а не усреднением LoRA A/B по отдельности.

## 5. Отрицательные или неперенесённые ветки

| Ветка | Наблюдение | Решение |
|---|---:|---|
| широкий threshold probe для FLAM | public около 0.9100 | не использовать |
| десятикомпонентный soup с дополнительным LR | public около 0.9056 | слабые ингредиенты разбавили ансамбль |
| дополнительная малая tail-партия | public около 0.8835 | остановить; данные/конвенция не прошли перенос |
| exact retrieval поверх A | public без изменения, `0.926164...` | не включать |
| rationale-loss V1 | OOF macro `0.899699 → 0.895204`; FLAM PR-AUC вырос | не менять verdict model |
| direct VLM residual | устойчивого matched OOF выигрыша для production не доказано | не включать в A/C |
| альтернативный backbone blend | complementarity измерялась относительно другого Qwen anchor | не экстраполировать на A |

Таблица является ретроспективным ledger, а не единой контролируемой таблицей
абляций: некоторые public точки меняли больше одной переменной.

## 6. Shortcut / group robustness

FLAM оказался особенно чувствителен к series membership. В retrospective OOF
F1 на повторяющихся сериях был существенно выше, чем на singleton-карточках;
модель могла запоминать устойчивые формулировки семейства вместо физики товара.
Это объясняет, почему случайный или row-level split оптимистичен, а маленькие
точечные партии могут не переноситься. Production recipe поэтому фиксирует
grouped folds и требует отдельного отчёта по singleton/worst-group страте.

## 7. Rationale decision

Обоснования нужны reviewer-у, но joint rationale training не показал прироста
классификации. Финальные A/C используют post-verdict renderer:

- frozen classifier остаётся неизменным;
- точные текстовые spans используются при безопасном совпадении;
- противоположный факт оформляется как конфликт, а не автоматический override;
- при недостатке опоры используется честный общий текст.

Такой выбор защищает F1 и faithfulness-инвариант, но ограничивает конкретность;
это прямо описано в `docs/RATIONALE.md`.

