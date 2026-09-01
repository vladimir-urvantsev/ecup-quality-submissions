# Данные и split policy

Модель обучалась на едином консолидированном наборе, включавшем официальные
данные соревнования и дополнительный вручную курированный контрастный набор.
Дополнительные карточки и исходный товарный контент в репозитории не
распространяются.

## Входы

Pipeline принимает:

1. официальный `data.csv` со столбцами `id`, `name`, `description`, `category`,
   `label`;
2. опциональный curated contrast CSV с теми же столбцами.

Сырые карточки и изображения не распространяются этим репозиторием. Production
модель text-only; `images.zip` для воспроизведения A не требуется.

## Группировка

- `dup_key`: hash нормализованных `name + description`;
- `split_group`: hash первых шести нормализованных слов названия;
- официальный train делится `StratifiedGroupKFold(n_splits=5, seed=42)`;
- строки одной product family не пересекают fold boundary;
- curated строки с совпавшей official family наследуют её fold, остальные
  получают `fold=-1` и используются только в train каждого fold;
- exact duplicates сворачиваются до одного текстового представителя;
- группа с `soft_label == 0.5` исключается как не имеющая majority label.

Вес представителя равен `log2(1 + duplicate_count)` и удваивается, если простой
rule baseline ошибается. Затем trainer нормирует class weights внутри каждой
категории по суммарной базовой массе, а не по числу строк.

## Frozen lineage

Исторический training JSONL имел 10 028 строк: 9 364 official representatives
и 664 curated FLAM examples. Его SHA-256:

```text
aeb8fa2bebbe7bbba011d6a7f27c3b506d5c8f42894aed45f910101c0e39df04
```

Набор не включён в Git. Если у вас есть законно полученная frozen-копия,
`prepare_dataset.py --verify-frozen` проверит hash и структуру. Современная
пересборка из новых входов создаёт новый manifest и не должна выдаваться за
byte-exact historical dataset.

## Проверки

Builder останавливается при неизвестной категории, нецелом label, пустом ID,
пересечении ID между источниками, семейной утечке между official folds и
невалидных весах. Manifest фиксирует source hashes, counts и output hash.
