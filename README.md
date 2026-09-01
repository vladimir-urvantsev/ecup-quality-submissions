# ECUP Quality Classifier — Solution C (`plus2`)

Исходный репозиторий отправленного решения текстовой классификации карточек для
категорий `БАД` и `Легковоспламеняющиеся`. Решение использует
`Qwen/Qwen3.5-4B`, пять grouped fold LoRA-адаптеров и точное среднее их
LoRA-delta в одном rank-80 адаптере.

Публичный leaderboard score исторического артефакта:
`0.9255540166204985`. Это результат одного публичного split, а не гарантия
качества на новых данных.

## Архитектура

```text
name + description
        │
        ▼
Qwen3.5-4B + rank-80 five-fold LoRA soup
        │ p("да") / [p("да") + p("нет")]
        ▼
порог категории → узкое проверенное исключение → бинарный вердикт
                                                │
                                                ▼
                             post-verdict rationale renderer
```

- пять rank-16 адаптеров обучаются по grouped 5-fold CV;
- `make_soup.py` конкатенирует LoRA-факторы так, что итоговый delta равен
  арифметическому среднему пяти delta без пяти forward-pass;
- итоговый адаптер имеет `rank=80`, `lora_alpha=160`;
- пороги исторического релиза: `0.2123` для БАД и `0.717` для FLAM;
- изображения, VLM, retrieval и генеративный CoT в production path не входят.

Подробности: [архитектура](docs/ARCHITECTURE.md),
[данные](docs/DATA.md), [rationale](docs/RATIONALE.md) и
[воспроизводимость](docs/REPRODUCIBILITY.md). История измеренных и закрытых
веток собрана в [experiment ledger](docs/EXPERIMENT_HISTORY.md).

## Точное воспроизведение отправленного ZIP

Каталог `submission/` содержит малые файлы ровно из отправленного Candidate C.
Большой `adapter_model.safetensors` публикуется отдельно как release asset и не
дублируется обычным Git blob. Если сохранённый финальный ZIP доступен локально,
его можно проверить, извлечь из него адаптер и побайтно пересобрать:

```bash
make verify ARCHIVE=/path/to/final_C_plus2_rationale.zip
make import-frozen ARCHIVE=/path/to/final_C_plus2_rationale.zip
make rebuild-submitted
```

Ожидаемый SHA-256 результата:
`281d5e381bc6356d983f5ee1ed7663e26b235e8f7069b22666efbabacca23ef4`.
Скрипт проверяет SHA-256 и размер каждого из 18 членов архива и завершится
ошибкой при любом расхождении.

Для запуска требуется также предоставленный организаторами образ и доступный в
нём shared checkpoint `Qwen/Qwen3.5-4B`; базовые веса в репозиторий не входят.

## Повторное обучение метода

Модель обучалась на едином консолидированном наборе, включавшем официальные
данные соревнования и дополнительный вручную курированный контрастный набор.
Дополнительные карточки и исходный товарный контент в репозитории не
распространяются.

Требуются Python 3.10+, CUDA GPU с BF16 и локально доступный
`Qwen/Qwen3.5-4B`. Базовые веса не входят в репозиторий.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Поместите разрешённые источники данных согласно [data/README.md](data/README.md),
затем выполните:

```bash
make prepare \
  OFFICIAL_DATA=data/official/data.csv \
  CURATED_DATA=data/curated/contrast.csv

make train MODEL_PATH=/shared_models/Qwen/Qwen3.5-4B
make evaluate MODEL_PATH=/shared_models/Qwen/Qwen3.5-4B
make soup
make package
```

Команды нового обучения принимают явные пути. Exact snapshot сохраняет
исторические пути внутри metadata адаптера ради побайтного соответствия, но
runtime разрешает базовую модель через `$SHARED_MODELS_PATH` и от этих путей не
зависит.

## Инференс

```bash
SHARED_MODELS_PATH=/shared_models \
python -u build/submission/run.py \
  --test-data-path /path/to/data.csv \
  --output-path /path/to/result.csv
```

Входные столбцы: `id`, `name`, `description`, `category`. Выход содержит `id`
и строку:

```text
<комментарий>ВЫВОД: ... ОСНОВАНИЕ: ... КОНТРАСТ: ...<вердикт>бан|не бан
```

`label=1` соответствует принадлежности товара к указанной категории риска и
выводится как `не бан` согласно контракту конкурса.

Проверка полноты ID, длины rationale и wire grammar:

```bash
python scripts/validate_output.py --input /path/to/data.csv --output /path/to/result.csv
```

## Rationale

Обоснование строится после выбора метки и не имеет доступа к score, logits или
gold. Точная цитата используется только как дословный span исходного текста.
Формальный аудит проверил 18 728 комбинаций карточка × возможный verdict:
структуру, длину 50–300 символов и неизменность решения.

Renderer намеренно консервативен. При отсутствии безопасной точной опоры он
выдаёт общий fallback; статический PASS не означает, что каждое обоснование
информативно для человека. См. [docs/RATIONALE.md](docs/RATIONALE.md).

## Уровни воспроизводимости

- побайтная проверка и пересборка отправленного финального ZIP из tracked source
  и сохранённого финального адаптера;
- точный inference-код, пороги, deterministic override и rationale отправленной
  посылки;
- формат и статические гарантии rationale;
- современный рецепт подготовки данных, обучения, оценки и упаковки новой модели.

Byte-exact переобучение исторических весов не заявляется: пять исходных
fold-checkpoint, их logs и OOF CSV локально не сохранились, а точное
происхождение четырёхзначного порога `0.2123` не замкнуто отдельным threshold
report. Финальный артефакт остаётся авторитетным frozen release.
Это ограничивает только повторное обучение тех же байтов, но не воспроизведение
уже отправленного runtime и его весов. Новое обучение должно получить новый
manifest и не наследовать исторический leaderboard score.

## Большие файлы и лицензии

В Git не включаются базовая модель, adapter weights, официальный датасет,
контрастный набор и изображения. Точный финальный `safetensors` либо ZIP нужно
передать жюри как release asset или через Git LFS вместе с опубликованным
SHA-256.

Исходный репозиторий распространяется с ограничением из [LICENSE](LICENSE).
Qwen3.5-4B лицензирован отдельно по Apache-2.0. Перед публикацией адаптера или
конкурсного runtime подтвердите соответствующие права. См.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
