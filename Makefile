.PHONY: install test audit prepare train evaluate soup package verify import-frozen rebuild-submitted

PYTHON ?= python3
MODEL_PATH ?= /shared_models/Qwen/Qwen3.5-4B
OFFICIAL_DATA ?= data/official/data.csv
CURATED_DATA ?= data/curated/contrast.csv
ADAPTER_MODEL ?= artifacts/frozen-adapter/adapter_model.safetensors
SUBMITTED_OUTPUT ?= build/final_C_plus2_rationale.zip

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest

audit:
	$(PYTHON) scripts/check_public_tree.py .

prepare:
	$(PYTHON) src/prepare_dataset.py --official $(OFFICIAL_DATA) --curated $(CURATED_DATA) --output data/derived/lora_dataset_plus.jsonl --manifest data/derived/manifest.json

train:
	bash scripts/train_all.sh "$(MODEL_PATH)" data/derived/lora_dataset_plus.jsonl

evaluate:
	bash scripts/evaluate_oof.sh "$(MODEL_PATH)" data/derived/lora_dataset_plus.jsonl

soup:
	$(PYTHON) src/make_soup.py --adapters 'runs/fold*/best' --out runs/soup5

package:
	$(PYTHON) src/build_submission.py --adapter runs/soup5 --recipe configs/recipe.json --output-dir build/submission
	$(PYTHON) src/package_final_submission.py --package-dir build/submission --output-zip build/solution-c.zip

verify:
	$(PYTHON) scripts/verify_frozen_release.py --manifest manifests/frozen_release.json --archive "$${ARCHIVE:?set ARCHIVE=/path/to/final_C_plus2_rationale.zip}"

import-frozen:
	$(PYTHON) scripts/import_frozen_adapter.py --manifest manifests/frozen_release.json --archive "$${ARCHIVE:?set ARCHIVE=/path/to/final_C_plus2_rationale.zip}" --output artifacts/frozen-adapter

rebuild-submitted:
	$(PYTHON) scripts/rebuild_submitted_release.py --adapter-model "$(ADAPTER_MODEL)" --output-zip "$(SUBMITTED_OUTPUT)"
