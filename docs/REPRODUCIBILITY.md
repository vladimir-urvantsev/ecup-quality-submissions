# Reproducibility levels

## Level 1 — exact frozen release

Given the historical ZIP, `scripts/verify_frozen_release.py` verifies its
SHA-256, safe member paths, expected adapter rank/alpha, thresholds and critical
member hashes. This is byte-exact.

## Level 2 — exact assembly of the submitted release

`submission/` stores every small member of the submitted archive byte-for-byte.
Given the frozen `adapter_model.safetensors`,
`scripts/rebuild_submitted_release.py` verifies the complete 18-member
inventory and creates the same deterministic ZIP. The command refuses any
source, adapter or final archive hash drift.

## Level 3 — methodological retraining

The repository provides the complete current commands for preparation, five
folds, OOF scoring, threshold reporting, a full-data endpoint, exact delta soup
and packaging. Random seed, prompt, grouping, hyperparameters and source hashes
are recorded. This path produces a new model artifact and must be evaluated and
calibrated as a new experiment.

It does not promise identical historical weights because the original five
fold endpoints/logs are unavailable and the sixth full-data checkpoint was not
independently hash-pinned. Modern library versions may also change kernels and
serialization. A new run must therefore receive a new manifest and must not
reuse the historical leaderboard number as its measured result.

## Historical recipe

```text
base model          Qwen/Qwen3.5-4B
folds               5 grouped folds, seed 42
rank / alpha        16 / 32 per ingredient
dropout             0.05
target modules      q,k,v,o,gate,up,down projections
max length          1024, description-only truncation
batch / accumulation 2 / 16
learning rate       1.5e-4
epochs              2 for fold endpoints
class-weight power  1.0
full-data endpoint  512 optimizer steps, seed 42
final soup          five best fold endpoints + one full-data endpoint
thresholds          БАД 0.2123; FLAM 0.717
```
