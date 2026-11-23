# Ozon E-CUP 2026 — Quality Measurement Track (8th Place)

This repository contains the source code, configurations, and documentation for the two selected final submissions in the **Ozon E-CUP 2026 Quality Measurement** track (8th place overall). Each submission is stored on its own branch to keep source snapshots, configs, and reproduction notes isolated.

| Branch | Solution | Public Score | Submitted ZIP SHA-256 |
|---|---|---:|---|
| `solution-a-soup6` | Rank-96 LoRA model soup (5 fold endpoints + 1 full-data endpoint) | `0.9261642156862745` | `912440247a450c09163d74c44a533292c9ffc5a8182cd6b411548da305c74b27` |
| `solution-c-plus2` | Rank-80 LoRA model soup (5 fold endpoints) | `0.9255540166204985` | `281d5e381bc6356d983f5ee1ed7663e26b235e8f7069b22666efbabacca23ef4` |

## Browsing the Submissions

```bash
git switch solution-a-soup6
# or
git switch solution-c-plus2
```

Each solution branch contains:

- Exact source snapshot of the corresponding submission;
- SHA-256 manifests and 18-member inventory verification checks;
- Code for data preparation, LoRA fine-tuning, evaluation, model-soup assembly, and submission packaging;
- Documentation covering architecture, training setup, design rationale, and limitations;
- Instructions to deterministically rebuild the submitted ZIP from the tracked source and saved final adapter.

## Data & Binary Artifacts

- The models were fine-tuned on a consolidated dataset combining the official competition data with an additional manually curated contrastive set. Competition product cards and curated data are not redistributed in this repository.
- Large binary artifacts—including final LoRA adapter weights, submission ZIP archives (checksums recorded in `RELEASE_CHECKSUMS.sha256`), and the base `Qwen/Qwen3.5-4B` checkpoint provided by the competition environment—are not committed to Git.
