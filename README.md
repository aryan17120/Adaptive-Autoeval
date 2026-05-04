# Adaptive AutoEval
## Importance-Weighted Model Evaluation under Unknown Covariate Shift

Official code for the paper:
> **AutoEval under Unknown Covariate Shift: Learning Importance Weights
> from Unlabeled Data** — Submitted to NeurIPS 2026

---

## The Problem

The AutoEval framework estimates model performance by combining
small human-labeled datasets with large synthetic-label datasets
via prediction-powered inference (PPI++). It works well — but only
when labeled and unlabeled data come from the same distribution.

In practice, this assumption is routinely violated. Annotators
select easier or higher-confidence examples. Benchmark datasets
are curated while deployment data is not. When covariate shift
occurs and importance weights are unknown, standard AutoEval
produces biased estimates and invalid confidence intervals.

**The headline failure**: under realistic confidence-biased
labeling on ImageNet, standard AutoEval coverage collapses from
nominal 90% to **1.7%** at n=500 as confidence intervals shrink
around a biased estimate. The error is invisible to standard
evaluation.

---

## Our Solution: Adaptive AutoEval

We learn importance weights directly from unlabeled data using
a discriminative classifier that distinguishes labeled from
unlabeled inputs, then integrate these weights into the PPI++
framework.

**Key results:**
- Coverage restored from **1.7% → 72–83%** on ImageNet
- **65% MSE reduction** vs. classical estimation  
- **Beats oracle weights** due to implicit regularization
- Validated on vision (ImageNet) and biology (ProteinGym)

---

## Quick Start

```bash
pip install -e .
```

```python
import numpy as np
from adaptive_autoeval import ppi_weighted, learn_importance_weights

# Features for density ratio estimation
feat_lab = ...   # (n, d) features of labeled data
feat_unl = ...   # (N, d) features of unlabeled data

# True losses and synthetic predictions
phi_lab  = ...   # (n,) true correctness on labeled data
syn_lab  = ...   # (n,) synthetic annotator predictions on labeled
syn_unl  = ...   # (N,) synthetic annotator predictions on unlabeled

# Step 1: Learn importance weights
weights = learn_importance_weights(feat_lab, feat_unl)

# Step 2: Adaptive AutoEval estimate + confidence interval
mu_hat, var_hat = ppi_weighted(phi_lab, syn_lab, syn_unl, weights)

from scipy.stats import norm
z = norm.ppf(0.95)   # 90% CI
ci = (mu_hat - z * np.sqrt(var_hat), mu_hat + z * np.sqrt(var_hat))
print(f"Estimate: {mu_hat:.3f}  90% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

---

## Results

### ImageNet — Synthetic Covariate Shift (250 trials)

| n   | Classical | PPI++ | Oracle | **Adaptive** |
|-----|-----------|-------|--------|--------------|
| 50  | 0.635 | 0.793 | 0.700 | **0.789** |
| 100 | 0.461 | 0.823 | 0.684 | **0.834** |
| 200 | 0.224 | 0.753 | 0.662 | **0.770** |
| 500 | 0.017 | 0.682 | 0.554 | **0.720** |

*Coverage of nominal 90% confidence intervals.*
Adaptive AutoEval outperforms even Oracle PPI++ (true weights)
due to implicit regularization of extreme density ratio values.

### ProteinGym SPG1 — Fitness-Biased Labeling (250 trials, β=1.0)

| n    | Classical | PPI++ | **Adaptive** | MSE reduction |
|------|-----------|-------|--------------|---------------|
| 200  | 0.198 | 0.166 | **0.556** | 46% |
| 1500 | 0.000 | 0.000 | **0.029** | 61% |

For full results including ablations see
[`results/experiments.md`](results/experiments.md).

---

## Repository Structure


adaptive-autoeval/
├── adaptive_autoeval/          # pip-installable library
│   ├── init.py
│   ├── estimators.py           # ppi_unweighted, ppi_weighted
│   └── weights.py              # learn_importance_weights
├── scripts/
│   ├── run_extension1_imagenet.py        # ImageNet experiment
│   ├── run_extension1_proteingym.py # ProteinGym experiment
│   ├── run_extension1_ablations.py       # Ablation studies
│   └── compute_spearman.py               # Spearman analysis
├── results/
│   ├── experiments.md          # Full cross-domain results table
│   ├── imagenet/               # Figures + CSVs
│   ├── proteingym/             # Figures + CSVs
│   └── ablations/              # Figures + CSVs
├── data/
│   ├── imagenet/README.md      # Download instructions
│   └── proteingym/README.md    # Download instructions
├── pyproject.toml
├── requirements.txt
└── README.md


---

## Running Experiments

```bash
# Set API keys (only needed for ProteinGym)
# No API keys needed for ImageNet experiments

# Install
git clone https://github.com/aryan17120/Adaptive-Autoeval.git
cd Adaptive-Autoeval
pip install -r requirements.txt

# ImageNet (no data download needed beyond numpy files)
python scripts/run_extension1_imagenet.py

# ProteinGym (download data first — see data/proteingym/README.md)
python scripts/run_extension1_proteingym_full.py

# Ablation studies
python scripts/run_extension1_ablations.py

# Spearman correlation analysis
python scripts/compute_spearman.py
```

All scripts use `np.random.seed(42)` for reproducibility.

---

## Data Setup

See [`data/imagenet/README.md`](data/imagenet/README.md) and
[`data/proteingym/README.md`](data/proteingym/README.md)
for download instructions.

---

## Citation

```bibtex
@inproceedings{adaptive_autoeval_2026,
  title     = {AutoEval under Unknown Covariate Shift: Learning
               Importance Weights from Unlabeled Data},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems
               (NeurIPS)},
  year      = {2026}
}
```

---

## License

Code: MIT License. Data and results: CC-BY-4.0.
