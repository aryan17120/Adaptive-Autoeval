# Adaptive AutoEval: Learning Importance Weights from Unlabeled Data

Official code repository for the paper:

> **AutoEval under Unknown Covariate Shift: Learning Importance
> Weights from Unlabeled Data**
> Submitted to NeurIPS 2026

---

## Overview

The AutoEval framework estimates model performance by combining
small human-labeled datasets with large synthetic-label datasets
via prediction-powered inference (PPI++). However, it assumes
labeled and unlabeled data come from the same distribution.

**Adaptive AutoEval** extends AutoEval to handle unknown covariate
shift by learning importance weights from unlabeled data using a
discriminative classifier.

**Key results:**
- Restores coverage from ~0% to 0.72–0.83 on ImageNet
- Reduces MSE by 65% vs. classical estimation
- Outperforms oracle weighted estimator due to implicit regularization
- Validated on ImageNet (vision) and ProteinGym (biology)

---

## Repository Structure

```bash
adaptive-autoeval/
├── scripts/
│   ├── run_extension1_imagenet.py       # Main ImageNet experiment
│   ├── run_extension1_proteingym_full.py # ProteinGym experiment
│   ├── run_extension1_ablations.py      # Ablation studies
│   ├── compute_spearman.py              # Spearman correlation analysis
├── results/
│   ├── imagenet/                        # ImageNet results + figures
│   ├── proteingym/                      # ProteinGym results + figures
│   ├── ablations/                       # Ablation results + figures
├── data/
│   ├── imagenet/README.md               # Data download instructions
│   ├── proteingym/README.md             # Data download instructions
├── requirements.txt
└── README.md

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/adaptive-autoeval.git
cd adaptive-autoeval
pip install -r requirements.txt
```

---

## Data Setup

See `data/imagenet/README.md` and `data/proteingym/README.md`
for download instructions.

After downloading, place data files in:

results/phi_imagenet/resnet{18,34,50,101,152}.npy

results/synthetic_imagenet/resnet{18,34,50,101,152}.npy

data/proteingym/SPG1_STRSG_Olson_2014.csv

data/proteingym/SPG1_STRSG_Olson_2014_zero_shot.csv

---

## Running Experiments

### Experiment 1: ImageNet Covariate Shift
```bash
python scripts/run_extension1_imagenet.py
```
Outputs: `results/extension1/ext1_results.csv` and figures.

### Experiment 2: ProteinGym Fitness Shift
```bash
python scripts/run_extension1_proteingym_full.py
```
Outputs: `results/extension1_proteingym/` directory.

### Experiment 3: Ablation Studies
```bash
python scripts/run_extension1_ablations.py
```
Outputs: `results/ablations/` directory.

### Spearman Correlation Analysis
```bash
python scripts/compute_spearman.py
```

---

## Results Summary

### ImageNet (250 trials, n = 50–500)

| n   | Classical | PPI++ | Oracle | **Adaptive** |
|-----|-----------|-------|--------|--------------|
| 50  | 0.635 | 0.793 | 0.700 | **0.789** |
| 100 | 0.461 | 0.823 | 0.684 | **0.834** |
| 200 | 0.224 | 0.753 | 0.662 | **0.770** |
| 500 | 0.017 | 0.682 | 0.554 | **0.720** |

### ProteinGym SPG1 (250 trials, fitness-biased shift β=1.0)

| n    | Classical | PPI++ | **Adaptive** | MSE reduction |
|------|-----------|-------|--------------|---------------|
| 200  | 0.198     | 0.166 | **0.556**    | 46%           |
| 1500 | 0.000     | 0.000 | **0.029**    | 61%           |

