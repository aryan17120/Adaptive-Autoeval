# Experimental Results

## ImageNet: Synthetic Covariate Shift

**Setup**: 50,000 images, 5 ResNet models, 250 Monte Carlo trials.
Shift: labeled data sampled ∝ ResNet-101 confidence score.

| n   | Classical | PPI++ | Oracle PPI++ | **Adaptive** | ESS  |
|-----|-----------|-------|--------------|--------------|------|
| 50  | 0.635     | 0.793 | 0.700        | **0.789**    | 1.22 |
| 100 | 0.461     | 0.823 | 0.684        | **0.834**    | 1.12 |
| 200 | 0.224     | 0.753 | 0.662        | **0.770**    | 1.04 |
| 300 | 0.098     | 0.734 | 0.609        | **0.768**    | 0.99 |
| 400 | 0.065     | 0.696 | 0.578        | **0.739**    | 0.97 |
| 500 | 0.017     | 0.682 | 0.554        | **0.720**    | 0.95 |

**Key finding**: Adaptive AutoEval outperforms Oracle PPI++ at
every sample size. Raw oracle weights have high variance (ESS < 1);
logistic regression implicitly regularizes the density ratio.

---

## ProteinGym SPG1: Fitness-Biased Labeling

**Setup**: 536,962 protein variants, 7 foundation models,
fitness-biased shift β=1.0, VESPA annotator, 250 trials.

| n    | Classical | PPI++ | **Adaptive** | MSE (Classical) | MSE (Adaptive) | ESS  |
|------|-----------|-------|--------------|-----------------|----------------|------|
| 200  | 0.198     | 0.166 | **0.556**    | 0.577           | **0.311**      | 0.89 |
| 600  | 0.007     | 0.005 | **0.194**    | 0.531           | **0.226**      | 0.82 |
| 1000 | 0.000     | 0.000 | **0.091**    | 0.500           | **0.202**      | 0.82 |
| 1500 | 0.000     | 0.000 | **0.029**    | 0.482           | **0.187**      | 0.80 |

**Key finding**: 46–60% MSE reduction vs. classical.
Coverage improves 2.8× at n=200 (0.198 → 0.556).

---

## Ablation Studies (ImageNet, n=200)

### A1: Classifier Capacity

| Classifier          | Coverage (n=200) | Coverage (n=500) |
|---------------------|------------------|------------------|
| Logistic Regression | 0.830            | 0.697            |
| MLP (64×64)         | 0.840            | 0.798            |
| Random Forest       | 0.834            | 0.828            |

### A2: Clipping Threshold

| ε     | Coverage | ESS   |
|-------|----------|-------|
| 0.001 | 0.830    | 1.234 |
| 0.010 | 0.823    | 1.225 |
| 0.050 | 0.845    | 1.231 |
| 0.100 | 0.834    | 1.237 |
| 0.200 | 0.834    | 1.457 |

### A3: Shift Severity

| β   | Classical Cov | **Adaptive Cov** | MSE Reduction |
|-----|--------------|------------------|---------------|
| 0.0 | 0.909        | 0.886            | 28%           |
| 0.5 | 0.749        | 0.877            | 54%           |
| 1.0 | 0.462        | 0.834            | 71%           |
| 2.0 | 0.067        | 0.777            | 82%           |
| 3.0 | 0.006        | 0.745            | 82%           |

---

## Spearman Rank Correlations (ProteinGym, n=600, 100 trials)

| Method    | Mean ρ | Std   |
|-----------|--------|-------|
| Classical | 0.941  | 0.053 |
| PPI++     | 0.938  | 0.054 |
| Adaptive  | 0.860  | 0.193 |

Note: All methods preserve gross ranking. Adaptive AutoEval's
advantage is in correcting **absolute** MSE estimates, not rankings.