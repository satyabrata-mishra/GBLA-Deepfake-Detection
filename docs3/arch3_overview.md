# Architecture 3 — Overview

## Context

| Architecture | What it is | Status |
|---|---|---|
| **Arch 1** | Full LayerNorm tuning + linear head + GBLA private LN offsets | **Published** (main model) |
| **Arch 2** | SB-LN + private **LN-value** offsets ± \(w_g\cdot\)CE / GBRA | **Scratched** after M12 |
| **Arch 3** | SB-LN + prototype head + **feature-space** private \(\delta_g\) + δ-linked loss | **Design locked** (M13 checks OK); pilot pending |

M12 showed that putting private offsets on LayerNorm values under SB-LN hurts NeuralTextures (A2 ΔNT = −2.5; A3 \(w_g\cdot\)CE ΔNT = −18.6). Arch 3 keeps parameter compression (SB-LN) but moves the private latent into **feature space** and defines a **new loss term through that latent**.

## Four design aims (set before Arch 3)

1. Introduce something new instead of “LayerNorms + linear head” alone.  
2. Introduce a new loss function.  
3. Link a new private latent space to that loss.  
4. Reduce below 104,450 trainable parameters if possible.

## One-sentence definition

**Arch 3** is a CLIP ViT-L/14 detector that adapts all 50 LayerNorms through a shared low-rank **SB-LN**, classifies with a **2-parameter prototype head**, and during training only maintains per-generator private vectors \(\delta_g\in\mathbb{R}^d\) that shift the fake centroid in feature space; the loss **attracts** fakes of generator \(g\) to that δ-dependent centroid and penalises \(\|\delta_g\|^2\). At inference all \(\delta_g\) are discarded.

## What is trained / deployed

| Piece | Train | Deploy |
|---|---|---|
| SB-LN bases \(B_\gamma,B_\beta\) + coeffs \(A,C\) | yes | yes |
| Prototype scalars \(b,\rho=\log\tau\) | yes | yes |
| Private \(\delta_g\) (one per train generator) | yes | **no** |
| Fixed orthonormal \(W\in\mathbb{R}^{1024\times d}\) | no (buffer) | unused at test (δ=0) |
| EMA / exact \(\mu_\text{real},\mu_\text{fake}\) | buffers | yes (refreshed) |
| Linear head inside the Detector module | present for `set_flat` compat | **not used** |

## Code entry points

| Item | Location |
|---|---|
| Checks notebook | `M13.ipynb` (§4.2 → `M13 CHECKS OK`) |
| Train notebook | `M14.ipynb` (pilot + decision) |
| State class | `Arch3State` |
| Loss | `arch3_loss` |
| Cluster spec | `runs/m13/arch3_spec.json` |
| Pilot run dir | `runs/m14/m14_arch3_r16_d64_ViT-L-14_seed0` |

## Reading order in this folder

1. This overview  
2. `arch3_spec.md` + `arch3_equations.md`  
3. `arch3_inference.md` / `arch3_training.md`  
4. `arch3_novelty.md` + `arch3_vs_prior.md`  
5. `arch3_checks.md` + `arch3_decision_rule.md`  
6. Diagram: `arch3_architecture_landscape.tex`
