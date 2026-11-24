# docs4 — Architecture 4 candidates (M15a / M15b / M15c)

Three **independent** designs after Arch 3 was rejected. Same four aims; different private-latent loci. Pick a winner after all three pilots.

| Aim | Requirement |
|---|---|
| 1 | Something new instead of LayerNorms + linear head alone |
| 2 | New loss function |
| 3 | Private latent linked to that loss |
| 4 | Reduce below 104,450 params if possible |

| Candidate | Notebook | TeX diagram | Private latent lives in | Shared adapter |
|---|---|---|---|---|
| **A** | `M15a.ipynb` | `arch4a_angular_centers.tex` | Angular fake centers \(c_g\) | SB-LN |
| **B** | `M15b.ipynb` | `arch4b_frozen_bottleneck.tex` | Bottleneck adapters \(U_g\) | Frozen CLIP (no LN tune) |
| **C** | `M15c.ipynb` | `arch4c_score_space.tex` | Score-space \((m_g,\tau_g)\) | Full LayerNorms |

Arch 1 (GBLA) stays the published main model until a candidate **KEEPs** its pre-registered rule.

## Compile diagrams

```bash
cd docs4
tectonic arch4a_angular_centers.tex
tectonic arch4b_frozen_bottleneck.tex
tectonic arch4c_score_space.tex
```

## Decision (same honesty as Arch 3)

For each pilot: **KEEP** iff NT ≥ A0 (81.21) and DFD ≥ 95.5; else REJECT that candidate. Compare survivors (or all three as negative evidence) after M15a–c finish.
