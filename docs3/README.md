# docs3 — Architecture 3 (Arch 3)

Complete documentation for **Architecture 3**, implemented and correctness-checked in `M13.ipynb`.

| Status | Detail |
|---|---|
| Design | Finalised after M13 checks (`M13 CHECKS OK`) |
| Spec on cluster | `runs/m13/arch3_spec.json` |
| Pilot train | **`M14.ipynb`** (M13 left as checks-only) |
| Arch 1 | Published (GBLA, full LN + linear head) — remains main model until pilot passes |
| Arch 2 | Scratched after M12 (SB-LN + LN-value offsets) |

## Files in this folder

| File | Contents |
|---|---|
| [README.md](README.md) | This index |
| [arch3_overview.md](arch3_overview.md) | Narrative: motivation, aims, what Arch 3 is |
| [arch3_spec.md](arch3_spec.md) | Locked technical specification |
| [arch3_equations.md](arch3_equations.md) | Maths: SB-LN, private latent, head, loss |
| [arch3_parameters.md](arch3_parameters.md) | Hyperparameters, param budget, schedule |
| [arch3_inference.md](arch3_inference.md) | Deployed / inference path (δ = 0) |
| [arch3_training.md](arch3_training.md) | Training path, private δ, data order |
| [arch3_vs_prior.md](arch3_vs_prior.md) | Arch 3 vs Arch 1 / Arch 2 / A0 / M11b |
| [arch3_novelty.md](arch3_novelty.md) | Claimed novelties N1–N6 and four aims |
| [arch3_checks.md](arch3_checks.md) | M13 correctness-check results |
| [arch3_decision_rule.md](arch3_decision_rule.md) | Pre-registered keep / reject rule for the pilot |
| [arch3_architecture_landscape.tex](arch3_architecture_landscape.tex) | 5-page landscape diagram (compile with tectonic / pdflatex) |

## Quick numbers (defaults)

- Rank \(r = 16\), private dim \(d = 64\), seed 0, 10 epochs  
- Deployed trainable: **34,370**  
- Train-time (incl. \(G \cdot d\)): **34,626**  
- Both &lt; 104,450  

## Compile the diagram

```bash
cd docs3
tectonic arch3_architecture_landscape.tex
# or: pdflatex arch3_architecture_landscape.tex
```
