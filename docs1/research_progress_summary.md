# Research progress summary (state at commit 06aedba, 5 Nov 2026)

Full record: `complete_research_progress_report.pdf` (source `complete_research_progress_report.tex`).
Every number below comes from the executed notebooks M0–M9b.

## Model actually used
- Frozen OpenAI CLIP ViT-L/14 visual encoder (`open_clip`, QuickGELU), CLS after `ln_post`, CLIP projection removed, L2-normalised, linear head 1024 → 2.
- Trainable: 50 LayerNorms (102,400) + head (2,050) = **104,450 of 303,181,826 (0.0345%)**; frozen 303,077,376.
- Not implemented: LoRA, codebook / vector quantisation, slerp. The seeded latent θ = θ0 + αWz was implemented in M2/M2b and failed.

## Method (GBLA)
Training: θ_g = θ_s + δ_g per generator, penalty λ‖δ_g‖², pure one-generator batches (4 blocks × 225 steps per epoch). Deployed: θ_s only (104,450 numbers). Chosen λ = 0.01.

## Main results (video AUC %, single seed)
| Test | LNCLIP-DF repro (M1, ep. 10) | Pooled, same batching (λ=∞) | GBLA λ=0.01 |
|---|---|---|---|
| FF++ FaceShifter (unseen) | 88.19 | 90.53 | **92.91** |
| LOGO mean held-out (5 folds) | – | 87.49 | **88.04** |
| LOGO held-out NeuralTextures | – | **64.40** | 62.29 |
| DFD | 95.88 | 96.05 | **96.71** [95.30, 98.21] |
| UADFV | 99.25 | 99.38 | 99.63 (not significant) |

The M1 reproduction at 20 epochs: seen 98.04, FaceShifter 88.37 (paper 98.0 / 88.2).

## Negative results
- M2/M2b: latent sizes d ≤ 1024 lose most of the adaptation; the compact-payload idea was dropped.
- M3: the curvature (Hessian) model of generator weighting failed its own checks. Only the geometric dominance measurement survived (F2F 0.58, NT ≈ 0, G_eff/G = 0.57).
- M7/M7b: G_eff does not predict OOD AUC once held-out identity is controlled (residual correlation 0.007).

## Reference values that could not be verified
786,432 trainable (0.2587%), LoRA on q_proj/v_proj (96 tensors), codebook 97/1024 and 46/1024, best epoch 2/10, accuracies 98.38 / 97.56 / 97.125%, val loss 0.134335. None of these appears in any project file. Two of the numbers do appear, with other meanings:
- 303,966,208 is the full ViT-L/14 tower including the projection.
- 786,432 is that unused projection.

## Open issues
Single seed; GBLA hurts held-out NT; UADFV at ceiling; λ chosen on FaceShifter; the mid-semester report (3 Nov) still describes the abandoned latent design.
