# Architecture 3 — Inference

Inference uses **only the shared model**. All private \(\delta_g\) are discarded (same philosophy as GBLA, different latent locus).

## Pipeline

1. Sample 32 evenly spaced frames; JPEG q95 (as M1).  
2. MTCNN largest face \(p\ge0.9\); \(1.3\times\) square crop → \(224\times224\).  
3. CLIP mean/std normalisation.  
4. Forward through frozen CLIP ViT-L/14 with **SB-LN** LayerNorms (\(\gamma,\beta\) from shared \(B,A,C\)).  
5. L2-normalise CLS → \(z\).  
6. Prototype head with **shared** centroids (\(\delta=0\)):

\[
c_r=\mathrm{norm}(\mu_r),\quad
c_f=\mathrm{norm}(\mu_f),\quad
s=\frac{(z-\bar m)^\top w-b}{\tau},\quad
p=\sigma(s).
\]

7. Video score = mean of frame \(p\) over the 32 frames.

## What is loaded at test time

| Tensor | Required |
|---|---|
| SB-LN: \(B_\gamma,B_\beta,A,C\) | yes |
| \(b,\rho\) | yes |
| \(\mu_r,\mu_f\) (exact-refreshed preferred) | yes |
| \(\delta_g\) | **no** |
| \(W\) | **no** (only used if δ≠0) |
| Linear `nn.Linear` head weights | unused (kept in checkpoint machinery only) |

## Code path (M13)

```text
predict_probs(state, arr)
  set_flat(state.theta())          # SB-LN → LayerNorms
  hp = state.head_parts(g=None)    # δ discarded
  z = encode(x)
  p = sigmoid(head_logit(z, b, rho, hp))
```

## Outputs reported

Same metrics tables as M4/M11/M12:

- FF++: seen combined, per-generator (DF, F2F, FS, NT), FaceShifter (unseen)  
- DFD, UADFV (optional flags `RUN_DFD`, `RUN_UADFV`)  
- Frame AUC, video AUC, video acc @ 0.5  
