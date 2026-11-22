# Architecture 3 vs Prior Models

Video AUC % (seed 0) for references; Arch 3 accuracy **pending pilot**.

## Structural comparison

| | Arch 1 (GBLA) | Arch 2 (scratched) | A0 (M11a) | M11b | **Arch 3** |
|---|---|---|---|---|---|
| LN path | full 50 LNs | SB-LN + LN-\(\delta\) | SB-LN | SB-LN + LN-\(\delta\) | **SB-LN only** |
| Head | linear 2,050 | linear / proto | linear 2,050 | prototype | **prototype** |
| Private latent | \(\delta\) on \(\gamma,\beta\) | \(\delta\) on \(\gamma,\beta\) under SB-LN | none | LN-\(\delta\) | **\(\delta\) in feature space** |
| Loss↔private | \(\lambda\|\delta\|^2\) | \(w_g\cdot\)CE / GBRA weights | — | GBRA + \(w_g\) | **PFAL \(\gamma\)-attract to \(c_f(\delta)\)** |
| Deployed \(P\) | 104,450 | ~34k–36k | 36,418 | 34,370 | **34,370** |
| Status | published | scratched (M12) | side result | side / rejected as main | **design locked** |

## Why Arch 2 was scratched (M12 deltas vs A0)

| Run | Δseen | ΔNT | ΔFSh | ΔDFD | Read |
|---|---:|---:|---:|---:|---|
| A1 prototype, no offsets | +0.22 | +0.86 | +0.56 | −0.33 | prototype OK |
| A2 linear + LN offsets | −0.60 | −2.47 | −0.51 | +0.50 | offsets hurt NT |
| A3 + \(w_g\cdot\)CE | −5.32 | −18.59 | −0.20 | +0.21 | weighting collapses NT |
| M11b full stack | — | 77.91 abs | 93.01 abs | 95.56 | FSh up, NT down |

Nothing under SB-LN + LN-offsets hit DFD ≥ 96. Path 3 of M12: keep Arch 1 as main; SB-LN as side result only.

## What Arch 3 reuses vs invents

| Reused | New |
|---|---|
| CLIP ViT-L/14 pipeline, data order, schedule | Feature-space \(\delta_g\) + fixed \(W\) |
| SB-LN construction (M11a) | \(c_f(g)=\mathrm{norm}(\mu_f+W\delta_g)\) |
| Prototype head idea (M10b / M12 A1) | PFAL: \(\gamma(1-\cos(z,c_f(g)))+\lambda\|\delta\|^2\) |
| Deploy/discard private tensors (GBLA pattern) | Private locus ≠ LN values (avoids M12 failure mode) |
| M1 CE+align+uniform | Combined with real-anchor + margin + PFAL |

## Reference absolute numbers (for the pilot decision)

| Model | Seen | NT | FSh | DFD |
|---|---:|---:|---:|---:|
| A0 SB-LN r=16 + linear | 94.54 | **81.21** | 92.25 | 95.15 |
| Arch 1 GBLA λ=0.01 | 95.55 | 85.06 | 92.91 | **96.71** |
| Full LN λ=∞ | 96.41 | 87.90 | 90.53 | 96.05 |
| M11b | 93.66 | 77.91 | 93.01 | 95.56 |
| Arch 3 | — | — | — | — |
