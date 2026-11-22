# Architecture 3 — Parameters, Budget, Schedule

## Hyperparameters (pilot defaults)

| Name | Code | Value | Notes |
|---|---|---|---|
| SB-LN rank | `RANK` | 16 | same as M11a A0 |
| Private dim | `DELTA_DIM` | 64 | feature-space \(\delta_g\) |
| LN / δ penalty | `LAMBDA` | 0.01 | \(\lambda\|\delta_g\|^2\) |
| Real pull | `ALPHA_REAL` | 0.1 | \(\alpha\) |
| Fake margin weight | `BETA_MARGIN` | 1.0 | \(\beta\) |
| Margin threshold | `MARGIN` | 0.1 | \(m\) |
| Attract to \(c_f(\delta)\) | `GAMMA_ATTR` | 0.5 | \(\gamma\) — δ↔loss link |
| Align / uniform | `W_ALIGN`, `W_UNIFORM` | 0.1, 0.5 | M1 |
| Prototype τ init | `TAU_INIT` | 0.1 | \(\rho=\log\tau\) |
| EMA momentum | `EMA` | 0.99 | running \(\mu\) |
| Proto warm count | `PROTO_WARM` | 512 | reserved for warm-start helpers |
| Batch size | `BATCH_SIZE` | 128 | |
| Epochs | `EPOCHS` | 10 | |
| LR schedule | `LR_MAX`→`LR_MIN` | \(3\cdot10^{-4}\)→\(10^{-5}\) | cosine, 1-ep warmup, 10-ep cycle |
| Seed | `SEED` | 0 | |
| Augment | `AUGMENT` | True | train only |
| Eval frames / video (val) | `VAL_FRAMES_PER_VIDEO` | 8 | monitor only |
| Test frames / video | — | 32 | as M1/M4 |

All of the above are fixed **before** the pilot; nothing is selected on test data.

## Parameter budget (r=16, d=64, G=4)

| Piece | Count | Deployed? |
|---|---:|:---:|
| \(B_\gamma,B_\beta\) (\(2\cdot1024\cdot16\)) | 32,768 | yes |
| \(A,C\) (\(2\cdot50\cdot16\)) | 1,600 | yes |
| \(b,\rho\) | 2 | yes |
| **Total deployed** | **34,370** | |
| \(\delta_g\) (\(4\cdot64\)) | 256 | no |
| \(W\) (fixed) | 0 trained | buffer |
| \(\mu_r,\mu_f\) | 0 trained | buffer |
| **Total train-time** | **34,626** | |

Comparison:

| Model | Deployed |
|---|---:|
| Arch 1 GBLA / full LN | 104,450 |
| A0 SB-LN + linear (M11a) | 36,418 |
| **Arch 3** | **34,370** |

Reduction vs Arch 1: \(104{,}450/34{,}370\approx3.04\times\).

## Data / loader (unchanged from M4)

| Item | Value |
|---|---|
| Train crops | 115,200 (real + 4 gens) |
| Steps / epoch | 900 (4×225 pure-generator blocks) |
| Real ownership | round-robin, \(1/G\) per generator, fixed for the run |
| FaceShifter | unseen (test only) |

## Optimiser

Custom Adam slots (no weight decay) on every trained tensor in `AdamSlots.tensors(state)`:

- Shared: `Bg`, `Bb`, `A`, `C`, `b`, `rho`  
- Private: `delta:Deepfakes`, `delta:Face2Face`, `delta:FaceSwap`, `delta:NeuralTextures`

Only the active generator’s \(\delta_g\) receives a nonzero gradient on that block.
