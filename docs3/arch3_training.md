# Architecture 3 — Training

## Data order

Identical to M4 / M11 / M12:

- Each epoch = 4 pure-generator blocks (shuffled order).  
- Block of generator \(g\): 225 steps × batch 128 = fakes of \(g\) + reals owned by \(g\).  
- 900 steps / epoch × 10 epochs.

## One step (generator \(g\))

1. Rebuild \(\theta\) from SB-LN: `set_flat(state.theta())`.  
2. Forward encoder → \(z=\mathrm{normalize}(\mathrm{CLS})\) (bf16 autocast).  
3. Build head from \(c_f(g)=\mathrm{norm}(\mu_f+W\delta_g)\) and shared \(c_r\).  
4. Compute `arch3_loss` (M1 + real-anchor + margin + attract-to-\(c_f\) + \(\lambda\|\delta_g\|^2\)).  
5. Backward:  
   - LN grads → `sb_grads` → \(\partial/\partial(A,B_\gamma,C,B_\beta)\)  
   - leaf grads for \(b,\rho,\delta_g\)  
6. Adam update on those tensors (other generators’ \(\delta_{g'}\) untouched).  
7. EMA-update \(\mu_r,\mu_f\) from this batch’s \(z\) (no grad).

## Prototypes

| Phase | Behaviour |
|---|---|
| During training | EMA (`EMA=0.99`) of real and **shared** fake means |
| After epoch 10 | Exact refresh on unaugmented train crops (`exact_prototypes`) with deployed SB-LN |
| Per-generator fake means | **Not** stored separately; generator identity enters only through \(\delta_g\) |

## Private latent lifecycle

```text
init:     δ_g = 0  ∀g
train:    only active g gets ∇_δ from CE(head(c_f(g))) + γ·attr + 2λδ
deploy:   discard all δ_g; head uses shared μ_f only
```

## Checkpoint contents (`Arch3State.save`)

- `architecture: "Arch3"`  
- `rank`, `delta_dim`  
- `phi`: \(B_\gamma,B_\beta,A,C,b,\rho\)  
- `delta`: all \(\delta_g\) (for resume; not for deploy metrics)  
- `W`, `proto` (\(\mu_r,\mu_f\), counts)  
- `theta`: flattened LN+head rebuild (compat)

## Planned run folder

```text
runs/m13/m13_arch3_r16_d64_ViT-L-14_seed0/
  config.json
  epoch_XX.pt / last.pt
  results.json
  …
```

Pilot train + eval live in **`M14.ipynb`** (`train_arch3`, decision vs A0 / Arch1). Run dir: `runs/m14/m14_arch3_r16_d64_…`. Resumable via `last.pt`. M13 stays checks-only.
