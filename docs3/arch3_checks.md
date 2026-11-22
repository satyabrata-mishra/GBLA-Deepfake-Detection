# Architecture 3 — M13 Correctness Checks

Source: executed `M13.ipynb` §4.2 on NVIDIA A100-SXM4-40GB.  
Result: **`M13 CHECKS OK`** → wrote `runs/m13/arch3_spec.json`.

## Gate table

| Check | Result |
|---|---|
| deployed and train-time \(P < 104{,}450\) | True (34,370 / 34,626) |
| start \(\theta = \theta_0\) | True |
| \(\delta=0\Rightarrow c_f=\) shared | True |
| nonzero \(\delta\) moves \(c_f\) | True |
| loss depends on \(\delta\) | True |
| step moves \(A,C,b,\rho,\delta_g\) | True |
| other generators’ \(\delta\) stay 0 | True |
| bases unchanged after step 1 | True |
| finite differences agree | True |

## Finite-difference table

Directional derivative on \(A,C,b,\rho,\delta\) (not bases), assembled vs central FD:

| eps | fd | auto | rel |
|---:|---:|---:|---:|
| 0.0100 | 0.040995 | 0.040954 | 0.001015 |
| 0.0030 | 0.040978 | 0.040954 | 0.000598 |
| 0.0010 | 0.040963 | 0.040954 | **0.000234** |
| 0.0003 | 0.040059 | 0.040954 | 0.021840 |

Best relative error **\(2.34\cdot10^{-4}\)** (threshold 0.02).

## First-step movement

`moved (expect): {'A': True, 'C': True, 'b': True, 'rho': True, 'delta:Deepfakes': True}`  
Bases `Bg`, `Bb` unchanged (chain rule at \(A=C=0\)).

## Param report (from §3.2)

| piece | trainable | deployed |
|---|---:|:---:|
| SB-LN (r=16) | 34,368 | yes |
| prototype (b, ρ) | 2 | yes |
| private δ_g (4×64) | 256 | no |
| W (fixed orthonormal) | 0 | no |
| TOTAL deployed | 34,370 | |
| TOTAL train-time | 34,626 | |

## What checks do **not** prove

- Test AUC / NT / DFD / FSh (needs pilot train).  
- That Arch 3 beats Arch 1 (decision rule in `arch3_decision_rule.md`).  
- Sensitivity to \(d\), \(\gamma\), or seed.
