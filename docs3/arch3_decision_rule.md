# Architecture 3 — Pre-Registered Decision Rule

Fixed **before** pilot training. Do not change after seeing test numbers.

## Keep Arch 3 as a candidate second architecture iff

| Criterion | Threshold | Reference |
|---|---|---|
| NeuralTextures video AUC | \(\ge 81.21\) | A0 (M11a SB-LN r=16) |
| DFD video AUC | \(\ge 95.5\) | soft floor below Arch 1’s 96.71 |

**Also report** (not hard gates): FaceShifter vs A0 (92.25) and vs Arch 1 (92.91); seen combined vs A0 (94.54).

## Else

- Arch 3 is a **negative / exploratory** result.  
- **Arch 1 (GBLA)** remains the only main / published architecture.  
- SB-LN A0/A1 may still appear as a **compression side result**, unrelated to claiming Arch 3.

## Pilot configuration (do not retune on test)

- \(r=16\), \(d=64\), seed 0, 10 epochs  
- Loss coeffs as in `arch3_parameters.md`  
- Same M4 data order / LR schedule  

## After a keep

Possible follow-ups (only if the rule passes):

1. Second seed  
2. \(d\in\{32,64,128\}\) ablation  
3. \(\gamma\in\{0,0.5,1.0\}\) (turns PFAL off/on)  
4. Celeb-DF / extra datasets as for Arch 1 paper extras  

## After a reject

Do not invent a new “Arch 3b” from the same failed run by changing thresholds. Either stop, or design a **new** architecture with a new pre-registered rule.
