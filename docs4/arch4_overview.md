# Architecture 4 — Three candidates

After Arch 3 **REJECT** (NT 78.2 vs A0 81.2), we do **not** retry SB-LN + feature/LN δ. Three new loci for the private latent:

| ID | Idea | Why it might work | Main risk |
|---|---|---|---|
| **4A / M15a** | Private **angular centers** \(c_g\) + SB-LN | Private object = classification geometry; ≪104k | SB-LN may still hurt NT |
| **4B / M15b** | Frozen CLIP + **bottleneck adapters** \(U_g\) | Cleanest “not LN+linear”; tiny private | Absolute AUC may lag Arch 1 badly |
| **4C / M15c** | Full LN + **score-space** \((m_g,\rho_g)\) | Keeps LN capacity; novelty in head/loss | Deployed still ~102k |

Shared pre-registered rule for each: **KEEP** iff NT ≥ 81.21 and DFD ≥ 95.5.

Diagrams: `arch4a_*.tex`, `arch4b_*.tex`, `arch4c_*.tex`.
