# Architecture 3 — Novelties

Accuracy claims are **pending** the M13 pilot. Implementation / gradient / budget claims are **checked**.

## N1 — Shared-Basis LayerNorm (SB-LN)

All 50 CLIP LayerNorm updates share one \(r\)-dimensional channel basis; each LN only learns \(2r\) coefficients. Deployed LN path: \(2\cdot1024r+100r\) (34,368 at \(r=16\)).

- **Vs Arch 1:** independent LNs (102,400).  
- **Role in Arch 3:** compression backbone; not the sole novelty.

## N2 — Prototype head (2 trained numbers)

Direction \(w\) and midpoint \(\bar m\) from real/fake centroids; only \(b,\tau\) trained. Replaces the 2,050-parameter linear head.

- **Vs Arch 1:** linear \(1024\to2\).  
- **Evidence so far:** M12 A1 ≈ A0 under SB-LN (prototype not the NT problem).

## N3 — Feature-space private latent

\(\delta_g\in\mathbb{R}^d\) with fixed orthonormal \(W\);  
\(c_f(g)=\mathrm{normalize}(\mu_f+W\delta_g)\).

- **Vs Arch 1 / Arch 2:** private structure is **not** an offset on LayerNorm \(\gamma,\beta\).  
- **Why:** M12 showed LN offsets under SB-LN hurt NT.

## N4 — δ-linked loss (PFAL)

**Private-Feature Attraction Loss:**

\[
\gamma\,\mathbb{E}_{\mathrm{fake}\in g}\big[1-\cos(z,c_f(g))\big]+\lambda\|\delta_g\|^2
\]

on top of M1 + real-anchor + margin. The private latent is part of the **definition** of the objective (through \(c_f(g)\)), not only a free penalty on LN space.

- **Vs Arch 1:** \(\lambda\|\delta\|^2\) on LN offsets.  
- **Vs Arch 2:** \(w_g\cdot\)CE from \(\|\delta\|\) (failed under SB-LN).

## N5 — Train / deploy split

Deploy SB-LN + prototype only (34,370). Discard \(\delta_g\) after training (GBLA-style split, new content of what is private).

## N6 — Parameter efficiency

\(3.04\times\) fewer deployed numbers than Arch 1; private train-time overhead only \(G\cdot d=256\).

---

## Four aims ↔ pieces

| Aim | How Arch 3 meets it |
|---|---|
| 1. New vs LN + linear head | SB-LN + prototype (N1, N2) |
| 2. New loss | M1 + real-anchor + margin + **PFAL** (N4) |
| 3. Private latent ↔ loss | \(\delta_g\) builds \(c_f(g)\); PFAL + training head use it (N3, N4) |
| 4. \(P<104{,}450\) | Deployed 34,370; train-time 34,626 (N5, N6) |
