# Architecture 3 — Locked Specification

Locked after M13 §4.2 (`M13 CHECKS OK`). Defaults below are the pilot configuration.

## Backbone

| Item | Value |
|---|---|
| Encoder | OpenCLIP ViT-L/14, `openai`, `force_quick_gelu=True` |
| Projection | removed (`encoder.proj = None`) → 1024-d CLS |
| Frozen | all encoder weights except LayerNorms rewritten by SB-LN |
| Input | \(224\times224\) face crop, CLIP mean/std |
| Feature | \(z = \mathrm{normalize}(\mathrm{CLS})\) (L2) |

## Shared adapter — SB-LN

For each of the \(L=50\) LayerNorms (width \(W=1024\)):

\[
\gamma_l = \gamma_l^0 + B_\gamma a_l,\qquad
\beta_l = \beta_l^0 + B_\beta b_l
\]

| Symbol | Shape | Notes |
|---|---|---|
| \(B_\gamma,B_\beta\) | \(1024\times r\) | shared bases; QR-init, scale \(\sqrt{W/r}\) |
| \(A=[a_l]\), \(C=[b_l]\) | \(50\times r\) | per-LN coefficients; init 0 |
| \(\gamma^0,\beta^0\) | from CLIP | frozen (stored in \(\theta_0\)) |
| \(r\) | 16 | default |

SB-LN parameter count: \(2\cdot1024\cdot r + 2\cdot50\cdot r = 34{,}368\) at \(r=16\).

## Prototype head (deployed)

Buffers (not trained as parameters):

- \(\mu_\text{real},\mu_\text{fake}\in\mathbb{R}^{1024}\) — EMA during train; exact refresh after training  
- \(c_r=\mathrm{normalize}(\mu_\text{real})\)  
- At inference: \(c_f=\mathrm{normalize}(\mu_\text{fake})\) (equivalent to \(\delta=0\))

Trained scalars: \(b\in\mathbb{R}\), \(\rho=\log\tau\in\mathbb{R}\) (\(\tau_0=0.1\)).

\[
w=\mathrm{normalize}(c_f-c_r),\quad
\bar m=\tfrac{c_r+c_f}{2},\quad
s=\frac{(z-\bar m)^\top w - b}{\tau},\quad
p=\sigma(s)
\]

Video score = mean of frame probabilities over 32 frames.

## Private latent (training only)

| Symbol | Shape | Notes |
|---|---|---|
| \(\delta_g\) | \(d\) per train generator \(g\) | init 0; Adam-trained |
| \(W\) | \(1024\times d\) | fixed orthonormal (QR); **not** trained |
| \(d\) | 64 | default |
| \(G\) | 4 | Deepfakes, Face2Face, FaceSwap, NeuralTextures |

\[
c_f(g)=\mathrm{normalize}\big(\mu_\text{fake}+W\delta_g\big)
\]

Identity: \(\delta_g=0\Rightarrow c_f(g)=c_f\) (shared). Inference always uses \(\delta=0\).

During a pure batch of generator \(g\), the **training** head uses \(c_f(g)\) (so \(w_g,\bar m_g\) depend on \(\delta_g\)).

## Loss (training)

\[
\begin{aligned}
\mathcal{L}
&= \mathcal{L}_{\mathrm{M1}}
+ \alpha\,\mathbb{E}_{\mathrm{real}}[1-\cos(z,c_r)]
+ \beta\,\mathbb{E}_{\mathrm{fake}}[\mathrm{relu}(\cos(z,c_r)-m)] \\
&\quad+ \gamma\,\mathbb{E}_{\mathrm{fake}}[1-\cos(z,c_f(g))]
+ \lambda\|\delta_g\|^2
\end{aligned}
\]

\(\mathcal{L}_{\mathrm{M1}}=\mathrm{CE}+0.1\cdot\mathrm{align}+0.5\cdot\mathrm{uniform}\) (same form as M1).

| Coeff | Symbol in code | Default |
|---|---|---|
| \(\alpha\) | `ALPHA_REAL` | 0.1 |
| \(\beta\) | `BETA_MARGIN` | 1.0 |
| \(m\) | `MARGIN` | 0.1 |
| \(\gamma\) | `GAMMA_ATTR` | 0.5 |
| \(\lambda\) | `LAMBDA` | 0.01 |

The \(\gamma\) term is the **private–loss link** (PFAL: private-feature attraction).

## Parameter totals

| Scope | Count |
|---|---|
| Deployed | \(34{,}368+2=34{,}370\) |
| Train-time | \(34{,}370+G\cdot d=34{,}626\) |
| Cap | both &lt; 104{,}450 |

## Implementation names (M13)

| Concept | Code |
|---|---|
| State | `Arch3State` |
| Rebuild LN θ | `state.theta()` |
| Chain-rule SB-LN grads | `state.sb_grads` |
| Centroids / head parts | `c_real`, `c_fake`, `head_parts` |
| Loss | `arch3_loss` |
| Step | `train_step` |
| Predict (δ=0) | `predict_probs` |
