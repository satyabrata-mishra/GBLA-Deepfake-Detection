# Architecture 3 — Equations

Notation: width \(W=1024\), \(L=50\) LayerNorms, rank \(r\), private dim \(d\), train generators \(\mathcal{G}=\{1,\ldots,G\}\), \(G=4\).

## 1. Shared-Basis LayerNorm (SB-LN)

Let \(\gamma_l^0,\beta_l^0\in\mathbb{R}^W\) be the frozen pretrained LN parameters for layer \(l\).

Shared bases \(B_\gamma,B_\beta\in\mathbb{R}^{W\times r}\) and coefficient rows \(a_l,b_l\in\mathbb{R}^r\):

\[
\gamma_l=\gamma_l^0+B_\gamma a_l,\qquad
\beta_l=\beta_l^0+B_\beta b_l.
\]

In matrix form (\(A\in\mathbb{R}^{L\times r}\) stacks \(a_l\)):

\[
\Delta\Gamma = A B_\gamma^\top\in\mathbb{R}^{L\times W},\quad \mathrm{rank}(\Delta\Gamma)\le r.
\]

Same for shifts with \(C,B_\beta\).

**Chain rule** (from flat LN grads \(G_\gamma,G_\beta\in\mathbb{R}^{L\times W}\)):

\[
\frac{\partial\mathcal{L}}{\partial A}=G_\gamma B_\gamma,\quad
\frac{\partial\mathcal{L}}{\partial B_\gamma}=G_\gamma^\top A,\quad
\frac{\partial\mathcal{L}}{\partial C}=G_\beta B_\beta,\quad
\frac{\partial\mathcal{L}}{\partial B_\beta}=G_\beta^\top C.
\]

At init \(A=C=0\), \(\partial\mathcal{L}/\partial B_\gamma=0\) (bases do not move on step 1).

## 2. Feature-space private latent

Fixed orthonormal \(W\in\mathbb{R}^{W\times d}\) (QR of a Gaussian matrix; not trained).  
Private \(\delta_g\in\mathbb{R}^d\) for each \(g\in\mathcal{G}\).

Shared buffers \(\mu_r,\mu_f\in\mathbb{R}^W\) (EMA of real / fake features).

\[
c_r=\frac{\mu_r}{\|\mu_r\|},\qquad
c_f(g)=\frac{\mu_f+W\delta_g}{\|\mu_f+W\delta_g\|}.
\]

Special case:

\[
\delta_g=0 \;\Rightarrow\; c_f(g)=\frac{\mu_f}{\|\mu_f\|}\equiv c_f^{\mathrm{shared}}.
\]

Inference uses \(c_f^{\mathrm{shared}}\) only.

## 3. Prototype head

For a batch labelled with generator \(g\) (training):

\[
w_g=\frac{c_f(g)-c_r}{\|c_f(g)-c_r\|},\qquad
\bar m_g=\frac{c_r+c_f(g)}{2}.
\]

Logit and probability (\(b\in\mathbb{R}\), \(\tau=e^\rho\)):

\[
s(z;g)=\frac{(z-\bar m_g)^\top w_g-b}{\tau},\qquad
p(\mathrm{fake}\mid z)=\sigma\big(s(z;g)\big).
\]

Two-class CE uses logits \((0,\,s)\). At inference, replace \(c_f(g)\) by \(c_f^{\mathrm{shared}}\).

## 4. Full training loss (PFAL-augmented)

On a pure batch of generator \(g\) with labels \(y\in\{0,1\}\) and features \(z\):

**M1 block**

\[
\mathcal{L}_{\mathrm{M1}}
=\mathrm{CE}(s,y)
+0.1\,\mathrm{Align}(z,y)
+0.5\,\mathrm{Uniform}(z).
\]

**Real anchor + margin** (off \(c_r\))

\[
\mathcal{L}_{\mathrm{real}}=\mathbb{E}_{y=0}\big[1-\cos(z,c_r)\big],\qquad
\mathcal{L}_{\mathrm{margin}}=\mathbb{E}_{y=1}\big[\mathrm{relu}(\cos(z,c_r)-m)\big].
\]

**Private-feature attraction (links \(\delta_g\) to \(\mathcal{L}\))**

\[
\mathcal{L}_{\mathrm{attr}}=\mathbb{E}_{y=1}\big[1-\cos\big(z,c_f(g)\big)\big],\qquad
\mathcal{L}_{\delta}=\lambda\|\delta_g\|_2^2.
\]

**Total**

\[
\mathcal{L}
=\mathcal{L}_{\mathrm{M1}}
+\alpha\mathcal{L}_{\mathrm{real}}
+\beta\mathcal{L}_{\mathrm{margin}}
+\gamma\mathcal{L}_{\mathrm{attr}}
+\mathcal{L}_{\delta}.
\]

Defaults: \(\alpha=0.1\), \(\beta=1.0\), \(m=0.1\), \(\gamma=0.5\), \(\lambda=0.01\).

Gradients w.r.t. \(\delta_g\) flow through \(c_f(g)\) into both the CE head (via \(w_g,\bar m_g\)) and \(\mathcal{L}_{\mathrm{attr}}\), plus \(2\lambda\delta_g\) from the penalty.

## 5. Deployed objective

After training, set \(\delta_g\leftarrow0\) for all \(g\), refresh \(\mu_r,\mu_f\) on unaugmented train crops with the shared SB-LN, and evaluate with the shared prototype head only.
