# First-Principles Diffusion Models (AI 623 — PA1)

A clean, end-to-end PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) built directly from mathematical foundations. This repository serves as a self-contained implementation and analytical playground designed to bridge the gap between textbook diffusion equations and raw PyTorch execution.

---

## Technical Architecture & Mechanics

### 1. Diffusion Core Logic (`diffusion/`)

* **`schedule.py`**: Computes and caches discrete variance schedule scalar chains ($\beta_i, \alpha_i, \overline{\alpha}_i$) across $L=1000$ timesteps.


* **`forward.py`**: Implements closed-form $q$-sampling to directly map clean data $x_0 \to x_i$ via $x_i = \sqrt{\overline{\alpha}_i}x_0 + \sqrt{1-\overline{\alpha}_i}\epsilon$.


* **`posterior.py`**: Evaluates true analytical Gaussian posterior parameters ($\tilde{\mu}_i, \tilde{\beta}_i$) to serve as the exact target for the reverse chain.


* **`ddpm.py`**: Implements the complete ancestral reverse sampler loop from $x_L \sim \mathcal{N}(0,I)$ down to $x_0$, enforcing zero noise injection at $i=1$.



### 2. Neural Network Backbone (`models/`)

* **`unet.py`**: A timestep-conditioned U-Net optimized on the unweighted noise-prediction loss ($\mathcal{L}_{simple}$).


* Configured with 3 resolution levels (channel multipliers: 32, 64, 128) using GroupNorm and SiLU activations.


* Sinusoidal positional embeddings are mapped via an MLP and injected directly as activation bias terms inside each residual block.



---

## Evaluation & Diagnostics Suite

To track convergence and protect against common diffusion pitfalls, the codebase implements the following rigorous verification pipelines:

* **Analytical Checks:** Monotonicity tracking of the Signal-to-Noise Ratio ($SNR(i) = \frac{\overline{\alpha}_i}{1-\overline{\alpha}_i}$) and empirical mean/variance validation of the forward chain.


* **Custom Dataset-FID/KID:** Rather than relying on mismatched ImageNet-Inception features, sample quality is quantitatively evaluated in a feature space learned by a classifier trained directly on the target dataset.


* **Memorization Audits:** Implements nearest-neighbor spatial checking ($l_2$ matching on raw pixel space and learned feature embeddings) against the training set to explicitly test for data memorization.


* **Trajectory Tracking:** Visualizes step-wise denoising progression by saving intermediate latents at scheduled intervals ($i \in \{L, \frac{3L}{4}, \frac{L}{2}, \frac{L}{4}, 1\}$).


---

## Setup & Dependencies

Ensure you have PyTorch and the following helper utilities installed:

* `torch` / `torchvision` 


* `einops` 


* `tqdm` 


* `matplotlib` 


* `scipy` / `scikit-learn`
