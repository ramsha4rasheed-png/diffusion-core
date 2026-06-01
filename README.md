# First-Principles Diffusion Models (AI 623 — PA1)

A clean, end-to-end PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) built directly from mathematical foundations. This repository contains a comprehensive Jupyter notebook that handles the entire pipeline—from mathematical schedules and training to ancestral sampling and advanced feature-space evaluation.

This setup serves as a self-contained implementation and analytical playground designed to bridge the gap between textbook diffusion equations and raw PyTorch execution.

---

## Technical Pipeline & Architecture

The notebook is structured sequentially to implement and verify every moving part of the DDPM framework:

### 1. Mathematical Foundation & Forward Process

* **Variance Scheduling**: Computes and caches discrete variance schedule scalar chains ($\beta_i, \alpha_i, \overline{\alpha}_i$) across $L=1000$ timesteps using a baseline linear setup.


* **Closed-Form Forward Noising**: Implements $q$-sampling to directly map clean data $x_0 \to x_i$ via $x_i = \sqrt{\overline{\alpha}_i}x_0 + \sqrt{1-\overline{\alpha}_i}\epsilon$.


* **Analytical True Posterior**: Evaluates closed-form true Gaussian posterior parameters ($\tilde{\mu}_i, \tilde{\beta}_i$) to serve as the exact target for the reverse chain.



### 2. Timestep-Conditioned U-Net

* **Architecture**: A localized U-Net backbone optimized on the unweighted noise-prediction loss ($\mathcal{L}_{simple}$).


* Implemented with 3 resolution levels (channel multipliers: 32, 64, 128) using GroupNorm and SiLU activations.


* **Time Injection**: Computes sinusoidal positional embeddings, passing them through an MLP to inject directly as activation bias terms inside each residual block.



### 3. Training & Ancestral Sampling

* Tracks optimization across a baseline training budget (at least 100k gradient steps / 50 epochs) with data mapped to the $[-1, 1]$ range.


* Implements the complete ancestral reverse sampler loop from $x_L \sim \mathcal{N}(0,I)$ down to $x_0$, enforcing zero noise injection at the final $i=1$ step.



---

## Evaluation & Diagnostics Suite

To track true generative quality and protect against common diffusion pitfalls, the notebook includes a dedicated evaluation section:

* **Analytical Checks**: Validates schedule scalars by plotting $\overline{\alpha}_i$ and the Signal-to-Noise Ratio ($SNR(i) = \frac{\overline{\alpha}_i}{1-\overline{\alpha}_i}$) to check convergence properties.


* **Custom Dataset-FID/KID**: Computes FID and KID scores using a feature space from a simple CNN classifier trained directly on the target dataset, avoiding mismatched ImageNet feature extractors.


* **Memorization Audits**: Runs spatial nearest-neighbor checking ($l_2$ matching on raw pixels and learned feature embeddings) against the training set to verify the model is generating novel samples rather than memorizing data.


* **Trajectory Tracking**: Visualizes step-wise denoising progression by saving intermediate latents at scheduled intervals ($i \in \{L, \frac{3L}{4}, \frac{L}{2}, \frac{L}{4}, 1\}$).



---

## Notebook Structure

```text
├── DDPM_First_Principles.ipynb   # Single monolithic pipeline containing:
│   ├── 1. Setup & Preprocessing  # Data scaling [-1, 1] and dataloading
│   ├── 2. Diffusion Mechanics    # Schedules, q-sampling, and posterior formulas
│   ├── 3. Model Architecture     # Timestep-conditioned U-Net implementation
│   ├── 4. Training Loop          # Simple loss optimization & diagnostic tracking
│   ├── 5. Reverse Sampling       # Ancestral generation & trajectory saving
│   └── 6. Advanced Evaluation    # Feature FID/KID and nearest-neighbor checks
└── README.md

```

---

## Dependencies

The notebook runs purely on top of standard scientific and deep learning libraries:

* `torch` / `torchvision` 


* `einops` 


* `tqdm` 


* `matplotlib` 


* `scipy` / `scikit-learn`
