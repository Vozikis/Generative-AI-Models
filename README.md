# Generative AI Models

Welcome to the **Generative AI Models** repository.

## Implemented Models

### 1. Variational Autoencoder (VAE)
The VAE is a generative model that learns to encode data into a latent space and decode it back to the original space, facilitating data generation.

**Reference Paper:**  
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — *Diederik P. Kingma and Max Welling* (ICLR 2014)

---

### 2. Vector Quantized Variational Autoencoder (VQ-VAE)
VQ-VAE introduces discrete latent variables using vector quantization, which addresses "posterior collapse" in traditional VAEs.

**Reference Paper:**  
- [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — *Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu* (NeurIPS 2017)

---

### 3. Attention-based VAE
This variant integrates attention mechanisms into the VAE framework to enhance the model's ability to focus on relevant parts of the input during encoding and decoding.

---

## Models Under Development

### 4. Diffusion Models
Diffusion models are generative models that learn to generate data by reversing a gradual noising process, resulting in high-quality sample generation.

**Reference Paper:**  
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — *Jonathan Ho, Ajay Jain, and Pieter Abbeel* (NeurIPS 2020)

---

### 5. Consistency Models
Consistency models enable fast one-step generation without adversarial training by mapping noise directly to data.

**Reference Paper:**  
- [Consistency Models](https://arxiv.org/abs/2303.01469) — *Yang Song et al.* (2023)

---

### 6. Flow Matching Models
Flow Matching models offer a simulation-free approach for training continuous normalizing flows by directly estimating a vector field that generates the data distribution.

**Reference Paper:**  
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — *Yaron Lipman et al.* (ICLR 2023)

---

## Repository Structure

```
Generative-AI-Models/
│
├── VAE/               # Variational Autoencoder Implementation
├── VQ-VAE/            # Vector Quantized VAE Implementation
├── Attention_VAE/     # Attention-based VAE Implementation
├── Diffusion_Models/  # (Planned) Diffusion Models
├── Consistency_Models/ # (Planned) Consistency Models
└── Flow_Matching_Models/ # (Planned) Flow Matching Models
```

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vozikis/Generative-AI-Models.git
   cd Generative-AI-Models
   ```

2. **Navigate to the desired model's directory:**
   ```bash
   cd VAE  # For Variational Autoencoder
   ```
---

## References
- Kingma, D.P., & Welling, M. (2014). *Auto-Encoding Variational Bayes*. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). *Neural Discrete Representation Learning*. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Song, Y., et al. (2023). *Consistency Models*. [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)
- Lipman, Y., et al. (2023). *Flow Matching for Generative Modeling*. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
