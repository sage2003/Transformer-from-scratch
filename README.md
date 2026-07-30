# Transformer from Scratch

Welcome to my journey of building a Transformer model entirely from scratch using nothing but PyTorch! This repository is currently in its very early stages, but it will soon contain all the code and experiments as I work through the architecture.

As a massive supporter of open source, free education, and a proud graduate of the prestigious University of YouTube 🎓, this project is primarily fueled by free online resources rather than expensive courses.

To that end, I am learning how to build this transformer by following [Umar Jamil's fantastic YouTube tutorial](https://www.youtube.com/watch?v=ISNdQcPhsts).

If you are interested in learning the theoretical foundations of how a transformer actually works and want a detailed analysis of the original "Attention is All You Need" paper, you should definitely check out another one of his brilliant explainer/paper walkthrough videos: [Attention Is All You Need - Paper Walkthrough](https://youtu.be/bCz4OMemCcA?si=KcbXLFJs5kZ9fHZO).

Honestly, I still don't know how half of this works. So, I'm using this file as a live diary while I figure it out. Every time I learn something new, I'll write it here. It's mostly just notes so I don't forget what I did, but hopefully it helps you skip the hard parts too.

Stay tuned for more updates!

---
## Formal Definition of _Residual Stream_

The residual stream is the $d_{model}$ dimensional vector that flows through the transformer, starting as the token embedding and accumulating "**additive**" updates from each attention head and MLP layer.

---

## Layer Normalization

Given an input vector $x \in \mathbb{R}^d$, layer normalization computes: 

$$ LN(x) = \gamma\ \odot\ \dfrac{x - \mu}{\sigma + \epsilon} + \beta$$

where $\mu = \frac{1}{d}\ \sum_{i=1}^{d}x_i$ is the mean, $\sigma = \sqrt{\frac{1}{d}\ \sum_{i=1}^{d}(x_i-\mu)^2}$, $\gamma$ and $\beta$ are learned per-dimension scale and shift parameters, $\epsilon$ is a small constant for numerical stability, and $\odot$ denotes element-wise multiplication. 

The residual stream accumulates additive updates from every attention head and MLP across all layers. In a model with $L$ layers, each contributing an update of typical magnitude $\delta$, the residual stream magnitude grows roughly as $O(L \cdot \delta)$. Without normalization, this growth has cascading consequences: the inputs to later sublayers become large, the dot products in attention scores become large, and the softmax saturates, producing nearly one-hot attention patterns with vanishing gradients.

Layer normalization constrains the scale of activations entering each sublayer. By normalizing to unit variance before every attention and MLP computation, it ensures that the inputs remain in a stable range regardless of depth. This is why transformers can be trained with dozens or hundreds of layers, and why removing layer normalization from a trained model causes immediate collapse.

### Pre-Norm vs. Post-Norm

The original transformer placed layer normalization after each residual connection (post-norm):

$$\textbf{r}^{l+1} = LN(\textbf{r}^l + Sublayer(\textbf{r}^l))$$

Most modern architectures, including GPT-2 and its descendants, instead place layer normalization before each sublayer (pre-norm):

$$\textbf{r}^{l+1} = \textbf{r}^l + Sublayer(LN(\textbf{r}^l))$$

For mechanistic interpretability, the pre-norm placement has a practical advantage: the residual stream after each sublayer addition is the raw sum of all previous contributions, not a normalized version. 

### RMSNorm

A variant of layer normalization that drops the mean-centering step and normalizes by the root mean square:

$$\text{RMSNorm}(x) = \gamma\ \odot\ \dfrac{x}{\text{RMS}(x) + \epsilon}, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\ \sum_{i=1}^{d}x_i^2}$$

### Why LN Matters?

Layer normalization creates three specific complications for mechanistic interpretability.

1. It breaks strict linearity of the residual stream decomposition. 
2. It couples all dimensions.
3. It erases magnitude information. 






