---
layout: post
title:  "Variational inference: chosing the variational family"
date:   2021-01-08 09:00:00 +0000
description: .
author: Massimiliano Patacchiola
type: bayesian methods
comments: false
published: false
---




Definition of the problem
--------------------------

Mean-field distributions
------------------------

Mean Field variational inference has its origins in the mean field theory of physics and it is considered a classic approach, still used in recent applications (e.g. Variational Auto-Encoders). A mean field distribution is based on independent factors, each one governed by its own variational parameters

$$
q_{\theta}(z) = \prod_{i=1}^{K} q_{\theta_i}(z_i).
$$

In practice, we are assuming independence between the components of the distribution. An example could be a multivatiate Gaussian where each component is parameterized by its own mean and variance, resulting in a diagonal covariance matrix. This can be an oversimplification, reducing the flexibility of the variational distribution which can be unable to model the real posterior. However, this simplification is also the strength of this family.

**ELBO of mean-field distributions.** To keep the notation compact, I will use $$q_{\theta}$$ as a shorthand for $$q_{\theta}(z)$$.

$$
\begin{align} 
\text{ELBO}
&= \int q_{\theta} \log p(x, z) - q_{\theta} \log q_{\theta} dz &\text{(1.1)}\\
&= \int q_{\theta} \log p(x, z) dz - \int q_{\theta} \log q_{\theta} dz  &\text{(1.2)}\\
&= \int \prod_{i=1}^{K} q_{\theta_i}\log p(x, z) dz - \int q_{\theta} \log \prod_{i=1}^{K} q_{\theta_i} dz  &\text{(1.3)}\\
&= \int \prod_{i=1}^{K} q_{\theta_i}\log p(x, z) dz - \int q_{\theta} \sum_{i=1}^{K} \log q_{\theta_i} dz  &\text{(1.4)}\\
&= \int q_{\theta_j} \prod_{i \neq j}^{K-1} q_{\theta_i}\log p(x, z) dz - \int q_{\theta} \bigg( \log q_{\theta_j} + \sum_{i \neq j}^{K-1} \log q_{\theta_i} \bigg) dz  &\text{(1.5)}\\
&= \int q_{\theta_j} \bigg[ \int \prod_{i \neq j}^{K-1} q_{\theta_i}\log p(x, z) dz_i \bigg] dz_j - \int q_{\theta_j} \log q_{\theta_j} dz_j - \int q_{\theta_i} \sum_{i \neq j}^{K-1} \log q_{\theta_i} dz_i  &\text{(1.6)}\\
&= \int q_{\theta_j} \mathbb{E}_{i \neq j} \big[ \log p(x, z) \big] dz_j - \int q_{\theta_j} \log q_{\theta_j} dz_j + \text{constant}  &\text{(1.7)}\\
&= \int q_{\theta_j} \log \tilde{p}(x, z_i) - q_{\theta_j} \log q_{\theta_j} dz_j + \text{constant}  &\text{(1.8)}\\
&= \text{KL}(q_{\theta_j} \vert \vert \tilde{p}(x, z_i)) + \text{constant} &\text{(1.9)}\\
\end{align}
$$


$$(1.1 \rightarrow 1.2)$$ Splitting the definition of ELBO using the linearity of integral.

$$(1.2 \rightarrow 1.3)$$ Applying the mean-field approximation, replacing $$q_{\theta}$$ with $$\prod_{i=1}^{K} q_{\theta_i}$$.

$$(1.3 \rightarrow 1.4)$$ Exploiting the properties of logarithms to turn the log of a product into the sum of logs.

$$(1.4 \rightarrow 1.5)$$ Carving out from the product and the sum a term indexed by $$j$$, while the remaining $$K-1$$ terms are indexed by $$i$$. Using the notation $$i \neq j$$ to specify that $$i$$ goes over all the $$K-1$$ indices but $$j$$. Note that, carving out the factor from the product correspond to multiplying the $$j$$-th term with the leftover, whereas carving out the factor from the sum of the logs correspond to sum the log of the $$j$$-th term with the leftover.

$$(1.5 \rightarrow 1.6)$$ First term: we introduce an inner integral over the residual indices. This inner integral may seems odd at first, but interpreting the multidimensional integral as a sequence of nested loops over the $$K$$ components, it becomes obvious that this is a legit operation. Second term: we split the integral in two components, isolating the two sets of variables.

$$(1.6 \rightarrow 1.7)$$ First term: the inner integral can be rewritten as an expectation. Third term: it is a constant, since we are going to maximize with respect to $$j$$ and this term does not contain any $$j$$ term.

$$(1.7 \rightarrow 1.8)$$ The two integrals can be joined. Additionally, the expectation can be rewritten as

$$
\mathbb{E}_{i \neq j} \big[ \log p(x, z) \big] = \log \tilde{p}(x, z_i),
$$

because it is an unnormalized log-likelihood, function of the factor $$z_i$$.

$$(1.8 \rightarrow 1.9)$$ The integral can be rewritten as the KL-divergence between $$q_{\theta_j}$$ and $$\tilde{p}(x, z_i)$$.

Conclusion
----------

Resources
------------

- [Stanford course](https://deepgenerativemodels.github.io/) notes on generative models [[link]](https://deepgenerativemodels.github.io/notes/flow/)
- [Will Wolf's blog](http://willwolf.io/) for a neat mean-field derivation [[link]](http://willwolf.io/2018/11/23/mean-field-variational-bayes/)
- Kleng's blog, e.g. [[link]](https://bjlkeng.github.io/posts/variational-autoencoders-with-inverse-autoregressive-flows)
- Kay Brodersen's slides [[PDF]](https://kaybrodersen.github.io/talks/Brodersen_2013_03_22.pdf)
- *"Graphical models, exponential families, and variational inference"* (2008) Wainwright and Jordan,  Now Publishers Inc., *Chapters-5: Mean Field Methods*

References
-----------


