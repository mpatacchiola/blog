---
layout: post
title:  "Variational inference, KL-divergence, and ELBO"
date:   2020-12-02 08:00:00 +0000
description: .
author: Massimiliano Patacchiola
type: bayesian methods
comments: false
published: false
---

In this post I will present a compact introduction to variational inference in latent variable models. Variational inference has been widely used in machine learning and it's the engine of many successful methods, e.g. Variational Auto-Encoders (VAEs). My goal here is to present an essential summary which is not meant to be exhaustive.


Definition of the problem
------------------------

We are dealing with latent variable models. By definition, [latent variables](https://en.wikipedia.org/wiki/Latent_variable) are hidden and they cannot be directly observed. However, we can *infer* latent variables by exploiting the observed variables. Examples of latent variables are intelligence and health. We cannot directly measure "intelligence" but we can infer its value by using appropriate tests (Raven's Progressive Matrices, SAT scores, etc). Same story for "health", we can measure various observable quantities (e.g. blood tests, pressure, etc) and from them we can infer an overall value. Here, I will refer to the latent variable as $$z$$ and to the observed variable as $$x$$. I will use the bold notation $$\mathbf{x}$$ and $$\mathbf{z}$$ to represent collection of variables 

$$
\mathbf{x} = \{x_{1}, \dots, x_{N} \}
 \quad \text{and} \quad
\mathbf{z} = \{z_{1}, \dots, z_{M} \}.
$$

Keep in mind that the observed variables $$\mathbf{x}$$ are often called just *data*.
We suppose that the latent variable *causes* the observed variable, therefore the probabilistic graphical model would be $$z \rightarrow x$$. For instance, "health" causes specific values to be recorded in the blood tests. In a Bayesian treatment we imply that latent variables govern the distribution of the data. In particular, in a Bayesian model we draw the latent variables from a prior density $$p(\mathbf{z})$$ and then relates them to the observations through the likelihood $$p(\mathbf{x} \vert \mathbf{z})$$.

**Inference.** Our goal is to perform inference. For instance, given various medical examinations (data) we want to infer "health", or given cognitive tests infer "intelligence". Formally, we want to estimate the following posterior 

$$
p(\mathbf{z} \vert \mathbf{x}) = \frac{p(\mathbf{z}, \mathbf{x})}{p(\mathbf{x})}.
$$

However, we have a problem. 

**Problem: intractable posterior.** Let's give a closer look at $$p(\mathbf{z} \vert \mathbf{x})$$. The numerator of the posterior is the joint distribution over the observed and latent variables $$p(\mathbf{z}, \mathbf{x})$$ which is efficient to compute. The denominator $$p(\mathbf{x})$$ of the posterior is called *marginal likelihood* or *evidence* and can be estimated as follows

$$
p(\mathbf{x}) = \int p(\mathbf{z}, \mathbf{x}) d \mathbf{z}.
$$

Here is the problem: in many cases this integral is intractable. Well, what does it mean intractable? Let's try to understand it with a concrete example. In a [previous post](https://mpatacchiola.github.io/blog/2020/07/31/gaussian-mixture-models.html) I have introduced Gaussian Mixture Models (GMMs). To estimating the evidence in GMMs we need to integrate (sum up) all configurations of the latent variables. This has an exponential cost in the number of latent variables, and it can rapidly become computationally intractable.


Variational Bayesian methods
----------------------------

We can exploit [variational Bayesian methods](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) to solve our problem. In particular, since the exact estimation of the integral over $$p(\mathbf{x})$$ is intractable we can try approximating the true posterior $$p(\mathbf{z} \vert \mathbf{x})$$ using a *variational distribution* over the latent variables $$q(\mathbf{z})$$. As a shorthand, I will often use just $$p$$ to refer to the true distribution and $$q$$ to refer to the approximated distribution.


**The variational distribution.** Let's define a *family of distributions* $$\mathcal{Q}$$ from which we are going to pick the ideal candidate $$q_{\ast} \in \mathcal{Q}$$ that better fits the true distribution $$p$$. Generally, we are going to choose a friendly distribution family, such as Gaussians, to simplify our life. The parameters describing $$q$$ (e.g. mean and standard deviation of the Gaussians) are grouped in a vector $$\boldsymbol{\theta}$$, therefore in practice we are looking for $$\boldsymbol{\theta}_{\ast}$$ the optimal set of parameters of the ideal distribution $$q_{\ast}$$. The parameters of the variational distribution $$\boldsymbol{\theta}$$ and the latent variables $$\mathbf{z}$$ are often grouped together and called *unobserved variables*. The problem now is how to find the best candidate $$q_{\ast}$$. We need a measure of similarity between $$p$$ and $$q$$ that we can use as a metric during our search. The Kullback-Leibler (KL) divergence is what we are looking for.


**The Kullback-Leibler (KL) divergence.** The KL divergence can be used to measure the similarity between two distributions. For instance, given our distributions $$p$$ and $$q$$ we define

$$
\text{KL} \big( q(\boldsymbol{z}) || p(\boldsymbol{z} | \boldsymbol{x}) \big) 
= \int q(\boldsymbol{z}) \log \frac{q(\boldsymbol{z})}{p(\boldsymbol{z} | \boldsymbol{x})} d\boldsymbol{z}
= - \int q(\boldsymbol{z}) \log \frac{p(\boldsymbol{z} | \boldsymbol{x})}{q(\boldsymbol{z})} d\boldsymbol{z}.
$$

In our particular case, we want to find the best candidate distribution $$q_{\ast}$$ that minimizes the KL divergence

$$
q_{\ast}(\mathbf{z}) = \text{argmin} \ \text{KL}\big(q(\mathbf{z}) \vert\vert p(\mathbf{z} \vert \mathbf{x})\big).
$$


**Short note on KL divergence.** The KL divergence is non-negative, that is $$\text{KL} \geq 0$$. Moreover, it is not symmetric and it does not satisfy the triangle inequality, meaning that it is not a proper distance metric. Asymmetry implies that

$$
\text{KL}(p \vert\vert q) \neq \text{KL}(q \vert\vert p),
$$

with $$\text{KL}(p \vert\vert q)$$ called the *forward* and $$\text{KL}(q \vert\vert p)$$ the *backward* (or reverse) KL divergence. The forward KL is mode averaging, whereas the backward is mode fitting. To remember this difference I came up with a mnemonic trick. In the forward KL we have $$\text{KL}(p \vert\vert q)$$ with the first term being $$p$$, the true distribution. The bump of the letter `p` points *forward* (left to right). In the backward KL instead the first distribution is $$q$$, the bump of the letter `q` points *backward* (right to left). To distinguish between mode averaging and mode fitting I use a similar trick. Imagine a bi-modal distribution with two symmetric modes, something like this `_/\___/\_`. In $$\text{KL}(p \vert\vert q)$$ the bumps of `p` and `q` point towards the center meaning that the resulting approximation will be mode averaging (positioned in between the two modes). In $$\text{KL}(q \vert\vert p)$$ the bumps `q` and `p` point outwards, meaning that the resulting approximation will be mode fitting (fitting one of the two modes).

**Problem: intractable KL divergence.** You may have noticed something shady going on here. Our goal was to find a way to avoid the estimation of $$p(\mathbf{z} \vert \mathbf{x})$$ because its evidence $$p(\mathbf{x})$$ was intractable. To do this we have introduced variational inference and the KL divergence $$\text{KL}\big(q(\mathbf{z}) \vert\vert p(\mathbf{z} \vert \mathbf{x})\big)$$. An obvious question is: how are we going to estimte this KL divergence if we cannot estimate $$p(\mathbf{z} \vert \mathbf{x})$$ in the first place? This question gets to the heart of the problem, our troubles are not ended. It turns out that using a variational approach by exploiting the KL divergence has the same intractability issues. Let's try to decompose the KL divergence to see how far we can go before we find an obstacle

$$
\begin{aligned}
\text{KL} \big( q(\boldsymbol{z}) || p(\boldsymbol{z} | \boldsymbol{x}) \big) 
&= \int q(\boldsymbol{z}) \log \frac{q(\boldsymbol{z})}{p(\boldsymbol{z} | \boldsymbol{x})} d\boldsymbol{z} &\text{(1.1)}\\
&= \int q(\boldsymbol{z}) \big( \log q(\boldsymbol{z}) - \log p(\boldsymbol{z} | \boldsymbol{x}) \big) d\boldsymbol{z} &\text{(1.2)}\\
&= \int q(\boldsymbol{z}) \log q(\boldsymbol{z}) - q(\boldsymbol{z}) \log p(\boldsymbol{z} | \boldsymbol{x}) d\boldsymbol{z} &\text{(1.3)}\\
&= \int q(\boldsymbol{z}) \log q(\boldsymbol{z}) d\boldsymbol{z} - \int q(\boldsymbol{z}) \log p(\boldsymbol{z} | \boldsymbol{x}) d\boldsymbol{z} &\text{(1.4)}\\ 
&= \mathbb{E}_{q} \big[ \log q(\boldsymbol{z}) \big] - \mathbb{E}_{q} \big[ \log p(\boldsymbol{z} | \boldsymbol{x}) \big] & \text{(1.5)}\\
&= \mathbb{E}_{q} \big[ \log q(\boldsymbol{z}) \big] - \mathbb{E}_{q} \bigg[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z}) }{p(\boldsymbol{x})} \bigg] &\text{(1.6)}\\
&= \mathbb{E}_{q} \big[ \log q(\boldsymbol{z}) \big] - \mathbb{E}_{q} \big[ \log p(\boldsymbol{x}, \boldsymbol{z}) - \log p(\boldsymbol{x}) \big] &\text{(1.7)}\\
&= \mathbb{E}_{q} \big[ \log q(\boldsymbol{z}) \big] - \mathbb{E}_{q} \big[ \log p(\boldsymbol{x}, \boldsymbol{z}) \big] + \mathbb{E}_{q} \big[ \log p(\boldsymbol{x}) \big] &\text{(1.8)}\\
&= \mathbb{E}_{q} \big[ \log q(\boldsymbol{z}) \big] - \mathbb{E}_{q} \big[ \log p(\boldsymbol{x}, \boldsymbol{z}) \big] + \underbrace{\log p(\boldsymbol{x})}_{\text{intractable}} &\text{(1.9)}\\
\end{aligned}
$$

Give a closer look to the last form in $$\text{(1.9)}$$. The intractable term $$p(\mathbf{x})$$ is still there as we suspected. It is necessary to find another way to reach our goal, but first I would like to give a step-by-step explanation of what I have done above: 

$$(1.1 \rightarrow 1.2)$$ Breaking the ratio by applying the property of logarithms

$$
\log \frac{f(x)}{g(x)} = \log f(x) - \log g(x).
$$

$$(1.2 \rightarrow 1.3)$$ Multiplying the two terms with $$q(\boldsymbol{z})$$ and expanding.

$$(1.3 \rightarrow 1.4)$$ Applying the property of integrals 

$$
\int f(x) + g(x) dx = \int f(x) dx + \int g(x) dx.
$$

$$(1.4 \rightarrow 1.5)$$ Applying the definition of expectation over the integrals.

$$(1.5 \rightarrow 1.6)$$ Straightforward application of the rule of probabilities

$$
p(\boldsymbol{z} | \boldsymbol{x}) = \frac{p(\boldsymbol{x}, \boldsymbol{z}) }{p(\boldsymbol{x})}.
$$

$$(1.6 \rightarrow 1.7)$$ Applying again the rule of logarithms to split the ratio.

$$(1.7 \rightarrow 1.8)$$ Applying the rule of expectations (linearity)

$$
\mathbb{E}[A + B] = \mathbb{E}[A] + \mathbb{E}[B].
$$

$$(1.8 \rightarrow 1.9)$$ There is no expected value for $$\mathbb{E}_{q} \big[ \log p(\boldsymbol{x}) \big]$$ since we are considering an expectation $$\mathbb{E}_{q}$$ over $$q(\boldsymbol{z})$$ and not over $$p(\boldsymbol{x})$$.




Evidence Lower BOund (ELBO)
-----------------------------

Let's go back to $$\text{(1.9)}$$ where we got

$$
\text{KL} \big( q(\boldsymbol{z}) \vert\vert p(\boldsymbol{z} \vert \boldsymbol{x}) \big)
= \mathbb{E}_{q} \big[ \log q(\boldsymbol{z}) \big] - \mathbb{E}_{q} \big[ \log p(\boldsymbol{x}, \boldsymbol{z}) \big] + \log p(\boldsymbol{x}).
$$

This decomposition is problematic because we still have two intractable terms $$\text{KL} ( q(\boldsymbol{z}) \vert\vert p(\boldsymbol{z} \vert \boldsymbol{x}) )$$ and $$\log p(\boldsymbol{x})$$.
However, rearranging those quantities by moving the intractable terms on the same side we get

$$
-\mathbb{E}_{q} \big[ \log q(\boldsymbol{z}) \big] + \mathbb{E}_{q} \big[ \log p(\boldsymbol{x}, \boldsymbol{z}) \big]
= \log p(\boldsymbol{x}) - \text{KL} \big( q(\boldsymbol{z}) || p(\boldsymbol{z} | \boldsymbol{x}) \big).
$$

Here is the idea: what if we maximize the quantity on the left side? This would be great because the terms on the left are tractable. Is this legit? Here is the interesting part, by maximizing the quantity on the left we are simultaneously (i) maximizing the evidence $$p(\boldsymbol{x})$$, and (ii) minimizing the KL divergence between our variational distribution $$q(\boldsymbol{z})$$ and the true posterior $$p(\boldsymbol{z} \vert \boldsymbol{x})$$, that is what we wanted to achieve with this machinery. Crucially, since the KL divergence is non-negative $$\text{KL} \geq 0$$ the left term is a lower-bound over the log-evidence $$p(\boldsymbol{x})$$ called the *Evidence Lower BOund (ELBO)*

$$
\text{ELBO}(q) 
= -\mathbb{E}_{q} \big[ \log q(\boldsymbol{z}) \big] + \mathbb{E}_{q} \big[ \log p(\boldsymbol{x}, \boldsymbol{z}) \big]
= \mathbb{E}_{q} \bigg[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{q(\boldsymbol{z})} \bigg].
$$

To obtain the last form I have first applied the linearity of the expectation and then the rule of the logarithms to get the ratio.

**Unpacking the ELBO.** The ELBO is the last trick in our arsenal. The idea is to avoid minimizing the KL divergence by maximizing an alternative quantity which is equivalent up to an added constant. The ELBO is such a quantity. The ELBO is also known as the *variational lower bound* or *negative variational free energy*. Here, we are going to unpack the ELBO until we get a decomposition that is easy to manage:

$$
\begin{aligned}
\mathrm{ELBO}(q) 
&=-\mathbb{E}_{q}[\log q(\boldsymbol{z})]+\mathbb{E}_{q}[\log p(\boldsymbol{z}, \boldsymbol{x})]  &\text{(2.1)}\\
&=-\mathbb{E}_{q}[\log q(\boldsymbol{z})]+\mathbb{E}_{q}[\log \big( p(\boldsymbol{z}) p(\boldsymbol{x} \vert \boldsymbol{z}) \big) ]  &\text{(2.2)}\\
&=-\mathbb{E}_{q}[\log q(\boldsymbol{z})]+\mathbb{E}_{q}[\log p(\boldsymbol{z}) + \log p(\boldsymbol{x} \vert \boldsymbol{z}) ]  &\text{(2.3)}\\
&=-\mathbb{E}_{q}[\log q(\boldsymbol{z})]+\mathbb{E}_{q}[\log p(\boldsymbol{z})]+\mathbb{E}_{q}[\log p(\boldsymbol{x} \mid \boldsymbol{z})]  &\text{(2.4)}\\
&=\mathbb{E}_{q}[\log p(\boldsymbol{x} \mid \boldsymbol{z})] + \int q(\boldsymbol{z}) \log p(\boldsymbol{z}) d\boldsymbol{z} - \int q(\boldsymbol{z}) \log q(\boldsymbol{z}) d\boldsymbol{z}  &\text{(2.5)}\\
&=\mathbb{E}_{q}[\log p(\boldsymbol{x} \mid \boldsymbol{z})] + \int q(\boldsymbol{z}) \log p(\boldsymbol{z}) - q(\boldsymbol{z}) \log q(\boldsymbol{z}) d\boldsymbol{z}  &\text{(2.6)}\\
&=\mathbb{E}_{q}[\log p(\boldsymbol{x} \mid \boldsymbol{z})] + \int q(\boldsymbol{z}) \big( \log p(\boldsymbol{z}) - \log q(\boldsymbol{z}) \big) d\boldsymbol{z}  &\text{(2.7)}\\
&=\mathbb{E}_{q}[\log p(\boldsymbol{x} \mid \boldsymbol{z})] + \int q(\boldsymbol{z}) \log \frac{p(\boldsymbol{z})}{q(\boldsymbol{z})} d\boldsymbol{z}  &\text{(2.8)}\\
&=\mathbb{E}_{q}[\log p(\boldsymbol{x} \mid \boldsymbol{z})]- \text{KL}(q(\boldsymbol{z}) \| p(\boldsymbol{z})) &\text{(2.9)}
\end{aligned}
$$

Let's take a closer look at $$\text{(2.9)}$$. The first term describes the probability of the data given the latent variable $$p(\boldsymbol{x} \mid \boldsymbol{z})$$. When we maximize the ELBO we also maximize this quantity which translate in picking those models $$q(\boldsymbol{z})$$ in the variational family $$\mathcal{Q}$$ that *better predict* the data $$\boldsymbol{x}$$. The second term, is the negative KL divergence between our variational model $$q(\boldsymbol{z})$$ and the prior over the latent variables $$p(\boldsymbol{z})$$. When we maximize the ELBO this term is pushed towards zero (because of the negative sign) meaning that the two distributions are forced to be close (identical if $$\text{KL}=0$$). In other words, the variational distribution is forced to be similar to the prior. Let's break down what has been done above:

$$(2.1 \rightarrow 2.2)$$ Factorizing the joint probability

$$
p(\boldsymbol{z}, \boldsymbol{x}) = p(\boldsymbol{z}) p(\boldsymbol{x} \vert \boldsymbol{z}).
$$

$$(2.2 \rightarrow 2.3)$$ Breaking the ratio by applying the property of logarithms

$$
\log(f(x) \ g(x)) = \log f(x) + \log g(x).
$$

$$(2.3 \rightarrow 2.4)$$ Applying the properties of expectation (linearity).

$$(2.4 \rightarrow 2.5)$$ Rearranging the terms and applying the definition of expectation to get the integral form.

$$(2.5 \rightarrow 2.6)$$ Applying the property of integrals to join the two terms.

$$(2.6 \rightarrow 2.7)$$ Collecting the common factor $$q(\mathbf{z})$$.

$$(2.7 \rightarrow 2.8)$$ Applying the properties of logarithm to arrange the difference as a ratio.

$$(2.8 \rightarrow 2.9)$$ The integral in the second term is equivalent to the definition of negative KL divergence.


Other routes to the ELBO
-------------------------

We are dealing with three quantities: 

- the log-evidence $$\log p(\boldsymbol{x})$$
- the KL divergence $$\text{KL} ( q(\boldsymbol{z}) \vert\vert p(\boldsymbol{z} \vert \boldsymbol{x}) )$$
- the ELBO $$-\mathbb{E}_{q} \big[ \log q(\boldsymbol{z}) \big] + \mathbb{E}_{q} \big[ \log p(\boldsymbol{x}, \boldsymbol{z}) \big]$$

As we saw above, those quantities are strictly intertwined. In previous sections I have followed one of the possible ways we can derive the ELBO, starting from the KL divergence. However, it is also possible to start from the evidence to get back the KL divergence and the ELBO. For completeness, we can see how to do it:

$$
\begin{aligned}
\log p(\boldsymbol{x})
&=\mathbb{E}_{q}[\log p(\boldsymbol{x})] &\text{(3.1)}\\
&=\mathbb{E}_{q} \bigg[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{p(\boldsymbol{z} \vert \boldsymbol{x})} \bigg] &\text{(3.2)}\\
&=\mathbb{E}_{q} \bigg[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{p(\boldsymbol{z} \vert \boldsymbol{x})} \frac{q(\boldsymbol{z})}{q(\boldsymbol{z})} \bigg] &\text{(3.3)}\\
&=\mathbb{E}_{q} \bigg[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{q(\boldsymbol{z})} \frac{q(\boldsymbol{z})}{p(\boldsymbol{z} \vert \boldsymbol{x})} \bigg] &\text{(3.4)}\\
&=\underbrace{\mathbb{E}_{q} \bigg[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{q(\boldsymbol{z})} \bigg]}_{\text{ELBO}} + \underbrace{\mathbb{E}_{q} \bigg[ \log \frac{q(\boldsymbol{z})}{p(\boldsymbol{z} \vert \boldsymbol{x})} \bigg]}_{\text{KL}} &\text{(3.5)}\\
\end{aligned}
$$

As expected, this gave us two terms: the ELBO and the KL divergence. Here is a quick explanation of what has been done above:

$$(3.1)$$ The expectation over $$q(\boldsymbol{z})$$ has no effect on $$p(\boldsymbol{x})$$ therefore

$$
\mathbb{E}_{q}[\log p(\boldsymbol{x})] = \log p(\boldsymbol{x}).
$$

$$(3.1 \rightarrow 3.2)$$ Applying a refactoring based on the basic probability rule

$$
p(\boldsymbol{x}) = \frac{p(\boldsymbol{x}, \boldsymbol{z})}{p(\boldsymbol{z} \vert \boldsymbol{x})}.
$$

$$(3.2 \rightarrow 3.3)$$ Including a new term for convenience. I will state the obvious: multiplying and dividing by the same quantity has no effect, therefore is a legit operation.

$$(3.3 \rightarrow 3.4)$$ Rearranging by switching the denominator of the two terms.

$$(3.4 \rightarrow 3.5)$$ Applying the rule of logarithms to turn a product into a sum, then splitting the expectation based on the linearity property.

**Derivation via Jensen’s inequality.** Another popular derivation starting from the evidence is based on the [Jensen’s inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality).


Conclusion
----------

In this post I have provided an introduction to variational inference and the ELBO. I have tried to be as compact as possible, explaining every step along the way. Below you can find some additional resources if you want to know more about variational inference.


Resources
------------

- *"Variational Inference"*, A.L. Popkes [[PDF]](http://alpopkes.com/files/variational_inference.pdf)
- *"Pattern Recognition and Machine Learning"*, Chapter 10, C. Bishop
- *"Variational Inference: A Review for Statisticians"*, D. Blei, A. Kucukelbir, and J.D. McAuliffe [[arXiv]](https://arxiv.org/pdf/1601.00670v1.pdf)



