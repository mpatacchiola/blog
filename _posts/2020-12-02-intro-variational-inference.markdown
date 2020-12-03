---
layout: post
title:  "Variational inference in a nutshell"
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

**Problem: intractable posterior.** The denominator $$p(\mathbf{x})$$ of the posterior distribution $$p(\mathbf{z} \vert \mathbf{x})$$ is called *evidence* and can be estimated as follows

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

The Evidence Lower BOund (ELBO) is the last trick in our arsenal. The idea is to avoid minimizing the KL divergence by maximizing an alternative quantity which is equivalent up to an added constant. The ELBO is such a quantity. The ELBO is also known as the *variational lower bound* or *negative variational free energy*. What is this ELBO in practice? Well, you may be surprised to know that the ELBO is just the form $$\text{(1.9)}$$ we have found in the previous section when we unpacked the KL divergence, $$\mathbb{E}_{q} [ \log q(\boldsymbol{z})] - \mathbb{E}_{q} [ \log p(\boldsymbol{x}, \boldsymbol{z}) ] + \log p(\boldsymbol{x})$$, but without the problematic term $$p(\boldsymbol{x})$$ and a reversed sign:

$$
\text{ELBO}(q) = - \mathbb{E}_{q} \big[ \log q(\boldsymbol{z}) \big] + \mathbb{E}_{q} \big[ \log p(\boldsymbol{x}, \boldsymbol{z}) \big].
$$

We got rid of our problematic term and more importantly by maximizing the ELBO we can minimize the KL divergence, that is what we wanted.
Ok this is amazing, but why are we allowed to do that? What's the theory behind it? To understand why this is legit we are going to unpack the ELBO:

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


**ELBO as evidence lower-bound.** As the name suggests the ELBO is a lower-bound over the evidence $$p(\boldsymbol{x})$$. This is crucial. Recall that the evidence is the intractable term that caused our troubles. First of all, let's show again the final form of the KL divergence and the ELBO:

$$
\text{KL} \big( q(\boldsymbol{z}) \vert\vert p(\boldsymbol{z} \vert \boldsymbol{x}) \big) = \mathbb{E}_{q} \big[ \log q(\boldsymbol{z}) \big] - \mathbb{E}_{q} \big[ \log p(\boldsymbol{x}, \boldsymbol{z}) \big] + \log p(\boldsymbol{x}),\\
\text{ELBO}(q) = - \mathbb{E}_{q} \big[ \log q(\boldsymbol{z}) \big] + \mathbb{E}_{q} \big[ \log p(\boldsymbol{x}, \boldsymbol{z}) \big].
$$

Note that, the KL divergence *contains* the (negative) ELBO:

$$
\text{KL} \big( q(\boldsymbol{z}) \vert\vert p(\boldsymbol{z} \vert \boldsymbol{x}) \big) = - \text{ELBO}(q) + \log p(\boldsymbol{x}) .
$$

Rearranging the terms we notice something interesting

$$
\log p(\boldsymbol{x}) - \text{KL} \big( q(\boldsymbol{z}) \vert\vert p(\boldsymbol{z} \vert \boldsymbol{x}) \big) = \text{ELBO}(q),
$$

by maximizing the ELBO we are simultaneously (i) maximizing the evidence $$p(\boldsymbol{x})$$, and (ii) minimizing the KL divergence between our variational distribution $$q(\boldsymbol{z})$$ and the true posterior $$p(\boldsymbol{z} \vert \boldsymbol{x})$$, that is what we wanted to achieve with this machinery. Crucially, since the KL divergence is non-negative $$\text{KL} \geq 0$$ the ELBO is a *lower-bound* over the log-evidence $$p(\boldsymbol{x})$$.

Conclusion
----------

I hope you enjoyed this post. I have tried to be as compact as possible, explaining every step along the way. Below you can find some additional resources if you want to know more about variational inference.


Resources
------------

- *"Variational Inference"*, A.L. Popkes [[PDF]](http://alpopkes.com/files/variational_inference.pdf)
- *"Pattern Recognition and Machine Learning"*, Chapter 10, C. Bishop
- *"Variational Inference: A Review for Statisticians"*, D. Blei, A. Kucukelbir, and J.D. McAuliffe [[arXiv]](https://arxiv.org/pdf/1601.00670v1.pdf)



