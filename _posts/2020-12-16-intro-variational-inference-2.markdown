---
layout: post
title:  "Variational inference: the gradient of the ELBO"
date:   2020-12-16 08:00:00 +0000
description: .
author: Massimiliano Patacchiola
type: bayesian methods
comments: false
published: false
---

This is the second post of the series on variational inference. In the previous post I have introduced the problem, and the three main characters at play: the evidence, the Kullback-Leibler (KL) divergence, and the Evidence Lower BOund (ELBO). I have showed how the ELBO can be considered as a surrogate objective to find the posterior distribution $$p(\mathbf{z} \vert \mathbf{x})$$ when the evidence is intractable. By maximizing the ELBO we can find the parameters $$\boldsymbol{\theta}$$ of a variational distribution $$q_{\theta}(\mathbf{z})$$ that better fit the posterior. However, I did not mention how the ELBO itself can by maximized in practice. One approach is to estimate the gradient of the ELBO with respect to $$\boldsymbol{\theta}$$ and then move in the direction of steepest ascent. Depending on the particular problem at hand and the choice of variational distribution, this could be challenging. 


In this post I will focus on this particular problem, showing how we can estimate the gradients of the ELBO by using two techniques: the reparametrization trick, and the score function estimators.


Definition of the problem
--------------------------

Every time we have an expectation we can use Monte Carlo to get an unbiased approximation of the expected value. Let's consider a generic density function $$p_{\theta}(x)$$ parameterized by a vector $$\boldsymbol{\theta}$$ and a function $$f(x)$$, the Monte Carlo approximation of $$\mathbb{E}_{p_{\theta}}[f(x)]$$ corresponds to

$$
\mathbb{E}_{p_{\theta}}[f(x)] \approx \frac{1}{N} \sum_{n=1}^{N} f(x_{n}) \quad \text{with} \quad x_{n} \sim p_{\theta}(x), 
$$

where $$N$$ is the to total number of samples. However, in many applications we are interested in something slightly different: we wish to get the *gradient of the expectation*. This raises an issue, the gradient of an expectation cannot be approximated via Monte Carlo as we usually do. Let's see why:

$$
\begin{align} 
\nabla_{\theta} \mathbb{E}_{p_{\theta}}[f(x)] 
&= \nabla_\theta \int p_{\theta}(x) f(x) dx &\text{(1.1)}\\ 
&= \int \underbrace{\nabla_\theta p_{\theta}(x)}_{\text{issue}} f(x) dx &\text{(1.2)}\\
\end{align}
$$

In the last step I have used the [Leibniz's rule](https://en.wikipedia.org/wiki/Leibniz_integral_rule) to move the gradient inside the integral. Here is the issue, most of the time the gradient of a density function is not itself a density function. In our particular case, the quantity $$\nabla_\theta p(x_{\theta})$$ may not be a proper density. It follows that the integral in $$\text{(1.2)}$$ cannot be casted into an expectation and therefore it cannot be approximated as usual via Monte Carlo.

When we try to get the gradient of the ELBO we incur in the same issue, since the gradient of the ELBO corresponds to the gradient of an expectation:

$$
\nabla_{\theta} \text{ELBO}(q)
= \nabla_{\theta} \mathbb{E}_{q} \bigg[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{q(\boldsymbol{z})} \bigg].
$$

In the next sections I will introduce some of the techniques we can use to solve this issue.


The score function estimator
-----------------------------

One way to solve the problem is to use the score function estimator. This technique has taken various names, such as: REINFORCE, likelihood-ratio estimator.

**What is the score?** In statistical terms, the [score](https://en.wikipedia.org/wiki/Score_(statistics)) (or score function) is just the gradient of the log-likelihood function with respect to the parameter vector. In our case this correspond to

$$
\text{score} = \nabla_{\theta} \log p_{\theta}(x).
$$



**The log-derivative trick.** The log-derivative trick just consists of applying the rule of the gradient on the logarithm of a function. If we apply this trick to the score function we get

$$
\nabla_\theta \log p_{\theta}(x) = \frac{\nabla_{\theta} p_{\theta}(x)}{p_{\theta}(x)}.
$$

The term on the right is called the *score ratio*, which explains why sometimes this estimator is also called the score-ratio estimator. The log-derivative trick can be quite helpful. For instance, exploiting this trick we can notice an interesting property of the score function, that is its expected value is equal to zero

$$
\mathbb{E}_{p_{\theta}}[\nabla_{\theta} \log p_{\theta}(x)]
=  \mathbb{E}_{p_{\theta}} \bigg[ \frac{\nabla_{\theta} p_{\theta}(x)}{p_{\theta}(x)} \bigg] 
= \int p_{\theta}(x) \frac{\nabla_{\theta} p_{\theta}(x)}{p_{\theta}(x)} dx
= \nabla_{\theta} \int  p_{\theta}(x) dx
= \nabla_{\theta} 1 = 0
$$

This property is not particularly relevant here but it is fundamental in the context of [control variates](https://en.wikipedia.org/wiki/Control_variates). Armed with the log-derivative trick, we are ready to derive the score function estimator.


**Derivation of the estimator.** We want to overcome the problem described in the previous section, where we saw that the gradient of an expectation cannot be approximated via Monte Carlo. Our goal will be to exploit the log-derivative trick to bypass this issue. In particular, we want to reach a friendly form of the integral and turn it into a proper expectation. Let's break it down

$$
\begin{align} 
\nabla_\theta \mathbb{E}_{p_{\theta}}[f(x)] 
&= \nabla_\theta \int p_{\theta}(x) f(x) dx &\text{(2.1)}\\
&= \int \nabla_\theta p_{\theta}(x) f(x) dx &\text{(2.2)}\\
&= \int \frac{p_{\theta}(x)}{p_{\theta}(x)} \nabla_\theta p_{\theta}(x) f(x) dx &\text{(2.3)}\\
&= \int p_{\theta}(x) \frac{\nabla_\theta p_{\theta}(x)}{p_{\theta}(x)} f(x) dx &\text{(2.4)}\\
&= \int p_{\theta}(x) \nabla_\theta \log p_{\theta}(x) f(x) dx &\text{(2.5)}\\
&= \mathbb{E}_{p_{\theta}} \big[ \underbrace{\nabla_\theta \log p_{\theta}(x)}_{\text{score function}} f(x) \big] &\text{(2.6)}
\end{align}
$$

Note that, the last form in $$(2.6)$$ is a proper expectation and we can now get the Monte Carlo approximation of the gradient as we wanted

$$
\mathbb{E}_{p_{\theta}} \big[ \nabla_\theta \log p_{\theta}(x) f(x) \big] \approx \frac{1}{N} \sum_{n=1}^{N} \nabla_\theta \log p_{\theta}(x_{n}) f(x_{n})
\quad \text{with} \quad 
x_{n} \sim p_{\theta}(x).
$$

Here is a step-by-step description of the procedure I have followed:

$$(2.1 \ \text{and} \ 2.2)$$ Those two steps are identical to the one described in the previous section. In the first step apply the definition of expectation and in the second the Leibniz's rule to move the gradient inside the integral.

$$(2.2 \rightarrow 2.3)$$ We use the identity trick by adding a new term at our convenience. The value of the new term is equal to 1, therefore it does not have any effect on the product.

$$(2.3 \rightarrow 2.4)$$ Rearranging the terms by switching the denominator.

$$(2.4 \rightarrow 2.5)$$ After rearranging the terms we notice that it is possible to use the log-derivative trick.

$$(2.5 \rightarrow 2.6)$$ The new form of the integral corresponds to a proper expectation, therefore we can rewrite in the equivalent form.


The reparameterization trick
----------------------------

One of the issues of the score function estimator is that it has a large variance, meaning that the resulting signal does a poor job at finding the parameters we are interested in.

Conclusion
----------



Resources
------------

- [Shakir's blog](http://blog.shakirm.com) in particular [[link-1]](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/) [[link-2]](http://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/)
- *"Pattern Recognition and Machine Learning"*, Chapter 10, C. Bishop


