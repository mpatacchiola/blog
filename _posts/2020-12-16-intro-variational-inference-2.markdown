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

This is the second post of the series on variational inference. In the previous post I have introduced the variational framework, and the three main characters at play: the evidence, the Kullback-Leibler (KL) divergence, and the Evidence Lower BOund (ELBO). I have showed how the ELBO can be considered as a surrogate objective for finding the posterior distribution $$p(\mathbf{z} \vert \mathbf{x})$$ when the evidence $$p(\mathbf{x})$$ is intractable. By maximizing the ELBO we can find the parameters $$\boldsymbol{\theta}$$ of a variational distribution $$q_{\theta}(\mathbf{z})$$ that better fit the posterior. However, I did not mention how the ELBO itself can by maximized in practice. One approach is to estimate the gradient of the ELBO with respect to $$\boldsymbol{\theta}$$ and then move in the direction of steepest ascent. Depending on the particular problem at hand and the choice of variational distribution, this could be challenging. 

In this post I will focus on this particular problem, showing how we can estimate the gradients of the ELBO by using two techniques: the score function estimator (a.k.a. REINFORCE) and the pathwise estimator (a.k.a. reparametrization trick).


Definition of the problem
--------------------------

Here we are interested in a system with two components: a stochastic component called *measure* and a *cost function*, with training consisting of two phases: a simulation phase and an optimisation phase. The entire system is stochastic since one of the element is stochastic. However, in many cases the system is said to be *doubly-stochastic* if for instance we are using stochastic gradient descent in the optimization component. In a doubly-stochastic system one source of randomness arises from the simulation phase (e.g. via the Monte Carlo estimators) and a second source arises in the optimization phase (e.g. sampling of datapoints in the mini-batch gradient descent).

Let's try now to better formalize our setting. Consider a generic density function $$p_{\theta}(x)$$ parameterized by a vector $$\boldsymbol{\theta}$$, representing the stochastic component (measure), and a function $$f(x)$$, representing the *cost function*. We assume $$p_{\theta}(x)$$ to be differentiable, but the cost function $$f(x)$$ is not necessarily differentiable, for instance it could be discrete or a black-box (e.g. the output is all we have). Since we are dealing with expectations (integrals) we can use Monte Carlo to get an unbiased approximation of the expected value. Monte Carlo numerically evaluates the integral by drawing samples $$x_1, \dots, x_N$$ from the distribution $$p_{\theta}(x)$$ and computing the average of the function evaluated at these points. In our case, the Monte Carlo approximation of $$\mathbb{E}_{p_{\theta}}[f(x)]$$ corresponds to

$$
\mathbb{E}_{p_{\theta}}[f(x)] \approx \frac{1}{N} \sum_{n=1}^{N} f(\hat{x}_{n}) \quad \text{with} \quad \hat{x}_{n} \sim p_{\theta}(x), 
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

Note that, the gradient of the ELBO can also be expressed in terms of measure and cost function: the variational distribution $$q$$ represents the measure and the log-ratio inside the brackets the cost function.

In the next sections I will first introduce some desirable properties of gradient estimators and then I will present the score function estimator and the pathwise gradient estimator, which are part of two different groups, the former differentiate through the measure and the latter through the cost function.


Properties of an estimator
----------------------------

Following [Mohamed et al. (2019)](https://arxiv.org/abs/1906.10652) I will use four properties to quantify the quality of an estimator:

1. *Consistency.* Increasing $$N$$, the number of samples, the estimate should converge to the true expected value (see [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers)). Typically, our estimators are consistent, meaning we will not focus too much on this property.
2. *Unbiasedness.* Repeating the estimate multiple times we see that *on average* the estimate is centered around the true expected value. This property is preferred because it guarantees the convergence of stochastic optimisation procedures.
3. *Minimum variance.* Estimators are random variables since they depend on a random variable. Given the same number of samples $$N$$, we prefer the estimator with lower variance. Low variance means that the gradient estimates are more accurate, therefore we have a higher chances of converging to useful local minima and we can use larger step sizes (faster convergence).
4. *Efficiency.* We prefer estimators that needs a low number of samples and can be easily parallelized.

The estimators that we will consider below differ in their variance and efficiency.

The score function estimator
-----------------------------

The score function estimator is also known as the likelihood-ratio estimator and as REINFORCE in the reinforcement learning literature. This estimator gets the gradient by computing the derivatives of the measure, meaning by direct differentiation through $$p_{\theta}(x)$$.

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


**Derivation of the estimator.** We want to overcome the problem described in the previous section, where we saw that the gradient of an expectation cannot be approximated via Monte Carlo. Our goal here will be to exploit the log-derivative trick to bypass this issue. In particular, we want to reach a friendly form of the integral and turn it into a proper expectation. Let's break it down

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

Note that, $$(2.5)$$ is a proper expectation and we can now get the Monte Carlo approximation of the gradient as we wanted

$$
\mathbb{E}_{p_{\theta}} \big[ \nabla_\theta \log p_{\theta}(x) f(x) \big] \approx \frac{1}{N} \sum_{n=1}^{N} \nabla_\theta \log p_{\theta}(\hat{x}_{n}) f(\hat{x}_{n})
\quad \text{with} \quad 
\hat{x}_{n} \sim p_{\theta}(x).
$$

Here is a step-by-step description of the procedure I have followed:

$$(2.1 \ \text{and} \ 2.2)$$ Those two steps are identical to the one described in the previous section. In the first step apply the definition of expectation and in the second the Leibniz's rule to move the gradient inside the integral.

$$(2.2 \rightarrow 2.3)$$ We use the identity trick by adding a new term at our convenience. The value of the new term is equal to 1, therefore it does not have any effect on the product.

$$(2.3 \rightarrow 2.4)$$ Rearranging the terms by switching the denominator.

$$(2.4 \rightarrow 2.5)$$ After rearranging the terms we notice that it is possible to use the log-derivative trick.

$$(2.5 \rightarrow 2.6)$$ The new form of the integral corresponds to a proper expectation, therefore we can rewrite in the equivalent form.

**[TODO] The score function estimator of the ELBO.**

The score function estimator is particularly popular because it is unbiased, flexible, and it does not impose any restriction over $$p_{\theta}(x)$$ or $$f(x)$$. However, a major issue is that is has large variance. Let's see what does this implies.


The variance of the gradient
---------------------------- 

In the previous section we saw how the score function estimator allows us to estimate the gradient of the ELBO. In practice, we have exploited the log-derivative trick to move the gradient estimate inside the expectation. This has solved one issue but it created another. 


Another issue is that the estimate $$\mathbb{E}_{p_{\theta}}[f(x)]$$ is done over the $$N$$ datapoints in our dataset. Given the size of many datasets this is often computationally expensive. What we do instead is to choose a random sample of those points $$M$$ with $$M \ll N$$, often called mini-batch. This is know as [stochastic variational inference](https://www.jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf). Using the same notation defined above, consider $$\nabla_\theta f(x_{m})$$ as the gradient w.r.t. a single datapoint, and $$\mathbb{E}_{p_{\theta}} [\nabla_\theta f(x)]$$ as the gradient w.r.t. all the $$N$$ datapoints. Through stochastic variational inference we get the following approximation of the gradient

$$
\mathbb{E}_{p_{\theta}} [\nabla_\theta f(x)] 
\approx \tilde{\mathbb{E}}_{p_{\theta}} = \frac{N}{M} \sum_{m=1}^{M} \nabla_\theta f(x_{m}).
$$

Stochastic variational inference solves one issue but it creates another. The number of datapoints in the mini-batch impacts the quality of the gradient. We can quantify this quality by introducing the variance of the gradient.


The pathwise gradient estimator
-------------------------------

Another way to estimate the gradient consists of differentiating the cost function $$f(x)$$ through the random variable $$x$$, which encodes the pathway from the target parameters $$\boldsymbol{\theta}$$. This approach is the one used in the pathwise gradient estimator. The pathwise estimator also appears under several names, such as process derivative, pathwise derivative, and more recently as the reparameterisation trick.

One of the issues with the score function estimator is that it has a large variance, meaning that the resulting signal does a poor job at finding the parameters we are interested in.

A well know application of the reparameterization trick in the machine learning context is on Variational Auto-Encoders (VAEs). In VAEs the variational distribution is a factorized Gaussian, with the gradients backpropagated to the encoder thanks to the reparameterization trick.

Conclusion
----------



Resources
------------

- [Shakir's blog](http://blog.shakirm.com) in particular [[link-1]](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/) [[link-2]](http://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/)
- [Yuge's blog](https://yugeten.github.io/) in particular [[link]](https://yugeten.github.io/posts/2020/06/elbo/)
- *"Pattern Recognition and Machine Learning"*, Chapter 10, C. Bishop
- *"Monte Carlo Gradient Estimation in Machine Learning"* (2019), S. Mohamed, M. Rosca, M. Figurnov, A. Mnih [arXiv](https://arxiv.org/abs/1906.10652)


