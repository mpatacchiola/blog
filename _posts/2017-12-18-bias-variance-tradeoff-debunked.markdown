---
layout: post
title:  "Bias-Variance tradeoff debunked"
date:   2017-12-17 09:00:00 +0000
description: This blog series explains the main ideas and techniques used in reinforcement learning. In this post Reinforcement Learning through non-linear function approximation (Neural Networks) and policy gradient methods. It includes complete Python code.
author: Massimiliano Patacchiola
type: computer vision
comments: false
published: false
---

The bias-variance tradeoff is the backbone of model selection, and it can be explained from different points of view. Here I will extensively use terms from probability theory and statistics but I will try to ground everything in concrete machine learning scenarios. If you are not happy about the explanations available out there then this post can (hopefully) give you a deeper inside on this topic. 


Intuition
----------
As a gentle introduction before the math, I would like to describe from an high level what we are talking about and why we should care.

Model selection.

Foundational concepts from probability theory and statistics are useful to characterize notions of generalization, underfitting and overfitting.

Estimators
----------
In [estimation theory](https://en.wikipedia.org/wiki/Estimation_theory) (a branch of statistics) the goal is to approximate the parameters of an unknown model using observed data taken from the model. We start from two main components, the vector of observed values $$\boldsymbol{x} = \{ x_{1}, ..., x_{n} \}$$ and the vector of parameters $$\boldsymbol{\theta} = \{ \theta_{1}, ..., \theta_{m} \}$$.
The parameters in $$\boldsymbol{\theta}$$ belong to an unknown model that generated the data in $$\boldsymbol{x}$$. Important: the parameters in $$\boldsymbol{\theta}$$ are not random variables! Those parameters are **constant values** belonging to the model.

Here we want the best possible guess about the unknown fixed parameters in $$\boldsymbol{\theta}$$, this guess is often called [point estimation](https://en.wikipedia.org/wiki/Point_estimation) or [statistic](https://en.wikipedia.org/wiki/Statistic). The point estimation is obtained through an **estimator**, that is a function of the observed data $$\boldsymbol{x}$$:

$$\boldsymbol{\hat{\theta}} = g(x_{1}, ..., x_{n})$$

The result of the point estimation is a vector of approximated parameters $$\boldsymbol{\hat{\theta}} = \{ \hat{\theta}_{1}, ..., \hat{\theta}_{m} \}$$. Important: the parameters in $$\boldsymbol{\hat{\theta}}$$ are **random variables**! Why is that? Those parameters are function of the data $$\boldsymbol{x} = \{ x_{1}, ..., x_{n} \}$$ and the data is drawn from a random distribution, therefore the parameters are random variables.

To ground all these concepts I will use the classic example of coin tossing. The outcomes of a single coin toss are head and tail, for this reason we can use a [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution) to model this phenomenon. 

$$f(k;\theta) = \theta^{k} (1-\theta)^{1-k} $$

As you can see this model has only a single parameter $$\theta$$ and this is why I am not using the vector notation (bold) to denote it. The value $$k$$ represent the possible outcomes and in our case they are tail or head (zero or one). The function $$f(k, \theta)$$ is called [probability mass function](https://en.wikipedia.org/wiki/Probability_mass_function) and it describes the probability that a discrete random variable is equal to some values (in our case zero and one). To understand how this function works, we suppose that **the coin is not fair**. That's it, head and tail do not have the same probability. In particular we assume that *head* has 80% probability to pop out. We translate this assumption assigning $$\theta = 0.8$$ to the model parameter. 

$$f(k; 0.8) = 0.8^{k} (1-0.8)^{1-k} $$

If I want to know what's the probability of having tail (zero) based on this model, I can plug zero in the function and check the result:

$$f(0; 0.8) = 0.8^{0} (1-0.8)^{1-0} = 0.2$$

The results says that I have 20% probability of obtaining tail. That's trivial given this simple model. Now let's suppose that we get a new unfair coin and we want to estimate the value of $$\theta$$ associated to its Bernoulli distribution. How to do it? We can toss the coin and record the outcomes, generating a dataset of observed values. This procedure is called an [experiment](https://en.wikipedia.org/wiki/Experiment_(probability_theory)) in the probability theory lingo. In the case of a Bernoulli distribution a single coin toss is defined as a [Bernoulli trial](https://en.wikipedia.org/wiki/Bernoulli_trial). Repeating this experiment multiple times we get a  **binomial experiment** having a well defined [binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution). The binomial distribution is the discrete probability distribution of the number of successes (outcome of one) in a sequence of independent experiments. In our case, multiple coin tosses can be modelled through a binomial distribution where the outcome *head* is considered a success. For instance, tossing the coin six times we get the following outcomes 

$$\boldsymbol{x} = \{ \text{head}, \text{ head}, \text{ tail}, \text{ head}, \text{ tail}, \text{ tail} \} = \{1, 1, 0, 1, 0, 0 \}$$ 

Having this dataset we can use an estimator to guess the value of $$\theta$$. There are many estimators available, but here I will use the [Maximum Likelihood Estimator (MLE)](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation). To remain on track I am not going to describe the MLE which may be material for a future post. The MLE is a powerful and generic method that can be applied to different distributions. In our case applying the MLE to the Bernoulli distribution reduces to a very clean equation:




[We have a dataset $$D = \{ x_{1}, ..., x_{n} \}$$ containing $$n$$ samples. We interpret the dataset as a collection of independent outcomes obtained from an underlying probability distribution. For example, tossing a coin six times we can obtain the following dataset $$D = \{ head, head, tail, head, tail, tail \}$$. We can imagine that there is a model having fixed parameters $$\theta$$ and that this model is the one that generated our dataset. Important: the parameters $$\theta$$ are **not random variables**! Those parameters are constant values belonging to the model. This golden model is what we are looking for, we want to estimate it and to do it we can use the data that it has generated (the dataset). To approximate the golden model we can use an **estimator**. An estimator can be defined as a function that generates approximated parameters $$\hat{\theta}$$ using the dataset. $$\hat{\theta} = g(x_{1}, ..., x_{n})$$ Important: the parameters $$\hat{\theta}$$ of the estimator are **random variables**! Why is that? Those parameters are function of the data $$\boldsymbol{x} = \{ x_{1}, ..., x_{n} \}$$ and the data is drawn from a random distribution, therefore the parameters are random variables. Roughly speaking, the function $$g()$$ is given by the algorithm and the training method we are using. For example when using a neural network as approximator, the function $$g()$$ is the result of a recursive application of the backpropagation. At the end of the training the backpropagation give us the weights of the network and those weights are simply $$\hat{\theta}$$.]


The bias
---------
What is an "unbiased" estimator? (see Deep Learning book, chapter 5.4)

Now let's suppose we defined our estimator and we obtained the parameters $$\hat{\theta}$$ thanks to a dataset. Now we want to know how good our approximation is, meaning how well it represents the true golden model. How to do it? The result of this comparison is the [**bias** of the estimator](https://en.wikipedia.org/wiki/Bias_of_an_estimator). The term *bias* must be intended as a property of the estimator, and it does not necessarily have a negative connotation. In fact most of the estimators used in machine learning are biased estimators. We will see later why in many contexts a biased estimator is better that an unbiased one. Now let's see how the bias is defined from the mathematical point of view:

$$\text{bias}(\hat{\theta}_{n}) = E[\hat{\theta}_{n}] - \theta $$


This equation is essential but it can be very confusing. At a first glance it seems we are comparing the parameters of the estimator $$\hat{\theta}$$ and the constant parameter $$\theta$$ of the generator.
However, we are taking the expectation of $$\hat{\theta}$$, what does it mean? In the previous section I stressed the fact that the parameters in $$\hat{\theta}$$ are random variables. Being random variables they should change, right? In fact they do! If we go back to the coin toss experiment, we can estimate the parameter $$\hat{\theta}$$ of the Bernoulli distribution observing the outcomes of consecutive coin tosses. Our approximation will change every time that we produced a different dataset. For instance if I repeat 200 Bernoulli trials, it is likely that every time I have a new batch of 200 observations this batch contains different combination of head-tail outcomes. The value estimated by MLE on the first batch will be different from the one estimated in the second batch, and so on so forth.

Our approximation of $$\hat{\theta}$$ will change with the incoming streaming of observations. On the other hand, the parameter $$\theta$$ of the underlying generator will remain exactly the same (constant). How can we compare a random variable and a constant? We have to take the expectation of the random variable. That's exactly what is done in the bias equation.

Imagine that you take one by one all the values contained in the dataset. In the coin tossing example it means that we are taking one by one all the $$n$$ outcomes contained in $$D$$ and we are using them to find $$\hat{\theta}$$. Let's do it:

$$\hat{\theta} = \hat{\theta}^{1} - (1-\hat{\theta})^{(1-1)} = $$

When we are dealing with a function estimator we can apply the same reasoning. Remember that in case of a function estimator our dataset contains input-output pairs. What we can do is simply take the input values contained in our dataset and see what our model generates, then we compare those outputs with the ones generated by the golden model (contained in the dataset). 



The variance
------------


The Bias-Variance decomposition
-------------------------------
How to find out that MSE = bias^2 + variance

Great, we know what is the bias and what is the variance. How are those quantities related?



Conclusions
-----------


Resources
----------



- Book: An introduction to statistical learning, chapter 2.2 "Assessing Model Accuracy", (2014) James, Witten, Hastie and Tibshirani [[pdf]](http://www-bcf.usc.edu/~gareth/ISL/)

- Book: The Elements of Statistical Learning (Second Edition), chapters 7 "Model Assessment and Selection", Hastie, Tibshirani, Friedman [[website]](https://web.stanford.edu/~hastie/ElemStatLearn/)

- Book: Deep Learning, chapter 5.4 "Estimators, bias and variance", (2016) Goodfellow, Bengio, Courville [[website]](http://www.deeplearningbook.org/)

- Video: A step by step explanation of the mathematical decomposition [[youtube]](https://www.youtube.com/watch?v=C3nIFH649wY)

References
------------




