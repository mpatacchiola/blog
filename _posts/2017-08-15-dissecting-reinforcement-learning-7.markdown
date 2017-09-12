---
layout: post
title:  "Dissecting Reinforcement Learning-Part.7"
date:   2017-08-15 09:00:00 +0000
description: This blog series explains the main ideas and techniques used in reinforcement learning. In this post Reinforcement Learning through function approximation. It includes complete Python code.
author: Massimiliano Patacchiola
type: reinforcement learning
comments: false
published: true
---

So far we have represented the utility function by a lookup table (or matrix if you prefer). This approach has a problem. When the underlying Markov decision process is large there are too many states and actions to store in memory. Moreover in this case it is extremely difficult to visit all the possible states, meaning that we cannot estimate the utility values for those states. The key issue is **generalization**, meaning how to produce a good approximation of a large state space experiencing only a small subset.
In this post I will show you how to use a **linear combination of features** in order to approximate the utility function.
This new technique will allow us to master new and old problems more efficiently.

![Books Reinforcement Learning]({{site.baseurl}}/images/books_reinforcement_learning_an_introduction.png){:class="img-responsive"}

The reference for this post is chapter 8 for the [Sutton and Barto's book]((https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)) called "Generalization and Function Approximation". Moreover a good resource is the [video-lesson 6 of David Silver's course](https://www.youtube.com/watch?v=UoPei5o4fps&t=5217s). A wider introduction to function approximation is given by any good machine learning textbook, I suggest [Pattern Recognition and Machine Learning](https://books.google.co.uk/books/about/Pattern_Recognition_and_Machine_Learning.html?id=kTNoQgAACAAJ&redir_esc=y) by Christopher Bishop.
I want to start this post with (another) brief excursion in the neuroscience world. Let's see how the function approximator concept relates to biological brains.


Brains (and grandmothers)
------------------------------------------ 
You couldn't read this post without using a powerful approximator: your brain. The first primordial brains, a bunch of nerve cells, gave a great advantage allowing elementary creatures to better perceive and react, considerably extending their lifespan. Evolution shaped brains for thousands of years optimising size, modularity, and connectivity. Having a brain seems a big deal. Why? **What's the purpose of having a brain?** We can consider the world as a huge and chaotic state-space, where the correct evaluation of a specific stimulus makes the difference between life and death. The brain stores information about the environment and allows an effective interaction with it. Let suppose that our brain is a massive lookup table, which can store in a single neuron (or cell) a single state. This is known as **local representation**. This theory is often called the [grandmother cell](https://en.wikipedia.org/wiki/Grandmother_cell). A grandmother cell is a hypothetical neuron that responds only to a specific and meaningful stimulus, such as the image of one's grandmother. The term is due to the cognitive scientist [Jerry Lettvin](https://en.wikipedia.org/wiki/Jerome_Lettvin) who used it to illustrate the inconsistency of the concept during a lecture at MIT. To describe the grandmother cell theory I will use the following example. Let's suppose we bring an experimental subject in an isolated room. The activity of a group of neurons in the subject brain is constantly monitored. In front of the subject there is a screen. Showing to the subject the picture of his grandmother we notice that a specific neuron fires. Showing the grandmother in different contexts (e.g. in a group picture) activates again the neuron. However showing on screen a neutral stimulus does not activate the neuron.

![Function Approximation Grandmother Cell]({{site.baseurl}}/images/reinforcement_learning_function_approximation_experiment_neuron_grandmother.png){:class="img-responsive"}

During the 1970s the grandmother cell moved into neuroscience journals and a proper scientific discussion started. In the same period [Gross et al. (1972)](http://jn.physiology.org/content/jn/35/1/96.full.pdf) observed neurons in the inferior temporal cortex of the monkey that fired selectively to hands and faces. The grandmother cell theory started to be seriously taken into account. The theory was appealing because **simple to grasp and pretty intuitive**. However a theoretical analysis of the grandmother cell confirmed many underlying weaknesses.
For instance, in this framework the loss of a cell means the loss of a specific chunk of information. Basic neurobiological observations strongly suggest the opposite. It is possible to hypothesise multiple grandmother cells, which codify the same information in a distributed way. Redundancy prevents loss. This explanation complicate even more the situation, because storing a single state requires multiple entries in the lookup table. To store $$N$$ states without the risk of information loss, at least $$2 \times N$$ cells are required. The paradox of the grandmother cell is that trying to simplify the brain functioning, it finishes to complicate it.

Which is the alternative to the grandmother cell hypothesis? We can suppose that information is stored in a distributed way, and that each single concept is represented through a **pattern of activity**. This theory was strongly sustained by researchers such as [Geoffrey Hinton](https://en.wikipedia.org/wiki/Geoffrey_Hinton) (one of the "godfather" of deep learning), and [James McClelland](https://stanford.edu/~jlmcc/). The **distributed representation** theory gives a big advantage. Having $$N$$ cells it is possible to represent more than $$N$$ states, whereas this is not true for a local representation. Moreover a distributed representation is robust against loss and it guaranties an implicit redundancy. Even though each active unit is less specific in its meaning, the combination of active units is far more specific. To understand the difference between the two representations think about a **computer keyboard**. In the local representation each single key can codify only a single character. In the distributed representation we can use a combination of keys (e.g. Shift and Ctrl) to associate multiple characters to the same key.
In the image below (inspired by Hinton, 1984) is represented how two stimuli (red and green dots) are codified in a local and distributed scheme. The local scheme is represented as a two dimensional grid, where it is always necessary to have two active units to codify a stimulus. We can think the distributed representation as an overlapping between radial units. The two stimuli are codified through an high level pattern, which is given by the units enclosed in a specific activation radius.

![Function Approximation local vs distributed]({{site.baseurl}}/images/reinforcement_learning_function_approximation_local_vs_distributed_representation.png){:class="img-responsive"}

How is it possible to explain the monkey selective-neurons described by [Gross et al. (1972)](http://jn.physiology.org/content/jn/35/1/96.full.pdf) using a distributed representation? A selective-neuron can be the visible part of an underlying network which encapsulate the information. Further research showed that those selective neurons had a large variation in their responsiveness and that it was connected to different aspects of faces. This observation suggested that those neurons embedded a distributed representation of faces. 


If you think that the grandmother cell theory is something born and dead in the Seventies you are wrong. In recent years the local representation theory received support from biological observations (see [Bowers 2009](http://web.stanford.edu/class/psych209a/ReadingsByDate/02_01/Bowers09GrandMotherCells.pdf)), however these results have been strongly criticised by [Plaut and McClelland (2009)](http://cnbc.cmu.edu/~plaut/papers/pdf/PlautMcClelland10PR.comment-on-Bowers.pdf). For a very recent survey I suggest you [this article](http://www.tandfonline.com/doi/abs/10.1080/23273798.2016.1267782).
From a machine learning perspective we know that the distributed representation works. The success of deep learning is based on neural networks, which are powerful function approximators. Moreover different methods, such as dropout, are tightly related to the distributed representation theory. Now it's time to go back to reinforcement learning, and see how a distributed representation can solve the problems due to local representation.



Function approximation intuition
--------------------------------

Here I will use again the **robot cleaning example** described in previous posts. This robot moves in a two-dimensional world we called grid-world. It has only 4 possible actions available (forward, backward, left, right) and its goal is to reach a charger (green cell) and avoid to fall on stairs (red cell). I define with $$U(s)$$ our usual utility function, and with $$Q(s,a)$$ the state-action function.
The grid-world is a discrete rectangular state space, having $$c$$ columns and $$r$$ rows. Using a tabular approach we can represent $$U(s)$$  using a table containing $$r \times c = N$$ elements, where $$N$$ represent the total number of states. To represent $$Q(s,a)$$ we need a table of size $$N \times M$$, where $$M$$ is the total number of actions. In the previous posts I always represented the **lookup tables** using **matrices**. As utility function I used a matrix having the same size of the world, whereas for the state-action function I used a matrix having $$N$$ columns (states) and $$M$$ rows (actions). In the first case, to get the utility we have to access the location of the matrix corresponding to the particular state where we are. In the second case, we use the state as index to access the column in the state-action matrix and from that column we return the utilities of all the available actions.

![Function Approximation Lookup Tables]({{site.baseurl}}/images/reinforcement_learning_function_approximation_lookup_tables.png){:class="img-responsive"}

How can we fit the function approximation mechanism inside this scheme?
Let's start with some definitions. Defining as $$S= \{ s_{1}, s_{2}, ... , s_{N} \} $$ the set of possible states, and as $$A= \{ a_{1}, a_{2}, ... , s_{M} \} $$ the set of possible actions, we define a utility function approximator $$\hat{U}(S,\boldsymbol{w})$$ having parameters stored in a vector $$\boldsymbol{w}$$. Here I use the hat on top of $$\hat{U}$$ to differentiate this function from the tabular version $$U$$.

Before going into details on how to create a function approximator it is helpful to visualise it as a **black box**. The method described below can be used on different approximators and for this reason we can easily apply it to the box content.
The black box takes as input the current state and returns the utility of the state or the state-action utilities. That's it. The main advantage is that we can approximate (with an arbitrary small error) the utilities using less parameters respect to the tabular approach. We can say that the number of elements stored in the vector $$\boldsymbol{w}$$ is smaller than $$N$$ the number of values in the tabular counterpart.

![Function Approximation Black Boxes]({{site.baseurl}}/images/reinforcement_learning_function_approximation_black_boxes.png){:class="img-responsive"}


I guess there is a question that came to your head: **what there is inside the black box?** This is a legitimate question and now I will try to give you the intuition. In the case of a blackbox which is approximating an utility function, the content of the box is $$\hat{U}(s, \boldsymbol{w})$$. You can imagine the **utility function** as a **music mixer** and the **vector of weights** $$\boldsymbol{w}$$ as the mixer **sliders**. We want to adjust the sliders in order to obtain a sound which is similar to a predefined tone. How to do it? Well we can move one of the slider and compare the output with the reference tone. If the output is more similar to the reference we know that we moved the right slider. Continuing this process many times we eventually obtain a tone which is very similar to the reference.

![Function Approximation Weights Update Intuition]({{site.baseurl}}/images/reinforcement_learning_function_approximation_weights_update_intuition.png){:class="img-responsive"}

Using a more formal view we can say that the vector $$\boldsymbol{w}$$ is adjusted at every iteration, moving the values of a quantity $$\Delta$$, in order to reach an objective which is minimising a function of cost. The cost is given by an **error measure** that we can obtain comparing the output of the function with a target. For instance, we know from previous posts that the actual utility value of state (4,1) in our gridworld is 0.388. Let's say that at time $$t$$ the output of the box is 0.352. After the **update step** the output will be 0.371, we moved closer to the target value.

![Function Approximation Methods]({{site.baseurl}}/images/reinforcement_learning_function_approximation_methods_overview.png){:class="img-responsive"}

Function approximation is an instance of [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning). In principle all the supervised learning techniques could be used in function approximation. However here I will consider only **differentiable** function approximators such as linear combination of features and neural networks. In this post I will focus on linear combination of features. Before describing the simplest case **the linear approximator**, I would like to introduce the general methodology used to adjust the vector of weights. The goal in function approximation is to move as close as possible to the real utility function adjusting the internal parameters stored in $$\boldsymbol{w}$$. To achieve this goal we need two things, first an **error measure** which can give us a feedback on how close we are to the target, second an **update rule** for adjusting the weights. In the next section I will describe these two components.

Method
--------

To improve the performance of our function approximator we need an error measure and an update rule. These two parts work tightly in the learning cycle of every supervised learning technique. Their use in reinforcement learning is not much different from how they are used in a classification task. In order to understand this section you need to refresh some concepts of [multivariable calculus](https://en.wikipedia.org/wiki/Multivariable_calculus) such as the [partial derivative](https://en.wikipedia.org/wiki/Partial_derivative) and [gradient](https://en.wikipedia.org/wiki/Gradient).

**Error Measure**: a common error measure is given by the [Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) between two quantities. For instance, if we have the optimal utility function $$U^{*}(S)$$ and an approximator function $$\hat{U}(s, \boldsymbol{w})$$, then the MSE is defined as follows:

$$ \text{MSE}( \boldsymbol{w} ) = \frac{1}{N} \sum_{s \in S} \big[ U^{*}(s) - \hat{U}(s, \boldsymbol{w}) \big]^{2}  $$

[comment]: <> (where $$P(s)$$ is a distribution weighting the errors of different states, such that $$\sum_{s} P(s) = 1$$. The number of parameters $$\boldsymbol{w}$$ is lower that the number of total states $$N$$. The function $$P(s)$$ allows gaining accuracy on some states instead of others.)

that's it, the MSE is given by the expectation $$\mathop{\mathbb{E}}[ (U^{*}(s) - \hat{U}(s, \boldsymbol{w}) \big)^{2} ]$$. 

**Update rule**: the update rule for differentiable approximator is [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent). The [gradient](https://en.wikipedia.org/wiki/Gradient) is a generalisation of the concept of derivative which applies to functions of several variables. You can imagine the gradient as the vector that points in the direction of the greatest rate of increase. Intuitively, if you want to reach the top of a mountain the gradient is an arrow that in each moment show you in which direction you should to go. The gradient is generally represented with the operator $$\nabla$$ and is also called **nabla**. 
The goal in gradient descent is to minimise the error measure. We can achieve this goal moving in the direction of the negative gradient vector, meaning that we are not moving anymore to the top of the mountain but downslope. At each step we adjust the parameter vector $$\boldsymbol{w}$$ moving a step closer to the valley.

$$\begin{eqnarray} 
\boldsymbol{w}_{t+1} &=&  \boldsymbol{w}_{t}  - \frac{1}{2} \alpha \nabla_{\boldsymbol{w}} \text{MSE}(\boldsymbol{w}) \\
&=& \boldsymbol{w}_{t}  - \frac{1}{2} \alpha \nabla_{\boldsymbol{w}} \mathop{\mathbb{E}} \big[ (U^{*}(s) - \hat{U}(s, \boldsymbol{w}) \big)^{2} \big]\\
&=& \boldsymbol{w}_{t}  + \alpha \mathop{\mathbb{E}} \big[ (U^{*}(s) - \hat{U}(s, \boldsymbol{w})) \nabla_{\boldsymbol{w}} \hat{U}(s, \boldsymbol{w}) \big] \\
&=&  \boldsymbol{w}_{t} + \alpha [ U^{*}(s) - \hat{U}(s, \boldsymbol{w}) ] \nabla_{\boldsymbol{w}} \hat{U}(s, \boldsymbol{w})
\end{eqnarray}$$

Linear approximator
--------------------

Here I will describe a **linear approximator** which is the simplest case of linear combinations, whereas in the next section I will describe some high order approximators. Before describing a liner approximator I want to clarify a crucial point. The linear approximator is a particular case of the broader class of linear combination of features.
A liner combination is based on a **polynomial** which can be or not a line. Using only a line to discriminate between states would be very limited. **Linear combination** means that the **parameters are linearly combined**. We are not saying anything about the input features, which in fact may be represented by and high-order polynomial.

To describe the linear approximator I will start from the usual robot cleaning example. Let's suppose we are in a very large state space, let's say a massive factory, and the cleaning robot has to find the charging stations which are in the corners on the right side of the room. At the centre of the factory there are some machines, we consider them as obstacles. On the left side there is the main entrance with many trucks and forklift moving around, better don't go there. In this scenario there is an high redundancy in storing all the possible states in a lookup table. What the robot has to learn is a pattern: the right corners of the room have maximum utility, the left corners of the room have negative utility. 

![Function Approximation Linear OR]({{site.baseurl}}/images/reinforcement_learning_function_approximation_linear_function_or_world.png){:class="img-responsive"}

**Choosing the features** is a crucial point. Features can be the position of a robot, position and speed of an inverted pendulum, configurations of the stones in a Go game, etc. Here I define $$\boldsymbol{x}$$ as the feature vector, and $$\boldsymbol{w}$$ as a vector of weights (or parameters) having the same dimension of $$\boldsymbol{x}$$. The feature vector contains the features isolated from the state-space. For instance in the cleaning robot example $$\boldsymbol{x}$$ may contain the position in terms of row and column indices.

The utility can be estimated through the dot product between $$\boldsymbol{x}$$ and $$\boldsymbol{w}$$, as follows:

$$ \hat{U}(s, \boldsymbol{w}) = \boldsymbol{x}(s)^{T} \boldsymbol{w} $$

If you are not used to linear algebra notation don't get scared. This is equivalent to the following sum:

$$ \hat{U}(s, \boldsymbol{w}) = x_{1} w_{1} + x_{2} w_{2} + ... + x_{N} w_{N} $$

where $$N$$ is the total number of features. Geometrically this solution is represented by a line (in two-dimensional space), a plane (in a three-dimensional space), or an hyper-plane (in hyper-spaces).  Now we know the content of the black box, which is given by the product of the vectors $$\boldsymbol{x}$$ and $$\boldsymbol{w}$$. However in order to apply the method described in the previous section we still need the MSE. I said that it can be estimated from the difference between the optimal function $$U^{*}(S)$$ and the content of the black box $$\hat{U}(s, \boldsymbol{w})$$. Now we have the content of the black box $$\hat{U}(s, \boldsymbol{w}) = \boldsymbol{x}(s)^{T} \boldsymbol{w}$$, but we do not have the optimal function. What we have is a rough approximation of the optimal function, which is given by our estimator. The formula for the RMSE reduces to:

$$ \text{MSE}( \boldsymbol{w} ) = \sum_{s \in S} \big[ \hat{U}(s, \boldsymbol{w}) - \boldsymbol{x}(s)^{T} \boldsymbol{w} \big]^{2}  $$

In the previous section I said that the MSE can be used as a feedback to adjust the values inside the black box.
However in our case we do not have the optimal utility function $$U^{*}(S)$$, what we have is an approximation of this function, given by the content of the black box $$\hat{U}(s, \boldsymbol{w})$$.

The **Generalised Policy Iteration (GPI)** (see [second post](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html)) applies here as well. Let's suppose we start with a random set of weights. At the very first step the agent follows an epsilon-greedy strategy. After the first step it is possible to update the weights using gradient descent. What's the effect of this adjustment? The effect is to slightly improve the utility function. At the next step the agent follows again a greedy strategy, then the weights are updated through gradient descent, and so on and so forth. As you can see we are applying the GPI scheme again. 


High-order approximators
------------------------

The linear approximator is the simplest form of approximation. It has many limits, and it cannot represent complex state spaces. High-order approximators may find useful links between multiple futures. An example of high-order approximator is the **quadratic approximator**. In the quadratic approximator we use a second order polynomial to model the utility function. 

$$ \hat{U}(s, \boldsymbol{w}) = x_{1} w_{1} + x_{2} w_{2} + x_{1}^{2} w_{3} + x_{2}^{2} w_{4} +... + x_{N-1} w_{M-1} + x_{N}^{2} w_{M} $$





Conclusions
-----------



Index
------

1. [[First Post]](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) Markov Decision Process, Bellman Equation, Value iteration and Policy Iteration algorithms.
2. [[Second Post]](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) Monte Carlo Intuition, Monte Carlo methods, Prediction and Control, Generalised Policy Iteration, Q-function. 
3. [[Third Post]](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html) Temporal Differencing intuition, Animal Learning, TD(0), TD(λ) and Eligibility Traces, SARSA, Q-learning.
4. [[Fourth Post]](https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html) Neurobiology behind Actor-Critic methods, computational Actor-Critic methods, Actor-only and Critic-only methods.
5. [[Fifth Post]](https://mpatacchiola.github.io/blog/2017/03/14/dissecting-reinforcement-learning-5.html) Evolutionary Algorithms introduction, Genetic Algorithm in Reinforcement Learning, Genetic Algorithms for policy selection.
6. [[Sixt Post]](https://mpatacchiola.github.io/blog/2017/08/14/dissecting-reinforcement-learning-6.html) Reinforcement learning applications, Multi-Armed Bandit, Mountain Car, Inverted Pendulum, Drone landing, Hard problems.
7. **[Seventh Post]** Function approximation, Neurobiology, Intuitive introduction, Linear approximator, High-order approximator, Applications.

Resources
----------

- The **complete code** for the Reinforcement Learning Function Approximation is available on the [dissecting-reinforcement-learning](https://github.com/mpatacchiola/dissecting-reinforcement-learning) official repository on GitHub.

- **Reinforcement learning: An introduction (Chapter 8 'Generalization and Function Approximation')** Sutton, R. S., & Barto, A. G. (1998). Cambridge: MIT press. [[html]](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)



References
------------

Bowers, J. S. (2009). On the biological plausibility of grandmother cells: implications for neural network theories in psychology and neuroscience. Psychological review, 116(1), 220.

Gross, C. G., Rocha-Miranda, C. E. D., & Bender, D. B. (1972). Visual properties of neurons in inferotemporal cortex of the Macaque. Journal of neurophysiology, 35(1), 96-111.

Gross, C. G. (2002). Genealogy of the “grandmother cell”. The Neuroscientist, 8(5), 512-518.

Hinton, G. E. (1984). Distributed representations.

Plaut, D. C., & McClelland, J. L. (2010). Locating object knowledge in the brain: Comment on Bowers’s (2009) attempt to revive the grandmother cell hypothesis.



