---
layout: post
title:  "Dissecting Reinforcement Learning-Part.7"
date:   2017-08-15 09:00:00 +0000
description: This blog series explains the main ideas and techniques used in reinforcement learning. In this post Reinforcement Learning through function approximation. It includes complete Python code.
author: Massimiliano Patacchiola
type: reinforcement learning
comments: false
published: false
---

So far we have represented the utility function by a lookup table (or matrix if you prefer). This approach has a problem. When the underlying Markov decision process is large there are too many states and actions to store in memory. Moreover in this case it is extremely difficult to visit all the possible states, meaning that we cannot estimate the utility values for those states. The key issue is **generalization**, meaning how to produce a good approximation of a large state space experiencing only a small subset.
In this post I will show you how to use a **linear combination of features** in order to approximate the utility function.
This new technique will allow us to master new and old problems more efficiently.

![Books Reinforcement Learning]({{site.baseurl}}/images/books_reinforcement_learning_an_introduction_pattern_recognition.png){:class="img-responsive"}

The reference for this post is chapter 8 for the [Sutton and Barto's book]((https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)) called "Generalization and Function Approximation". Moreover a good resource is the [video-lesson 6 of David Silver's course](https://www.youtube.com/watch?v=UoPei5o4fps&t=5217s). A wider introduction to function approximation is given by any good machine learning textbook, I suggest [Pattern Recognition and Machine Learning](https://books.google.co.uk/books/about/Pattern_Recognition_and_Machine_Learning.html?id=kTNoQgAACAAJ&redir_esc=y) by Christopher Bishop.
I want to start this post with a brief excursion in the neuroscience world. Let's see how a function approximator relates to biological brains.


Approximators (and grandmothers)
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

Here I will use again the **robot cleaning example** described in previous posts. The robot moves in a two-dimensional world we called gridworld. It has only 4 possible actions available (forward, backward, left, right) and its goal is to reach a charger (green cell) and avoid to fall on stairs (red cell). I define with $$U(s)$$ our usual utility function, and with $$Q(s,a)$$ the state-action function.
The grid-world is a discrete rectangular state space, having $$c$$ columns and $$r$$ rows. Using a tabular approach we can represent $$U(s)$$  using a table containing $$r \times c = N$$ elements, where $$N$$ represent the total number of states. To represent $$Q(s,a)$$ we need a table of size $$N \times M$$, where $$M$$ is the total number of actions. In the previous posts I always represented the **lookup tables** using **matrices**. As utility function I used a matrix having the same size of the world, whereas for the state-action function I used a matrix having $$N$$ columns (states) and $$M$$ rows (actions). In the first case, to get the utility we have to access the location of the matrix corresponding to the particular state where we are. In the second case, we use the state as index to access the column in the state-action matrix and from that column we return the utilities of all the available actions.

![Function Approximation Lookup Tables]({{site.baseurl}}/images/reinforcement_learning_function_approximation_lookup_tables.png){:class="img-responsive"}

How can we fit the function approximation mechanism inside this scheme?
Let's start with some definitions. Defining as $$S= \{ s_{1}, s_{2}, ... , s_{N} \} $$ the set of possible states, and as $$A= \{ a_{1}, a_{2}, ... , s_{M} \} $$ the set of possible actions, we define a utility function approximator $$\hat{U}(S,\boldsymbol{w})$$ having parameters stored in a vector $$\boldsymbol{w}$$. Here I use the hat on top of $$\hat{U}$$ to differentiate this function from the tabular version $$U$$.

Before explaining how to create a function approximator it is helpful to visualise it as a **black box**. The method described below can be used on different approximators and for this reason we can easily apply it to the box content.
The black box takes as input the current state and returns the utility of the state or the state-action utilities. That's it. The main advantage is that we can approximate (with an arbitrary small error) the utilities using less parameters respect to the tabular approach. We can say that the number of elements stored in the vector $$\boldsymbol{w}$$ is smaller than $$N$$ the number of values in the tabular counterpart.

![Function Approximation Black Boxes]({{site.baseurl}}/images/reinforcement_learning_function_approximation_black_boxes.png){:class="img-responsive"}


I guess there is a question that came to your head: **what there is inside the black box?** This is a legitimate question and now I will try to give you the intuition. In the case of a blackbox which is approximating an utility function, the content of the box is $$\hat{U}(s, \boldsymbol{w})$$. You can imagine the **utility function** as a **music mixer** and the **vector of weights** $$\boldsymbol{w}$$ as the mixer **sliders**. We want to adjust the sliders in order to obtain a sound which is similar to a predefined tone. How to do it? Well we can move one of the slider and compare the output with the reference tone. If the output is more similar to the reference we know that we moved the right slider. Continuing this process many times we eventually obtain a tone which is very similar to the reference.

![Function Approximation Weights Update Intuition]({{site.baseurl}}/images/reinforcement_learning_function_approximation_weights_update_intuition.png){:class="img-responsive"}

Using a more formal view we can say that the vector $$\boldsymbol{w}$$ is adjusted at every iteration, moving the values of a quantity $$\Delta$$, in order to reach an objective which is minimising a function of cost. The cost is given by an **error measure** that we can obtain comparing the output of the function with a target. For instance, we know from previous posts that the actual utility value of state (4,1) in our gridworld is 0.388. Let's say that at time $$t$$ the output of the box is 0.352. After the **update step** the output will be 0.371, we moved closer to the target value.

![Function Approximation Methods]({{site.baseurl}}/images/reinforcement_learning_function_approximation_methods_overview.png){:class="img-responsive"}

Function approximation is an instance of [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning). In principle all the supervised learning techniques could be used in function approximation. The vector $$\boldsymbol{w}$$ may be the set of connection weights of a neural network or the split points and leaf values of a decision tree. However here I will consider only **differentiable** function approximators such as linear combination of features and neural networks, which represents the most promising techniques nowadays. In this post I will focus on linear combination of features. Before describing the simplest case **the linear approximator**, I would like to introduce the general methodology used to adjust the vector of weights. The goal in function approximation is to move as close as possible to the real utility function adjusting the internal parameters stored in $$\boldsymbol{w}$$. To achieve this goal we need two things, first an **error measure** which can give us a feedback on how close we are to the target, second an **update rule** for adjusting the weights. In the next section I will describe these two components.

Method
--------

To improve the performance of our function approximator we need an error measure and an update rule. These two parts work tightly in the learning cycle of every supervised learning technique. Their use in reinforcement learning is not much different from how they are used in a classification task. In order to understand this section you need to refresh some concepts of [multivariable calculus](https://en.wikipedia.org/wiki/Multivariable_calculus) such as the [partial derivative](https://en.wikipedia.org/wiki/Partial_derivative) and [gradient](https://en.wikipedia.org/wiki/Gradient).


![Function Approximation Training]({{site.baseurl}}/images/reinforcement_learning_function_approximation_training_cycle.png){:class="img-responsive"}

**Error Measure**: a common error measure is given by the [Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) between two quantities. For instance, if we have the optimal utility function $$U^{*}(S)$$ and an approximator function $$\hat{U}(s, \boldsymbol{w})$$, then the MSE is defined as follows:

$$ \text{MSE}( \boldsymbol{w} ) = \frac{1}{N} \sum_{s \in S} \big[ U^{*}(s) - \hat{U}(s, \boldsymbol{w}) \big]^{2}  $$

[comment]: <> (where $$P(s)$$ is a distribution weighting the errors of different states, such that $$\sum_{s} P(s) = 1$$. The number of parameters $$\boldsymbol{w}$$ is lower that the number of total states $$N$$. The function $$P(s)$$ allows gaining accuracy on some states instead of others.)

that's it, the MSE is given by the expectation $$\mathop{\mathbb{E}}[ (U^{*}(s) - \hat{U}(s, \boldsymbol{w}) \big)^{2} ]$$ which quantifies the difference between the target and the approximator output. When the training is working correctly the MSE will decrease meaning that we are getting closer to the optimal utility function. The MSE is a common loss function used in supervised learning. However, in reinforcement learning it is often used a reinterpretation of the MSE called **Mean Squared Value Error (MSVE)**. The MSVE introduce a distribution $$\mu(s) \geq 0$$ which specifies how much we care about each state $$s$$. As I told you the function approximator is based on a set of weights $$\boldsymbol{w}$$ which contains less elements that the total number of states. For this reason adjusting a subset of the weights means improving the utility prediction of some states but loosing precision in others. We have limited resources and we have to manage them carefully. The function $$\mu(s)$$ gives us an explicit solution and using it we can rewrite the previous equation as follows:

$$ \text{MSVE}( \boldsymbol{w} ) = \frac{1}{N} \sum_{s \in S}  \mu(s) \big[ U^{*}(s) - \hat{U}(s, \boldsymbol{w}) \big]^{2}  $$


**Update rule**: the update rule for differentiable approximator is [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent). The [gradient](https://en.wikipedia.org/wiki/Gradient) is a generalisation of the concept of derivative which applies to functions of several variables. You can imagine the gradient as the vector that points in the direction of the greatest rate of increase. Intuitively, if you want to reach the top of a mountain the gradient is an arrow that in each moment show you in which direction you should walk. The gradient is generally represented with the operator $$\nabla$$ and is also called **nabla**. 
The goal in gradient descent is to minimise the error measure. We can achieve this goal moving in the direction of the negative gradient vector, meaning that we are not moving anymore to the top of the mountain but downslope. At each step we adjust the parameter vector $$\boldsymbol{w}$$ moving a step closer to the valley. First of all, we have to estimate the gradient vector for $$ \text{MSE}( \boldsymbol{w} )$$ or $$ \text{MSVE}( \boldsymbol{w} )$$. Those error functions are based on $$\boldsymbol{w}$$. In order to get the gradient vector we have to calculate the partial derivative of each weight with respect to all the other weights. Secondly, once we have the gradient vector we have to adjust the value of all the weights in accordance with the negative direction of the gradient.
In mathematical terms, we can update the vector $$\boldsymbol{w}$$ at $$t+1$$ as follows:

$$\begin{eqnarray} 
\boldsymbol{w}_{t+1} &=&  \boldsymbol{w}_{t}  - \frac{1}{2} \alpha \nabla_{\boldsymbol{w}} \text{MSE}(\boldsymbol{w}_{t}) \\
&=& \boldsymbol{w}_{t}  - \frac{1}{2} \alpha \nabla_{\boldsymbol{w}} \mathop{\mathbb{E}} \big[ (U^{*}(s) - \hat{U}(s, \boldsymbol{w}_{t}) \big)^{2} \big]\\
&=& \boldsymbol{w}_{t}  + \alpha \mathop{\mathbb{E}} \big[ (U^{*}(s) - \hat{U}(s, \boldsymbol{w}_{t})) \nabla_{\boldsymbol{w}} \hat{U}(s, \boldsymbol{w}_{t}) \big] \\
&=&  \boldsymbol{w}_{t} + \alpha \big[ U^{*}(s) - \hat{U}(s, \boldsymbol{w}_{t}) \big] \nabla_{\boldsymbol{w}} \hat{U}(s, \boldsymbol{w}_{t})
\end{eqnarray}$$

At this point you might think we have all we need to start the learning procedure, however there is an important part missing. We supposed it was possible to use the optimal utility function $$U^{*}$$ as target in the error estimation step. **We do not have the optimal utility function**. Think about that, having this function would mean we do not need an approximator at all. Moving in our gridworld we could simply call $$U^{*}(s_{t})$$ at each time step $$t$$ and get the actual utility value of that state. What we can do to overcome this problem is to build a target function $$U^{\sim}$$ which represent an **approximated target** and plug it in our formula:

$$ \boldsymbol{w}_{t+1} =  \boldsymbol{w}_{t} + \alpha \big[ U^{\sim}(s) - \hat{U}(s, \boldsymbol{w}) \big] \nabla_{\boldsymbol{w}} \hat{U}(s, \boldsymbol{w}) $$

How can we estimate the approximated target? We can follow different approaches, for instance using Monte Carlo or TD learning. In the next section I will introduce these methods.


Target estimation
------------------

In the previous section we came to the conclusion that we need approximated target functions $$U^{\sim}(s)$$ and $$Q^{\sim}(s,a)$$ to use in the error evaluation and update rule. The type of target used is at the heart of function approximation in reinforcement learning. There are two main approaches:

**Monte Carlo target**: an approximated value for the target can be obtained through a direct interaction with the environment. Using a Monte Carlo approach (see the [second post](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html)) we can generate an episode and update the function $$U^{\sim}(s)$$ based on the states encountered along the way. The estimation of the optimal function $$U^{*}(s)$$ is unbiased because $$\mathop{\mathbb{E}}[U^{\sim}(s)] = U^{*}(s)$$, meaning that the *prediction is guaranteed to converge*.

**Bootstrapping target**: the other approach used to build the target is called bootstrapping and I introduced it in the [third post](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html). In bootstrapping methods we do not have to complete an episode for getting an estimation of the target, we can directly update the approximator parameters after each visit. The simplest form of bootstrapping target is the one based on TD(0) which is defined as follows:

$$ U^{\sim}(s_{t}) = \hat{U}(s_{t+1}, \boldsymbol{w}) \qquad Q^{\sim}(s_{t}, a) = \hat{Q}(s_{t+1}, a, \boldsymbol{w})$$

That's it, the target is obtained through the approximation given by the estimator itself at $$s_{t+1}$$.



I already wrote about the differences between the two approaches, however here I would like to discuss it again in the new context of function approximation.
In both cases the functions $$U^{\sim}(s)$$ and $$Q^{\sim}(s,a)$$ are based on the vector of weights $$\boldsymbol{w}$$. For this reason the correct notation we are going to use from now on is $$U^{\sim}(s,\boldsymbol{w})$$ and $$Q^{\sim}(s,a,\boldsymbol{w})$$.
We have to be particularly careful when using the bootstrapping methods in gradient-based approximators. Bootstrapping methods are not true instances of gradient descent because they only care about the parameters in $$\hat{U}(s, \boldsymbol{w})$$. At training time we adjust $$\boldsymbol{w}$$ in the estimator $$\hat{U}(s,\boldsymbol{w})$$ based on a measure of error but we are not changing the parameters in the target function $$U^{\sim}(s,\boldsymbol{w})$$ based on an error measure. Bootstrapping ignores the effect on the target, taking into account only the gradient of the estimation. For this reason bootstrapping techniques are called **semi-gradient methods**. Due to this issue semi-gradient methods **does not guarantee the convergence**. At this point you may think that it is better to use Monte Carlo methods because at least they are guaranteed to converge. Bootstrapping gives two main advantages. First of all they learn online and it is not required to complete the episode in order to update the weights. Secondly they are faster to learn and computationally friendly. 

The **Generalised Policy Iteration (GPI)** (see [second post](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html)) applies here as well. Let's suppose we start with a random set of weights. At the very first step the agent follows an epsilon-greedy strategy moving in the state with the highest utility. After the first step it is possible to update the weights using gradient descent. What's the effect of this adjustment? The effect is to slightly improve the utility function. At the next step the agent follows again a greedy strategy, then the weights are updated through gradient descent, and so on and so forth. As you can see we are applying the GPI scheme again.



Linear approximator
--------------------

Now it's time to put everything together. We have built a method based on an error measure and an update rule and we know how to estimate a target. Now I will show you how to build an approximator, the content of the black box, represented by the function $$\hat{U}(s, \boldsymbol{w})$$.
Here I will describe a **linear approximator** which is the simplest case of linear combinations, whereas in the next section I will describe some high order approximators. Before describing a liner approximator I want to clarify a crucial point in order to avoid a **common missunderstanding**. The linear approximator is a particular case of the broader class of linear combination of features.
A liner combination is based on a **polynomial** which can be or not a line. Using only a line to discriminate between states can be very limited. **Linear combination** means that the **parameters are linearly combined**. We are not saying anything about the input features, which in fact may be represented by and high-order polynomial. Hopefully this distinction will be clear at the end of the post.

Here we model the state as a vector called $$\boldsymbol{x}$$. This vector contains the current state values at time $$t$$. We call these values **features**. Features can be the position of a robot, position and speed of an inverted pendulum, configurations of the stones in a Go game, etc. Here I define also $$\boldsymbol{w}$$ as the vector of weights (or parameters) of our linear approximator, having the same number of elements of $$\boldsymbol{x}$$. Great, we have two vectors and we want to use them in a linear function. How to do it? Simple, we have to perform the [dot product](https://en.wikipedia.org/wiki/Dot_product) between $$\boldsymbol{x}$$ and $$\boldsymbol{w}$$ as follows:

$$ \hat{U}(s, \boldsymbol{w}) = \boldsymbol{x}(s)^{T} \boldsymbol{w} $$

If you are not used to linear algebra notation don't get scared. This is equivalent to the following sum:

$$ \hat{U}(s, \boldsymbol{w}) = x_{1} w_{1} + x_{2} w_{2} + ... + x_{N} w_{N} $$

where $$N$$ is the total number of features. Geometrically this solution is represented by a line (in two-dimensional space), a plane (in a three-dimensional space), or an hyper-plane (in hyper-spaces).  Now we know the content of the black box, which is given by the product of the vectors $$\boldsymbol{x}$$ and $$\boldsymbol{w}$$. However in order to apply the method described in the previous section we still need the error measure, the update rule and the target. Using the MSE we can write the **error measure** as follows:

$$ \text{MSE}( \boldsymbol{w} ) = \sum_{s \in S} \big[ U^{\sim}(s, \boldsymbol{w}) - \boldsymbol{x}(s)^{T} \boldsymbol{w} \big]^{2}  $$

Using the TD(0) definition we can define the **target** as follows:

$$ U^{\sim}(s, \boldsymbol{w}) = \boldsymbol{x}(s_{t+1})^{T} \boldsymbol{w}  $$

The **update rule** defined previously can be reused here as well, however we have to introduce the reward $$r_{t+1}$$ and gamma as required by the reinforcement learning update definition:

$$ \boldsymbol{w}_{t+1} =  \boldsymbol{w}_{t} + \alpha \big[ r_{t+1} + \gamma \boldsymbol{x}(s_{t+1})^{T} \boldsymbol{w} - \boldsymbol{x}(s)^{T} \boldsymbol{w} \big] \nabla_{\boldsymbol{w}} \hat{U}(s, \boldsymbol{w}) $$

Great, we have almost all we need. I said almost because a last piece is missing. The update rule requires the gradient $$\nabla_{\boldsymbol{w}} \hat{U}(s, \boldsymbol{w}) $$. How to find it? It turns out that the gradient of the linear approximator simplify to a very nice form. First of all, based on the previous definitions we can rewrite the gradient as follows:

$$\nabla_{\boldsymbol{w}} \hat{U}(s, \boldsymbol{w}) =  \nabla_{\boldsymbol{w}} ( x_{1} w_{1} + x_{2} w_{2} + ... + x_{N} w_{N}) $$

now we have to find the partial derivatives of each weight $$ w_{1}, w_{2}, ..., w_{N} $$. If you do not remember how a partial derivative is calculated I will refresh your memory. For each unknown we have to find the derivative considering the other unknowns as constants. For instance, the partial derivative of the first unknown $$w_{1}$$ is simply $$x_{1}$$ because all the other values are considered constant values and the derivative of a constant is zero by definition:

$$ \frac{\partial \hat{U}}{\partial w_{1}} = x_{1} + 0 + 0 + ... + 0 = x_{1}$$

Applying the same process to all the other weights we end up with the following gradient vector:

$$\nabla_{\boldsymbol{w}} \hat{U}(s, \boldsymbol{w}) =  x_{1} + x_{2} + ... + x_{N} = \boldsymbol{x}(s) $$

Meaning that we can rewrite the update rule as follows:

$$ \boldsymbol{w}_{t+1} =  \boldsymbol{w}_{t} + \alpha \big[ r_{t+1} + \boldsymbol{x}(s_{t+1})^{T} \boldsymbol{w} - \boldsymbol{x}(s)^{T} \boldsymbol{w} \big] \boldsymbol{x}(s) $$

Great, we have all we need now. Let's get the party started! 


Application: gridworld
-----------------------
In this section I will give a geometric interpretation of the linear approximator using again the robot cleaning example. The geometric interpretation will give us a demonstration of what a linear approximator can and cannot do. Moreover it will justify the use of tricks such as the bias unit. Let's suppose we have a square gridworld where charging stations and stairs are disposed in multiple locations. The position of the positive and negative cells can vary giving rise to four worlds which I called: OR-world, AND-world, NAND-world, XOR-world. The rule of the worlds are similar to the one defined in the previous posts. The robot has four action available: forward, backward, left, right.  When an action is performed, with a probability of 0.2 it can lead to a wrong movement. The reward is positive (+1.0) for green cells, and negative (-1.0) for red cells. A negative cost of living of -0.04 is applied to all the other states. In the following illustration I represented the four worlds:

![Function Approximation Linear OR]({{site.baseurl}}/images/reinforcement_learning_function_approximation_linear_function_boolean_worlds.png){:class="img-responsive"}
 
If you are familiar with [Boolean algebra](https://en.wikipedia.org/wiki/Boolean_algebra) you have already noticed that there is a pattern in the worlds which reflects basic Boolean operations. From the **geometrical** point of view, when we apply a linear approximator to the boolean worlds, we are trying to find a **plane in a three-dimensional space** which can discriminate between states with high utility (green cells) and states with low utility (red cells). In the three-dimensional space the **x-axis** is represented by the **columns** of the world, whereas the **y-axis** is represented by the **rows**. The **utility value** is given by the **z-axis**. During the gradient descent we are changing the weights, adjusting the inclination of the plane and the utilities associated to each state.


The current definition of the approximator does not take into account an important factor, the **translation of the plane**. Having only two weights we can rotate the plane on the x and y axes but we cannot translate it. This problem becomes clear if you think about a scenario where the position (0,0) of the gridworld should have an utility different from 0. The input vector will be $$\boldsymbol{x} = \{ 0, 0 \}$$. Given this input, no matter which value we choose for the weights, we are going to end up with an utility of zero. Introducing a translation we can shift the plane and associate with the position (0,0) the utility value which best fit the world.
To introduce the translation we have to introduce the **bias unit**. The bias unit can be represented as an additional input which is always equal to 1. Using the bias unit the input vector becomes $$\boldsymbol{x} = \{ 0, 0, 1 \}$$. The new vector when multiplied with the weights
will return 
The core of the python code is the update rule defined in the previous section, which can be summarised in a single line thanks to Numpy:

```python
def update(w, x, x_t1, reward, alpha, gamma):
  '''Return the updated weight vector w_t1

  @param w the weights before the update
  @param x the feature vector observed at t
  @param x_t1 the feature vector observed at t+1
  @param reward the reward observed after the action
  @param alpha the step size (learning rate)
  @param gamma the discount factor
  @return w_t1 the weights at t+1
  '''
  w_t1 = w + alpha * ((reward + gamma*(np.dot(x_t1.T,w)) - np.dot(x.T,w)) * x)
  return w_t1
```

The function `numpy.dot()` is an implementation of the dot product, whereas the `.T` operator represent the transpose. 


High-order approximators
------------------------

The linear approximator is the simplest form of approximation. The linear case is appealing not only for its simplicity but also because it is guaranteed to converge. However, there is an important limit implicit in the linear model: it cannot represent complex relationships between features. That's it, the linear form does not allow representing the interaction between features. Such a complex interaction naturally arise in physical systems. Some features may be informative only when other features are absent. For example, in the inverted pendulum  position and angular speed are tightly connected.


In general, we also need features for combinations of these natural qualities. This is because the linear form prohibits the representation of interactions between features, such as the presence of feature i being good only in the absence of feature j. For example, in the pole-balancing task (Example 3.4), a high angular velocity may be either good or bad depending on the angular position. If the angle is high, then high angular velocity means an imminent danger of falling—a bad state—whereas if the angle is low, then high angular velocity means the pole is righting itself—a good state. In cases with such interactions one needs to introduce features for combinations of feature values when using linear function approximation methods. In the following subsections we consider a variety of general ways of doing this.

High-order approximators may find useful links between multiple futures. An example of high-order approximator is the **quadratic approximator**. In the quadratic approximator we use a second order polynomial to model the utility function. 

$$ \hat{U}(s, \boldsymbol{w}) = x_{1} w_{1} + x_{2} w_{2} + x_{1}^{2} w_{3} + x_{2}^{2} w_{4} +... + x_{N-1} w_{M-1} + x_{N}^{2} w_{M} $$


Application: inverted pendulum
-------------------------------
Using linear approximators in a discrete state space was easy. However many problems have a continuous state space. In the last post I showed how to create partitions and cast the state space inside specific bins. Here I would like to introduce a method which is based on the distributed representation we were talking about in the introduction. Partitioning the state space in bins of the same dimension 


Conclusions
-----------

In this post I introduced function approximation and we saw how to build a methodology based on an error measure, an update rule, and a target. This methodology is extremely flexible and we are going to use it again in future posts. Moreover I introduced linear methods, which are the simplest approximators. Linear function approximation is limited because it cannot capture important relationships between features. Using an high-order polynomial can often solve the problem but is still a limited approach because modeling the relationship between features remain a manual task. In complex physical systems multiple elements are interacting and it will be difficult to find the right polynomial which may describe those relationships. How to solve this problem? 


Index
------

1. [[First Post]](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) Markov Decision Process, Bellman Equation, Value iteration and Policy Iteration algorithms.
2. [[Second Post]](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) Monte Carlo Intuition, Monte Carlo methods, Prediction and Control, Generalised Policy Iteration, Q-function. 
3. [[Third Post]](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html) Temporal Differencing intuition, Animal Learning, TD(0), TD(λ) and Eligibility Traces, SARSA, Q-learning.
4. [[Fourth Post]](https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html) Neurobiology behind Actor-Critic methods, computational Actor-Critic methods, Actor-only and Critic-only methods.
5. [[Fifth Post]](https://mpatacchiola.github.io/blog/2017/03/14/dissecting-reinforcement-learning-5.html) Evolutionary Algorithms introduction, Genetic Algorithm in Reinforcement Learning, Genetic Algorithms for policy selection.
6. [[Sixt Post]](https://mpatacchiola.github.io/blog/2017/08/14/dissecting-reinforcement-learning-6.html) Reinforcement learning applications, Multi-Armed Bandit, Mountain Car, Inverted Pendulum, Drone landing, Hard problems.
7. **[Seventh Post]** Function approximation, Intuition, Linear approximator, High-order approximators, Applications.

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



