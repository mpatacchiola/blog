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

So far we have represented the utility function by a lookup table (or matrix if you prefer). This approach has a problem. When the underlying Markov decision process is large there are too many states and actions to store in memory. Moreover in this case it is extremely difficult to visit all the possible states, meaning that we cannot estimate the utility values for those states.

![Books Reinforcement Learning]({{site.baseurl}}/images/books_reinforcement_learning_an_introduction.png){:class="img-responsive"}

The reference for this post is chapter 8 for the [Sutton and Barto's book]((https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)) called "Generalization and Function Approximation". Moreover a good resource is the video of [lesson 6 of David Silver's course](https://www.youtube.com/watch?v=UoPei5o4fps&t=5217s).

Defining the function approximator
----------------------------------

I define with $$U(s)$$ our usual utility function, and with $$Q(s,a)$$ the state-action function.
Let's suppose we are in a discrete rectangular state space, having $$c$$ columns and $$r$$ rows. Using a tabular approach we can represent $$U(s)$$  using a matrix containing $$r \times c = N$$, where $$N$$ represent the total number of states. To represent $$Q(s,a)$$ we need a table of size $$N \times M$$, where $$M$$ is the total number of actions. In the cleaning robot example I represented the lookup tables using matrices. As utility function I used a matrix having the same size of the world, whereas for the state-action function I used a matrix having $$N$$ columns (states) and $$M$$ rows (actions). In the first case, to get the utility we have to access the location of the matrix corresponding to the particular state where we are. In the second case, we use the state as index to access the column in the state-action matrix and from that column we return the utilities of all the available actions.

![Function Approximation Lookup Tables]({{site.baseurl}}/images/reinforcement_learning_function_approximation_lookup_tables.png){:class="img-responsive"}

How can we fit the function approximation mechanism inside this scheme?
Let's start with some definitions. Defining as $$S= \{ s_{1}, s_{2}, ... , s_{N} \} $$ the set of possible states, and as $$A= \{ a_{1}, a_{2}, ... , s_{M} \} $$ the set of possible actions, we define a utility function approximator $$\hat{U}(S,\boldsymbol{w})$$ having parameters stored in a vector $$\boldsymbol{w}$$. Here I use the hat on top of $$\hat{U}$$ to differentiate this function from the tabular version $$U$$.

Before going into details on how to create a function approximator it is helpful to visualise it as a **black box**. The black box takes as input the current state and returns the utility of the state or the state-action utilities. That's it. The main advantage is that we can approximate (with an arbitrary small error) the utilities using less parameters respect to the tabular approach. We can say that the number of elements stored in the vector $$\boldsymbol{w}$$ is smaller than $$N$$ the number of values in the tabular counterpart.

![Function Approximation Black Boxes]({{site.baseurl}}/images/reinforcement_learning_function_approximation_black_boxes.png){:class="img-responsive"}

As you can see there is a price to pay, the value returned by the black box are not so precise as the tabular ones. However the price we pay is lower than the reward we get. With an approximator problems that were extremely hard become manageable.
I guess there is a question that came to your head: **what there is inside the black box?** This is a legitimate question and now I will try to give you the intuition.


There are many different function approximators: linear combination of features, neural networks, decision trees, nearest neighbour, etc.
Here We will consider only **differentiable** function approximators such as the linear combination and neural networks.

Linear function approximation
------------------------------


Let's suppose we are in a very large state space, let's say a massive factory, and the cleaning robot has to find the charging stations which are in the corners on the right side of the room. At the centre of the factory there are some machines, we consider them as obstacles. On the left side there is movement of trucks and forklift, better don't go there. In this scenario there is an high redundancy in storing all the possible states. What we need to know is a pattern: the right corners of the room have maximum utility, the left corners of the room have negative utility. 

![Function Approximation Linear OR]({{site.baseurl}}/images/reinforcement_learning_function_approximation_linear_function_or_world.png){:class="img-responsive"}

In Linear function approximation our goal is to represent the utility function or the state-action function through a linear combination of features. I defined $$\boldsymbol{x}$$ as the feature vector, and $$\boldsymbol{w}$$ as a vector of weights (or parameters) having the same dimension of $$\boldsymbol{x}$$. The utility can be estimated through the dot product between $$\boldsymbol{x}$$ and $$\boldsymbol{w}$$, as follows:

$$ \hat{U}(s, \boldsymbol{w}) = \boldsymbol{x}(s)^{T} \boldsymbol{w} $$

If you are not used to linear algebra notation don't get scared. This is equivalent to the following sum:

$$ \hat{U}(s, \boldsymbol{w}) = x_{1} w_{1} + x_{2} w_{2} + ... + x_{N} w_{N} $$

where $$N$$ is the total number of features. 

The **Generalised Policy Iteration (GPI)** (see [second post](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html)) applies here as well. Let's suppose we start with a random set of weights. At the very first step the agent follows an epsilon-greedy strategy. After the first step it is possible to update the weights using gradient descent. What's the effect of this adjustment? The effect is to slightly improve the utility function. At the next step the agent follows again a greedy strategy, then the weights are updated through gradient descent, and so on and so forth. As you can see we are applying the GPI scheme again. 

**Choosing the features** is a crucial point for an effective function approximation. Features can be the position of a robot, position and speed of an inverted pendulum, configurations of the stones in a Go game, etc. To simplify the explanation I will cast the features into a vector $$\boldsymbol{x}$$ containing the value observed at state $$s$$. For instance in the cleaning robot example a possible feature vector could be $$\boldsymbol{x}=(5,4)$$, this vector contains the two values that identify the position of the robot in the environment (column and row).

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
7. **[Seventh Post]** Function approximation.

Resources
----------

- The **complete code** for the Reinforcement Learning applications is available on the [dissecting-reinforcement-learning](https://github.com/mpatacchiola/dissecting-reinforcement-learning) official repository on GitHub.

- **Reinforcement learning: An introduction (Chapter 11 'Case Studies')** Sutton, R. S., & Barto, A. G. (1998). Cambridge: MIT press. [[html]](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)

- **History of Inverted-Pendulum Systems** Lundberg, K. H., & Barton, T. W. (2010). [[pdf]](http://ecee.colorado.edu/~taba7194/CPIFAC2oct11.pdf)

- **Reinforcement Learning on autonomous humanoid robots** Schuitema, E. (2012). [[pdf]](https://repository.tudelft.nl/islandora/object/uuid:986ea1c5-9e30-4aac-ab66-4f3b6b6ca002/datastream/OBJ)

- **Generalization in reinforcement learning: Successful examples using sparse coarse coding** Sutton, R. S. (1996). [[pdf]](http://papers.nips.cc/paper/1109-generalization-in-reinforcement-learning-successful-examples-using-sparse-coarse-coding.pdf)

References
------------

Abbeel, P., Coates, A., Quigley, M., & Ng, A. Y. (2007). An application of reinforcement learning to aerobatic helicopter flight. In Advances in neural information processing systems (pp. 1-8).

Kober, J., Bagnell, J. A., & Peters, J. (2013). Reinforcement learning in robotics: A survey. The International Journal of Robotics Research, 32(11), 1238-1274.

Lundberg, K. H., & Barton, T. W. (2010). History of inverted-pendulum systems. IFAC Proceedings Volumes, 42(24), 131-135.

Sutton, R. S. (1996). Generalization in reinforcement learning: Successful examples using sparse coarse coding. In Advances in neural information processing systems (pp. 1038-1044).

Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25(3/4), 285-294.


