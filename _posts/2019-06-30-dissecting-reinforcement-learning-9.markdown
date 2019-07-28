---
layout: post
title:  "Dissecting Reinforcement Learning-Part.9"
date:   2019-06-30 09:00:00 +0000
description: This blog series explains the main ideas and techniques used in reinforcement learning. In this post Reinforcement Learning through non-linear function approximation (Neural Networks) and policy gradient methods. It includes complete Python code.
author: Massimiliano Patacchiola
type: reinforcement learning
comments: false
published: false
---


In the [last post](https://mpatacchiola.github.io/blog/2018/12/28/dissecting-reinforcement-learning-8.html) I introduced **neural networks** as non-linear approximators for the utility function. In this post I will describe an approach based on neural networks, that can be considered a form of reinforcement learning on steroids: **Deep Reinforcement Learning (DRL)**. Back in 2013, the introduction of DRL represented a quantum leap and it allowed to face a series of complex problems that were inaccessible with standard methods. At the end of the [previous post](https://mpatacchiola.github.io/blog/2018/12/28/dissecting-reinforcement-learning-8.html) I described some of the issues of classic techniques, here I briefly review them:

1. **Size of the model**. Standard neural networks, based on fully-connected layers, have number of parameters that grows rapidly with the number of units. Therefore, dealing with high-dimensional inputs (e.g. images, audio, etc) is not convenient.

2. **Update rule**. Training a neural network with [stochastic gradient descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) requires samples that are independent and identically distributed (i.i.d.). However, the samples gathered in standard approaches are part of a temporal sequence and therefore strongly related.

3. **Training instability**. Training a neural network is unstable in standard reinforcement learning algorithms (see for instance Q-learning and its bootstrap mechanism).

DRL overcome all these issues! First, to reduce the size of the network DRL adopts [Convolutional Neural Networks (CNNs)](https://en.wikipedia.org/wiki/Convolutional_neural_network) as function approximator. Secondly, to break the correlation between samples DRL introduces an external memory, called *Replay Memory*, a bucket from which past experiences are sampled and then used for training. Thirdly, the instability issue is mitigated cloning the network and using the frozen clone as target. In this post I will carefully explain each one of these tricks giving you an idea of why they are necessary.


![Books Reinforcement Learning]({{site.baseurl}}/images/books_reinforcement_learning_an_introduction_deep_learning.png){:class="img-responsive"}

The reference for this post is chapter 8 for the [Sutton and Barto's book]((https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)) called "Generalization and Function Approximation". Recently the second version of the book has been released, you can easily find a pre-print for free on [DuckDuckGo](https://duckduckgo.com/).
A wider introduction to CNNs is given in the book ["Deep Learning"](http://www.deeplearningbook.org/) written by Goodfellow, Bengio, and Courville.
I also strongly suggest to read the original DRL paper [published on Nature](https://www.nature.com/articles/nature14236/), that you can easily find with a [quick search on DuckDuckGo](https://duckduckgo.com/?q=pdf+Human-level+control+through+deep+reinforcement+learning). 


First issue: size of the model
------------------------------

-Why MLP is not good for images

-How a CNN work

-Description of the CNN used in DRL

-why the CNN does not have pooling (because otherwise it will miss fine details)

Second issue: i.i.d. samples
-----------------------------------------

**What is a *Replay Memory*?** As stated in the introduction, one of the issues affecting standard reinforcement learning methods that are based on neural networks is the correlation between samples. In order to work correctly the samples of a mini-batch must be [independent and identically distributed (i.i.d.)](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables). What does it mean? It means that the samples must belong to the same probability distribution, and that they must be mutually independent. In other words, the input to the neural network has to be a mini-batch of samples that is representative of the function we want to approximate. Samples that are very similar (like in a temporal sequence) are a good representation of a function locally, but they fail to represent that function globally. 
In supervised learning i.i.d. samples are obtained shuffling the dataset, such that images belonging to the same class are evenly spread at training time. However, in reinforcement learning we cannot use the same trick because we do not have a proper dataset to shuffle. The solution adopted in DRL is the introduction of a buffer called *Replay Memory*. Differently from a standard dataset the replay memory does not contains simple images but *Experiences* that are gradually accumulated during the interaction with the environment. The replay memory has a fixed size and experiences are added and removed following a [first-in-first-out (FIFO)](https://en.wikipedia.org/wiki/FIFO_(computing_and_electronics)) scheme.
Before going into details I would like to briefly discuss the issue of non i.i.d. samples in the context of temporal sequences.

**Catastrophic forgetting**. Let's suppose we are training a network on an environment that has two incremental levels. This is a typical scenario in many video games, where the difficulty of the task increases sequentially. A typical example is the [Atari game Combat](https://en.wikipedia.org/wiki/Combat_(Atari_2600)). This game has 27 games in one, with three main branches: tank, biplane, and jet. Even though those games have similar control commands, they visually look different and the difficulty level can also vary substantially. Here is a screenshot of the games tank and jet for two levels of difficulty:

![Atari Combat]({{site.baseurl}}/images/reinforcement_learning_atari_tank_jet_levels.png){:class="img-responsive"}

Let's suppose that the agent starts from the level *Thank simple*. The neural network will receive only states belonging to this level and it will perform pretty well after a while. However, when the new level *Thank advanced* is used for training the same agent, a completely new series of states will update the network weights. What happens at this point is a phenomenon called [catastrophic forgetting](https://en.wikipedia.org/wiki/Catastrophic_interference). The network will "forget" the old task and will start to perform better and better on the second one. Such a problem may be even more evident if we want to train the agent on a new environment like *Jet simple* or *Jet advanced*.
The replay memory is useful to contrast this phenomenon, since it accumulates in a reservoir past experiences from previous levels and it randomly feeds the network with these experiences.

**Hippocampus**. The use of a replay memory has been often compared to the hippocampus, a fundamental component of the mammalian brain. The [hippocampus](https://en.wikipedia.org/wiki/Hippocampus) plays a crucial role in moving the information acquired by the short-term memory to the long-term memory. Patients with a damage in the hippocampus cannot acquire new information. A famous case is the one of [Henry Molaison](https://en.wikipedia.org/wiki/Henry_Molaison), an American man who received a temporal lobectomy that caused a severe form of [anterograde amnesia](https://en.wikipedia.org/wiki/Anterograde_amnesia). You may also know the [movie "Memento"](https://en.wikipedia.org/wiki/Memento_(film)) where the protagonist is affected by the same amnesia and his memory reset approximately every five minutes.


**Content of an Experience**. An experience is composed of four components: a stack of images representing the state $$s_t$$, another stack of images representing the state $$s_{t+1}$$, the action $$a$$ performed to pass from $$s_{t}$$ to $$s_{t+1}$$, and $$r$$ the reward obtained. Depending on the environment we may have to store an additional Boolean variable $$d$$ that identifies the *done* condition, meaning a terminal state (goal reached). Formally an experience is represented as a tuple

$$e_t = \big( s_{t}, a, r, s_{t+1} \big), $$

and the replay memory as a collection of tuples

$$D = \{ e_1, e_2, ..., e_N \}. $$

The experience contains two stacks of images. A stack of images is simply a pile of the last frames (generally between 3 and 6) obtained during the interaction with the environment. This is a paradigm shift with respect to previous approaches, where only the last state was considered. Why accumulating more frames? The reason why we want to accumulate more frames is because in many environments looking at one frame is not informative. A clear example is the [Atari game "Pong"](https://en.wikipedia.org/wiki/Pong). The game is simple, you have to move vertically a paddle in order to hit a small ball. You compete against another player who controls another paddle in the other side of the screen. One point is earned if the opponent fails in returning the ball. Looking at the following image you can understand immediately why a single frame is not enough for training a competitive agent.


![Atari pong reinforcement learning]({{site.baseurl}}/images/reinforcement_learning_single_vs_multi_frame.png){:class="img-responsive"}

Let's say that you are the player controlling the left paddle. Given only a **single frame** it is not easy to estimate in which direction the ball is moving. In fact it is not even possible to understand if the player just hit the ball or if the opponent did it. Is the ball moving to the left or to the right? Even if we know that the ball is moving toward us, it is not trivial to estimate the trajectory. It may be that the opponent hit the ball top-down such that it is directed to the bottom-left corner, or it may be the other way around. Let's consider now a stack of the **last three frames**. In this condition it is much easier to understand what is happening. The opponent hit the ball, and then moved bottom-up, meaning that the ball is directed towards the left paddle. This was just a simple case but there are many other scenarios where such a temporal information is necessary. For instance, if you are training an agent to drive a car based on camera frames, it is necessary to know in which direction the other cars are moving in order to avoid collisions. A single frame cannot give you this information whereas a stack can do the job.

**Types of *Replay Memory*.** During the last years there have been different improvements on the type of replay memory used in DRL and here I will describe some of them.

**Tips and tricks.** In my experience with deep Q-learning I find out that a well tuned replay memory is very important. Crucial factors to take into account are the size of the memory (how many experiences it is possible to store), the image stack size (how many frames are stored for each experience), and the buffer type (standard, prioritized, partitioned, etc).



Third issue: training instability
-----------------------------------------

**Q-learning update rule.** To understand this issue I have to refresh your memory about the update rule used in standard Q-learning that I introduced in the [third post](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html) of the series. Deep Q-learning uses this update rule at its core

$$ Q(s_{t}, a_{t}) \leftarrow Q(s_{t}, a_{t}) + \alpha \big[ \text{r}_{t+1} + \gamma \underset{a}{\text{ max }} Q(s_{t+1}, a) - Q(s_{t}, a_{t}) \big], $$

where $$Q(s, a)$$ is our state-action function that is parameterised as a neural network, $$\gamma$$ is the discount factor, $$s$$ is the state, $$a$$ the action, and $$r$$ the reward.
I hope you recall that part of this rule represents the *Target*:

$$ \text{Target} = \text{r}_{t+1} + \gamma \underset{a}{\text{ max }} Q(s_{t+1}, a).$$

While in supervised learning we have a clear target, represented by the label of the image, in reinforcement learning such a label does not exist. What we do instead is to *bootstrap*, meaning that the model itself provides the target. This is done looking one step ahead at $$s_{t+1}$$ and asking *"what the current model would do in the next state?"* The answer to this question is a state-action value. This state-action value is then used to recalibrate the approximation done in the previous state.

The issue with bootstrapping is that when the environment and the neural network are complex, the estimate of $$s_{t+1}$$ provided by the model can be pretty wrong. If you ever played with standard classification models in supervised learning you know that there may be large oscillations in the loss function at training time. These oscillations are caused by the fact that the loss function is defined in a high-dimensional space where local and global minima are hard to find. The same problems remain in reinforcement learning, with the additional difficulty of not having a stable target like in supervised learning. 

**Target-Network.** The solution proposed to solve this issue is to save a snapshot of the neural network every $$k$$ iterations, and use this twin model to estimate the target value. Without too much creativity this network has been called *target network* in the original paper. There are therefore two networks. The former is the actual model used to predict the action and whose weights are updated. We can call it *Net*. The latter is the twin network, the *Target-Net*, a snapshot of *Net* that is not updated during the training. Here I underlined where *Net* and *Target-Net* are used in the update rule:

$$ \underbrace{Q(s_{t}, a_{t})}_{\text{Net}} \leftarrow \underbrace{Q(s_{t}, a_{t})}_{\text{Net}} + \alpha \big[ \text{r}_{t+1} + \gamma \underset{a}{\text{ max }} \underbrace{Q(s_{t+1}, a)}_{\text{Target-Net}} - \underbrace{Q(s_{t}, a_{t})}_{\text{Net}} \big]. $$

As you can see the *Target-Net* is only used to predict the state-action value at $$s_{t+1}$$, providing an estimate that is more stable in time.


Index
------

1. [[First Post]](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) Markov Decision Process, Bellman Equation, Value iteration and Policy Iteration algorithms.
2. [[Second Post]](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) Monte Carlo Intuition, Monte Carlo methods, Prediction and Control, Generalised Policy Iteration, Q-function. 
3. [[Third Post]](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html) Temporal Differencing intuition, Animal Learning, TD(0), TD(λ) and Eligibility Traces, SARSA, Q-learning.
4. [[Fourth Post]](https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html) Neurobiology behind Actor-Critic methods, computational Actor-Critic methods, Actor-only and Critic-only methods.
5. [[Fifth Post]](https://mpatacchiola.github.io/blog/2017/03/14/dissecting-reinforcement-learning-5.html) Evolutionary Algorithms introduction, Genetic Algorithm in Reinforcement Learning, Genetic Algorithms for policy selection.
6. [[Sixt Post]](https://mpatacchiola.github.io/blog/2017/08/14/dissecting-reinforcement-learning-6.html) Reinforcement learning applications, Multi-Armed Bandit, Mountain Car, Inverted Pendulum, Drone landing, Hard problems.
7. [[Seventh Post]](https://mpatacchiola.github.io/blog/2017/12/11/dissecting-reinforcement-learning-7.html) Function approximation, Intuition, Linear approximator, Applications, High-order approximators.
8. [[Eighth Post]](https://mpatacchiola.github.io/blog/2018/12/28/dissecting-reinforcement-learning-8.html) Non-linear function approximation, Perceptron, Multi Layer Perceptron, Applications, Policy Gradient.
9. **[Ninth Post]** Deep Reinforcement Learning, Convolutional Networks, Buffer Replay, Applications

Resources
----------

- The **complete code** for the Deep Reinforcement Learning episode is available on the [dissecting-reinforcement-learning](https://github.com/mpatacchiola/dissecting-reinforcement-learning) official repository on GitHub.

- **Reinforcement learning: An introduction (Chapter 8 'Generalization and Function Approximation')** Sutton, R. S., & Barto, A. G. (1998). Cambridge: MIT press. [[html]](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)


References
------------

Goodfellow, I., Bengio, Y., Courville, A., & Bengio, Y. (2016). Deep learning. Cambridge: MIT press.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Petersen, S. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529.


