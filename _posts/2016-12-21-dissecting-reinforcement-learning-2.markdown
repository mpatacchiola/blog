---
layout: post
title:  "Dissecting Reinforcement Learning-Part.2"
date:   2016-12-21 19:00:00 +0000
description: Explaining the basic ideas behind reinforcement learning. In particular, Markov Decision Process, Bellman equation, Value iteration and Policy Iteration algorithms, policy iteration through linear algebra methods. It includes full working code written in Python.
author: Massimiliano Patacchiola
comments: false
published: true
---

[Introduction here and review of the first part]
[Talk about dynamic programming and their limits]
[difference in the terminology used by Norvig e Sutton]

Welcome to the second part of the series **dissecting reinforcement learning**. If you managed to survive to the [first part](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) then congratulations! You learnt the foundation of reinforcement learning, the **dynamic programming** approach.
As I promised in the second part I will go deep in model-free reinforcement learning for prediction, giving an overview on **Monte Carlo (MC)** and **Temporal Differencing (TD)** methods. This post is (weakly) connected with [part one](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html), and I will use the same terminology, examples and mathematical notation.
In this post I will merge some of the ideas presented by Russel and Norvig in **Artificial Intelligence: A Modern Approach** and the classical **Reinforcement Learning, An Introduction** by **Sutton and Barto**. In particular I will focus on chapter 21 (second edition) of the former and on chapter 5 (first edition) of the latter. Moreover you can follow [lecture 4 of David Silver's course](https://www.youtube.com/watch?v=PnHCvfgC_ZA). For open versions of the books look at the resources section.

![Russel and Norvig and Sutton and Barto]({{site.baseurl}}/images/artificial_intelligence_a_modern_approach_reinforcement_learning_an_introduction.png){:class="img-responsive"}

This post is a gentle introduction to model-free reinforcement learning. With the same spirit of the previous part I am going to dissect all the concepts we will step through.

Beyond dynamic programming
--------------------------
In the first post I showed you the two main algorithms for computing optimal policies namely value iteration and policy iteration. We modelled the environment as a Markov decision process (MDP), and we used a transition model to describe the probability of moving from one state to the other. The transition model was stored in a matrix `T` and used to find the utility function $$ U^{*} $$ and the best policy $$ \pi^{*} $$. Here we must be careful with the mathematical notation. In the book of Sutton and Barto the utility function is called value function and is indicated with the letter $$ V $$. To keep uniformity with the previous post I will use the notation of Russel and Norvig which uses the letter $$ U $$ to identify the utility (value) function. The two notations have the same meaning, but in literature it is often used the $$ V $$ notation. The reader should get used to different notations, it is a good form of mental gymnastics. 

Having said that I would like to give a proper definition of model-free reinforcement learning and in particular of the **passive reinforcement learning** approach we are using in this post. In model-free reinforcement learning the first thing we miss is a **transition model**. In fact the name model-free stands for transition-model-free. The second thing we miss is the **reward function** $$ R(s) $$ which gives to the agent the reward associated to a particular state. What we have is a **policy** $$ \pi $$ which the agent can use to move in the environment. This last statement is part of the passive approach, in state $$ s $$ the agent always produce the action $$ a $$ given by the policy $$ \pi $$. The **goal of the agent** in passive reinforcement learning is to learn the utility function $$ U^{\pi}(s) $$. I will use again the example of the **cleaning robot** from the first post with a different starting setup. 

![Passive Model-Free RL]({{site.baseurl}}/images/reinforcement_learning_model_free_passive_simple_world.png){:class="img-responsive"}

The robot is in a 3x4 world with an unknown transition model. The only information about the environment is the states availability. Since the robot does not have the reward function it does not know which state contains the charging station (+1) and which state contains the stairs (-1). The robot does not even have any clue about the policy, it can be a good policy or a bad one. Finally the transition model, since the robot does not know what it is going to happen after each action it can only give unknown probabilities to each possible outcome. 

Bayesian reinforcement learning
-------------------------------
The first thing the robot can do is to estimate the transition model. Moving in the environment and looking to the reaction to its actions. Once the transition model has been estimated the robot can use either value iteration or policy iteration to get the utility function. Estimating the values of a transition model can be expensive. In our 3x4 world for example it means to estimate the values for a 12x12x4 table.

This approach is inspired by the Bayes rule and is then called by Russel and Norvig **Bayesian reinforcement learning** (chapter 21.2.2).

Monte Carlo methods
--------------------------
[model free RL]
[Why using Montecarlo name for this methods?]
[Bootstrap: MC does not need to bootstrap (to guess the state at t+1)]
[Backup: The MC needs to beackup the full episode, whereas TD only t and t+1]

Active Learning
---------------


The exploration-exploitation dilemma
------------------------------------


On-Policy and Off-Policy
------------------------

Temporal-Difference learning
---------------------------


Conclusions
-----------



Resources
----------

The [dissecting-reinforcement-learning](https://github.com/mpatacchiola/dissecting-reinforcement-learning) official repository.

*"Artificial Intelligence: a Modern Approach"* [[github]](https://github.com/aimacode)

References
------------

Bellman, R. (1957). A Markovian decision process (No. P-1066). RAND CORP SANTA MONICA CA.

Russell, S. J., Norvig, P., Canny, J. F., Malik, J. M., & Edwards, D. D. (2003). Artificial intelligence: a modern approach (Vol. 2). Upper Saddle River: Prentice hall.



