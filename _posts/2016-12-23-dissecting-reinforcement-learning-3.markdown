---
layout: post
title:  "Dissecting Reinforcement Learning-Part.2"
date:   2016-12-23 19:00:00 +0000
description: Explaining the basic ideas behind reinforcement learning. In particular, Markov Decision Process, Bellman equation, Value iteration and Policy Iteration algorithms, policy iteration through linear algebra methods. It includes full working code written in Python.
author: Massimiliano Patacchiola
comments: false
published: false
---



This post is particularly relevant because it includes a description of important concepts like the General Policy Iteration, Bootstrap and On-Off-Policy which are hard to find in online resources. With the same spirit of the previous part I am going to dissect all the concepts we will step through.

Generalised Policy Iteration
--------------------------
Here I will focus on the policy iteration because its structure is the core of many reinforcement learning methods. The policy iteration allowed finding the utility values for each state and at the same time the optimal policy $$ \pi^{*} $$. The approach we used in policy iteration included two main stages:

1. Policy evaluation: $$ U \rightarrow U^{\pi} $$
2. Policy improvement: $$ \pi \rightarrow greedy(U) $$

[Here we must be careful with the mathematical notation. In the book of Sutton and Barto the utility function is called value function and is indicated with the letter $$ V $$ while here I used the notation of Russel and Norvig which uses the letter $$ U $$. The two notations have the same meaning, but in literature it is often used the $$ V $$ notation. The reader should get used to different notations, it is a good form of mental gymnastics.]

Policy iteration uses these two steps until the utility and policy functions are optimal. The **first step** makes the utility (or value) function consistent with the current policy (evaluation). The **second step** makes the policy $$ \pi $$ *greedy* with respect to the current utility function (improvement). **What does it means greedy?** I said something about it in the policy iteration section of the first post. We saw that at certain point the algorithm found a suboptimal policy and stick to it for some iterations. In that case the agent was following a greedy behaviour, meaning that it was short sighted preferring a small reward in the short-term instead of a big reward in the long-term.

All reinforcement learning methods can be described in terms of policy iteration. Sutton and Barto used a specific name for this general idea that they called **Generalised Policy Iteration (GPI)**. 

The drawbacks of the dynamic programming approach are clear when we think of MDP with an high number of states. In this scenario the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) can make the use of these techniques of limited applicability. However modern GPUs can be used to handle these limitations, making the dynamic programming approach feasible.

[Policy Improvement Theorem]

Conclusions
-----------



Resources
----------

The [dissecting-reinforcement-learning](https://github.com/mpatacchiola/dissecting-reinforcement-learning) repository.

The [setosa blog](http://setosa.io/) containing a good-looking simulator for Markov chains.

Official [github repository](https://github.com/aimacode) for the book *"Artificial Intelligence: a Modern Approach"*.

References
------------

Bellman, R. (1957). A Markovian decision process (No. P-1066). RAND CORP SANTA MONICA CA.

Russell, S. J., Norvig, P., Canny, J. F., Malik, J. M., & Edwards, D. D. (2003). Artificial intelligence: a modern approach (Vol. 2). Upper Saddle River: Prentice hall.



