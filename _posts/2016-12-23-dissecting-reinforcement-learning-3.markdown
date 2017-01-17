---
layout: post
title:  "Dissecting Reinforcement Learning-Part.3"
date:   2017-01-16 19:00:00 +0000
description: This blog series explains the basic ideas behind reinforcement learning. In particular Temporal Difference Learning, Sarsa, Q-Learning, On-Policy and Off-Policy. It includes full working code written in Python.
author: Massimiliano Patacchiola
comments: false
published: false
---

Welcome to the third part of the "Disecting Reinforcement Learning" series. In the [first](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) and [second](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) post we dissected dynamic programming and  Monte Carlo (MC) methods. The third (and last) group of techniques in reinforcement learning is generally called **Temporal Differencing (TD)** learning. This term was first used by Sutton back in 1988, who also proved the convergence in the mean for the tabular form of the algorithm. TD learning solves some of the problem arising in MC learning. 

[MC problem and possible ways to solve them with TD][6.2 pag. 138]
[Bootstrapping]
[What TD Learning is]
[TD(0) the simplest form of TD learning][TD(0) was proven to converge by Sutton(1988) pag. 159 6.1-2]
[TD(lamda) Eligibility traces]
[TD generally converge faster the MC on stochastic task][Example 6.2 pag. 139]
[Batch Learning]
[SARSA (on-policy) TD control]
[On-Policy and Off-Policy]
[for "off-policy example" cite Deep Reinforcement Learning for Robotic Manipulation with Asynchronous Off-Policy Updates]
[Q-Learning]
[Conclusion][if Mc and TD both converge, which one is faster, and efficient? "pag. 139"]

[Value Function Approximation]
[Action-Value Function Approximation]
[Linear Approximation]
[Neural Network Approximation]
[Example in python with tensorflow]
[Policy Gradient][Lesson 7 of David Silver's course][See pong from pixels post]
[Policy Optimization without gradient (e.g. genetic algoritm which are actor only model)]

[Deep Reinforcement Learning]
[Playing Atari with Deep Reinforcement Learning - 2013]
[Human-level control through deep reinforcement learning - 2015]
[Replay Memory]
[Improving Replay Memory with Prioritized Sweeping][9.4, pag 238; cit in "Human level control through deep RL" in section "Training algorithm for deep Q-networks"]
[Example in Python, cleaning robot 3D]

[Actor-Critic]
[Mastering the game of Go with deep neural networks and tree search - 2016]


Conclusions
-----------



Resources
----------

- The **complete code** for MC prediction and MC control is available on the [dissecting-reinforcement-learning](https://github.com/mpatacchiola/dissecting-reinforcement-learning) official repository on GitHub.

- Dadid Silver's course (DeepMind) in particular **lesson 4** [[pdf]](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MC-TD.pdf)[[video]](https://www.youtube.com/watch?v=PnHCvfgC_ZA&t=438s) and **lesson 5** [[pdf]](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf)[[video]](https://www.youtube.com/watch?v=0g4j2k_Ggc4&t=2s).

- **Artificial intelligence: a modern approach. (chapters 17 and 21)** Russell, S. J., Norvig, P., Canny, J. F., Malik, J. M., & Edwards, D. D. (2003). Upper Saddle River: Prentice hall. [[web]](http://aima.cs.berkeley.edu/) [[github]](https://github.com/aimacode)

- **Reinforcement learning: An introduction.** Sutton, R. S., & Barto, A. G. (1998). Cambridge: MIT press. [[html]](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)

- **Reinforcement learning: An introduction (second edition).** Sutton, R. S., & Barto, A. G. (in progress). [[pdf]](https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf)

References
------------

Bellman, R. (1957). A Markovian decision process (No. P-1066). RAND CORP SANTA MONICA CA.

Russell, S. J., Norvig, P., Canny, J. F., Malik, J. M., & Edwards, D. D. (2003). Artificial intelligence: a modern approach (Vol. 2). Upper Saddle River: Prentice hall.

Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. Machine learning, 3(1), 9-44.



