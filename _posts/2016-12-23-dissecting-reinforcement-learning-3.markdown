---
layout: post
title:  "Dissecting Reinforcement Learning-Part.3"
date:   2017-01-16 19:00:00 +0000
description: This blog series explains the basic ideas behind reinforcement learning. In particular Temporal Difference Learning, Sarsa, Q-Learning, On-Policy and Off-Policy. It includes full working code written in Python.
author: Massimiliano Patacchiola
comments: false
published: false
---

Welcome to the third part of the series "Disecting Reinforcement Learning". In the [first](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) and [second](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) post we dissected **dynamic programming** and  **Monte Carlo (MC)** methods. The third group of techniques in reinforcement learning is  called **Temporal Differencing (TD)** methods. TD learning solves some of the problem arising in MC learning. In the conclusions of the [second part](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) I described one of this problem. Using MC methods it is necessary to wait until the end of the episode before updating the utility function. This is a serious problem because some applications can have very long episodes and delaying learning until the end is too slow. We will see how TD methods solve this issue.


![Russel and Norvig and Sutton and Barto and Mitchel]({{site.baseurl}}/images/artificial_intelligence_a_modern_approach_reinforcement_learning_an_introduction_machine_learning.png){:class="img-responsive"}

In this post I will start from a **general introduction** to the TD approach and then pass to the most famous (and used) TD techniques, namely **Sarsa** and **Q-Learning**. TD had a huge impact on reinforcement learning and most of the last publications (included Deep Reinforcement Learning) are based on the TD approach.
If you want to read more about TD learning you can use the Sutton and Barto's book (chapter 6). If you are not still satisfied you can read the article of Sutton entitled ["Learning to predict by the methods of temporal differences"](http://link.springer.com/article/10.1007/BF00115009) which contains most of the concepts reported in the book and in this post. If you want to read more about Sarsa and Q-Learning you can use the book of Russel and Norvig (chapter 21.3.2). A short introduction to reinforcement learning and Q-Learning is also provided by Mitchell in his book *Machine Learning* (1997) (chapter 13). Links to these resources are available in the last section of the post.



Temporal Differencing (and rabbits)
------------------------------

The term **Temporal Differencing** was first used by **Sutton** back in 1988. Sutton has such an interesting background. Libertarian, psychologist and Computer scientist interested in understanding what we mean by intelligence and goal-directed behaviour. Give a look to his [personal page](http://richsutton.com) if you want to know more. The interesting thing about Sutton's research is that he motivated and explained TD from the point of view of **animal learning theory** and showed that the TD model solves many problems with a simple time-derivative approach. Many of you heard about the famous [Pavlov's experiment](https://en.wikipedia.org/wiki/Classical_conditioning) in **classical conditioning**. Presenting to a dog some food will cause salivation in the dog's mouth. This association is called **unconditioned response (UR)** and it is caused by an **unconditioned stimulus (US)**. The UR it is a natural reaction that does not depend on previous experience. In a second phase before presenting the food we ring a bell. After a while the dog will associate the ring to the food salivating in advance. The bell is called **conditioned stimulus (CS)** and the response is the **conditioned response (CR)**. 

![Eyeblinking Conditioning in Rabbits]({{site.baseurl}}/images/reinforcement_learning_eyeblink_conditioning_rabbits.png){:class="img-responsive"}

The same effect is studied with [eyeblink conditioning](https://en.wikipedia.org/wiki/Eyeblink_conditioning) in rabbits. A mild puff of air is directed to the rabbit's eyes. The UR in this case is closing the eyelid, whereas the US is the puff of air. During conditioning the red light (CS) is turned on before the air puff, forming an association between the light and the eye blink. The time between CS onset and US onset is an important variable which is called the **interstimulus interval (ISI)**.

Studying the results on eyblink conditioning, Sutton and Barto (1990) found a correlation with the TD framework. For rabbits **learning** means to accurately predict at each point in time the **imminence-weighted sum of future US intensity levels**. Experimentally has been observed that rabbits learn a weaker prediction for CSs presented far in advance of the US. Reinforcement is weighted according to its imminence, when slightly delayed it carries slightly less weight, when long-delayed it carries very little weight, and so on so forth. If you read the previous posts you should find some similarities whit the concept of **discounted rewards**. That's it, the general rule behind the TD approach applies to rabbits and to artificial agents. This **general rule** can be expressed as follow:

$$ \text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize} \big[ \text{Target} - \text{OldEstimate} \big] $$

The expression $$\big[ \text{Target} - \text{OldEstimate} \big] $$ is the **estimation error** that can be reduced moving of a step toward the real value ($$ \text{Target} $$). The $$ \text{StepSize} $$ is a parameter which changes at each time step. Processing the $$ k $$th reward the parameter is updated as $$ \frac{1}{k} $$. **What is the $$ Target $$ in our case?** From the [second post](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) we know that we can estimate the real utility of a state as the expectation of the returns for that state. The $$ Target $$ is the expected return of the state:

$$ \text{Target} = E_{\pi} \Bigg[  \sum_{k=0}^{\infty} \gamma^{k} r_{t+k+1} \Bigg]   $$

In MC method to estimate the Target we take into account all the states visited until the end of the episode:

$$ \text{Target} = E_{\pi} \Bigg[  r_{t+1} + \gamma r_{t+2} + \gamma^{2} r_{t+3} + ... + \gamma^{k} r_{t+k+1}\Bigg]   $$


In the TD algorithm we want to **update the utility function after each visit**, because of this we do not have all the states and we do not have the values of the rewards. The only information available is $$ r_{t+1} $$ the reward at t+1 and the utilities estimated before. If we find a way to express the target using only those values we are done. To solve the issue we can **bootstrap** meaning that we can use the estimates to build new estimates. This is the most important part, if we group $$ \gamma $$ we obtain exactly the equation of $$ U(s_{t+1}) $$:

$$ \text{Target} = E_{\pi} \Bigg[  r_{t+1} + \gamma \big( r_{t+2} + \gamma r_{t+3} + ... + \gamma^{k-1} r_{t+k+1} \big)  \Bigg] = E_{\pi} \Bigg[  r_{t+1} + \gamma U(s_{t+1})  \Bigg]  $$

We got what we want. The $$ \text{Target} $$ is now expressed by two quantities: $$ r_{t+1} $$ and $$ U(s_{t+1}) $$ and both of them are known.
Taking into account all these considerations we can finally write the **complete update rule**: 


$$ U(s_{t}) \leftarrow U(s_{t}) + \alpha \big[ \text{r}_{t+1} + \gamma U(s_{t+1}) - U(s_{t}) \big] $$


This update rule is fascinating. At the very first iteration we are updating the utility table using trash. We initialised the utility values with random values (or zeros) and what we are doing is taking one of this values at $$ t+1 $$ to update the state at $$ t $$. **How can the algorithm converge to the real values?** The magic happens when **the agent meet a terminal state for the first time**. In this particular case the return obtained by TD and MC coincides. Using again our cleaning robot we can easily see what is the difference between TD and MC learning and what each one does at each step...

TD(0) Python implementation
---------------------------

The update rule found in the previous part is the simplest form of TD learning, the **TD(0)** algorithm. TD(0) allow estimating the utility values following a specific policy. We are in the **passive learning** case for **prediction**, and we are in model-free reinforcement learning, meaning that we do not have the transition model. To estimate the utility function we can only move in the world. Using again the **cleaning robot example** I want to show you what does it mean to apply the TD algorithm to a single episode. I am going to use the episode of the [second post](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) where the robot starts at (1,1) and reaches the terminal state at (4,3) after seven steps.

![Reinforcement Learning TD(0) first episode]({{site.baseurl}}/images/reinforcement_learning_model_free_passive_td_first_episode.png){:class="img-responsive"}

Applying the algorithm to the episode leads to the following changes in the utility matrix:

![Reinforcement Learning TD(0) first episode utilities]({{site.baseurl}}/images/reinforcement_learning_model_free_passive_td_first_episode_utilities.png){:class="img-responsive"}

The red frame highlights the utility value that has been updated at each visit. The matrix is initialised with zeros. At k=0 the state (1,1) is updated since the robot is in the state (1,2) and the first reward (-0.04) is available.

TD($$ \lambda $$) and eligibility traces
----------------------------------------

SARSA: Temporal Differencing control
------------------------------------

Now it is time to extend the TD method to the control case. Here we are in the **active** scenario, we want to **estimate the optimal policy** starting from a random one. We saw in the introduction that the final update rule for the TD(0) case was:

$$ U(s_{t}) \leftarrow U(s_{t}) + \alpha \big[ \text{r}_{t+1} + \gamma U(s_{t+1}) - U(s_{t}) \big] $$

In this update rule we need the utility at t+1. The update rule works having the tuple **State-Action-Reward-State**. Now we are in the control case. Here we follow the Generalised Policy Iteration (GPI) framework (see [second post](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html)) using the Q function to estimate the best policy. The Q function requires as input a state-action pair. The TD algorithm for control is straightforward, giving a look at the update rule will give you immediately the idea of how it works:

$$ Q(s_{t}, a_{t}) \leftarrow Q(s_{t}, a_{t}) + \alpha \big[ \text{r}_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_{t}, a_{t}) \big] $$

That's it, we simply replaced $$ U $$ with $$ Q $$ in our updating rule. We must be careful because there is a difference. Now we need a new value which is the action at t+1. This is not a problem because it is contained in the Q-matrix. In TD control the estimation is based on the tuple **State-Action-Reward-State-Action** and this tuple gives the name to the algorithm: **SARSA**.



Q-Learning: off-policy control
-----------------------------

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

- **Machine Learning** Mitchell T. (1997) [[web]](http://www.cs.cmu.edu/~tom/mlbook.html)

- **Artificial intelligence: a modern approach. (chapters 17 and 21)** Russell, S. J., Norvig, P., Canny, J. F., Malik, J. M., & Edwards, D. D. (2003). Upper Saddle River: Prentice hall. [[web]](http://aima.cs.berkeley.edu/) [[github]](https://github.com/aimacode)

- **Reinforcement learning: An introduction.** Sutton, R. S., & Barto, A. G. (1998). Cambridge: MIT press. [[html]](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)

- **Reinforcement learning: An introduction (second edition).** Sutton, R. S., & Barto, A. G. (in progress). [[pdf]](https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf)



References
------------

Bellman, R. (1957). A Markovian decision process (No. P-1066). RAND CORP SANTA MONICA CA.

Russell, S. J., Norvig, P., Canny, J. F., Malik, J. M., & Edwards, D. D. (2003). Artificial intelligence: a modern approach (Vol. 2). Upper Saddle River: Prentice hall.

Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. Machine learning, 3(1), 9-44.

Sutton, R. S., & Barto, A. G. (1990). Time-derivative models of pavlovian reinforcement.

