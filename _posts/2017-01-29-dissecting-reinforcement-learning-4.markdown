---
layout: post
title:  "Dissecting Reinforcement Learning-Part.4"
date:   2017-01-29 07:00:00 +0000
description: This blog series explains the main ideas and techniques used in reinforcement learning. In this post Actor-Critic methods, Neuroscience and Actor-Critic, animal learning, Actor only and Critic only methods. It includes complete Python code.
author: Massimiliano Patacchiola
comments: false
published: false
---

Here we are, the fourth episode of the "Dissecting Reinforcement Learning" series. In this post I will introduce the last group of techniques used in reinforcement learning: **Actor-Critic (AC) methods**. I often define Actor-Critic as a **meta-technique** which uses the methods introduced in the previous posts in order to learn.
AC based algorithms are among the most popular methods in reinforcement learning. For example, the [Deep Determinist Policy Gradient](https://arxiv.org/abs/1509.02971) algorithm introduced recently by some researchers of Google DeepMind is an actor-critic, model-free algorithm. Moreover the AC framework has many links with neuroscience and animal learning, in particular with models of basal ganglia ([Takahashi et al. 2008](http://journal.frontiersin.org/article/10.3389/neuro.01.014.2008/full)). 

![Books Reinforcement Learning an Introduction]({{site.baseurl}}/images/books_reinforcement_learning_an_introduction.png){:class="img-responsive"}

AC methods are not accurately described in the books I generally provide. For instance in [Russel and Norvig](http://aima.cs.berkeley.edu/) and in the [Mitchell's](http://www.cs.cmu.edu/~tom/mlbook.html) book they are not covered at all. In the classical [Sutton and Barto's book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html) there are only three short paragraphs (2.8, 6.6, 7.7), however in the [second edition](https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf) a wider description of neuronal AC methods has been added in chapter 15 (neuroscience). A meta-classification of reinforcement learning techniques is covered in the article ["Reinforcement Learning in a Nutshell"](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.69.9557&rep=rep1&type=pdf).


Actor-Critic methods (and rats)
-----------------------------------

Reinforcement learning is deeply connected with **neuroscience**, and often the research in this area pushed the implementation of new algorithms in the computational field. Following this observation I will introduce AC methods with a brief excursion in the neuroscience world. 
To understand this introduction you should be familiar with the basic structure of the nervous system. What is a [neuron](https://en.wikipedia.org/wiki/Neuron)? How do neurons communicate using [synapses](https://en.wikipedia.org/wiki/Synapse) and [neurotransmitters](https://en.wikipedia.org/wiki/Neurotransmitter)? What is the [cerebral cortex](https://en.wikipedia.org/wiki/Cerebral_cortex)? You do not need to know the details, here I want you to get the general scheme. 
Let's start from Dopamine. **Dopamine** is a [neuromodulator](https://en.wikipedia.org/wiki/Neuromodulation) which is implied in some of the most important process in human and animal brains. You can see the dopamine as a messenger which allows neurons to comunicate. Dopamine has an important role in different processes in the mammalian brain (e.g. learning, motivation, addiction), and it is produced in two specific areas: **substantia nigra pars compacta** and **ventral tegmental area**. These two areas have direct projection to another area of the brain, the **striatum**. The striatum is divided in two parts: **ventral striatum** and **dorsal striatum**. The output of the striatum is directed to **motor areas** and **prefrontal cortex**, and it is involved in motor control and planning.

![Reinforcement Learning Actor-Critic Neural Implementation]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_neural_implementation_no_group.png){:class="img-responsive"}

Most of the areas cited before are part of the **basal ganglia**.
There are different models that found a connection between basal ganglia and learning. In particular it seems that the phasic activity of the dopaminergic neurons can deliver an error between an old and a new estimate of expected future rewards. This error is very similar to the error in Temporal Differencing (TD) learning which I introduced in the [third post](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html). Before going into details I would like to simplify the basal ganglia mechanism distinguish between two groups:


1. Ventral striatum, substantia nigra, ventral tegmental area
2. Dorsal striatum and motor areas

There are no specific biological names for these groups but I will create two labels for the occasion.
Since the first group can evaluate the saliency of a stimulus and the reward associated to it, I will call it the **critic**. The second group has direct access to actions, because of this I will call it the **actor**. 

![Reinforcement Learning Actor-Critic Neural Implementation]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_neural_implementation.png){:class="img-responsive"}

The interaction between actor and critic has an important role in learning. In particular a consistent research showed that it is involved in Pavlovian learning (see [third post](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html)) and in [procedural (explicit) memory](https://en.wikipedia.org/wiki/Procedural_memory), meaning unconscious memories such as skills and habits. On the other hand the acquisition of [declarative (implicit) memory](https://en.wikipedia.org/wiki/Explicit_memory), which is implied in the recollection of factual information, seems to be connected with another area called hippocampus. 

[Explaining here how the mechanism work, reward, and action selection]

The only way actor and critic can communicate is through the dopamine released from the substantia nigra after the stimulation of the ventral striatum. The idea of [Takahashi et al. (2008)](http://journal.frontiersin.org/article/10.3389/neuro.01.014.2008/full) was to silence the critic breaking the connection between actor and critic. 




Let's try to represent the neuronal AC structure using the reinforcement learning techniques we studied. First of all, **how can we represent the critic?**
The critic does not have access to the actions. Input to the critic is the information obtained through the cerebral cortex which is compare to the information obtained by the agent through the sensors (state estimation). Moreover the critic receives as input a reward, which arrives directly from the environment. The critic can be represented by an utility function, which is updated based on the reward signal received at each iteration. In model free reinforcement learning we can use the **TD(0) algorithm** to represent the **critic**. The dopaminergic output from substantia nigra and ventral tegmental area can be represented by the two signals which TD(0) returns, meaning the update value and the error estimation $$ \delta $$. In practice we use the update signal to improve the utility function and the error to update the actor. 
**How can we represent the actor?** In the neural system the actor receives an input from the cerebral cortex, which we can translate in a signal sent from the sensors (current state). The dorsal striatum projects to the motor areas and executes an action. Similarly we can use a state-action matrix containing the possible actions for each state. The action can be selected with an epsilon-greedy (or softmax) strategy and then updated using the error returned by the critic. As usual a picture is worth a thousand words:

![Reinforcement Learning Actor-Critic Neural Implementation]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_neural_implementation_outcome.png){:class="img-responsive"}

As you can see there is a tight correspondence between the reinforcement learning and the neural version.
However the elegance of the model should not prevent us to criticize it. In fact different experiments did not confirm the model. For example, has been observed that some form of stimulus-reward learning can take place in the absence of dopamine. Moreover dopamine cells can fire before the stimulus, meaning that its value cannot be used for the update. For a good review I suggest you to read the article of [Joel et al. (2002)](http://www.sciencedirect.com/science/article/pii/S0893608002000473).



Actor-Critic Python implementation
----------------------------------

Actor-only Critic-only methods
-------------------------------

Conclusions
-----------


Index
------

1. [[First Post]]((https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html)) Markov Decision Process, Bellman Equation, Value iteration and Policy Iteration algorithms.
2. [[Second Post]](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) Monte Carlo Intuition, Monte Carlo methods, Prediction and Control, Generalised Policy Iteration, Q-function. 
3. [[Third Post]](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html) Temporal Differencing intuition, Animal Learning, TD(0), TD(λ) and Eligibility Traces, SARSA, Q-learning.

Resources
----------

- The **complete code** for the Actor-Critic examples is available on the [dissecting-reinforcement-learning](https://github.com/mpatacchiola/dissecting-reinforcement-learning) official repository on GitHub.

- **Reinforcement learning: An introduction.** Sutton, R. S., & Barto, A. G. (1998). Cambridge: MIT press. [[html]](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)

- **Reinforcement learning: An introduction (second edition).** Sutton, R. S., & Barto, A. G. (in progress). [[pdf]](https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf)



References
------------

Heidrich-Meisner, V., Lauer, M., Igel, C., & Riedmiller, M. A. (2007, April). Reinforcement learning in a nutshell. In ESANN (pp. 277-288).

Joel, D., Niv, Y., & Ruppin, E. (2002). Actor–critic models of the basal ganglia: New anatomical and computational perspectives. Neural networks, 15(4), 535-547.

Takahashi, Y., Schoenbaum, G., & Niv, Y. (2008). Silencing the critics: understanding the effects of cocaine sensitization on dorsolateral and ventral striatum in the context of an actor/critic model. Frontiers in neuroscience, 2, 14.

