---
layout: post
title:  "Dissecting Reinforcement Learning-Part.4"
date:   2017-01-29 07:00:00 +0000
description: This blog series explains the main ideas and techniques used in reinforcement learning. In this post Actor-Criti models. It includes complete Python code.
author: Massimiliano Patacchiola
comments: false
published: false
---

Here we are, the fourth episode of the "Dissecting Reinforcement Learning" series. In this post I will introduce the last group of techniques used in reinforcement learning: **Actor-Critic (AC) methods**. I often define Actor-Critic as a **meta-technique** which uses the methods introduced in the previous posts in order to learn. This post is a summary of what we did and what we will do.
Policy-gradient-based actor-critic algorithms are among the most popular algorithms in the reinforcement learning framework. For example, the [Deep Determinist Policy Gradient](https://arxiv.org/abs/1509.02971) algorithm (DDPG) introduced recently by some researchers of Google DeepMind is an actor-critic, model-free algorithm. The AC approach has many links with neuroscience and animal learning, in particular with models of basal ganglia ([Takahashi et al. 2008](http://journal.frontiersin.org/article/10.3389/neuro.01.014.2008/full)). 

![Books Reinforcement Learning an Introduction]({{site.baseurl}}/images/books_reinforcement_learning_an_introduction.png){:class="img-responsive"}

AC methods are not accurately described in the books I generally provide. For instance in [Russel and Norvig](http://aima.cs.berkeley.edu/) and in the [Mitchell's](http://www.cs.cmu.edu/~tom/mlbook.html) book they are not covered at all. In the classical [Sutton and Barto's book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html) there are only three short paragraphs (2.8, 6.6, 7.7), however in the [second edition](https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf) a wider description of neuronal AC methods has been added in chapter 15 (Neuroscience). A meta-classification of reinforcement learning techniques is covered in the article ["Reinforcement Learning in a Nutshell"](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.69.9557&rep=rep1&type=pdf).



[Numerous models have addressed the basal ganglia's role in learning, based on the uncanny resemblance of dopaminergic cell firing to the requirements of an error signal in temporal difference (TD) learning. Such models map the Actor-Critic implementation of TD learning on to the basal ganglia (Barto, 1995; Houk, Adams & Barto, 1995; Montague, Dayan & Sejnowski, 1996; Schultz, Dayan & Montague, 1997), where, roughly, the Actor is mapped on to the selection function of the basal ganglia, and the Critic is mapped on to the RL circuit in Figure 2. As such the dopamine signal is envisaged as the teaching signal that alters the Actor's responses to maximise future reward.
Joel, Niv and Ruppin (2002) evaluated both anatomical and computational perspectives of Actor-Critic models. They review several models and conclude that they are not compatible with current anatomical data.
Though computationally elegant in approach, critics of these models have variously argued for its biological implausibility, citing:
- dopamine cells increase firing to aversive, neutral, and rewarding stimuli, and thus cannot uniquely signal reward prediction error (Pennartz, 1996, however, see Ungless et al. 2004 for claims that the neurons responding to aversive stimuli are, in fact, not dopaminergic)
- dopamine cells fire before the stimulus can be fully perceived, and thus its value cannot be known at the time (Redgrave, Gurney & Prescott, 1999b)
- stimulus-reward learning can take place in the absence of dopamine, and thus it is not necessary for learning (Berridge, 2007)]


Actor-Critic methods (and rats)
-----------------------------------

Reinforcement learning is deeply connected with **neuroscience**, and often the research in this area pushed the implementation of new algorithms in the computational field. Following this observation I will introduce AC methods with a brief excursion in the neuroscience world, based on the model of [Takahashi](http://journal.frontiersin.org/article/10.3389/neuro.01.014.2008/full). 
To understand this introduction you should be familiar with the basic structure of the nervous system. What is a [neuron](https://en.wikipedia.org/wiki/Neuron)? How do neurons communicate using [synapses](https://en.wikipedia.org/wiki/Synapse) and [neurotransmitters](https://en.wikipedia.org/wiki/Neurotransmitter)? What is the [cerebral cortex](https://en.wikipedia.org/wiki/Cerebral_cortex)? You do not need to know the details, here I want you to get the general scheme. 
Let's start from Dopamine. **Dopamine** is a [neuromodulator](https://en.wikipedia.org/wiki/Neuromodulation) which is implied in some of the most important process in human and animal brains. You can see the dopamine as a messenger which allows neurons to comunicate. Dopamine has an important role in different processes in the mammalian brain (e.g. learning, motivation, addiction), and it is produced in two specific areas: **substantia nigra pars compacta** and **ventral tegmental area**. These two areas have direct projection to another area of the brain, the **striatum**. The striatum is divided in two parts: **ventral striatum** and **dorsal striatum**. The striatum input is mainly from different parts of the **cerebral cortex**, whereas the output is mainly to **motor areas** and **prefrontal cortex**. 

![Reinforcement Learning Actor-Critic Neural Implementation]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_neural_implementation_no_group.png){:class="img-responsive"}

Most of the areas cited before are part of the **basal ganglia**.
There are different models that found a connection between basal ganglia and learning. In particular it seems that the phasic activity of the dopaminergic neurons can deliver an error between an old and a new estimate of expected future rewards. This error is very similar to the error in Temporal Differencing (TD) learning which I introduced in the [third post](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html). Before going into details I would like to simplify the mechanism involved in learning distinguish between two groups:


1. Ventral striatum, substantia nigra, ventral tegmental area
2. Dorsal striatum and motor areas

There are no specific biological names for these groups but I will create two labels for the occasion.
Since the first group can evaluate the saliency of a stimulus and the reward associated to it, I will call it the **critic**. The second group has direct access to actions, because of this I will call it the **actor**. 

![Reinforcement Learning Actor-Critic Neural Implementation]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_neural_implementation.png){:class="img-responsive"}

It is the interaction between actor and critic that has an important role in learning. 

[Explaining here how the mechanism work, reward, and action selection]

The only way actor and critic can communicate is through the dopamine released from the substantia nigra after the stimulation of the ventral striatum. The idea of [Takahashi](http://journal.frontiersin.org/article/10.3389/neuro.01.014.2008/full) was to silence the critic breaking the connection between actor and critic. 

Conclusions
-----------


Index
------

1. [[First Post]]((https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html)) Markov Decision Process, Bellman Equation, Value iteration and Policy Iteration algorithms.
2. [[Second Post]](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) Monte Carlo Intuition, Monte Carlo methods, Prediction and Control, Generalised Policy Iteration, Q-function. 
3. [[Third Post]](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html) Temporal Differencing intuition, Animal Learning, TD(0), TD(λ) and Eligibility Traces, SARSA, Q-learning.

Resources
----------

- The **complete code** for MC prediction and MC control is available on the [dissecting-reinforcement-learning](https://github.com/mpatacchiola/dissecting-reinforcement-learning) official repository on GitHub.

- Dadid Silver's course (DeepMind) in particular **lesson 4** [[pdf]](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MC-TD.pdf)[[video]](https://www.youtube.com/watch?v=PnHCvfgC_ZA&t=438s) and **lesson 5** [[pdf]](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf)[[video]](https://www.youtube.com/watch?v=0g4j2k_Ggc4&t=2s)

- **Christopher Watkins** doctoral dissertation, which introduced the **Q-learning** for the first time [[pdf]](https://www.researchgate.net/profile/Christopher_Watkins2/publication/33784417_Learning_From_Delayed_Rewards/links/53fe12e10cf21edafd142e03/Learning-From-Delayed-Rewards.pdf)

- **Machine Learning** Mitchell T. (1997) [[web]](http://www.cs.cmu.edu/~tom/mlbook.html)

- **Artificial intelligence: a modern approach. (chapters 17 and 21)** Russell, S. J., Norvig, P., Canny, J. F., Malik, J. M., & Edwards, D. D. (2003). Upper Saddle River: Prentice hall. [[web]](http://aima.cs.berkeley.edu/) [[github]](https://github.com/aimacode)

- **Reinforcement learning: An introduction.** Sutton, R. S., & Barto, A. G. (1998). Cambridge: MIT press. [[html]](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)

- **Reinforcement learning: An introduction (second edition).** Sutton, R. S., & Barto, A. G. (in progress). [[pdf]](https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf)



References
------------

Heidrich-Meisner, V., Lauer, M., Igel, C., & Riedmiller, M. A. (2007, April). Reinforcement learning in a nutshell. In ESANN (pp. 277-288).

Joel, D., Niv, Y., & Ruppin, E. (2002). Actor–critic models of the basal ganglia: New anatomical and computational perspectives. Neural networks, 15(4), 535-547.

Takahashi, Y., Schoenbaum, G., & Niv, Y. (2008). Silencing the critics: understanding the effects of cocaine sensitization on dorsolateral and ventral striatum in the context of an actor/critic model. Frontiers in neuroscience, 2, 14.

