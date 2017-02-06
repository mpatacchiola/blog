---
layout: post
title:  "Dissecting Reinforcement Learning-Part.4"
date:   2017-01-29 07:00:00 +0000
description: This blog series explains the main ideas and techniques used in reinforcement learning. In this post Actor-Critic methods, Neuroscience and Actor-Critic, animal learning, Actor only and Critic only methods. It includes complete Python code.
author: Massimiliano Patacchiola
comments: false
published: false
---

Here we are, the fourth episode of the "Dissecting Reinforcement Learning" series. In this post I will introduce the last group of techniques used in reinforcement learning: **Actor-Critic (AC) methods**. I often define AC as a **meta-technique** which uses the methods introduced in the previous posts in order to learn.
AC based algorithms are among the most popular methods in reinforcement learning. For example, the [Deep Determinist Policy Gradient](https://arxiv.org/abs/1509.02971) algorithm introduced recently by some researchers of Google DeepMind is an actor-critic, model-free method. Moreover the AC framework has many links with neuroscience and animal learning, in particular with models of basal ganglia ([Takahashi et al. 2008](http://journal.frontiersin.org/article/10.3389/neuro.01.014.2008/full)). 

![Books Reinforcement Learning an Introduction]({{site.baseurl}}/images/books_reinforcement_learning_an_introduction.png){:class="img-responsive"}

AC methods are not accurately described in the books I generally provide. For instance in [Russel and Norvig](http://aima.cs.berkeley.edu/) and in the [Mitchell's](http://www.cs.cmu.edu/~tom/mlbook.html) book they are not covered at all. In the classical [Sutton and Barto's book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html) there are only three short paragraphs (2.8, 6.6, 7.7), however in the [second edition](https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf) a wider description of neuronal AC methods has been added in chapter 15 (neuroscience). A meta-classification of reinforcement learning techniques is covered in the article ["Reinforcement Learning in a Nutshell"](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.69.9557&rep=rep1&type=pdf). Here I will introduce AC methods starting from neuroscience. You can consider this post as the neuro-physiological counterpart of the [third](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html) one, which introduced Temporal Differencing (TD) methods from a psychological and behavioristic point of view.


Actor-Critic methods (and rats)
-----------------------------------

Reinforcement learning is deeply connected with **neuroscience**, and often the research in this area pushed the implementation of new algorithms in the computational field. Following this observation I will introduce AC methods with a brief excursion in the neuroscience world. 
To understand this introduction you should be familiar with the basic structure of the nervous system. What is a [neuron](https://en.wikipedia.org/wiki/Neuron)? How do neurons communicate using [synapses](https://en.wikipedia.org/wiki/Synapse) and [neurotransmitters](https://en.wikipedia.org/wiki/Neurotransmitter)? What is the [cerebral cortex](https://en.wikipedia.org/wiki/Cerebral_cortex)? You do not need to know the details, here I want you to get the general scheme. 
Let's start from Dopamine. **Dopamine** is a [neuromodulator](https://en.wikipedia.org/wiki/Neuromodulation) which is implied in some of the most important process in human and animal brains. You can see the dopamine as a messenger which allows neurons to comunicate. Dopamine has an important role in different processes in the mammalian brain (e.g. learning, motivation, addiction), and it is produced in two specific areas: **substantia nigra pars compacta** and **ventral tegmental area**. These two areas have direct projection to another area of the brain, the **striatum**. The striatum is divided in two parts: **ventral striatum** and **dorsal striatum**. The output of the striatum is directed to **motor areas** and **prefrontal cortex**, and it is involved in motor control and planning.

![Reinforcement Learning Actor-Critic Neural Implementation]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_neural_implementation_no_group.png){:class="img-responsive"}

Most of the areas cited before are part of the **basal ganglia**.
There are different models that found a connection between basal ganglia and learning. In particular it seems that the phasic activity of the dopaminergic neurons can deliver an error between an old and a new estimate of expected future rewards. This error is very similar to the error in TD learning which I introduced in the [third post](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html). Before going into details I would like to simplify the basal ganglia mechanism distinguish between two groups:


1. Ventral striatum, substantia nigra, ventral tegmental area
2. Dorsal striatum and motor areas

There are no specific biological names for these groups but I will create two labels for the occasion.
Since the first group can evaluate the saliency of a stimulus and the reward associated to it, I will call it the **critic**. The second group has direct access to actions, because of this I will call it the **actor**. 

![Reinforcement Learning Actor-Critic Neural Implementation]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_neural_implementation.png){:class="img-responsive"}

The interaction between actor and critic has an important role in learning. In particular a consistent research showed that it is involved in Pavlovian learning (see [third post](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html)) and in [procedural (explicit) memory](https://en.wikipedia.org/wiki/Procedural_memory), meaning unconscious memories such as skills and habits. On the other hand the acquisition of [declarative (implicit) memory](https://en.wikipedia.org/wiki/Explicit_memory), which is implied in the recollection of factual information, seems to be connected with another area called hippocampus. 

The only way actor and critic can communicate is through the dopamine released from the substantia nigra after the stimulation of the ventral striatum. Drug use can have an effect on the dopaminergic system, altering the communication between actor and critic. Experiments of [Takahashi et al. (2007)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2480493/) showed that cocaine sensitization in rats can have as effect maladaptive decision-making. In particular rather than being influenced by long-term goal the rats are driven by immediate rewards. This issue is present in any standard computational frameworks and is know as the **credit assignment problem**. For example, when playing chess it is not easy to isolate the critical actions that lead to the final victory.
To understand how the neuronal actor-critic mechanism was involved in the credit assignment problem, [Takahashi et al. (2008)](http://journal.frontiersin.org/article/10.3389/neuro.01.014.2008/full) observed the performances of rats pre-sensitised with cocaine in a **Go/No-Go task**. The procedure of a Go/No-Go task it is simple. The rat is in a small metallic box and it has to learn to push a button when a specific odour (cue) is released in order to get a reward (sugar). If the rat presses the button when the negative odour is present, then it is punished with a bitter substance (e.g. quinine). If the rat does not move then neither reward nor punishment is given. The possible conditions are:

- Go + Positive Odour = Reward (sugar)
- Go + Negative Odour = Punishment (quinine)
- No-Go + Positive Odour = Nothing
- No-Go + Negative Odour = Nothing

Has been observed that rats pre-sensitised with cocaine do not learn this task, probably because cocaine damages the basal ganglia. To test this hypotesis [Takahashi et al. (2008)](http://journal.frontiersin.org/article/10.3389/neuro.01.014.2008/full) sensitised a group of rats 1-3 months before the experiment and then compared it with a non-sensitised control group.

![Reinforcement Learning Actor-Critic Neural Implementation]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_go_nogo_rats.png){:class="img-responsive"}

The results of the experiment showed that the **rat in the control group** could **learn how to obtain the reward** when the positive odour was presented and how to avoid the punishment with a no-go strategy when the negative odour was presented. The observation of the basal ganglia showed that the ventral striatum (critic) developed some cue-selectivity neurons, which fired when the odour appeared. This neurons developed during the training and they precede the accurate responding in the dorsal striatum (actor).
On the other hand the **sensitised rat** did not show any kind of cue-selectivity during the training. Moreover post-mortem analysis showed that those rats did not developed cue-selective neurons in the ventral striatum (critic). These results confirm the hypothesis that the critic learns the value of the cue and it trains the actor regarding the action to execute.


Let's try now to represent the neuronal AC structure using the reinforcement learning techniques we studied. First of all, **how can we represent the critic?**
The critic does not have access to the actions. Input to the critic is the information obtained through the cerebral cortex which we can compare to the information obtained by the agent through the sensors (state estimation). Moreover the critic receives as input a reward, which arrives directly from the environment. The critic can be represented by an **utility function**, which is updated based on the reward signal received at each iteration. In model free reinforcement learning we can use the **TD(0) algorithm** to represent the critic. The dopaminergic output from substantia nigra and ventral tegmental area can be represented by the two signals which TD(0) returns, meaning the update value and the error estimation $$ \delta $$. In practice we use the update signal to improve the utility function and the error to update the actor. 
**How can we represent the actor?** In the neural system the actor receives an input from the cerebral cortex, which we can translate in a signal sent from the sensors (current state). The dorsal striatum projects to the motor areas and executes an action. Similarly we can use a **state-action matrix** containing the possible actions for each state. The action can be selected with an ε-greedy (or softmax) strategy and then updated using the error returned by the critic. As usual a picture is worth a thousand words:

![Reinforcement Learning Actor-Critic Neural Implementation]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_neural_implementation_outcome.png){:class="img-responsive"}

We can summarise the steps of the AC algorithm as follow:

1. Produce the action $$ a_{t} $$ for the current state $$ s_{t} $$
2. Observe next state $$ s_{t+1} $$ and the reward $$ r $$
3. Update the utility of state $$ s_{t} $$ (critic)
4. Update the probability of the action using $$ \delta $$ (actor)

In **step 1**, the agent produces an action following the current policy. In **step 2** we observe the new state and the reward. In **step 3** we plug the reward, the utility of $$ s_{t} $$ and $$ s_{t+1} $$ in the standard update formula used in TD(0) (see [third post](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html)):

$$ U(s_{t}) \leftarrow U(s_{t}) + \alpha \big[ \text{r}_{t+1} + \gamma U(s_{t+1}) - U(s_{t}) \big] $$

In **step 4** we use the error estimation $$ \delta $$ to update the policy. In the previous posts I used an ε-greedy strategy to selct the action and to update the policy. Here I will select a certain action with probability $$ p(s_{t}, a_{t}) $$ using a [softmax function](https://en.wikipedia.org/wiki/Softmax_function). In this case step 4 consists in strengthening or weakening the probability of the action using the error $$ \delta $$ and a positive step-size parameter $$ \beta $$:

$$ p(s_{t}, a_{t}) \leftarrow p(s_{t}, a_{t}) + \beta \delta_{t} $$

Like in the TD case, we can integrate the **eligibility traces** mechanism (see [third post](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html)) in AC methods. However in the AC case we need two set of traces, one for the actor and one for the critic. For the **critic** we need to store a trace for each state and update them as follow:

$$e_{t}(s) = \begin{cases} \gamma \lambda e_{t-1}(s) & \text{if}\ s \neq s_{t}; \\ \gamma \lambda e_{t-1}(s)+1 & \text{if}\ s=s_{t}; \end{cases}$$

There is nothing different from the TD(λ) method I introduced in the [third post](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html). Once we estimated the trace we can update the state as follow:

$$ U(s_{t}) \leftarrow U(s_{t}) + \alpha \delta_{t} e_{t}(s) $$

For the **actor** we have to store a trace for each state-action pair, similarly to SARSA and Q-learning. The traces can be updated as follow:

$$e_{t}(s,a) = \begin{cases} \gamma \lambda e_{t-1}(s,a)+1 & \text{if}\ s=s_{t} \text{ and } a=a_{t};  \\ \gamma \lambda e_{t-1}(s,a) & \text{otherwise}; \end{cases}$$ 

Finally the probability of choosing an action is updated as follow:

$$ p(s_{t}, a_{t}) \leftarrow p(s_{t}, a_{t}) + \alpha \delta_{t} e_{t}(s) $$

In this section I showed you how the AC framework is deeply correlated with the neurobiology of mammalian brain. This computational model is elegant and it can explain phenomena like Pavlovian learning and drug addiction.
However the elegance of the model should not prevent us to criticize it. In fact different experiments did not confirm it. For example, some form of stimulus-reward learning can take place in the absence of dopamine. Moreover dopamine cells can fire before the stimulus, meaning that its value cannot be used for the update. For a good review of neuronal AC models and their limits I suggest you to read the article of [Joel et al. (2002)](http://www.sciencedirect.com/science/article/pii/S0893608002000473).
In the next section I will show you how to implement an AC algorithm in Python and how to apply it to the cleaning robot example.



Actor-Critic Python implementation
----------------------------------

Using the knowledge acquired in the previous posts we can easily create a Python script to implement an AC agent in the usual 4x3 grid world. First of all I will focus on the critic, then on the actor, finally I will put it all together.


Actor-only Critic-only methods
-------------------------------

In the article ["Reinforcement Learning in a Nutshell"](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.69.9557&rep=rep1&type=pdf) the different techniques I introduced until now are divided in three categories: AC methods, Critic-only, Actor-only. Following a similar approach I will briefly introduce some other techniques which are generally not covered in a reinforcement learning course. 

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

Takahashi, Y., Roesch, M. R., Stalnaker, T. A., & Schoenbaum, G. (2007). Cocaine exposure shifts the balance of associative encoding from ventral to dorsolateral striatum. Frontiers in integrative neuroscience, 1(11).

Takahashi, Y., Schoenbaum, G., & Niv, Y. (2008). Silencing the critics: understanding the effects of cocaine sensitization on dorsolateral and ventral striatum in the context of an actor/critic model. Frontiers in neuroscience, 2, 14.

