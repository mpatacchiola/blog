---
layout: post
title:  "Dissecting Reinforcement Learning-Part.4"
date:   2017-01-29 07:00:00 +0000
description: This blog series explains the main ideas and techniques used in reinforcement learning. In this post Actor-Critic methods, Neurobiology of Actor-Critic, animal learning, Actor only and Critic only methods. It includes complete Python code.
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

Reinforcement learning is deeply connected with **neuroscience**, and often the research in this area pushed the implementation of new algorithms in the computational field. Following this observation I will introduce AC methods with a brief excursion in the neuroscience field. If you have a pure computational background you will learn something new. My objective is to give you a deeper insight into the reinforcement learning (extended) world. 
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
The only way actor and critic can communicate is through the dopamine released from the substantia nigra after the stimulation of the ventral striatum. **Drug abuse** can have an effect on the **dopaminergic system**, altering the communication between actor and critic. 
Experiments of [Takahashi et al. (2007)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2480493/) showed that cocaine sensitisation in rats can have as effect maladaptive decision-making. In particular rather than being influenced by long-term goal the rats are driven by immediate rewards. This issue is present in any standard computational frameworks and is know as the **credit assignment problem**. For example, when playing chess it is not easy to isolate the critical actions that lead to the final victory.

![Reinforcement Learning Actor-Critic Rats cocaine effect]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_cocaine_rats.png){:class="img-responsive"}

To understand how the neuronal actor-critic mechanism was involved in the credit assignment problem, [Takahashi et al. (2008)](http://journal.frontiersin.org/article/10.3389/neuro.01.014.2008/full) observed the performances of rats pre-sensitised with cocaine in a **Go/No-Go task**. The procedure of a Go/No-Go task it is simple. The rat is in a small metallic box and it has to learn to poke a button with the nose when a specific odour (cue) is released. If the rat pokes the button when a positive odour is present it gets a reward (sugar). If the rat pokes the button when a negative odour is present it gets a bitter substance (e.g. quinine). Here positive and negative odours do not mean that they are pleasant or unpleasant, we can consider them as neutral. Learning means to associate a specific odour to a reward and another specific odour to punishment.
Finally, if the rat does not move (No-Go) then neither reward nor punishment are given. In total there are four possible conditions.

![Reinforcement Learning Actor-Critic Go No-Go four condition]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_go_nogo_rats.png){:class="img-responsive"}

Has been observed that rats pre-sensitised with cocaine do not learn this task, probably because cocaine damages the basal ganglia and the signal returned by the critic became awry. To test this hypothesis [Takahashi et al. (2008)](http://journal.frontiersin.org/article/10.3389/neuro.01.014.2008/full) sensitised a group of rats 1-3 months before the experiment and then compared it with a non-sensitised control group.
The results of the experiment showed that the **rat in the control group** could **learn how to obtain the reward** when the positive odour was presented and how to avoid the punishment with a no-go strategy when the negative odour was presented. The observation of the basal ganglia showed that the ventral striatum (critic) developed some cue-selectivity neurons, which fired when the odour appeared. This neurons developed during the training and they precede the accurate responding in the dorsal striatum (actor).

![Reinforcement Learning Actor-Critic Go No-Go normal rats result]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_normal_rats.png){:class="img-responsive"}

On the other hand the **sensitised rat** did not show any kind of cue-selectivity during the training. Moreover post-mortem analysis showed that those rats did not developed cue-selective neurons in the ventral striatum (critic). These results confirm the hypothesis that the critic learns the value of the cue and it trains the actor regarding the action to execute.

![Reinforcement Learning Actor-Critic Go No-Go sensitised rats result]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_sensitised_rats.png){:class="img-responsive"}

In this section I showed you how the AC framework is deeply correlated with the **neurobiology of mammalian brain**. This computational model is elegant and it can explain phenomena like Pavlovian learning and drug addiction.
However the elegance of the model should not prevent us to criticize it. In fact different experiments did not confirm it. For example, some form of stimulus-reward learning can take place in the absence of dopamine. Moreover dopamine cells can fire before the stimulus, meaning that its value cannot be used for the update. For a good review of neuronal AC models and their limits I suggest you to read the article of [Joel et al. (2002)](http://www.sciencedirect.com/science/article/pii/S0893608002000473).

Now it is time to turn our attention to math and code. How can we obtain a generic computational model from the biological one? 

Rewiring Actor-Critic methods
-----------------------------

In the last section I presented a neuronal model of the basal ganglia based on the AC framework. Here I will rewire that model using the reinforcement learning techniques we studied until now. The objective is to obtain a computational version which can be used in generic cases (e.g. the 4x3 grid world). First of all, **how can we represent the critic?**
In the neural version the critic does not have access to the actions. Input to the critic is the information obtained through the cerebral cortex which we can compare to the information obtained by the agent through the sensors (state estimation). Moreover the critic receives as input a reward, which arrives directly from the environment. The critic can be represented by an **utility function**, which is updated based on the reward signal received at each iteration. In model free reinforcement learning we can use the **TD(0) algorithm** to represent the critic. The dopaminergic output from substantia nigra and ventral tegmental area can be represented by the two signals which TD(0) returns, meaning the update value and the error estimation $$ \delta $$. In practice we use the update signal to improve the utility function and the error to update the actor. 
**How can we represent the actor?** In the neural system the actor receives an input from the cerebral cortex, which we can translate in sensor signals (current state). The dorsal striatum projects to the motor areas and executes an action. Similarly we can use a **state-action matrix** containing the possible actions for each state. The action can be selected with an ε-greedy (or softmax) strategy and then updated using the error returned by the critic. As usual a picture is worth a thousand words:

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

Great, we obtained our generic computational model to use in a standard reinforcement learning scenario. Now I would like to close the loop giving an answer to a simple question: **does the computational model explain the neurobiological observation?** Apparently yes. In the previous section we saw how [Takahashi et al. (2008)](http://journal.frontiersin.org/article/10.3389/neuro.01.014.2008/full) observed some anomalies in the interaction between actor and critic in rats sensitised with cocaine. Drug abuse seems to deteriorate the dopaminergic feedback from the critic to the actor. From the computational point of view we can observe a similar result when all the $$ U(s) $$ are the same regardless of the current state. In this case the prediction error $$ \delta $$ generated by the critic (with $$ \gamma=1 $$) reduces to the immediate available reward:

$$ \delta_{t} = r_{t+1} + U(s_{t+1}) - U(s_{t}) = r_{t+1} $$

This result explains why the **credit assignment problem** emerges during the training of cocaine sensitised rats. The rats prefer the immediate reward and do not take into account the long-term drawbacks. A learning signal based only on immediate reward it is not sufficient to learn a complex Go/No-Go tasks. Paradoxically the same result explains why in simple task in which actions are immediately rewarded learning can be faster in cocaine sensitised rats. However for a neuroscientist this explanation could be too tidy. Recent work has highlighted the existence of multiple learning systems operating in parallel in the mammalian brain. Some of these systems (e.g. amygdala and/or nucleus accumbens) can replace a malfunctioning critic and compensate the damage caused by cocaine sensitisation. In conclusion, additional experiments are needed in order to shed light on the neuronal AC architecture.  
Now it is time for coding. In the next section I will show you how to implement an AC algorithm in Python and how to apply it to the cleaning robot example.



Actor-Critic Python implementation
----------------------------------

Using the knowledge acquired in the previous posts we can easily create a Python script to implement an AC agent in the usual 4x3 grid world. As usual I will use the robot cleaning example. To understand this example you have to read the rules of the grid world introduced in the [first post](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html). First of all I will describe the general architecture, then I will describe a step of the algorithm in a single episode. Finally I will implement everything in Python.
In the complete architecture we can represent the **critic** using a utility function (state matrix). The matrix is initialised with zeros and updated at each iteration through TD learning. For example, after the first step the robot moves from (1,1) to (1,2) and obtains a reward of -0.04.
The **actor** is represented by a state-action matrix similar to the one introduced to model the Q-function. Each time a new state is observed an action is returned and the robot moves. The values inside the table can be initialised at zero or with random values.


![Reinforcement Learning Actor-Critic Robot example overview]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_robot_actor_critic.png){:class="img-responsive"}


In the episode considered here the robot starts in the bottom-left corner at state (1,1) and it reaches the charging station (reward=+1.0) after seven steps. 

![Reinforcement Learning Actor-Critic Robot example critic]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_robot_first_episode.png){:class="img-responsive"}

The first thing do is to decide an action. A query to the state-action table (actor) allows returning the action vector for the current state which in our case is `[0.48, 0.08, 0.15, 0.37]`. The action vector is passed to the softmax function which returns a probability distribution `[0.30, 0.20, 0.22, 0.27]`. Sampling from the distribution returns UP. 

![Reinforcement Learning Actor-Critic Robot example actor]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_robot_actor.png){:class="img-responsive"}

The [softmax function](https://en.wikipedia.org/wiki/Softmax_function) takes as input the N-dimensional action vector $$ \boldsymbol{x} $$, which may contain arbitrary real values, and returns an N-dimensional vector of real values in the range [0, 1] that add up to 1. The softmax function can be easily implemented in Python:

```python
def softmax(x):
    '''Compute softmax values of array x.

    @param x the input array
    @return the softmax array
    '''
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
```

After the action a new state is reached and a reward is available (-0.04). It is time for the critic to update the state value and to estimate the error $$ \delta $$. The value used are $$ \gamma=0.9 $$ and $$ \alpha=0.1 $$. Applying the update rule showed in the previous section we obtain the new value for the state (1,1): `0.0 + 0.1[-0.04 + 0.9(0.0) - 0.0] = -0.004`. At the same time it is possible to calculate the error $$ \delta $$ as follow:  `-0.04 + 0.9(0.0) - 0.0 = -0.04`

![Reinforcement Learning Actor-Critic Robot example critic]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_robot_critic.png){:class="img-responsive"}

The robot is in a new state and the critic evaluated the error which now must be used to update the state-action table of the actor. In this step the action UP for state (1,1) is weakened, adding the negative term $$ \delta $$. In case of a positive $$ \delta $$ the action would be strengthened.

![Reinforcement Learning Actor-Critic Robot example actor]({{site.baseurl}}/images/reinforcement_learning_model_free_active_actor_critic_robot_actor_2.png){:class="img-responsive"}

Now we can imagine to repeat the same steps until the end of the episode. All the action will be weakened but the last one, which will be strengthened by a factor of +1.0. Repeating the process for many other episodes leads to an optimal utility matrix and an optimal policy. 

Actor-only Critic-only methods
-------------------------------

In the article ["Reinforcement Learning in a Nutshell"](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.69.9557&rep=rep1&type=pdf) the different techniques I introduced until now are divided in three categories: AC methods, Critic-only, Actor-only. Following a similar approach I will briefly introduce some other techniques which are generally not covered in a reinforcement learning course. 

Conclusions
-----------


Index
------

1. [[First Post]](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) Markov Decision Process, Bellman Equation, Value iteration and Policy Iteration algorithms.
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

