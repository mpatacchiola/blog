---
layout: post
title:  "Dissecting Reinforcement Learning-Part.2"
date:   2016-12-21 19:00:00 +0000
description: Explaining the basic ideas behind reinforcement learning. In particular, Markov Decision Process, Bellman equation, Value iteration and Policy Iteration algorithms, policy iteration through linear algebra methods. It includes full working code written in Python.
author: Massimiliano Patacchiola
comments: false
published: false
---

Welcome to the second part of the series **dissecting reinforcement learning**. If you managed to survive to the [first part](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) then congratulations! You learnt the foundation of reinforcement learning, the **dynamic programming** approach.
As I promised in the second part I will go deep in model-free reinforcement learning (for prediction), giving an overview on **Monte Carlo (MC)** and **Temporal Differencing (TD)** methods. This post is (weakly) connected with [part one](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html), and I will use the same terminology, examples and mathematical notation.
In this post I will merge some of the ideas presented by Russel and Norvig in **Artificial Intelligence: A Modern Approach** and the classical **Reinforcement Learning, An Introduction** by **Sutton and Barto**. In particular I will focus on chapter 21 (second edition) of the former and on chapter 5 (first edition) of the latter. Moreover you can follow [lecture 4 of David Silver's course](https://www.youtube.com/watch?v=PnHCvfgC_ZA). For open versions of the books look at the resources section.

![Russel and Norvig and Sutton and Barto]({{site.baseurl}}/images/artificial_intelligence_a_modern_approach_reinforcement_learning_an_introduction.png){:class="img-responsive"}

This post is a gentle introduction to model-free reinforcement learning for prediction. With the same spirit of the previous part I am going to dissect all the concepts we will step through.

Beyond dynamic programming
--------------------------
In the first post I showed you the two main algorithms for computing optimal policies namely value iteration and policy iteration. We modelled the environment as a Markov decision process (MDP), and we used a transition model to describe the probability of moving from one state to the other. The transition model was stored in a matrix `T` and used to find the utility function $$ U^{*} $$ and the best policy $$ \pi^{*} $$. Here we must be careful with the mathematical notation. In the book of Sutton and Barto the utility function is called value function or state-value function and is indicated with the letter $$ V $$. To keep uniformity with the previous post I will use the notation of Russel and Norvig which uses the letter $$ U $$ to identify the utility function. The two notations have the same meaning and they define the value of a state as the expected cumulative future discounted reward starting from that state. The reader should get used to different notations, it is a good form of mental gymnastics. 

Having said that I would like to give a proper definition of model-free reinforcement learning and in particular of the **passive reinforcement learning** approach we are using in this post. In model-free reinforcement learning the first thing we miss is a **transition model**. In fact the name model-free stands for transition-model-free. The second thing we miss is the **reward function** $$ R(s) $$ which gives to the agent the reward associated to a particular state. What we have is a **policy** $$ \pi $$ which the agent can use to move in the environment. This last statement is part of the passive approach, in state $$ s $$ the agent always produce the action $$ a $$ given by the policy $$ \pi $$. The **goal of the agent** in passive reinforcement learning is to learn the utility function $$ U^{\pi}(s) $$. I will use again the example of the **cleaning robot** from the first post with a different starting setup. 

![Passive Model-Free RL]({{site.baseurl}}/images/reinforcement_learning_model_free_passive_simple_world.png){:class="img-responsive"}

The robot is in a 4x3 world with an unknown transition model. The only information about the environment is the states availability. Since the robot does not have the reward function it does not know which state contains the charging station (+1) and which state contains the stairs (-1). The robot does not even have any clue about the policy, it can be a good policy or a bad one. Finally the transition model, since the robot does not know what it is going to happen after each action it can only give unknown probabilities to each possible outcome. 

Bayesian reinforcement learning
-------------------------------
The first thing the robot can do is to estimate the transition model. Moving in the environment and looking to the reaction to its actions. Once the transition model has been estimated the robot can use either value iteration or policy iteration to get the utility function. Estimating the values of a transition model can be expensive. In our 3x4 world for example it means to estimate the values for a 12x12x4 (states x states x actions) table.

This approach is inspired by the Bayes rule and is then called by Russel and Norvig **Bayesian reinforcement learning** (chapter 21.2.2).

Monte Carlo methods
--------------------------
[model free RL]
[Why using Montecarlo name for this methods?]
[Bootstrap: MC does not need to bootstrap (to guess the state at t+1)]
[Backup: The MC needs to beackup the full episode, whereas TD only t and t+1]

The Monte Carlo (MC) method was used for the first time in 1930 by [Enrico Fermi](https://en.wikipedia.org/wiki/Enrico_Fermi) who was studying neutron diffusion. Fermi did not publish anything on it, the modern version is due to [Stanislaw Ulam](https://en.wikipedia.org/wiki/Stanislaw_Ulam) who invented it during the 1940s at Los Alamos. The idea behind MC is simple: using randomness to solve problems. For example it is possible to use MC to estimate a multidimensional definite integral, a technique which is called [MC integration](https://en.wikipedia.org/wiki/Monte_Carlo_integration). In artificial intelligence we can use MC tree search to find the best move in a game. The [DeepMind AlphaGo](https://deepmind.com/research/alphago/) defeated the Go world champion Lee Seedol using MC tree search combined with convolutional networks and deep reinforcement learning. Later on in this series we will discover how it was possible.

Now let's go back to our cleaning robot and let's see what does it mean to apply the MC method to this scenario.
The robot starts at state (1, 1) and it follows its internal policy. At each step it records the reward obtained and saves an history of all the states visited until reaching a terminal state. We call an **episode** the sequence of states from the starting state to the terminal state. Now let's suppose that our robot recorded the following three episodes:

![Passive Model-Free RL Monte Carlo Three Episodes]({{site.baseurl}}/images/reinforcement_learning_model_free_monte_carlo_three_episodes_fast.gif){:class="img-responsive"}

The robot followed its internal policy but **the unknown transition model perturbed the trajectory** leading to undesired states. In the first and second episode, after some fluctuation the robot eventually reached the terminal state obtaining a positive reward. In the third episode the robot moved along a wrong path reaching the stairs and falling down (reward: -1.0). The following is another representation of the three episodes, useful if you are reading the pdf version of the post.

![Passive Model-Free RL Monte Carlo Three Episodes]({{site.baseurl}}/images/reinforcement_learning_model_free_monte_carlo_three_episodes_linear.png){:class="img-responsive"}


Each occurrence of a state during the episode is called **visit**. The concept of visit is important because it permits defining two different MC approaches:

1. **First-Visit MC**: $$ U^{\pi}(s) $$ is defined as the average of the returns following the *first visit* to $$ s $$ in a set of episodes.

2. **Every-Visit MC**: $$ U^{\pi}(s) $$ is defined as the average of the returns following *all the visit* to $$ s $$ in a set of episodes.

What does **return** means? The return is the sum of discounted reward. I already presented the return in the first post when I introduced the Bellman equation and the utility of a state history.
I will now introduce the equation used in MC methods, which give us the utility of a state following the policy $$ \pi $$:

$$ U^{\pi}(s) = E \Bigg[ \sum_{t=0}^{\infty} \gamma^{t} R(S_{t})  \Bigg]  $$

There is nothing new. We have the discount factor $$ \gamma $$, the reward function $$ R(s) $$ and $$ S_{t} $$ the state reached at time $$ t $$. Here is where the MC terminology steps into. We can define $$ S_{t} $$ to be a [discrete random variable](https://en.wikipedia.org/wiki/Random_variable) which can assume all the available states with a certain probability. Every time our robot steps into a state is like if we are picking a value for the random variable $$ S_{t} $$. For each state of each episode we can calculate the return and store it in a list. Repeating this process for a large number of times is **guaranteed to converge to the true utility**. How is that possible? This is the result of a famous theorem known as the [law of large number](https://en.wikipedia.org/wiki/Law_of_large_numbers). Understanding the law of large number is crucial. Rolling a six-sided dice produces one of the numbers 1, 2, 3, 4, 5, or 6, each with equal probability. The [expectation](https://en.wikipedia.org/wiki/Expected_value) is 3.5 and can be calculated as the arithmetic mean: (1+2+3+4+5+6)/6=3.5. Using a MC approach we can obtain the same value, let's do it in Python:

```python
import numpy as np

#Trowing a dice for N times and evaluating the expectation
dice = np.random.randint(low=1, high=7, size=10)
print("Expectation (rolling 10 times): " + str(np.mean(dice)))
dice = np.random.randint(low=1, high=7, size=100)
print("Expectation (rolling 100 times): " + str(np.mean(dice)))
dice = np.random.randint(low=1, high=7, size=1000)
print("Expectation (rolling 1000 times): " + str(np.mean(dice)))
dice = np.random.randint(low=1, high=7, size=100000)
print("Expectation (rolling 100000 times): " + str(np.mean(dice)))
```

```
Expectation (rolling 10 times): 2.9
Expectation (rolling 100 times): 3.47
Expectation (rolling 1000 times): 3.481
Expectation (rolling 100000 times): 3.49948
```

As you can see the estimation of the expectation converges to the true value of 3.5. What we are doing in MC reinforcement learning is exactly the same but in this case we want to **estimate the utility for each state based on the return of each episode**. As for the dice, more episodes we take into account more accurate our estimation will be.

Following the order on the Sutton and Barto's book **I will focus only on the First-Visit MC method in this post**. As usual we will implement the algorithm in Python. I wrote a class called `GridWorld` which is contained in the module `gridworld.py`. Using this class it is possible to create a grid world of any size and add obstacles and terminal states. The world contains a cleaning robot which will move following a specified policy. Let's see how to create the world of our example:

```python
import numpy as np
from gridworld import GridWorld

#Declare our environmnet variable
#The world has 3 rows and 4 columns
env = GridWorld(3, 4)
#Define the state matrix
#Adding obstacle at position (1,1)
#Adding the two terminal states
state_matrix = np.zeros((3,4))
state_matrix[0, 3] = 1
state_matrix[1, 3] = 1
state_matrix[1, 1] = -1
#Define the reward matrix
#The reward is -0.04 for all states but the terminal
reward_matrix = np.full((3,4), -0.04)
reward_matrix[0, 3] = 1
reward_matrix[1, 3] = -1
#Define the transition matrix
#For each one of the four actions there is a probability
transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                              [0.1, 0.8, 0.1, 0.0],
                              [0.0, 0.1, 0.8, 0.1],
                              [0.1, 0.0, 0.1, 0.8]])
#Define the policy matrix
#0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
policy_matrix = np.array([[1, 1, 1, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 3],
                          [0, 3, 3, 3]])
#Set the matrices 
env.setStateMatrix(state_matrix)
env.setRewardMatrix(reward_matrix)
env.setTransitionMatrix(transition_matrix)
```

In a few lines I defined a grid world with the properties of our example.
Now it is time to reset the environment (move the robot to starting position) and using the `render` method to display the world.

```python
#Reset the environment
observation = env.reset()
#Display the world printing on terminal
env.render()
```

Running the snippet above we get the following print on screen.

```
 -  -  -  * 
 -  #  -  * 
 ○  -  -  -
```

I represented free positions with `-` the two terminal states with `*` obstacles with `#` and the robot with `○`.
Now we can run an episode using a for loop:

```python
for _ in range(1000):
    action = policy_matrix[observation[0], observation[1]]
    observation, reward, done = env.step(action)
    print("")
    print("ACTION: " + str(action))
    print("REWARD: " + str(reward))
    print("DONE: " + str(done))
    env.render()
    if done: break
```

If you are familiar with [OpenAI Gym](https://gym.openai.com/) you will find many similarities with my code. I used the same structure and I implemented the same methods `step` `reset` and `render`. In particular the method `step` moves forward at t+1 and returns the **reward**, an **observation** (the position of the robot), and a variable called `done` which is `True` when the episode is finished (the robot reached a terminal state). Now we have all we need to implement the MC method.


Temporal-Difference learning
---------------------------

The exploration-exploitation dilemma
------------------------------------

Conclusions
-----------



Resources
----------

The [dissecting-reinforcement-learning](https://github.com/mpatacchiola/dissecting-reinforcement-learning) official repository.

*"Artificial Intelligence: a Modern Approach"* [[github]](https://github.com/aimacode)

References
------------

Russell, S. J., Norvig, P., Canny, J. F., Malik, J. M., & Edwards, D. D. (2003). Artificial intelligence: a modern approach (Vol. 2). Upper Saddle River: Prentice hall.

Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: An introduction (Vol. 1, No. 1). Cambridge: MIT press.



