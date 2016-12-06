---
layout: post
title:  "Reinforcement Learning dissected limb by limb"
date:   2016-12-4 19:00:00 +0000
description: What is Reinforcement Learning?
author: Massimiliano Patacchiola
comments: true
published: false
---



This post is an introduction to **Reinforcement Learning** and it is meant to be the starting point for a reader who already have some machine learning background and is confident with a little bit of math and Python programming. When I started to study reinforcement learning I did not find any good resource which explained from the basis what reinforcement learning really is. Most of the material out there focus on the modern approaches (Deep Reinforcement Learning) and introduce the [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation) without a satisfying explanation. Fortunately there are some exceptions, one of those is the book of **Russel and Norvig** called **Artificial Intelligence: A Modern Approach**. This post is based on **chapters 17 and 21** of the second edition. I will use the same mathematical notation of the authors, in this way you can use the book to cover some missing part or vice versa.

![Russel and Norvig]({{site.baseurl}}/images/artificial_intelligence_a_modern_approach.png){:class="img-responsive"}

In the next section I will introduce **Markov chains** and **Markov Decision Processes**, if you already know this concepts you can skip the section...

In the beginning was Andrey Markov
----------------------------------
[Andrey Markov](https://en.wikipedia.org/wiki/Andrey_Markov) was a Russian mathematician who studied stochastic processes. Markov was particularly interested in systems that follow a chain of linked events. In 1906 Markov produced interesting results about discrete processes that he called **chain**. A **Markov Chain** has a set of **states**  $$ S = \{ s_0, s_1, ... , s_r \} $$ and a **process** that can move successively from one state to another. Each move is a single **step** and is based on a **transition model** $$ T $$. You should make some effort in remembering the keywords in bold because we will use them extensively during the rest of the article. To summarise a Markov chain is defined by:

1. Set of possible States: $$ S $$
2. Initial State: $$ s_0 $$
3. Transition Model: $$ T(s, s^{'}) $$

There is something peculiar in a Markov chain that I did not mention. A Markov chain is based on the **Markov Property**. The Markov property states that **given the present, the future is conditionally independent of the past**. That's it, the state in which the process is now it is dependent only from the state it was at $$ t-1 $$. An example can simplify the digestion of Markov chain. Let's suppose we have a chain with only two states $$ s_0 $$ and $$ s_1 $$, where $$ s_0 $$ is the initial state. For 90% of the time the process is in $$ s_0 $$  and in the remaining 10% of the time it can move to $$ s_1 $$. When the process is in state $$ s_1 $$ it will remain there 50% of the time. Given this data we can create a **Transition Matrix** $$ T $$ as follow: 

$$
T = 
\begin{bmatrix}
 0.90 & 0.10 \\
 0.50 & 0.50
\end{bmatrix}
$$

The transition matrix is always a square matrix, and since we are dealing with probability distributions all the entries are included between 0 and 1 and a single row should sum up to 1. **We can graphically represent the Markov chain**. In the following representation each state of the chain is a node and the transition probabilities are edges. Highest probabilities have a thickest edge:

![Simple Markov Chain]({{site.baseurl}}/images/simple_markov_chain.png){:class="img-responsive"}

Until now we did not mentioned **time**, but we have to do it because Markov chains are dynamical processes which evolve in time.
Let's suppose we have to guess were the process will be after 3 steps and after 50 steps. How can we do it? We are interested in chains that have a finite number of states and are  time-homogeneous meaning that the transition matrix does not change over time. Given these assumptions **we can compute the k-step transition probability as the k-th power of the transition matrix**, let's do it in Numpy:

```python
import numpy as np

#Declaring the Transition Matrix T
T = np.array([[0.90, 0.10],
              [0.50, 0.50]])

#Obtaining T after 3 steps
T_3 = np.linalg.matrix_power(T, 3)
#Obtaining T after 50 steps
T_50 = np.linalg.matrix_power(T, 50)
#Obtaining T after 100 steps
T_100 = np.linalg.matrix_power(T, 100)

#Printing the matrices
print("T: " + str(T))
print("T_3: " + str(T_3))
print("T_50: " + str(T_50))
print("T_100: " + str(T_100))
```

```shell
T: [[ 0.9  0.1]
    [ 0.5  0.5]]

T_3: [[ 0.844  0.156]
      [ 0.78   0.22 ]]

T_50: [[ 0.83333333  0.16666667]
       [ 0.83333333  0.16666667]]

T_100: [[ 0.83333333  0.16666667]
        [ 0.83333333  0.16666667]]
```



Now we define a **starting distribution** which represent the state of the system at k=0. Our system is composed of two states and we can model the starting distribution as a vector with two elements, the first element of the vector represents the probability of staying in the state $$ s_0 $$ and the second element the probability of staying in state $$ s_1 $$. Let's suppose that we start from $$ s_0 $$, the vector $$ \mathbf{u} $$ representing the starting distribution will have this form:

$$ \mathbf{u} = (1, 0) $$

We can calculate **what's the probability of being in a specific state after k iterations** multiplying the starting distribution and the transition matrix: $$ \mathbf{u} \cdot T^{k} $$. Let's do it in Numpy:

```python
import numpy as np

#Declaring the starting distribution
u = np.array([[1.0, 0.0]])
#Declaring the Transition Matrix T
T = np.array([[0.90, 0.10],
              [0.50, 0.50]])

#Obtaining T after 3 steps
T_3 = np.linalg.matrix_power(T, 3)
#Obtaining T after 50 steps
T_50 = np.linalg.matrix_power(T, 50)
#Obtaining T after 100 steps
T_100 = np.linalg.matrix_power(T, 100)

#Printing the starting distribution
print("u: " + str(u))
print("u_1: " + str(np.dot(u,T)))
print("u_3: " + str(np.dot(u,T_3)))
print("u_50: " + str(np.dot(u,T_50)))
print("u_100: " + str(np.dot(u,T_100)))
```

```shell
u: [[ 1.  0.]]

u_1: [[ 0.9  0.1]]

u_3: [[ 0.844  0.156]]

u_50: [[ 0.83333333  0.16666667]]

u_100: [[ 0.83333333  0.16666667]]

```
**What's going on?** The process starts in $$ s_0 $$ and after one iteration we can be 90% sure it is still in that state. This is easy to grasp, our transition model says that the process can stay in $$ s_0 $$ with 90% probability, nothing new. Looking to the state distribution at k=3 we noticed that there is something different. We are moving in the future and different branches are possible. If we want to find the probability of being in state $$ s_0 $$ after three iteration we should sum all the possible branches that lead to $$ s_0 $$. A picture is worth a thousand words:

![Markov Chain Tree]({{site.baseurl}}/images/markov_chain_tree.png){:class="img-responsive"}

The possibility to be in $$ s_0 $$ at $$ k=3 $$ is given by (0.729 + 0.045 + 0.045 + 0.025) which is equal to 0.844 we got the same result. Now let's suppose that at the beginning we have some uncertainty about the starting state of our process, let's define another starting vector as follow: 

$$ \mathbf{u} = (0.5, 0.5) $$

That's it, with a probability of 50% we can start from $$ s_0$$. Running again the Python script we print the results after 1, 3, 50 and 100 iterations:

```shell
u: [[ 0.5, 0.5]]

u_1: [[ 0.7  0.3]]

u_3: [[ 0.812  0.188]]

u_50: [[ 0.83333333  0.16666667]]

u_100: [[ 0.83333333  0.16666667]]

```
This time the probability of being in $$ s_0 $$ at k=3 is lower (0.812), but in the long run we have the same outcome (0.8333333). **What is happening in the long run?** The result after 50 and 100 iterations are the same and `u_50` is equal to `u_100` no matter which starting distribution we have. The chain **converged to equilibrium** meaning that as the time progresses it forgets about the starting distribution. But we have to be careful, the convergence is not always guaranteed. The dynamics of a Markov chain can be very complex, in particular it is possible to have **transient and recurrent states**. For our scope what we saw is enough. I suggest you to give a look at the [setosa.io](http://setosa.io/) blog because they have an interactive page for Markov chain visualization.


Markov Devision Process
-----------------------

In reinforcement learning it is often used a concept which is affine to Markov chain, I am talking about **Markov Decision Process (MDP)**. A MDP is a reinterpretation of Markov chains which includes an **agent** and a **decision making** process. A MDP is defined by four components:

1. Set of possible States: $$ S $$
2. Initial State: $$ s_0 $$
3. Transition Model: $$ T(s, a, s^{'}) $$
4. Reward Function: $$ R(s) $$

As you can see we are introducing some new elements respect to Markov chains, in particular the Transition Model depends on the current state, the next state and the action of the agent. The Transition Model returns the probability of reaching the state $$ s^{'} $$ if the action $$ a $$ is done in state $$ s $$. But given $$ s $$ and $$ a $$ the model is conditionally independent of all previous states and actions (Markov Property). Moreover there is the **Reward function** $$ R(s) $$ which return a real value to the agent every times it moves from one state to the other (Attention: defining the Reward function to depend only from $$ s $$ can be confusing, Russel and Norvig used this notation in the book to simplify the description, it does not change the problem in any significant way). Since we have a reward function we can say that **some states are more desirable that others** because when the agent move in those states it receives an higher reward. On the opposite there are **states that are not desirable at all**, because when the agent moves there it receives a negative reward.

- **Problem** the agent has to maximise the reward avoiding states which return negative values and choosing the one which return positive values. 

- **Solution** find a **policy** $$ \pi(s) $$ which returns the action with the highest reward.

The agent can try different policies but only one of those can be considered an **optimal policy**, denoted by $$ \pi^{*} $$, which yields to the highest expected utility. It is time to introduce an example that I am going to use along all the post. This example is inspired by the simple environment presented by Russel and Norving in chapter 17.1 of their book. Let suppose we have a **cleaning robot** which has to reach a charging station. Our simple world is a 3x4 matrix where the starting point $$ s_0 $$ is at (1,1), the charging station at (4,3) and a dangerous trap at (4,2). **The robot has to find the best way to reach the charging station** (Reward +1) and avoid the trap (Reward -1). Every time the robot takes a decision it is possible to have the interference of a stochastic factor (ex. the ground is slippery, an evil cat is stinging the robot), which makes the robot diverge from the original path 20% of the time. If the robot decides to go ahead in 10% of the cases it will finish on the left and in 10% of the cases on the right state. If the robot hits the wall it will bounce back to the previous position. The main characteristics of this world are the following:

- Discrete time and space
- Infinite horizon (time goes on forever)
- The Transition Model is known by the agent


A representation of this world and of the Transition Model is reported below:

![Simple World]({{site.baseurl}}/images/reinforcement_learning_simple_world.png){:class="img-responsive"}

I said that the aim of the robot is to find **the best way to reach the charging station**, but what does it mean **the beast way**? Depending on the type of reward the robot is receiving for each intermediate state we can have different optimal policies $$ \pi^{*} $$. In particular we can have four different cases:

1. $$ R(s) \leq -1.6284 $$ **painful life**
2. $$ -0.4278 \leq R(s) \leq -0.085 $$ **quite unpleasant life**
3. $$ -0.0221 \leq R(s) \leq 0 $$ **slightly dreary life**
4. $$ R(s) > 0 $$ **positive enjoyable life**

Based on these points we can try to guess which policy the agent will choose. In the **painful life** scenario the agent only wants to stop the pain as soon as possible. Life is so painful that finishing in the trap is a good choice.  In the **quite unpleasant life** scenario the agent takes the shortest path to the charging station, it does not care about falling in the trap. In the **slightly dreary life** case the robot does not take risks at all and it avoids the trap at cost of banging against the wall. Finally in the **positive enjoyable life** case the agent avoids both the exits and remain in a steady state receiving a positive reward at each time step. The optimal policies for the four cases are graphically represented above.

![Four Policies Comparison]({{site.baseurl}}/images/reinforcement_learning_four_policies.png){:class="img-responsive"}

Until now we know which kind of policies can emerge in specific environments with defined rewards, but there is still something I did not talk about: how can the agent choose the best policy?

The Bellman equation
---------------------------------------

The previous section terminated with a question: **how can the agent choose the best policy?** To give an answer to this question I will present the Bellman equation. First of all we have to find a way to compare two policies. We can use the reward given at each state to obtain a measure of the **utility of a state sequence**. We define the utility of the states **history** $$ h $$ as: 

$$ U_{h} = ([s_0, s_1,...,s_n]) = R(s_0) + \gamma R(s_1) + \gamma^{2} R(s_2) + ... + \gamma^{n} R(s_n) $$

The previous formula defines the **Discounted Rewards** of a state sequence, where $$ \gamma \in [0,1]  $$ is called the **discount factor**. The discount factor describe the preference of the agent for the current rewards over future rewards. A discount factor of 1.0 leads to a degeneration of the previous formula to **additive rewards**.
The discounted rewards formula is not the only way we can estimate the utility, but it is the one which gives less problems. For example in the case of an infinite sequence of states the discounted reward leads to a **finite utility** (using the sum of infinite series), moreover we can also compare infinite sequences using the average reward obtained per time step. What if we want to compare the **utility of single states**? The utility $$ U(s) $$ can be defined as:

$$ U(s) = E\bigg[ \sum_{t=0}^{\infty} \gamma^t R(s_t) \bigg] $$

Let's remember that the utility is defined with respect of a policy $$ \pi $$ which for brevity I did not mention.
Once we have the utilities how can we choose the best action for the next state? Using the **maximum expected utility** principle which says that a rational agent should choose an action that maximise it's expected utility. We are a step closer the Bellman equation, what we missed is to remember that the utility of a state $$ s $$ is correlated with the utility of its neighbours at $$ s^{'} $$, meaning: 

$$ U(s) = R(s) + \gamma \underset{a}{\text{ max }}  \sum_{s^{'}}^{} T(s,a,s^{'}) U(s^{'}) $$

We just derived the **Bellman equation**! Using the Bellman equation an agent can estimate the best action to take and find the optimal policy. Let's try to dissect this equation. **First**, the term $$ R(s) $$ is something we have to add for sure in the equation. We are in state $$ s $$ and we know the reward given for that state, the utility must take it into account. **Second**, notice that the equation is using the transition model $$ T $$ which is multiplied times the utility of the next state $$ s^{'} $$. If you think about that it makes sense, a state which has a low probability to happen (like the 10% probability of moving on the left and on the right in our simplified world) will have a lowest weight in the summation. 

To empirically test the Bellman equation we are going to use our cleaning robot in the simplified 4x3 environment. In this example the reward for each non-terminal state is $$ R(s) = -0.04 $$. We can imagine to have the utility values for each one of the states, **for the moment you do not need to know how we got these values, imagine they appeared magically**. In the same magical way we obtained the optimal policy for the world (to double-check if what we will obtain from the Bellman equation makes sense).

![Example with R(s)=0.04]({{site.baseurl}}/images/reinforcement_learning_example_r004.png){:class="img-responsive"}

In our example we suppose **the robot starts from the state (1,1)**. First of all the robot has to **find which is the best action** between UP, LEFT, DOWN and RIGHT. The robot does not know the optimal policy, but it knows the transition model and the utility values for each state. You have to remember two main things. First, if the robot bounce on the wall it goes back to the previous state. Second, the selected action is executed only with a probability of 80% in accordance with the transition model. Instead of dealing with crude numbers I want to show you with a graphical illustration which are the possible outcomes of the robot action.

![Example with R(s)=0.04]({{site.baseurl}}/images/reinforcement_learning_simple_world_bellman_example.png){:class="img-responsive"}

For each possible outcome I reported the utility and the probability given by the transition model. The next step in the Bellman equation is to calculate the **product between the utility and the transition probability, then sum up the value for each action**.

![Example with R(s)=0.04]({{site.baseurl}}/images/reinforcement_learning_simple_world_bellman_example_2.png){:class="img-responsive"}

We found out that **the action UP is the one with the highest value**. This is in accordance with the optimal policy we magically got. This part of the Bellman equation returns the action that maximizes the expected utility of the subsequent state, which is what an optimal policy should do:

$$ \pi^{*}(s) = \underset{a}{\text{ argmax }} \sum_{s^{'}}^{} T(s,a,s^{'}) U(s^{'}) $$

We have all the elements, we can add the values we got in the Bellman equation and find the utility of the state (1,1):

$$ U(s_{11}) = -0.04 + 1.0 \times 0.7456 = 0.705$$

The Bellman equation works! What we need is a Python implementation of the equation to use in our simulated world. We are going to use the same terminology of the first sections. Our world has 4x3=12 possible states. The starting vector contains 12 values and the transition matrix is a huge 12x4x12 matrix where most of the values are zeros (we can move only from one state to its neighbours).

```python
import numpy as np

#Starting state vector
#The agent starts from (1, 1)
v = np.array([[0.0, 0.0, 0.0, 0.0, 
               0.0, 0.0, 0.0, 0.0, 
               1.0, 0.0, 0.0, 0.0]])

#Transition matrix loaded from file
#(It is too big to write here)
T = np.load("T.npy")

#Utility matrix
U = np.array([[0.812, 0.868, 0.918,   1.0],
               [0.762,   0.0, 0.660,  -1.0],
               [0.705, 0.655, 0.611, 0.388]])

def return_state_utility(v, T, U, reward, gamma):
    action_array = np.zeros(4)
    for action in range(0, 4):
        action_array[action] = np.sum( np.multiply(U, np.dot(v, T[:,:,action]).reshape(3,4)))
    return reward + gamma * np.max(action_array)
   
```



That's great, but we supposed that the utility values appeared magically. Instead of using a magician we want to **find an algorithm to obtain these values**. There is a problem. For $$ n $$ possible states there are $$ n $$ Bellman equations, and each equation contains $$ n $$ unknown. Using any linear algebra package would be possible to solve these equations, the problem is that they are not linear because of the $$ \text{max} $$ operator. What to do? We can use the value iteration algorithm...

The value iteration algorithm
---------------------------------------

The Bellman equation is the basis of the value iteration algorithm for solving a MDP. **Our objective is to find the utility values for each state**. As we said we cannot use a linear algebra library, we need an iterative approach. We start with arbitrary initial utility values (usually zeros). Then we calculate the utility of a state using the Bellman equation and we assign it to the state. This iteration is called **Bellman update**. Applying the Bellman update infinitely often we are **guaranteed to reach an equilibrium**. Once we reached the equilibrium we have the utility values we were looking for and we can use them to estimate which is the best move for each state.


References
------------

Bellman, R. (1957). A Markovian decision process (No. P-1066). RAND CORP SANTA MONICA CA.

Russell, S. J., Norvig, P., Canny, J. F., Malik, J. M., & Edwards, D. D. (2003). Artificial intelligence: a modern approach (Vol. 2). Upper Saddle River: Prentice hall.



