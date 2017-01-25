---
layout: post
title:  "Dissecting Reinforcement Learning-Part.3"
date:   2017-01-16 19:00:00 +0000
description: This blog series explains the main ideas and techniques behind reinforcement learning. In particular Temporal Difference Learning, Animal Learning, Eligibility Traces, Sarsa, Q-Learning, On-Policy and Off-Policy. It includes complete Python code.
author: Massimiliano Patacchiola
comments: false
published: false
---

Welcome to the third part of the series "Disecting Reinforcement Learning". In the [first](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) and [second](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) post we dissected **dynamic programming** and  **Monte Carlo (MC)** methods. The third group of techniques in reinforcement learning is  called **Temporal Differencing (TD)** methods. TD learning solves some of the problem arising in MC learning. In the conclusions of the [second part](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) I described one of this problem. Using MC methods it is necessary to wait until the end of the episode before updating the utility function. This is a serious problem because some applications can have very long episodes and delaying learning until the end is too slow. We will see how TD methods solve this issue.


![Russel and Norvig and Sutton and Barto and Mitchel]({{site.baseurl}}/images/artificial_intelligence_a_modern_approach_reinforcement_learning_an_introduction_machine_learning.png){:class="img-responsive"}

In this post I will start from a **general introduction** to the TD approach and then pass to the most famous (and used) TD techniques, namely **Sarsa** and **Q-Learning**. TD had a huge impact on reinforcement learning and most of the last publications (included Deep Reinforcement Learning) are based on the TD approach.
We will see how TD is correlated with psychology and neuroscience through **animal learning** experiments. If you want to read more about these topics there is a chapter (14) in the second edition of the Sutton and Barto's book ([pdf](https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf)) and another chapter of the same authors entitled *"Time-derivative models of pavlovian reinforcement"* which you can easily find on Google.
Some parts of this post are based on chapters 6 and 7 of the classical "Reinforcement Learning: An Introduction". If after reading this post you are not satisfied I suggest you to give a look to the article of Sutton entitled ["Learning to predict by the methods of temporal differences"](http://link.springer.com/article/10.1007/BF00115009). If you want to read more about **Sarsa** and **Q-learning** you can use the book of Russel and Norvig (chapter 21.3.2). A short introduction to reinforcement learning and Q-Learning is also provided by Mitchell in his book *Machine Learning* (1997) (chapter 13). Links to these resources are available in the last section of the post.



Temporal Differencing (and rabbits)
------------------------------

The term **Temporal Differencing** was first used by **Sutton** back in 1988. Sutton has such an interesting background. Libertarian, psychologist and Computer scientist interested in understanding what we mean by intelligence and goal-directed behaviour. Give a look to his [personal page](http://richsutton.com) if you want to know more. The interesting thing about Sutton's research is that he motivated and explained TD from the point of view of **animal learning theory** and showed that the TD model solves many problems with a simple time-derivative approach. Many of you heard about the famous [Pavlov's experiment](https://en.wikipedia.org/wiki/Classical_conditioning) in **classical conditioning**. Presenting to a dog some food will cause salivation in the dog's mouth. This association is called **unconditioned response (UR)** and it is caused by an **unconditioned stimulus (US)**. The UR it is a natural reaction that does not depend on previous experience. In a second phase before presenting the food we ring a bell. After a while the dog will associate the ring to the food salivating in advance. The bell is called **conditioned stimulus (CS)** and the response is the **conditioned response (CR)**. 

![Eyeblinking Conditioning in Rabbits]({{site.baseurl}}/images/reinforcement_learning_eyeblink_conditioning_rabbits.png){:class="img-responsive"}

The same effect is studied with [eyeblink conditioning](https://en.wikipedia.org/wiki/Eyeblink_conditioning) in rabbits. A mild puff of air is directed to the rabbit's eyes. The UR in this case is closing the eyelid, whereas the US is the air puff. During conditioning the red light (CS) is turned on before the air puff, forming an association between the light and the eye blinks. There are two types of arrangements of stimuli in classical conditioning experiments. In **delay conditioning**, the CS extends throughout the US without any interval. In **trace conditioning** there is a time interval, called the trace interval, between CS and US. The delay between CS and US is an important variable which is called the **interstimulus interval (ISI)**.

![Delay-Trace Conditioning]({{site.baseurl}}/images/reinforcement_learning_delay_trace_conditioning.png){:class="img-responsive"}

Learning about predictive relationships among stimuli is extremely important for surviving, this is the reason why it is widely present among species ranging from mice to humans. **Learning** means to accurately predict at each point in time the **imminence-weighted sum of future US intensity levels**. In the eyblinking experiement has been observed that rabbits learn a weaker prediction for CSs presented far in advance of the US. Studying the results on eyeblink conditioning, Sutton and Barto (1990) found a correlation with the TD framework. Reinforcement is weighted according to its imminence (length of the ISI), when slightly delayed it carries slightly less weight, when long-delayed it carries very little weight, and so on so forth. This assumption is the core of the **TD model of classical conditioning** and it is an extension of the [Rescorla-Wagner model](https://en.wikipedia.org/wiki/Rescorla%E2%80%93Wagner_model) (1972). If you read the previous posts you should find some similarities whit the concept of **discounted rewards**. The general rule behind TD applies to rabbits and to artificial agents. This **general rule** can be expressed as follow:

$$ \text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize} \big[ \text{Target} - \text{OldEstimate} \big] $$

The expression $$ \big[ \text{Target} - \text{OldEstimate} \big] $$ is the **estimation error** or $$ \delta $$ which can be reduced moving of a step toward the real value ($$ \text{Target} $$). The $$ \text{StepSize} $$ (sometimes called learning rate) is a parameter which determines to what extent the error has to be integrated in the new estimation. If $$ \text{StepSize} = 0 $$ the agent does not learn at all. If $$ \text{StepSize} = 1 $$ the agent considers only the most recent information. In some application the $$ \text{StepSize} $$ changes at each time step. Processing the $$ k $$th reward the parameter is updated as $$ \frac{1}{k} $$. However in practice it is often used a constant value such as 0.1 for all the steps. **What is the $$ Target $$ in our case?** From the [second post](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) we know that we can estimate the real utility of a state as the expectation of the returns for that state. The $$ Target $$ is the expected return of the state:

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

The update rule found in the previous part is the simplest form of TD learning, the **TD(0)** algorithm. TD(0) allows estimating the utility values following a specific policy. We are in the **passive learning** case for **prediction**, and we are in model-free reinforcement learning, meaning that we do not have the transition model. To estimate the utility function we can only move in the world. Using again the **cleaning robot example** I want to show you what does it mean to apply the TD algorithm to a single episode. I am going to use the episode of the [second post](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) where the robot starts at (1,1) and reaches the terminal state at (4,3) after seven steps. 

![Reinforcement Learning TD(0) first episode]({{site.baseurl}}/images/reinforcement_learning_model_free_passive_td_first_episode.png){:class="img-responsive"}

Applying the TD algorithm means to move step by step considering only the state at t and the state at t+1. That's it, after each step we get the utility value and the reward at t+1 and we update the value at t. The **TD(0)** algorithm **ignores the past states** and this is shown by the shadow I added above those states. Applying the algorithm to the episode leads to the following changes in the utility matrix:

![Reinforcement Learning TD(0) first episode utilities]({{site.baseurl}}/images/reinforcement_learning_model_free_passive_td_first_episode_utilities.png){:class="img-responsive"}

The red frame highlights the **utility value** that has been **updated at each visit**. The matrix is initialised with zeros. At k=0 the state (1,1) is updated since the robot is in the state (1,2) and the first reward (-0.04) is available. The calculation for updating the utility at (1,1) is: `0.0 + 0.1 (-0.04 + 0.9 (0.0) - 0.0) = -0.004`. Similarly to (1,1) the algorithm updates the state at (1,2). At k=2 the robot goes back and the calculation take the form: `0.0 + 0.1 (-0.04 + 0.9 (-0.004) - 0.0) = -0.00436`. At k=3 the robot changes again its direction. In this case the algorithm update for the second time the state (1,2) as follow: `-0.004 + 0.1 (-0.04 + 0.9 (-0.00436) + 0.004) = -0.0079924`. The same process is applied until the end of the episode.

In the **Python implementation** we have to create a grid world as we did in the [second post](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html), using the class `GridWorld` contained in the module `gridworld.py`. I will use again the 4x3 world with a charging station at (4,3) and the stairs at (4,2).  The **optimal policy** and the **utility values** of this world are the same we obtained in the previous posts:

```
Optimal policy:			Utility Matrix:

 >   >   >   * 			0.812  0.868  0.918  1.0
 ^   #   ^   * 			0.762  0.0    0.660 -1.0
 ^   <   <   <. 		0.705  0.655  0.611  0.388
```

The **update rule** of TD(0) can be implemented in a few lines:

```python
def update_utility(utility_matrix, observation, new_observation, 
                   reward, alpha, gamma):
    '''Return the updated utility matrix

    @param utility_matrix the matrix before the update
    @param observation the state observed at t
    @param new_observation the state observed at t+1
    @param reward the reward observed after the action
    @param alpha the step size (learning rate)
    @param gamma the discount factor
    @return the updated utility matrix
    '''
    u = utility_matrix[observation[0], observation[1]]
    u_t1 = utility_matrix[new_observation[0], new_observation[1]]
    utility_matrix[observation[0], observation[1]] += \
        alpha * (reward + gamma * u_t1 - u)
    return utility_matrix
```

The **main loop** is much simpler than the one of MC methods. In this case we do not have any first-visit constraint and the only thing to do is to apply the update rule.

```python
for epoch in range(tot_epoch):
    #Reset and return the first observation
    observation = env.reset(exploring_starts=True)
    for step in range(1000):
        #Take the action from the action matrix
        action = policy_matrix[observation[0], observation[1]]
        #Move one step in the environment and get obs and reward
        new_observation, reward, done = env.step(action)
        #Update the utility matrix using the TD(0) rule
        utility_matrix = update_utility(utility_matrix, 
                                        observation, new_observation, 
                                        reward, alpha, gamma)
        observation = new_observation
        if done: break #return
```
The complete code, called `temporal_differencing_prediction.py`, is available in the [GitHub repository](https://github.com/mpatacchiola/dissecting-reinforcement-learning). For the moment it is important to get the general idea behind the algorithm. Running the complete code with `gamma=0.999`, `alpha=0.1` and following the optimal policy for a reward of -0.04 we obtain:

```
Utility matrix after 1 iterations:
[[-0.004  -0.0076  0.1     0.    ]
 [ 0.      0.      0.      0.    ]
 [ 0.      0.      0.      0.    ]]

Utility matrix after 2 iterations:
[[-0.00835924 -0.00085     0.186391    0.        ]
 [-0.0043996   0.          0.          0.        ]
 [-0.004       0.          0.          0.        ]]

Utility matrix after 3 iterations:
[[-0.01520748  0.01385546  0.2677519   0.        ]
 [-0.00879473  0.          0.          0.        ]
 [-0.01163916 -0.0043996  -0.004      -0.004     ]]

...

Utility matrix after 100000 iterations:
[[ 0.83573452  0.93700432  0.94746457  0.        ]
 [ 0.77458346  0.          0.55444341  0.        ]
 [ 0.73526333  0.6791969   0.62499965  0.49556852]]

...

Utility matrix after 300000 iterations:
[[ 0.85999294  0.92663558  0.99565229  0.        ]
 [ 0.79879005  0.          0.69799246  0.        ]
 [ 0.75248148  0.69574141  0.65182993  0.34041743]]

```

We can now compare the utility matrix obtained with TD(0) and the one obtained with Dynamic Programming in the [first post](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html):

![Reinforcement Learning Dynamic Programming VS TD(0)]({{site.baseurl}}/images/reinforcement_learning_utility_estimation_dp_vs_td.png){:class="img-responsive"}

Most of the values are similar. The main difference between the two table is the **estimation for the two terminal states**. The TD(0) does not work for terminal states because we need reward and utility of the next state at t+1. For definition after a terminal states there is not another state. However this is not a big issue. What we want to know is the utility of the states nearby the terminal states. 

Great we saw how TD(0) works, however there is something I did not talk about: **what the (0) contained in the name of the algorithm means?** To understand what that zero does mean I have to introduce the eligibility traces.

TD(λ) and eligibility traces
----------------------------------------

As I told you in the previous section, the TD(0) algorithm does not take into account past states. What matters in TD(0) is the current state and the state at t+1. However would be useful to extend what learned at t+1 also to previous states, so to accelerate learning. To achieve this objective it is necessary to have a short-term memory mechanism to store the states which have been visited in the last steps. For each state $$ s $$ at time $$ t $$ we can define $$ e_{t}(s) $$ as the **eligibility trace**:

$$e_{t}(s) = \begin{cases} \gamma \lambda e_{t-1}(s) & \text{if}\ s \neq s_{t}; \\ \gamma \lambda e_{t-1}(s)+1 & \text{if}\ s=s_{t}; \end{cases}$$

Here $$ \gamma $$ is the discount rate and $$ \lambda \in [0,1] $$ is a decay parameter called **trace-decay** or **accumulating trace** which defines the update weight for each state visited. When $$ 0 <\lambda <\ 1 $$ the traces decrease in time. This allow giving a small weight to infrequent states.
For $$ \lambda = 0 $$ we have the TD(0) case, and only the immediately preceding prediction is updated. For $$ \lambda = 1$$ we have TD(1) where all the preceding predictions are equally updated. TD(1) can be considered an **extension of MC methods using a TD framework**. In MC methods we need to wait the end of the episode in order to update the states. In TD(1) we can update all the previous states online, we do not need the end of the episode. Let's see now what happens to a specific state trace during an episode. I will take into account an episode with seven visits where five states are visited. The state $$ s_1 $$ is visited twice during the episode. Let's see what happens to its trace.

![Reinforcement Learning TD(lamda) first episode Trace Decay]({{site.baseurl}}/images/reinforcement_learning_model_free_passive_td_lambda_first_episode_trace_decay.png){:class="img-responsive"}

At the beginning the trace is equal to zero. After the first visit to $$ s_1 $$ (second step) the trace goes up to 1 and then it starts decaying. After the second visit (fourth step) +1 is added to the current value (0.25) obtaining a final trace of 1.25. After that point the state $$ s_1 $$ is no more visited and the trace slowly goes to zero. **How does TD(λ) update the utility function?** 
In TD(0) we saw that a uniform shadow was added in the graphical illustration to represent the inaccessibility of previous states. In TD(λ) the **previous states** are accessible but they are **updated based on the eligibility trace value**. States with a small eligibility trace will be updated of a small amount whereas states with high eligibility traces will be substantially updated. 

![Reinforcement Learning TD(lamda) first episode Eligibility Traces]({{site.baseurl}}/images/reinforcement_learning_model_free_passive_td_first_episode_eligibility_traces.png){:class="img-responsive"}

Graphically we can represent T(λ) with a non-uniform shadow which partially hides the old states and shows the recent ones. Now it's time to define the **update rule for TD(λ)**. Remembering that the estimation error $$ \delta $$ was defined in the previous section as:

$$ \delta_{t} = r_{t+1} + \gamma U(s_{t+1}) - U(s_{t})  $$

we can update the utility function as follow:

$$ U(s_{t}) = U(s_{t}) + \alpha \delta_{t} e_{t}(s) \qquad  \text{for all } s \in S $$

The **Python implementation** of TD(λ) is straightforward. We only need to add an eligibility matrix and a new update rule for the utility matrix.

```python
def update_utility(utility_matrix, trace_matrix, alpha, delta):
    '''Return the updated utility matrix

    @param utility_matrix the matrix before the update
    @param alpha the step size (learning rate)
    @param delta the error (Taget-OldEstimte) 
    @return the updated utility matrix
    '''
    utility_matrix += alpha * delta * trace_matrix
    return utility_matrix

def update_eligibility(trace_matrix, gamma, lambda_):
    '''Return the updated trace_matrix

    @param trace_matrix the eligibility traces matrix
    @param gamma discount factor
    @param lambda_ the decaying value
    @return the updated trace_matrix
    '''
    trace_matrix = trace_matrix * gamma * lambda_
    return trace_matrix
```

The main loop introduces some new components compared with the previous TD(0) case. We have the estimation of `delta` in a separate line and the management of the `trace_matrix` in two lines. First of all the states are increased (+1) and then they are decayed.

```python
for epoch in range(tot_epoch):
  #Reset and return the first observation
  observation = env.reset(exploring_starts=True)
  for step in range(1000):
    #Take the action from the action matrix
    action = policy_matrix[observation[0], observation[1]]
    #Move one step in the environment and get obs and reward
    new_observation, reward, done = env.step(action)
    #Estimate the error delta (Target - OldEstimate)
    delta = reward + gamma * \
        utility_matrix[new_observation[0], new_observation[1]] - \
        utility_matrix[observation[0], observation[1]]
    #Adding +1 in the trace matrix (only the state visited)
    trace_matrix[observation[0], observation[1]] += 1
    #Update the utility matrix (all the states)
    utility_matrix = update_utility(utility_matrix, trace_matrix, alpha, delta)
    #Update the trace matrix (decaying) (all the states)
    trace_matrix = update_eligibility(trace_matrix, gamma, lambda_)
    observation = new_observation
    if done: break #return
```

The complete code is available on the [GitHub repository](https://github.com/mpatacchiola/dissecting-reinforcement-learning) and it is called `temporal_differencing_prediction_trace.py`. Running the script we obtain the following utility matrices:

```
Utility matrix after 1 iterations:
[[ 0.       0.04595  0.1      0.     ]
 [ 0.       0.       0.       0.     ]
 [ 0.       0.       0.       0.     ]]

...

Utility matrix after 101 iterations:
[[ 0.90680695  0.98373981  1.05569002  0.        ]
 [ 0.8483302   0.          0.6750451   0.        ]
 [ 0.77096419  0.66967837  0.50653039  0.22760573]]

...

Utility matrix after 100001 iterations:
[[ 0.86030512  0.91323552  0.96350672  0.        ]
 [ 0.80914277  0.          0.82155788  0.        ]
 [ 0.76195244  0.71064599  0.68342933  0.48991829]]

...

Utility matrix after 300000 iterations:
[[ 0.87075806  0.92693723  0.97192601  0.        ]
 [ 0.82203398  0.          0.87812674  0.        ]
 [ 0.76923169  0.71845851  0.7037472   0.52270127]]
```

Comparing the final utility matrix with the one obtained without the use of eligibility traces in TD(0) you will notice similar values. One could ask: **what's the advantage of using eligibility traces?** The advantage become clear when dealing with sparse reward in a large state space. In this case the eligibility trace mechanism can speed up learning propagating what learnt at t+1 back to the last states visited. 


SARSA: Temporal Differencing control
------------------------------------

Now it is time to extend the TD method to the control case. Here we are in the **active** scenario, we want to **estimate the optimal policy** starting from a random one. We saw in the introduction that the final update rule for the TD(0) case was:

$$ U(s_{t}) \leftarrow U(s_{t}) + \alpha \big[ \text{r}_{t+1} + \gamma U(s_{t+1}) - U(s_{t}) \big] $$

The update rule is based on the tuple **State-Reward-State**. Remember that now we are in the **control case**. Here we use the **Q-function** (see [second post](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html)) to estimate the best policy. The Q-function requires as input a state-action pair. The TD algorithm for control is straightforward, giving a look at the update rule will give you immediately the idea of how it works:

$$ Q(s_{t}, a_{t}) \leftarrow Q(s_{t}, a_{t}) + \alpha \big[ \text{r}_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_{t}, a_{t}) \big] $$

That's it, we simply replaced $$ U $$ with $$ Q $$ in our updating rule. We must be careful because there is a difference. Now we need a new value which is the action at t+1. This is not a problem because it is contained in the Q-matrix. In **TD control** the estimation is based on the tuple **State-Action-Reward-State-Action** and this tuple gives the name to the algorithm: **SARSA**.

![Reinforcement Learning SARSA first episode]({{site.baseurl}}/images/reinforcement_learning_model_free_passive_td_sarsa_first_episode.png){:class="img-responsive"}

To get the intuition behind the algorithm we consider again a single episode of the cleaning robot in the grid world. The robot starts at (1,1) and after seven visits it reaches the charging station at (4,3). As you can see for each state we have an action. Moving forward the algorithm takes into account only the state at t and t+1. In the standard implementation of SARSA the **previous states are ignored**, as shown by the shadow on top of them in the graphical illustration. This is in line with the TD framework as explained in the TD(0) section. 

**Which are the steps of the algorithm?** The steps of the algorithm are the following:

1. Move one step selecting $$ a_{t} $$ from $$ \pi(s_{t}) $$
2. Observe: $$ r_{t+1} $$, $$ s_{t+1} $$, $$ a_{t+1} $$
3. Update the state-action function $$ Q(s_{t}, a_{t}) $$
4. Update the policy $$ \pi(s_{t}) \leftarrow \underset{a}{\text{ argmax }} Q(s_{t},a_{t}) $$

In **step 4** we are using the same mechanism of MC for control (see [second post](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html)), the **policy $$ \pi $$ is updated at each visit** choosing the action with the highest state-action value. We are making the policy **greedy**. 


[In the [second post](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) we used the assumption of **exploring starts** to guarantee a uniform exploration of all the state-action pairs. Without random exploration the policy could get stuck in a sub-optimal solution. However, exploring start is a constraint because in a very big world we cannot easily explore all the possible states. This issue is known as the **exploration-exploitation dilemma**.
The solution is called **ε-greedy policy**. An ε-greedy policy explores all the states picking an action from a specific probability distribution.] 

The Python implementation of SARSA is based on the same code of TD(0) but with a different update rule.

```python
def update_state_action(state_action_matrix, observation, new_observation, 
                   action, new_action, reward, alpha, gamma):
    '''Return the updated utility matrix

    @param state_action_matrix the matrix before the update
    @param observation the state observed at t
    @param new_observation the state observed at t+1
    @param action the action at t
    @param new_action the action at t+1
    @param reward the reward observed after the action
    @param alpha the step size (learning rate)
    @param gamma the discount factor
    @return the updated state action matrix
    '''
    #Getting the values of Q at t and at t+1
    col = observation[1] + (observation[0]*4)
    q = state_action_matrix[action ,col]
    col_t1 = new_observation[1] + (new_observation[0]*4)
    q_t1 = state_action_matrix[new_action ,col_t1]
    #Applying the update rule
    state_action_matrix[action ,col] += \
        alpha * (reward + gamma * q_t1 - q)
    return state_action_matrix
```
Here you must remember that we define the **state-action matrix** has having the one state for each column, and one action for each row (see [second post](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html)). For instance, with the query `state_action_matrix[0, 2]` we get the state-action value for the state (3,1) (top-left corner) and action DOWN of the 4x3 grid world. With the query `state_action_matrix[11, 0]` we get the state-action value for the state (4,1) (bottom-right corner) and action UP. As usual we used the convention of Russel and Norvig for naming the states. The bottom-left corner is the state (1,1), while in Python we use the Numpy convention where `[0, 0]` defines the top-left value of the grid world.

**Does SARSA always converge to the optimal policy?** The answer is yes, SARSA converges with probability 1 as long as all the state-action pairs are visited an infinite number of times. In practice it is not possible to run our algorithm forever and for large state-action spaces the convergence is not always guaranteed. Now it is time to introduce a second algorithm for TD control: Q-learning.

Q-learning: off-policy control
-----------------------------

**Q-learning** is one of the most important algorithm in reinforcement learning. However most of the time it is not explained in details. Understanding how it works means understanding most of the ideas now on. Here I will dissect the algorithm focusing on its deep meaning. Before proceeding you should have clear in your mind the following concepts:

- The Generalised Policy Iteration (GPI) ([second post](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html))
- The $$ \text{Target} $$ term in TD learning (first section)
- The update rule of SARSA (previous section)

Now we can proceed. In the control case we always used the policy $$ \pi $$ to learn on the job, meaning that we updated $$ \pi $$ from experiences sampled from $$ \pi $$. This approach is called **on-policy** learning. However there is another way to learn about $$ \pi $$ which is called **off-policy** learning. In off-policy learning the policy $$ \pi $$ is updated based on the observation of a second policy $$ \mu $$ that is **not updated**. For instance considering the first four iterations in a 4x3 grid world we can see how after the random initialisation of $$ \pi $$ the states are updated step by step, whereas the policy $$ \mu $$ does not change at all.

![Reinforcement Learning Q-learning policies comparison]({{site.baseurl}}/images/reinforcement_learning_model_free_active_td_qlearning_policies_update.png){:class="img-responsive"}

Which are the advantages of off-policy learning? First of all using off-policy it is possible to learn about an **optimal policy** while following an **exploratory policy**. Off-policy means **learning by observation**. For example our cleaning robot could find a policy looking to another robot. It is also possible to learn about **multiple policies** while following one policy (e.g. multi-robot scenario). Moreover in deep reinforcement learning we will see how off-policy allows **re-using old experiences** generated from old policies to improve the current policy (experience replay).

The most famous **off-policy TD algorithm for control** is called **Q-Learning**. To understand how Q-learning works let's consider its update rule:

$$ Q(s_{t}, a_{t}) \leftarrow Q(s_{t}, a_{t}) + \alpha \big[ \text{r}_{t+1} + \gamma \underset{a}{\text{ max }} Q(s_{t+1}, a) - Q(s_{t}, a_{t}) \big] $$

Comparing the update rule of SARSA and the one of Q-learning you will notice that the only difference is in the $$ \text{Target} $$ term. Here I report both of them to simplify the comparison:

$$ \text{Target}[\text{SARSA}] = \text{r}_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) $$

$$ \text{Target}[\text{Q-learning}] = \text{r}_{t+1} + \gamma \underset{a}{\text{ max }} Q(s_{t+1}, a) $$

SARSA uses GPI to improve the policy $$ \pi $$. The $$ \text{Target} $$ is estimated through $$ Q(s_{t+1}, a_{t+1}) $$ which is based on the action $$ a_{t+1} $$ sampled from the policy $$ \pi $$. In SARSA improving $$ \pi $$ means improving the estimation returned by $$ Q(s_{t+1}, a_{t+1}) $$. In Q-learning we have two policies $$ \pi $$ and $$ \mu $$.The value of $$ a_{t} $$ necessary to estimate $$ Q(s_{t},a_{t}) $$ is sampled from the exploratory policy $$ \mu $$. The value of $$a_{t+1} $$ at $$ Q(s_{t+1}, a_{t+1}) $$ cannot be sampled from $$ \mu $$ because $$ \mu $$ is not updated during the training and using it would **break the GPI scheme**. The value of $$a_{t+1} $$ cannot even be sampled from $$ \pi $$ because there would be a difference between the $$ \text{Target} $$ estimated through $$ \pi $$ and the $$ \text{OldEstimate} $$ estimated through $$ \mu $$.  To solve this issue Q-learning gets the Q-value at t+1 through $$ \underset{a}{\text{ max }} Q(s_{t+1}, a) $$. The evaluation of $$ \underset{a}{\text{ max }} Q(s_{t+1}, a) $$ is independent from both $$ \pi $$ and $$ \mu $$, and it is respectful of the GPI because obtained through bootstrapping. Let's see now all the Q-learning **steps**:

1. Move one step selecting $$ a_{t} $$ from $$ \mu(s_{t}) $$
2. Observe: $$ r_{t+1} $$, $$ s_{t+1} $$
3. Update the state-action function $$ Q(s_{t}, a_{t}) $$
4. Update the policy $$ \pi(s_{t}) \leftarrow \underset{a}{\text{ argmax }} Q(s_{t},a_{t}) $$

There are some differences between the steps followed in SARSA and the one followed in Q-learning. Unlike in SARSA in the **step 2** of Q-learning we are not considering $$ a_{t+1} $$ the action at the next step. In this sense Q-learning updates the state-action function using the tuple **State-Action-Reward-State**.
If you compare **step 1** and **step 4** in SARSA you can see that in **step 1** the action is sampled from $$ \pi $$ and then the same policy is updated at **step 4**. In **step 1** and **step 4** of Q-learning we are sampling the action from the **exploration policy** $$ \mu $$ while we are updating the policy $$ \pi $$ at **step 4**.

An **example** will clarify what expressed until now. Let's suppose our cleaning robot observed the movements of a second robot in the 4x3 grid world. 
What is interesting about Q-learning is that while following a policy $$ \mu $$ which may be sub-optimal it can estimates the optimal policy $$ \pi^{*} $$. 


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

Index
------

1. [[First Post]]((https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html)) Markov Decision Process, Bellman Equation, Value iteration and Policy Iteration algorithms.
2. [[Second Post]](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) Monte Carlo Intuition, Monte Carlo methods, Prediction and Control, Generalised Policy Iteration, Q-function. 
3. **[Third Post]** Temporal Differencing intuition, Animal Learning, TD(0), TD(λ) and Eligibility Traces, SARSA, Q-learning.

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

Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and nonreinforcement. Classical conditioning II: Current research and theory, 2, 64-99.

Russell, S. J., Norvig, P., Canny, J. F., Malik, J. M., & Edwards, D. D. (2003). Artificial intelligence: a modern approach (Vol. 2). Upper Saddle River: Prentice hall.

Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. Machine learning, 3(1), 9-44.

Sutton, R. S., & Barto, A. G. (1990). Time-derivative models of pavlovian reinforcement.

