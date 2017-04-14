---
layout: post
title:  "Dissecting Reinforcement Learning-Part.6"
date:   2017-03-14 09:00:00 +0000
description: This blog series explains the main ideas and techniques used in reinforcement learning. In this post Reinforcement Learning applications, Black jack, Mountain Car, Robotic Arm, Bomb Disposal Rover. It includes complete Python code.
author: Massimiliano Patacchiola
comments: false
published: false
---

Hello folks! Welcome to the sixth episode of the "Dissecting Reinforcement Learning" series. Until now we saw how reinforcement learning works. However we applied most of the techniques to the robot cleaning example. I decided to follow this approach because I think that the same example applied to different techniques can help the reader to better understand what changes from one scenario to the other.
Now it is time to apply this knowledge to taugh problems. In each one of the following sections I will introduce a reinforcement learning problem and I will show you how to tackle it. First of all I will explaining the history behind the application, then I will implement it in Python and I will apply a reinforcement learning technique to solve it. I will follow an incremental approach, starting from the simplest case and arriving to the most complicated one. Here we are going to use a discrete representation for both the utility and the action-value function, meaning that I will represent these functions with a matrix (or look-up table if you prefer). The discrete approach is the only one you know until this point in the series. In the next post I am going to introduce function approximation which is a powerful tool for dealing with challenging problems.

![Books Reinforcement Learning]({{site.baseurl}}/images/books_reinforcement_learning_an_introduction_statistical_reinforcement_learning.png){:class="img-responsive"}

The references for this post are the [Sutton and Barto's book]((https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)) (chapter 11, case studies), and ["Statistical Reinforcement Learning"](https://www.crcpress.com/Statistical-Reinforcement-Learning-Modern-Machine-Learning-Approaches/Sugiyama/p/book/9781439856895) by Masashi Sugiyama which contains a good description of some of the applications we are going to encounter.


Blak Jack Monte Carlo
----------------------
Discrete state space, discrete action space.

Tic-Tac-Toe
-------------
Discrete state space, discrete action space.

Bomb disposal autonomous robot
-------------------------------------
Discrete state space, discrete action space.
For a series of a robotic applications: Connel and Mahadevan (1993), Robot Learning
Maybe use this example in the next post, when talking about function approximation (linear approximation).
The parameter which is possible to take into account are distance from obstacles and presence of a bomb.

Mountain Car
------------
The mountain car is a classic reinforcement learning problem. This problem was first described by [Andrew Moore in his PhD thesis](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.2654) and is defined as following: a mountain car is moving on a two-hills landscape. The engine of the car does not have enough power to cross a steep climb. The driver has to find a way to reach the top of the hill. 

![Reinforcement Learning Mountain Car illustration]({{site.baseurl}}/images/reinforcement_learning_discrete_applications_mountain_car_photo.png){:class="img-responsive"}

A good explanation of the probelm is presented in [chapter 4.5.2](https://www.crcpress.com/Statistical-Reinforcement-Learning-Modern-Machine-Learning-Approaches/Sugiyama/p/book/9781439856895) of Sugiyama's book. I will follow the same mathematical convention here.
The state space is defined by the position $$ x $$ obtained through the function $$ sin(3x) $$ in the domain [-1.2, +0.5] ($$ m $$) and the velocity $$ \dot{x} $$ defined in the interval [-1.5, +1.5] ($$m/s$$). There are three possible actions $$ a = $$ [-2.0, 0.0, +2.0] which are the values of the force applied to the car (left, no-op, right). The mass of the car is $$ m = 0.2 \ kg $$, the gravity is $$ g = 9.8 \ m/s^2 $$, the friction is defined by $$ k = 0.3 \ N $$, and the time step is $$ \Delta t =0.1 \ s $$. Given all these parameters the position and velocity of the car at $$ t+1 $$ are updated using the following equations:

$$ x_{t+1} = x_{t} + \dot{x}_{t+1} \Delta t $$

$$ \dot{x}_{t+1} = \dot{x}_{t} + (g \ m \ cos(3x_{t}) + \frac{a_{t}}{m} - k \ \dot{x}_{t}) \Delta t $$

The mountain car environment has been implemented in [OpenAI Gym](https://gym.openai.com/), however here I will build everything from scratch for pedagogical reason. In the repository you will find the file `mountain_car.py` which contains a class called `MountainCar`. I built this class using only `Numpy` and `matplotlib`. The class contains methods which are similar to the one used in OpenAI Gym. The main method is called `step()` and allows executing an action in the environment. This method returns the state at t+1 the reward and a value called `done` which is `True` if the car reach the goal. The method contains the implementations of the equation of motion and it uses the parameters previously defined.


```python
def step(self, action):
    """Perform one step in the environment following the action.
        
    @param action: an integer representing one of three actions [0, 1, 2]
     where 0=move_left, 1=do_not_move, 2=move_right
    @return: (postion_t1, velocity_t1), reward, done
     where reward is always negative but when the goal is reached
     done is True when the goal is reached
    """
    if(action >= 3):
        raise ValueError("[MOUNTAIN CAR][ERROR] The action value "
                         + str(action) + " is out of range.")
    done = False
    reward = -0.01
    action_list = [-0.2, 0, +0.2]
    action_t = action_list[action]
    velocity_t1 = self.velocity_t + \
                  (-self.gravity * self.mass * np.cos(3*self.position_t)
                   + (action_t/self.mass) 
                   - (self.friction*self.velocity_t)) * self.delta_t
    position_t1 = self.position_t + (velocity_t1 * self.delta_t)
    # Check the limit condition (car outside frame)
    if position_t1 < -1.2:
        position_t1 = -1.2
        velocity_t1 = 0
    # Assign the new position and velocity
    self.position_t = position_t1
    self.velocity_t= velocity_t1
    self.position_list.append(position_t1)
    # Reward and done when the car reaches the goal
    if position_t1 >= 0.5:
        reward = +1.0
        done = True
    # Return state_t1, reward, done
    return [position_t1, velocity_t1], reward, done
```

In the object initialisation it is possible to define different parameters for the simulation. Here I will define a new car object setting the parameters we want:

```python
from mountain_car import MountainCar

my_car = MountainCar(mass=0.2, friction=0.3, delta_t=0.1)
```

I added an useful method called `render()` which can save the episode animation in a gif or a video. This method can be called every k episodes in order to save the animation and monitor the improvements. For example to save an mp4 video it is possible to call the method with the following parameters:

```python
my_car.render(file_path='./mountain_car.mp4', mode='mp4')
```

Now let's try to use the class and let's build an agent which uses a random policy for choosing the actions. Here I will use a time step of 0.1 seconds and a total of 100 steps (which means 10 seconds of simulation). The code is very compact, and at this point of the series you can easily understand it without any additional comment:

```python
from mountain_car import MountainCar
import random

my_car = MountainCar(mass=0.2, friction=0.3, delta_t=0.1)
cumulated_reward = 0
print("Starting random agent...")
for step in range(100):
    action = random.randint(a=0, b=2)
    observation, reward, done = my_car.step(action)
    cumulated_reward += reward
    if done: break
print("Finished after: " + str(step+1) + " steps")
print("Cumulated Reward: " + str(cumulated_reward))
print("Saving the gif in: ./mountain_car.gif")
my_car.render(file_path='./mountain_car.gif', mode='gif')
print("Complete!")
```

Observing the behaviour of the car in the animation generated by the script it is possible to see how difficult is the task. Moving randomly the car remains at the bottom of the valley and will never reach the goal. 

![Reinforcement Learning Random Mountain Car]({{site.baseurl}}/images/reinforcement_learning_discrete_applications_radom_mountain_car.gif){:class="img-responsive"}

The **optimal policy** here is to move on the left accumulating inertia and then to push as much as possible to the right.


Inverted Pendulum Cart
-----------------------
Continuous state space (two joint position, two joint velocities) and discrete action space (left, right, nothing).

Acrobot
--------- 
[Continuous state space and continuous action space.] The acrobot is planar robot represented by two links of equal length. The  first link is connected to a fixed joint. The second link is connected to the first and both of them can swing freely and can pass by each other. The robot can control the torque applied to the second joint in order to swing and move the system. The state space is represented by the two positions of the links and by the two velocities. The action space is represented by the the amount of torque the robot can apply to the joint (+1.0, -1.0, 0.0). The goal is to swing the links until the tip passes a specific boundary. The reward given is negative (-1.0) until the robot reaches the terminal state.
The state space of the acrobot is large and it is challenging for our discrete approach. Following the classical implementation I constrained the angular velocities to $$ \theta_{1} \in [-4 \pi, +4 \pi] $$ and $$ \theta_{2} \in [-9 \pi, +9 \pi] $$. A time step of $$ 0.05 sec $$ was used in all the experiments.

The acrobot is described in chapter 11.3 of the [Sutton and Barto's book]((https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)). 
Similarly to Sutton I will use SARSA(lambda) to solve the problem, however I will not use any kind of linear approximation.

The Python implementation of the acrobot may seems complicated because of many mathematical terms. Here I will describe the equations of motion of the acrobot and how it is possible to translate them in code. If you want to understand this part I suggest you to refresh you physics background, in particular [ordinary differential equations (ODEs)](https://en.wikipedia.org/wiki/Ordinary_differential_equation). If you are not interested in the math behind the problem you can skip this part and go directly to the results. The acrobot is a physical system which can be formalised through some parameters. In the real world the links have masses, which I will represent with two constants $$ m_{1} = m_{2} = 1 kg $$. At the same way I define the length of the links as $$ l_{1} = l_{2} = 1 m $$. The gravity constant $$ g $$ is equal to $$ 9.8 m/s^{2} $$.

[If the acrobot with Q-Lerning does not work I can explain why a discrete representation is limited. I can say that a large state space is challenging because it is difficult to visit all the states. Using function approximation it is possible to have actions for states that have never been visited before but which are similar to previously visited states.]



Index
------

1. [[First Post]](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) Markov Decision Process, Bellman Equation, Value iteration and Policy Iteration algorithms.
2. [[Second Post]](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) Monte Carlo Intuition, Monte Carlo methods, Prediction and Control, Generalised Policy Iteration, Q-function. 
3. [[Third Post]](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html) Temporal Differencing intuition, Animal Learning, TD(0), TD(λ) and Eligibility Traces, SARSA, Q-learning.
4. [[Fourth Post]](https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html) Neurobiology behind Actor-Critic methods, computational Actor-Critic methods, Actor-only and Critic-only methods.
5. [[Fifth Post]](https://mpatacchiola.github.io/blog/2017/03/14/dissecting-reinforcement-learning-5.html) Evolutionary Algorithms introduction, Genetic Algorithm in Reinforcement Learning, Genetic Algorithms for policy selection.
6. **[Sixt Post]** Reinforcement learning applications, Black Jack Monte Carlo, Acrobot in SARSA(lambda)

Resources
----------

- The **complete code** for the Reinforcement Learning applications is available on the [dissecting-reinforcement-learning](https://github.com/mpatacchiola/dissecting-reinforcement-learning) official repository on GitHub.

- **Reinforcement learning: An introduction (Chapter 11 'Case Studies')** Sutton, R. S., & Barto, A. G. (1998). Cambridge: MIT press. [[html]](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)


References
------------


