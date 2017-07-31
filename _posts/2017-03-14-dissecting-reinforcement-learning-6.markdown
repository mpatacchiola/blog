---
layout: post
title:  "Dissecting Reinforcement Learning-Part.6"
date:   2017-05-30 09:00:00 +0000
description: This blog series explains the main ideas and techniques used in reinforcement learning. In this post Reinforcement Learning applications, Black jack, Mountain Car, Robotic Arm, Bomb Disposal Rover. It includes complete Python code.
author: Massimiliano Patacchiola
type: reinforcement learning
comments: false
published: false
---

Hello folks! Welcome to the sixth episode of the "Dissecting Reinforcement Learning" series. Until now we saw how reinforcement learning works. However we applied most of the techniques to the robot cleaning example. I decided to follow this approach because I think that the same example applied to different techniques can help the reader to better understand what changes from one scenario to the other.
Now it is time to apply this knowledge to other problems. In each one of the following sections I will introduce a reinforcement learning problem and I will show you how to tackle it. First of all I will explaining the history behind the application, then I will implement it in Python and I will apply a reinforcement learning technique to solve it. I will follow an incremental approach, starting from the simplest case and arriving to the most complicated one. Here we are going to use a discrete representation for both the utility and the action-value function, meaning that I will represent these functions with a matrix (or look-up table if you prefer). The **discretization** is the only method we can use at this point in the series. In the next post I am going to introduce function approximation which is a powerful tool for dealing with complex problems.

![Books Reinforcement Learning]({{site.baseurl}}/images/books_reinforcement_learning_an_introduction_statistical_reinforcement_learning.png){:class="img-responsive"}

The references for this post are the [Sutton and Barto's book]((https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)) (chapter 11, case studies), and ["Statistical Reinforcement Learning"](https://www.crcpress.com/Statistical-Reinforcement-Learning-Modern-Machine-Learning-Approaches/Sugiyama/p/book/9781439856895) by Masashi Sugiyama which contains a good description of some of the applications we are going to encounter.


Multi-Armed Bandits
---------------

An armed bandit is a fancy way to call slot machines in Las Vegas. They are *bandits* because they steal your money!
In 1950s Mosteller and Bush were studying the effect of reward on mice in a [T-maze](https://en.wikipedia.org/wiki/T-maze). In order to compare the performance with humans they realised a two-armed bandit experiment. The subjects could choose to pull the left or right arm in order to receive a reward. One of the two arm was more generous. 


![Reinforcement Learning Multi-Armed Bandit illustration]({{site.baseurl}}/images/reinforcement_learning_multi_armed_bandit_photo.png){:class="img-responsive"}

The subject had to find a good balance between **exploration and exploitation**. Let's suppose the subject play a single round finding out that the left arm is more generous. How to proceed? You must remember that the machines are stochastic and the best one may not return the prize for a while in a short sequence. Should the subject explore the option that looks inferior or exploit the current best option?

During the year has been proposed many solutions to the multi-armed bandit problem.
In the previous posts we already saw one of those solutions: the **epsilon-greedy** strategy. We can count how many times an arm returns a positive reward. At each time step we are going to select the most generous arm with probability $$p = \epsilon$$ (exploitation) and we are going to randomly choose one of the other arms with probability $$q = 1 - \epsilon$$ (exploration). Which value should we choose for epsilon? The best thing to do is to set $$\epsilon = 1$$ at the beginning and then decrease it linearly during the game. In this way we will explore a lot at the beginning and we will focus on the most generous arm in the end. this strategy is called **epsilon-decreasing**.


Formally we can define the problem as a Markov decision process with a single state (see the [first post](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html)). There are $$N$$ arms which is possible to pull and each one as a certain probability of returning a prize. We have a single state and $$N$$ possible actions (one action for each arm). At each round the agent chooses one arm to pull and it receives a reward. The goal of the agent is to maximise the reward.

We can start from the machine $$n_{1}$$ and check how often it returns the prize, then we can switch to the machine $$n_{2}$$ and count again the number of prizes obtained, then machine $$n_{3}$$, etc. However we must be careful because the machines are stochastic and the best one may not return the prize for a while in a short sequence. 

Multi-armed bandit problems are in our daily life. The doctor that has to choose the best treatment for a patient, the web-designer who has to find the best template for maximising the AdSense clicks, or the entrepreneur who has to decide how to distribute the budget among different companies for maximising the incomes.

Mountain Car
------------
The mountain car is a classic reinforcement learning problem. This problem was first described by [Andrew Moore in his PhD thesis](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.2654) and is defined as following: a mountain car is moving on a two-hills landscape. The engine of the car does not have enough power to cross a steep climb. The driver has to find a way to reach the top of the hill. 

![Reinforcement Learning Mountain Car illustration]({{site.baseurl}}/images/reinforcement_learning_discrete_applications_mountain_car_photo.png){:class="img-responsive"}

A good explanation of the probelm is presented in [chapter 4.5.2](https://www.crcpress.com/Statistical-Reinforcement-Learning-Modern-Machine-Learning-Approaches/Sugiyama/p/book/9781439856895) of Sugiyama's book. I will follow the same mathematical convention here.
The state space is defined by the position $$ x $$ obtained through the function $$ sin(3x) $$ in the domain [-1.2, +0.5] ($$ m $$) and the velocity $$ \dot{x} $$ defined in the interval [-1.5, +1.5] ($$m/s$$). There are three possible actions $$ a = $$ [-2.0, 0.0, +2.0] which are the values of the force applied to the car (left, no-op, right). The reward obtained is positive 1.0 only if the car reaches the goal. A negative cost of living of -0.01 is applied at every time step. The mass of the car is $$ m = 0.2 \ kg $$, the gravity is $$ g = 9.8 \ m/s^2 $$, the friction is defined by $$ k = 0.3 \ N $$, and the time step is $$ \Delta t =0.1 \ s $$. Given all these parameters the position and velocity of the car at $$ t+1 $$ are updated using the following equations:

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

I added an useful method called `render()` which can save the episode animation in a gif or a video (it requires [imagemagick](https://www.imagemagick.org) and [avconv](https://libav.org/avconv.html)). This method can be called every k episodes in order to save the animation and check the improvements. For example, to save an mp4 video it is possible to call the method with the following parameters:

```python
my_car.render(file_path='./mountain_car.mp4', mode='mp4')
```

If you want an animated gif instead you can call the method in this way:

```python
my_car.render(file_path='./mountain_car.gif', mode='gif')
```

Now let's try to use the class and let's build an **agent which uses a random policy** for choosing the actions. Here I will use a time step of 0.1 seconds and a total of 100 steps (which means a 10 seconds long episode). The code is very compact, and at this point of the series you can easily understand it without any additional comment:

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

Observing the behaviour of the car in the animation generated by the script it is possible to see how difficult is the task. Using a purely **random policy** the car remains at the bottom of the valley and it does not reach the goal.  The **optimal policy** is to move on the left accumulating inertia and then to push as much as possible to the right. 

![Reinforcement Learning Random Mountain Car]({{site.baseurl}}/images/reinforcement_learning_discrete_applications_radom_mountain_car.gif){:class="img-responsive"}

How can we deal with this problem using a discrete approach? We said that the state space is continuous, meaning that we have infinite values to take into account. What we can do is dividing the continuous state-action space in chunks. This is called **discretization**. If the car moves in a continuous space enclosed in the range $$[-1.2, 0.5]$$ it is possible to create 10 bins to represent the **position**. When the car is at -1.10 it is in the first bin, when at -0.9 in the second, etc.

![Reinforcement Learning Mountain Car Discretization]({{site.baseurl}}/images/reinforcement_learning_discrete_applications_mountain_car_discretization.png){:class="img-responsive"}

In our case both position and velocity must be discretized and for this reason we need two arrays to store all the states.
Here I call **bins** the discrete containers (entry of the array) where both position and velocity are stored. In Numpy is easy to create these containers using the `numpy.linspace()` function. The two arrays can be used to define a policy matrix. In the script I defined the policy matrix as a square matrix having size `tot_bins`, meaning that both velocity and position have the same number of bins. However it is possible to differently discretize velocity and position, obtaining a rectangular matrix.

```python
tot_bins = 10  # the number of bins to use for the discretization
# Generates two arrays having bins of equal size
velocity_state_array = np.linspace(-1.5, +1.5, num=tot_bins-1, endpoint=False)
position_state_array = np.linspace(-1.2, +0.5, num=tot_bins-1, endpoint=False)
# Random policy as a square matrix of size (tot_bins x tot_bins)
# Three possible actions represented by three integers
policy_matrix = np.random.randint(low=0, 
                                  high=3, 
                                  size=(tot_bins,tot_bins)).astype(np.float32)
```

When a new observation arrives it is possible to assign it to a specific bin using the Numpy method `numpy.digitize()` which takes as input the observation (velocity and position) and cast it inside the containers previously declared. The digitized observation can be used to access the policy matrix at specific address.

```python
# Digitizing the continuous observation
observation = (np.digitize(observation[1], velocity_state_array), 
               np.digitize(observation[0], position_state_array))
# Accessing the policy using observation as index
action = policy_matrix[observation[0], observation[1]]
```

Now it is time to use reinforcement learning for mastering the mountain car problem.
Here I will use the temporal differencing method called **SARSA** which I introduced in the [third Post](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html). I suggest you to use other methods to check the different performances you may obtain. Only a few changes are required in order to run the code of the previous posts. In this example I trained the policy for $$10^{5}$$ episodes (`gamma`=0.999, `tot_bins`=12), using an epsilon decayed value (from 0.9 to 0.1) which helped exploration in the first part of the training. The script automatically saves gif and plots every $$10^{4}$$ episodes. The following is the graph of the **cumulated reward per episode**, where the light-red line is the raw data and the dark line a moving average of 500 episodes:

![Reinforcement Learning Mountain Car Reward]({{site.baseurl}}/images/reinforcement_learning_mountain_car_sarsa_reward_plot.png){:class="img-responsive"}

As it is possible to see a stable policy is obtained around episode $$65 \times 10^{3}$$. At the same time a significant reduction in the number of **steps** necessary to finish the episode is showed in the step plot.

![Reinforcement Learning Mountain Car Step]({{site.baseurl}}/images/reinforcement_learning_mountain_car_sarsa_step_plot.png){:class="img-responsive"}

One of the policy that the algorithm obtained is very efficient and allows to reach the target position in 6.7 seconds with a cumulated reward of 0.34. The script prints on **terminal** the policy, which is represented by three actions-symbols (`<`=left, `O`=noop, `>`=right), the position (columns) and velocity (rows).

```
Episode: 100001
Epsilon: 0.1
Episode steps: 67
Cumulated Reward: 0.34
Policy matrix: 
 O   <   O   O   O   <   >   <   >   O   O   <  
 <   <   >   <   <   >   <   <   >   >   O   >  
 O   >   <   <   <   <   <   <   >   <   O   <  
 O   <   <   <   >   >   <   <   >   >   <   O  
 O   >   <   <   >   >   >   <   >   O   >   >  
 O   >   >   <   >   O   >   <   >   >   <   <  
 <   O   >   <   >   >   <   <   >   >   >   >  
 O   >   <   <   >   >   >   >   >   >   >   >  
 <   >   >   >   >   >   >   >   >   >   >   >  
 O   >   >   >   >   >   >   >   >   >   O   >  
 <   <   >   >   >   >   O   >   >   >   >   >  
 >   O   >   >   >   >   >   >   >   <   >   < 
```

The policy obtained is a sub-optimal policy. As it is possible to see from the step plot (light-blue curve) there are policies that can reach the goal in around 40 steps (4 seconds). The policy can be observed in the gif generated at the end of the training:

![Reinforcement Learning Sarsa Policy Mountain Car]({{site.baseurl}}/images/reinforcement_learning_mountain_car_sarsa_final_policy.gif){:class="img-responsive"}

The discretization method worked well in the mountain car problem. 
However there are some possible issues that may occur. First of all, it is hard to decide how many bins one should use for the discretization. An high number of bins leads to a fine control but they can cause a [combinatorial explosion](https://en.wikipedia.org/wiki/Combinatorial_explosion). The second issue is that it may be necessary to visit all the states in order to obtain a good policy and this can lead to long training time. We will see in the rest of the series how to deal with these problems. For the moment you should download the complete code, which is called `sarsa_mountain_car.py`, from the [official repository](https://github.com/mpatacchiola/dissecting-reinforcement-learning) and play with it changing the hyper-parameters for obtaining a better performance.

Inverted Pendulum
------------------
The [inverted pendulum](https://en.wikipedia.org/wiki/Inverted_pendulum) is another classical problem, which is considered a benchmark in control theory. [James Roberge](http://ecee.colorado.edu/~taba7194/CPIFAC2oct11.pdf) was probably the first author to present a solution to the problem in his bachelor thesis back in 1960. The problem consists of a pole hinged on a cart which must be moved in order to keep the pole in vertical position. The inverted pendulum is well described in chapter 4.5.1 of [Sugiyama's book](https://www.crcpress.com/Statistical-Reinforcement-Learning-Modern-Machine-Learning-Approaches/Sugiyama/p/book/9781439856895). Here I will use the same mathematical notation.
The **state space** consists of the **angle** $$ \phi \in [\frac{-\pi}{2}, \frac{\pi}{2}]$$ (rad) (which is zero when the pole is perfectly vertical) and the **angular velocity** $$ \dot{\phi} \in [-\pi, \pi]$$ (rad/sec). The **action space** is discrete and it consists of **three forces** [-50, 0, 50] (Newton) which can be applied to the cart in order to swing the pole up.

![Reinforcement Learning Inverted Pendulum Illustration]({{site.baseurl}}/images/reinforcement_learning_discrete_applications_inverted_pendulum_photo.png){:class="img-responsive"}


The system has different parameters which can decide the dynamics. The mass $$ m = 2 kg $$ of the pole, the mass $$ M = 8 kg $$ of the cart, the lengh $$ d = 0.5 m $$ of the pole, and time step $$ \Delta t = 0.1 s $$.
Given these parameters the angle $$ \phi $$ and the angular velocity $$ \dot{\phi} $$ at $$ t+1 $$ are update as following:

$$ \phi_{t+1} = \phi_{t} + \dot{\phi}_{t+1} \Delta t $$

$$ \dot{\phi_{t+1}} = \dot{\phi_{t}} + \frac{g \ sin(\phi_t) - \alpha \ m \ d \ (\dot{\phi}_t)^2 \  sin(2\phi_t)/2 + \alpha \ cos(\phi_t) \ a_t }{ 4l/3 - \alpha \ m \ d \ cos^2(\phi_t)} \Delta t$$

Here $$ \alpha = 1 / M+m $$, and $$ a_t $$ is the action at time $$ t $$.
The **reward** is updated considering the cosine of the angle $$ \phi $$, meaning larger the angle lower the reward. The reward is 0.0 when the pole is horizontal and 1.0 when vertical. When the pole is completely horizontal the episode finishes. Like in the mountain car example we can use discretization to enclose the continuous state space in pre-defined bins. For example, the **position** is codified with an **angle** in the range $$[\frac{-\pi}{2}, \frac{\pi}{2}]$$ and it can be discretized in 4 bins. When the pole has an angle of $$\frac{-\pi}{5}$$ it is in the third bin, when it has an angle of $$\frac{\pi}{6}$$ it is in the second bin, etc.

![Reinforcement Learning Pendulum Discretization]({{site.baseurl}}/images/reinforcement_learning_discrete_applications_inverted_pendulum_discretization.png){:class="img-responsive"}

I wrote a special module called `inverted_pendulum.py` containing the class `InvertedPendulum`. Like in the mountain car module there are the methods `reset()`, `step()`, and `render()` which allow starting the episode, moving the pole and saving a gif. The animation is produced using Matplotlib and can be imagined as a camera centred on the cartpole main joint which moves in accordance with it. To create a new environment it is necessary to create a new instance of the `InvertedPendulum` object, defining the main parameters (masses, pole length and time step).

```python
from inverted_pendulum import InvertedPendulum

# Defining a new environment with pre-defined parameters
my_pole = InvertedPendulum(pole_mass=2.0, 
                           cart_mass=8.0, 
                           pole_lenght=0.5, 
                           delta_t=0.1)
```

We can test the performance of an agent which follows a random policy. The code is called `random_agent_inverted_pendulum.py` and is available on the [repository](https://github.com/mpatacchiola/dissecting-reinforcement-learning). Using a random strategy on the pole balancing environment leads to unsatisfactory performances. The best I got running the script multiple times is a very short episode of 1.5 seconds. 

![Reinforcement Learning Random Inverted Pendulum]({{site.baseurl}}/images/reinforcement_learning_inverted_pendulum_random_agent.gif){:class="img-responsive"}

The **optimal policy** consists in compensating the angle and speed variations keeping the pole as much vertical as possible.
Like for the mountain car I will deal with this problem using discretization. Both velocity and angle are discretized in bins of equal size and the resulting arrays are used as indices of a square policy matrix. As algorithm I will use **first-visit Monte Carlo** for control, which has been introduced in the [second post of the series](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html).
I trained the policy for $$5 \times 10^5$$ episodes (`gamma`=0.999, `tot_bins`=12). In order to encourage exploration I used an $$\epsilon$$-greedy strategy with $$\epsilon$$ linearly decayed from 0.99 to 0.1. Each episode was 100 steps long (10 seconds). The maximum reward that can be obtained is 100 (the pole is kept perfectly vertical for all the steps). The reward plot shows that the algorithm could rapidly find good solutions, reaching an average score of 45. 

![Reinforcement Learning Inverted Pendulum Reward]({{site.baseurl}}/images/reinforcement_learning_inverted_pendulum_montecarlo_reward_plot.png){:class="img-responsive"}

The final policy has a very good performance and with a favourable starting position can easily manage to keep the pole in balance for the whole episode (10 seconds).

![Reinforcement Learning Inverted Pendulum Final Policy]({{site.baseurl}}/images/reinforcement_learning_inverted_pendulum_montecarlo_final_policy.gif){:class="img-responsive"}

The complete code is called `montecarlo_control_inverted_pendulum.py` and is included in the [Github repository](https://github.com/mpatacchiola/dissecting-reinforcement-learning) of the project. Feel free to change the parameters and check if they have an impact on the learning. Moreover you should test other algorithms on the pole balancing problem and verify which one gets the best performance.

Bomb disposal autonomous robot
-------------------------------------
There are plenty of possible applications for reinforcement learning. One of the most interesting is robot control. Reinforcement learning offers a wide set of techniques for the implementation of complex policies for [humanoid control](https://repository.tudelft.nl/islandora/object/uuid:986ea1c5-9e30-4aac-ab66-4f3b6b6ca002/datastream/OBJ), [helicopter acrobatic maneuvers](http://heli.stanford.edu/papers/nips06-aerobatichelicopter.pdf). For a recent survey I suggest you to read the article of [Kober et at. (2013)](http://journals.sagepub.com/doi/full/10.1177/0278364913495721).


When I was working as robotics engineer I was part of the team involved in the realisation of an **autonomous robot** used for bomb disposal. The robot had a sensor which allowed finding land mine concealed under the ground. This sensor had a limited operative range and it was very important to carefully read it in order to control the robot movements. The goal of the robot was to find the bomb and sign the position with a red mark using a sprayer. Here I will reproduce the same scenario using a modified version of the gridworld module developed in the previous posts.

Drone landing
--------------

In the cleaning robot example the robot moved in a flat 2D environment. Now it is time to add another dimension. We have to train an autonomous **drone** to **land on a ground marker**. The drone moves in a discrete 3D world, representing a cubic room. The marker is always in the same point (the centre of the room). The rules are similar to the gridworld, if the drone hits one of the wall it bounces back to the previous position. Landing on the marker leads to a positive reward of +1.0, while landing on another point leads to a negative reward of -1.0. A negative cost of living of -0.01 is applied at each time step.

Hard problems
--------------

The problem that I described above are difficult but not extremely difficult. In the end we managed to find good policies using a tabular approach. Which kind of problems are hard to solve using reinforcement learning? 

An example is the **acrobot**. The acrobot is a planar robot represented by two links of equal length. The  first link is connected to a fixed joint. The second link is connected to the first and both of them can swing freely and can pass by each other. The robot can control the torque applied to the second joint in order to swing and move the system. The state space is represented by the two positions of the links and by the two velocities. 
The action space is represented by the amount of torque the robot can apply to the joint. At the beginning of the episode the two links point downward. The goal is to swing the links until the tip passes a specific boundary. The reward given is negative (-1.0) until the robot reaches the terminal state (+1.0).

![Reinforcement Learning Acrobot]({{site.baseurl}}/images/reinforcement_learning_discrete_applications_acrobot_photo.png){:class="img-responsive"}

The state space of the acrobot is large and it is challenging for our discrete approach. It is like having **two inverted pendula** which interact in the same system. Moreover the positive **reward is sparse**, meaning that it can be obtained only after a long series of coordinated movements.
The acrobot is described in chapter 11.3 of the [Sutton and Barto's book]((https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)). Sutton used SARSA($$\lambda$$) and a **linear approximator** to solve the problem. We still do not have the right tools for mastering this problem, only in the next post we will see what a linear approximator is. If you are interested in the Sutton's solution you can [read this paper]((http://papers.nips.cc/paper/1109-generalization-in-reinforcement-learning-successful-examples-using-sparse-coarse-coding.pdf)).

[comment]: <> (Following the classical implementation I constrained the angular velocities to $$ \theta_{1} \in [-4 \pi, +4 \pi] $$ and $$ \theta_{2} \in [-9 \pi, +9 \pi] $$. A time step of $$ 0.05 sec $$ was used in all the experiments. The Python implementation of the acrobot may seems complicated because of many mathematical terms. Here I will describe the equations of motion of the acrobot and how it is possible to translate them in code. If you want to understand this part I suggest you to refresh you physics background, in particular [ordinary differential equations (ODEs)](https://en.wikipedia.org/wiki/Ordinary_differential_equation). If you are not interested in the math behind the problem you can skip this part and go directly to the results. The acrobot is a physical system which can be formalised through some parameters. In the real world the links have masses, which I will represent with two constants $$ m_{1} = m_{2} = 1 kg $$. At the same way I define the length of the links as $$ l_{1} = l_{2} = 1 m $$. The gravity constant $$ g $$ is equal to $$ 9.8 m/s^{2} $$.)

Think about videogames. The state space may be very large and hard to formalise using a lookup matrix. Moreover the transition matrix is not given.

Conclusions
-----------

Here I presented some classical reinforcement learning problems showing how the techniques of the previous posts can be used to obtain stable policies. However we always started from the assumption of a discretized state space which was described by a lookup table or matrix. The main limitation of this approach is that in many application the state space is extremely large and it is not possible to visit all the states. To solve this problem it is possible to use **function approximation**. In the next post I will introduce function approximation and I will show you how a **neural network** can be used in order to describe a large state space. The use of neural networks open up new horizons and it is the first step toward modern methods such as **deep reinforcement learning**.


Index
------

1. [[First Post]](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) Markov Decision Process, Bellman Equation, Value iteration and Policy Iteration algorithms.
2. [[Second Post]](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) Monte Carlo Intuition, Monte Carlo methods, Prediction and Control, Generalised Policy Iteration, Q-function. 
3. [[Third Post]](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html) Temporal Differencing intuition, Animal Learning, TD(0), TD(λ) and Eligibility Traces, SARSA, Q-learning.
4. [[Fourth Post]](https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html) Neurobiology behind Actor-Critic methods, computational Actor-Critic methods, Actor-only and Critic-only methods.
5. [[Fifth Post]](https://mpatacchiola.github.io/blog/2017/03/14/dissecting-reinforcement-learning-5.html) Evolutionary Algorithms introduction, Genetic Algorithm in Reinforcement Learning, Genetic Algorithms for policy selection.
6. **[Sixt Post]** Reinforcement learning applications, Mountain Car in SARSA

Resources
----------

- The **complete code** for the Reinforcement Learning applications is available on the [dissecting-reinforcement-learning](https://github.com/mpatacchiola/dissecting-reinforcement-learning) official repository on GitHub.

- **Reinforcement learning: An introduction (Chapter 11 'Case Studies')** Sutton, R. S., & Barto, A. G. (1998). Cambridge: MIT press. [[html]](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)

- **History of Inverted-Pendulum Systems** Lundberg, K. H., & Barton, T. W. (2010). [[pdf]](http://ecee.colorado.edu/~taba7194/CPIFAC2oct11.pdf)

- **Reinforcement Learning on autonomous humanoid robots** Schuitema, E. (2012). [[pdf]](https://repository.tudelft.nl/islandora/object/uuid:986ea1c5-9e30-4aac-ab66-4f3b6b6ca002/datastream/OBJ)

- **Generalization in reinforcement learning: Successful examples using sparse coarse coding** Sutton, R. S. (1996). [[pdf]](http://papers.nips.cc/paper/1109-generalization-in-reinforcement-learning-successful-examples-using-sparse-coarse-coding.pdf)

References
------------

Abbeel, P., Coates, A., Quigley, M., & Ng, A. Y. (2007). An application of reinforcement learning to aerobatic helicopter flight. In Advances in neural information processing systems (pp. 1-8).

Kober, J., Bagnell, J. A., & Peters, J. (2013). Reinforcement learning in robotics: A survey. The International Journal of Robotics Research, 32(11), 1238-1274.

Lundberg, K. H., & Barton, T. W. (2010). History of inverted-pendulum systems. IFAC Proceedings Volumes, 42(24), 131-135.

Sutton, R. S. (1996). Generalization in reinforcement learning: Successful examples using sparse coarse coding. In Advances in neural information processing systems (pp. 1038-1044).


