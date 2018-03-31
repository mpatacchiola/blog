---
layout: post
title:  "Dissecting Reinforcement Learning-Part.8"
date:   2018-03-30 09:00:00 +0000
description: This blog series explains the main ideas and techniques used in reinforcement learning. In this post Reinforcement Learning through non-linear function approximation (Neural Networks) and policy gradient methods. It includes complete Python code.
author: Massimiliano Patacchiola
type: reinforcement learning
comments: false
published: false
---


In the [last post]((https://mpatacchiola.github.io/blog/2017/12/11/dissecting-reinforcement-learning-7.html)) I introduced **function approximation** as a method for representing the utility function in a reinforcement learning setting. The simple approximator we used was based on a linear combination of features and it was quite limited because it could not model complex state spaces (like the XOR gridworld). In this post I will introduce **Artificial Neural Networks** as non-linear function approximators and I will show you how we can use a neural network to model a Q-function. I will start from basic architecture called **Perceptron** and then move to the standard feed-forward model called **Multi Layer Perceptron**. Moreover, I will introduce **policy gradient methods** that are (most of the time) based on neural network policies. I will use pure Numpy to implement the network and the update rule, in this way you will have a transparent code to study. This post is important because it allows understanding the **deep models** (e.g. Convolutional Neural Networks) used in Deep Reinforcement Learning, that I will introduce in the next post.

![Books Reinforcement Learning]({{site.baseurl}}/images/books_reinforcement_learning_an_introduction_pattern_recognition.png){:class="img-responsive"}

The reference for this post is chapter 8 for the [Sutton and Barto's book]((https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)) called "Generalization and Function Approximation". Moreover a good resource is the [video-lesson 6 of David Silver's course](https://www.youtube.com/watch?v=UoPei5o4fps&t=5217s). A wider introduction to function approximation is given by any good machine learning textbook, I suggest [Pattern Recognition and Machine Learning](https://books.google.co.uk/books/about/Pattern_Recognition_and_Machine_Learning.html?id=kTNoQgAACAAJ&redir_esc=y) by Christopher Bishop.
I want to start this post with a brief excursion in the history of neural networks. I will show you how the first neural networks had the same limitation of the linear models considered in the [previous post]((https://mpatacchiola.github.io/blog/2017/12/11/dissecting-reinforcement-learning-7.html)).


The life and death of the Perceptron
------------------------------------- 
In 1957 the American psychologist [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt) presented a report entitled *"The Perceptron: a perceiving and recognizing automaton"* to the Cornell Aeronautical Laboratory commission in New York. The Perceptron was not a proper software, it was implemented in a custom hardware (as big as a wardrobe), and it could discriminate between two types of marked cards. The first series of cards was marked on the left side, whereas the second series was marked on the right. A grid of 400 photo-cells was able to locate the marks and activate mechanical relays. Adjusting the parameters of the Perceptron was not so easy as today. In Python and Tensorflow we simply define a bunch of variables and then we search for the best combinations. In the hardware implementation the parameters to tune were physical handles (potentiometers) adjusted through electric motors.

For your joy I found the original article of **The New York Times** describing the official presentation of the Perceptron by Rosenblatt and his staff:

![Perceptron NYT]({{site.baseurl}}/images/reinforcement_learning_approximators_perceptron_nyt.png){:class="img-responsive"}

As you have read the power of the Perceptron was a bit overestimated, especially when talking about artificial brains that *"would be conscious of their existence"* (we could start here a long discussion about what consciousness really is, but it would take another blog series). However, some of the predictions made by Rosenblatt were correct:

*"Later perceptrons will be able to recognize people and call out their names and instantly translate speech in one language to speech or writing in another language"*

Well, it took several decades but we now have it. What is the Perceptron from a technical point of view? You might be surprised to know that **the Perceptron is a linear function approximator** like the one described in the [previous post](https://mpatacchiola.github.io/blog/2017/12/11/dissecting-reinforcement-learning-7.html). Let's try to formalise this point. The input to the Rosenblat's Perceptron were 400 binary photo-cells that we can model as a vector of units $$ \boldsymbol{x} = ( x_{1}, x_{2,}, ..., x_{400} )$$. The output was given by an electric tension that we can call $$y$$. The connection between input and output were a series of wires and potentiometers that we can model through another vector $$\boldsymbol{w}$$ containing 400 real values (the resistance of the potentiometers). I said that the Perceptron is a linear system, and exactly like the linear model described in the previous post we can define it as the dot product between two vectors:

$$y = \sigma \big( \boldsymbol{x}^{T} \boldsymbol{w} \big)$$

The only difference between the linear approximator that I have already described and the Perceptron is the update rule.  The original perceptron used a [sign function](https://en.wikipedia.org/wiki/Sign_function) in the output unit to generate a binary output of the type zero/one. The sign function $$\sigma$$ is applied to the result of the dot product and it generates the binary output.
The problem here is the sign function itself. Using a sign function is **not possible to apply gradient descent** because this function is not differentiable. Moreover, another important detail I should tell you is that gradient based techniques (such as the backpropagation) were unknown at that time. 

**How did Rosenblatt train the Perceptron?** Here is the cool part, Rosenblatt used a form of [Hebbian learning](https://en.wikipedia.org/wiki/Hebbian_theory). The psychologist [Donald Hebb](https://en.wikipedia.org/wiki/Donald_O._Hebb) had published a few years before (1949) the work entitled *"The Organization of Behavior"* where he explained cognitive processes from the point of view of neural connectivity patterns. The principle is easy to grasp: **when two neurons fire at the same time their connection is strengthened**.
Ronsenblat was inspired by this work when he defined the update rule of the Perceptron. The update rule directly manages the three possible outcomes of the training phase and it is repeated for a certain number of epochs:

1. If the output $$y$$ is correct, leave the weights $$w$$ *unchanged*
2. If the output $$y$$ is incorrect (zero instead of one), *add* the input vector $$x$$ to the weight vector $$w$$
3. If the output $$y$$ is incorrect (one instead of zero), *subtract* the input vector $$x$$ from the weight vector $$w$$

The **geometric interpretation** of the update rule can help you understand what is going on. I said that $$w$$ is a vector, meaning that we can imagine this vector in a three-dimensional space. The starting point of the vector is at the origin of the axes, whereas the end of the vector is at the coordinates specified by its values. The set of all the input vectors $$x$$ can also be considered as a bunch of vectors that occupy part of the same three-dimensional space. What we are doing during the update rule is moving around the weight vector changing its end-coordinates. The optimal position of the vector is inside a cone (with apex in the origin) where the first criterion of the learning rule is always satisfied. Said differently, when the weight vector reach this cone the second and third conditions of the update rule are no more triggered.

Implementing a Perceptron and its update rule in **Python** is straightforward. Here You can find an example:

```python
def perceptron(x_array, w_array):
    '''Perceptron model

    Given an input array and a weight array
    it returns the output of the model.
    @param: x_array a binary input array
    @param: w_array numpy array of weights
    @return: the output of the model
    '''
    import numpy as np
    #Dot product of input and weights
    y = np.dot(x_array, w_array)
    #Applying the sign function
    if(y>0): return 1
    else: return 0

def update_rule(x_array, w_array, y_output, y_target)
    '''Perceptron update rule

    Given input, weights, output and target
    it returns the updated set of weights.
    @param: x_array a binary input array
    @param: w_array numpy array of weights
    @param: y_output the Perceptron output
    @param: y_target the target value for the input
    @return: the updated set of weights
    '''
    if(y_output == y_target):
        return w_array #first condition
    elif(y_output == 0 and y_target == 1):
        return x_array + w_array #second condition
    elif(y_output == 1 and y_target == 0)
        return x_array - w_array #third condition
```

In the [previous post](https://mpatacchiola.github.io/blog/2017/12/11/dissecting-reinforcement-learning-7.html) I mentioned the fact that linear approximators are limited because they can only be used in linearly separable problems. The same considerations apply for the Perceptron. Historically, the research community was not aware of this problem and the work on the Perceptron continued for several years with good success. Interesting results were achieved by **Widrow and Hoff** in 1960 at Stanford with an evolution of the Perceptron called [ADALINE](https://en.wikipedia.org/wiki/ADALINE). The ADALINE network had multiple inputs and outputs, the activation function used on the outputs was a linear function and the update rule was the [Delta Rule](https://en.wikipedia.org/wiki/Delta_rule). The Delta Rule is a particular case of backpropagation and it is based on a gradient descent procedure. At that time this was an important success but the main problems remained. ADALINE was still a linear model and the Delta Rule was not applicable to non-linear problems.

**There is a problem:** the publication of a book called [*"Perceptrons: an introduction to computational geometry"*](https://en.wikipedia.org/wiki/Perceptrons_(book)) in 1969 . In this book the authors, Marvin Minsky and Seymour Papert, mathematically proved that Perceptron-like models could only solve linearly separable problems and that they could not be applied to a non-linear dataset such as the XOR one. In the [previous post](https://mpatacchiola.github.io/blog/2017/12/11/dissecting-reinforcement-learning-7.html) I carefully chosen the **XOR grid world** as an example of non-linear problem, showing you how a linear approximator was not able to describe the utility function of this world. This is what Minsky and Papert proved in their book. The so called **XOR affair** signed the end of the Perceptron era. The funding to artificial neural network projects gradually disappeared and just a few researchers continued to study these models.

Revival (Multi Layer Perceptron)
--------------------------------

After the winter the spring comes back. We must wait until 1985 to see the resurgence of neural networks. **Rumelhart, Hinton and Williams** published an article entitled *"Learning Internal Representations by Error Propagation"* on the use of a generalized delta rule for training a Perceptron having multiple layers. The **Multi Layer Perceptron (MLP)** is an extension of the classical Perceptron having one or more intermediate layers. The authors experimentally verified that using an additional layer (called hidden) and a new update rule, the network was able to solve the XOR problem. This result ignited again the research on neural networks. The authors stated:

*"In short, we believe that we have answered Minsky and Papert's challenge and have found a learning result sufficiently powerful to demonstrate that their pessimism about learning in multilayer machines was misplaced."*

The MLP in its classical form, is based on an input layer, an hidden layer and an output layer. The transfer function used between the layers is generally a [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function). The **error function** can be defined as the **mean squared error (MSE)** between the output and the labels. Each layer of the MLP can be represetned as a vector-matrix multiplication between an input vector $$\boldsymbol{x}$$ and a weight matrix $$\boldsymbol{W}$$. The resulting value is added to a bias and passed to an activation function, generating an output vector $$\boldsymbol{y}$$. These operations are equivalent to the weighted sum of the input values used for the linear approximators. The main innovation behind the MLP is the **update rule** called **backpropagation**. The idea of backpropagation was not completely new, but the applying it to feedforward networks was not immediate.

Similarly to what we saw in the [previous post](https://mpatacchiola.github.io/blog/2017/12/11/dissecting-reinforcement-learning-7.html) regarding the update rule for linear approximators, it is possible to think about backpropagation as an iterative application of the **chain rule** on the different layers of the network. Intuitively you can see backpropagation as the process of opening a set of black [Chinese boxes](https://en.wikipedia.org/wiki/Chinese_boxes) until a red box appear.

![Backpropagation math]({{site.baseurl}}/images/reinforcement_learning_approximators_backprop_math.png){:class="img-responsive"}




Method
--------

To improve the performance of our function approximator we need an error measure and an update rule. These two components work tightly in the learning cycle of every supervised learning technique. Their use in reinforcement learning is not much different from how they are used in a classification task. In order to understand this section you need to refresh some concepts of [multivariable calculus](https://en.wikipedia.org/wiki/Multivariable_calculus) such as the [partial derivative](https://en.wikipedia.org/wiki/Partial_derivative) and [gradient](https://en.wikipedia.org/wiki/Gradient).

![Function Approximation Training]({{site.baseurl}}/images/reinforcement_learning_function_approximation_training_cycle.png){:class="img-responsive"}

**Error Measure**: a common error measure is given by the [Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) between two quantities. For instance, if we have the optimal utility function $$U^{*}(S)$$ and an approximator function $$\hat{U}(s, \boldsymbol{w})$$, then the MSE is defined as follows:

$$ \text{MSE}( \boldsymbol{w} ) = \frac{1}{N} \sum_{s \in S} \big[ U^{*}(s) - \hat{U}(s, \boldsymbol{w}) \big]^{2}  $$

[comment]: <> (where $$P(s)$$ is a distribution weighting the errors of different states, such that $$\sum_{s} P(s) = 1$$. The number of parameters $$\boldsymbol{w}$$ is lower that the number of total states $$N$$. The function $$P(s)$$ allows gaining accuracy on some states instead of others.)

that's it, the MSE is given by the expectation $$\mathop{\mathbb{E}}[ (U^{*}(s) - \hat{U}(s, \boldsymbol{w}) \big)^{2} ]$$ that quantifies the difference between the target and the approximator output. When the training is working correctly the MSE will decrease meaning that we are getting closer to the optimal utility function. The MSE is a common loss function used in supervised learning. However, in reinforcement learning it is often used a reinterpretation of the MSE called **Mean Squared Value Error (MSVE)**. The MSVE introduce a distribution $$\mu(s) \geq 0$$ that specifies how much we care about each state $$s$$. As I told you the function approximator is based on a set of weights $$\boldsymbol{w}$$ that contains less elements than the total number of states. For this reason adjusting a subset of the weights means improving the utility prediction of some states but loosing precision in others. We have limited resources and we have to manage them carefully. The function $$\mu(s)$$ gives us an explicit solution and using it we can rewrite the previous equation as follows:

$$ \text{MSVE}( \boldsymbol{w} ) = \frac{1}{N} \sum_{s \in S}  \mu(s) \big[ U^{*}(s) - \hat{U}(s, \boldsymbol{w}) \big]^{2}  $$


**Update rule**: the update rule for differentiable approximator is [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent). The [gradient](https://en.wikipedia.org/wiki/Gradient) is a generalisation of the concept of derivative applied to scalar-valued functions of multiple variables. You can imagine the gradient as the vector that points in the direction of the greatest rate of increase. Intuitively, if you want to reach the top of a mountain the gradient is a signpost that in each moment show you in which direction you should walk. The gradient is generally represented with the operator $$\nabla$$ also known as **nabla**. 
The goal in gradient descent is to minimise the error measure. We can achieve this goal moving in the direction of the negative gradient vector, meaning that we are not moving anymore to the top of the mountain but downslope. At each step we adjust the parameter vector $$\boldsymbol{w}$$ moving a step closer to the valley. First of all, we have to estimate the gradient vector for $$ \text{MSE}( \boldsymbol{w} )$$ or $$ \text{MSVE}( \boldsymbol{w} )$$. Those error functions are based on $$\boldsymbol{w}$$. In order to get the gradient vector we have to calculate the partial derivative of each weight with respect to all the other weights. Secondly, once we have the gradient vector we have to adjust the value of all the weights in accordance with the negative direction of the gradient.
In mathematical terms, we can update the vector $$\boldsymbol{w}$$ at $$t+1$$ as follows:

$$\begin{eqnarray} 
\boldsymbol{w}_{t+1} &=&  \boldsymbol{w}_{t}  - \frac{1}{2} \alpha \nabla_{\boldsymbol{w}} \text{MSE}(\boldsymbol{w}_{t}) \\
&=& \boldsymbol{w}_{t}  - \frac{1}{2} \alpha \nabla_{\boldsymbol{w}} \big[ U^{*}(s) - \hat{U}(s, \boldsymbol{w}_{t}) \big]^{2}\\
&=& \boldsymbol{w}_{t}  + \alpha \big[ U^{*}(s) - \hat{U}(s, \boldsymbol{w}_{t}) \big] \nabla_{\boldsymbol{w}} \hat{U}(s, \boldsymbol{w}_{t}) \\
\end{eqnarray}$$

The last step is an application of the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) that is necessary because we are dealing with a function composition. We want to find the gradient vector of the error function with respect to the weights, and the weights are part of our function approximator $$\hat{U}(s, \boldsymbol{w}_{t})$$. The minus sign in front of the quantity 1/2 is used to change the direction of the gradient vector. Remember that the gradient points to the top of the hill, while we want to go to the bottom (minimizing the error). In conclusion the upate rule is telling us that all we need is the output of the approximator and its gradient. Finding the gradient of a linear approximator is particularly easy, whereas in non-linear approximators (e.g. neural networks) it requires more steps.

At this point you might think we have all we need to start the learning procedure, however there is an important part missing. We supposed it was possible to use the optimal utility function $$U^{*}$$ as target in the error estimation step. **We do not have the optimal utility function**. Think about that, having this function would mean we do not need an approximator at all. Moving in our gridworld we could simply call $$U^{*}(s_{t})$$ at each time step $$t$$ and get the actual utility value of that state. What we can do to overcome this problem is to build a target function $$U^{\sim}$$ which represent an **approximated target** and plug it in our formula:

$$ \boldsymbol{w}_{t+1} =  \boldsymbol{w}_{t} + \alpha \big[ U^{\sim}(s) - \hat{U}(s, \boldsymbol{w}) \big] \nabla_{\boldsymbol{w}} \hat{U}(s, \boldsymbol{w}) $$

How can we estimate the approximated target? We can follow different approaches, for instance using Monte Carlo or TD learning. In the next section I will introduce these methods.


Target estimation
------------------

In the previous section we came to the conclusion that we need approximated target functions $$U^{\sim}(s)$$ and $$Q^{\sim}(s,a)$$ to use in the error evaluation and update rule. The type of target used is at the heart of function approximation in reinforcement learning. There are two main approaches:

**Monte Carlo target**: an approximated value for the target can be obtained through a direct interaction with the environment. Using a Monte Carlo approach (see the [second post](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html)) we can generate an episode and update the function $$U^{\sim}(s)$$ based on the states encountered along the way. The estimation of the optimal function $$U^{*}(s)$$ is unbiased because $$\mathop{\mathbb{E}}[U^{\sim}(s)] = U^{*}(s)$$, meaning that the *prediction is guaranteed to converge*.

**Bootstrapping target**: the other approach used to build the target is called bootstrapping and I introduced it in the [third post](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html). In bootstrapping methods we do not have to complete an episode for getting an estimation of the target, we can directly update the approximator parameters after each visit. The simplest form of bootstrapping target is the one based on TD(0) which is defined as follows:

$$ U^{\sim}(s_{t}) = \hat{U}(s_{t+1}, \boldsymbol{w}) \qquad Q^{\sim}(s_{t}, a) = \hat{Q}(s_{t+1}, a, \boldsymbol{w})$$

That's it, the target is obtained through the approximation given by the estimator itself at $$s_{t+1}$$.

I already wrote about the differences between the two approaches, however here I would like to discuss it again in the new context of function approximation.
In both cases the functions $$U^{\sim}(s)$$ and $$Q^{\sim}(s,a)$$ are based on the vector of weights $$\boldsymbol{w}$$. For this reason the correct notation we are going to use from now on is $$U^{\sim}(s,\boldsymbol{w})$$ and $$Q^{\sim}(s,a,\boldsymbol{w})$$.
We have to be particularly careful when using the bootstrapping methods in gradient-based approximators. Bootstrapping methods are not true instances of gradient descent because they only care about the parameters in $$\hat{U}(s, \boldsymbol{w})$$. At training time we adjust $$\boldsymbol{w}$$ in the estimator $$\hat{U}(s,\boldsymbol{w})$$ based on a measure of error but we are not changing the parameters in the target function $$U^{\sim}(s,\boldsymbol{w})$$ based on an error measure. Bootstrapping ignores the effect on the target, taking into account only the gradient of the estimation. For this reason bootstrapping techniques are called **semi-gradient methods**. Due to this issue semi-gradient methods **does not guarantee the convergence**. At this point you may think that it is better to use Monte Carlo methods because at least they are guaranteed to converge. Bootstrapping gives two main advantages. First of all they learn online and it is not required to complete the episode in order to update the weights. Secondly they are faster to learn and computationally friendly. 

The **Generalised Policy Iteration (GPI)** (see [second post](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html)) applies here as well. Let's suppose we start with a random set of weights. At the very first step the agent follows an epsilon-greedy strategy moving in the state with the highest utility. After the first step it is possible to update the weights using gradient descent. What's the effect of this adjustment? The effect is to slightly improve the utility function. At the next step the agent follows again a greedy strategy, then the weights are updated through gradient descent, and so on and so forth. As you can see we are applying the GPI scheme again.



Application: Multi Layer XOR
--------------------------------------

Here I will reproduce the architecture used by Rumelhart and al. to solve the XOR problem. 


High-order approximators
------------------------

The linear approximator is the simplest form of approximation. The linear case is appealing not only for its simplicity but also because it is guaranteed to converge. However, there is an important limit implicit in the linear model: it cannot **represent complex relationships between features**. That's it, the linear form does not allow representing the interaction between features. Such a complex interaction naturally arise in physical systems. Some features may be informative only when other features are absent. For example, the inverted pendulum  angular position and velocity are tightly connected. A high angular velocity may be either good or bad depending on the position of the pole. If the angle is high then high angular velocity means an imminent danger of falling, whereas if the angle is low then high angular velocity means the pole is righting itself.

Solving the XOR problem is very easy when an additional feature is added. 

$$ \hat{U}(s, \boldsymbol{w}) = x_{1} w_{1} + x_{2} w_{2} + x_{1} x_{2} w_{3} + w_{4} $$

If you look to the equation what I added is the new term $$x_{1} x_{2} w_{3}$$. This term introduces a relationship between the two features $$x_{1}$$ and $$x_{2}$$. Now the surface represented by the equation is no more a plane but an [hyperbolic paraboloid](https://en.wikipedia.org/wiki/Paraboloid), a saddle-like surface which perfectly adapt to the XOR-world. We do not need to rewrite the update function because it remains unchanged. We always have a linear combination of features and the gradient is always equal to the input vector. In the repository you will find another script called `xor_paraboloid.py` containing an implementation of this new approximator. Running the script with the same parameters used for the linear case we end up with the following plot:

![Function Approximation XOR Hyperbolic]({{site.baseurl}}/images/reinforcement_learning_function_approximation_linear_function_xor_hyperbolic.png){:class="img-responsive"}

Here the paraboloid is represented using four different perspectives. The result obtained at the end of the training shows that the utilities are very good. 

```
w: [ 0.36834857  0.36628493 -0.18575494 -0.73988694]
[[ 0.73  0.36 -0.02 -0.4  -0.77]
 [ 0.37  0.17 -0.02 -0.21 -0.4 ]
 [-0.   -0.01 -0.01 -0.02 -0.02]
 [-0.37 -0.19 -0.01  0.17  0.35]
 [-0.74 -0.37 -0.01  0.36  0.73]]
```

We should have -1 in the bottom-left and top-right corners, the approximator returned -0.74 and -0.77 which are pretty good estimations. Similar results have been obtained for the positive states in the top-left and bottom-right corners, where the approximator returned 0.73 and 0.77 which are very close to the true utility of 1.0. I suggest you to run the script using different hyper-parameters (e.g. the learning rate alpha) to see the effects on the final plot and on the utility table.

The geometrical intuition is helpful because it gives an immediate intuition of the different approximators. We saw that using additional features and more complex functions it is possible to better describe the utility space. High-order approximators may find useful links between futures whereas a pure linear approximator could not. An example of high-order approximator is the **quadratic approximator**. In the quadratic approximator we use a second order polynomial to model the utility function. 

$$ \hat{U}(s, \boldsymbol{w}) = x_{1} w_{1} + x_{2} w_{2} + x_{1}^{2} w_{3} + x_{2}^{2} w_{4} +... + x_{N-1} w_{M-1} + x_{N}^{2} w_{M} $$


It is not easy to **choose the right polynomial**. A simple approximator like the linear one can miss the relevant relations between features and target, whereas an high order approximator can fail to generalise to new unseen states. The optimal balance is achieved through a delicate tradeoff known in machine learning as the [bias-variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff).



Conclusions
-----------



Index
------

1. [[First Post]](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) Markov Decision Process, Bellman Equation, Value iteration and Policy Iteration algorithms.
2. [[Second Post]](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) Monte Carlo Intuition, Monte Carlo methods, Prediction and Control, Generalised Policy Iteration, Q-function. 
3. [[Third Post]](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html) Temporal Differencing intuition, Animal Learning, TD(0), TD(λ) and Eligibility Traces, SARSA, Q-learning.
4. [[Fourth Post]](https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html) Neurobiology behind Actor-Critic methods, computational Actor-Critic methods, Actor-only and Critic-only methods.
5. [[Fifth Post]](https://mpatacchiola.github.io/blog/2017/03/14/dissecting-reinforcement-learning-5.html) Evolutionary Algorithms introduction, Genetic Algorithm in Reinforcement Learning, Genetic Algorithms for policy selection.
6. [[Sixt Post]](https://mpatacchiola.github.io/blog/2017/08/14/dissecting-reinforcement-learning-6.html) Reinforcement learning applications, Multi-Armed Bandit, Mountain Car, Inverted Pendulum, Drone landing, Hard problems.
7. [[Seventh Post]](https://mpatacchiola.github.io/blog/2017/12/11/dissecting-reinforcement-learning-7.html) Function approximation, Intuition, Linear approximator, Applications, High-order approximators.
8. **[Eighth Post]** Non-linear function approximation, Perceptron, Multi Layer Perceptron, Applications, Policy Gradient.

Resources
----------

- The **complete code** for the Reinforcement Learning Function Approximation is available on the [dissecting-reinforcement-learning](https://github.com/mpatacchiola/dissecting-reinforcement-learning) official repository on GitHub.

- **Reinforcement learning: An introduction (Chapter 8 'Generalization and Function Approximation')** Sutton, R. S., & Barto, A. G. (1998). Cambridge: MIT press. [[html]](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)



References
------------




