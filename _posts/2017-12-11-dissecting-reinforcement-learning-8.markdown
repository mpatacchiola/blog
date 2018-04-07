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

![Books Reinforcement Learning]({{site.baseurl}}/images/books_reinforcement_learning_an_introduction_deep_learning.png){:class="img-responsive"}

The reference for this post is chapter 8 for the [Sutton and Barto's book]((https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)) called "Generalization and Function Approximation". Moreover a good resource is the [video-lesson 7 of David Silver's course](https://www.youtube.com/watch?v=KHZVXao4qXs&t=4781s). A wider introduction to neural networks is given in the book ["Deep Learning"](http://www.deeplearningbook.org/) written by Goodfellow, Bengio, and Courville.
I want to start this post with a brief excursion in the history of neural networks. I will show you how the first neural network had the same limitation of the linear models considered in the [previous post]((https://mpatacchiola.github.io/blog/2017/12/11/dissecting-reinforcement-learning-7.html)). Then I will present the standard neural network known as **Multi Layer Perceptron** and I will show you how we can use it to model any utility function. This post can be heavy compared to the previous ones but I suggest you to study it carefully, because the use of neural networks as function approximators is the main approach used today.


The life and death of the Perceptron
------------------------------------- 
In 1957 the American psychologist [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt) presented a report entitled *"The Perceptron: a perceiving and recognizing automaton"* to the Cornell Aeronautical Laboratory commission in New York. The Perceptron was not a proper software, it was implemented in a custom hardware (as big as a wardrobe), and it could discriminate between two types of marked cards. The first series of cards was marked on the left side, whereas the second series was marked on the right. A grid of 400 photo-cells was able to locate the marks and activate mechanical relays. Adjusting the parameters of the Perceptron was not so easy as today. In Python and Tensorflow we simply define a bunch of variables and then we search for the best combinations. In the hardware implementation the parameters to tune were physical handles (potentiometers) adjusted through electric motors.

For your joy I found the original article of **The New York Times** describing the official presentation of the Perceptron by Rosenblatt and his staff:

![Perceptron NYT]({{site.baseurl}}/images/reinforcement_learning_approximators_perceptron_nyt.png){:class="img-responsive"}

As you have read the power of the Perceptron was a bit overestimated, especially when talking about artificial brains that *"would be conscious of their existence"* (we could start here a long discussion about what consciousness really is, but it would take another blog series). However, some of the predictions made by Rosenblatt were correct:

*"Later perceptrons will be able to recognize people and call out their names and instantly translate speech in one language to speech or writing in another language"*

Well, it took several decades but we now have it. What is the Perceptron from a technical point of view? You might be surprised to know that **the Perceptron is a linear function approximator** like the one described in the [previous post](https://mpatacchiola.github.io/blog/2017/12/11/dissecting-reinforcement-learning-7.html). Let's try to formalise this point. The input to the Rosenblat's Perceptron were 400 binary photo-cells that we can model as a vector of units $$ \boldsymbol{x} = ( x_{1}, x_{2,}, ..., x_{400} )$$. The output was given by an electric tension that we can call $$y$$. The connection between input and output were a series of wires and potentiometers that we can model through another vector $$\boldsymbol{w}$$ containing 400 real values (the resistance of the potentiometers). I said that the Perceptron is a linear system, and exactly like the linear model described in the previous post we can define it as the dot product between two vectors $$\boldsymbol{x}^{T} \boldsymbol{w}$$. However, there is an intermediate step: the use of an **activation function** called $$\phi$$ that takes as input the result of the weighted sum and generates the output $$y$$:

$$y = \phi \big( \boldsymbol{x}^{T} \boldsymbol{w} \big)$$

We can also use an additional **bias unit** called $$x_{b}$$ and its weight $$w_{b}$$. We can append the additional input to the vector $$\boldsymbol{x}$$, and we can append the additional weight to the vector $$\boldsymbol{w}$$. The reason for a bias unit has been extensively discussed in the [previous post](https://mpatacchiola.github.io/blog/2017/12/11/dissecting-reinforcement-learning-7.html) and I suggest you to give a look if you do not remeber why we need it.
The last equation is the final form of the Perceptron. It may be useful to graphically represent the Perceptron as a **directed graph** where each node is a unit (or neuron) and each edge is a weight (or synapse). Regarding the **name convention** there is not a clear rule in the community. I personally prefer to use the term *unit* and *weight* instead of *neuron* and *synapse* to avoid confusion with the biological counterpart. 

The following image represent the Rosenblatt's Perceptron. The 400 photo-cell fires only when a mark is identified on the card. Each photo-cell can be thought as an input of the Perceptron. Here I represented the neurons using a green colour for the active units (state=1), a red colour for the inactive units (state=0), and an orange colour for the units with undefined state. When I say *"undefined state"* I mean that the output of that unit has not been computed yet. For instance, this is the state of the output unit $$y$$ before the weighted sum and the transfer function are applied. In the image you must notice that the bias unit is always active, if you remember the [last post](https://mpatacchiola.github.io/blog/2017/12/11/dissecting-reinforcement-learning-7.html), we used this trick in order to add the bias in our model.

![Perceptron Graph Chain]({{site.baseurl}}/images/reinforcement_learning_approximators_perceptron_graph_chain.png){:class="img-responsive"}

As I told you above, the only difference between the linear approximator and the Perceptron is the activation function $$\phi$$. The original perceptron uses a [sign function](https://en.wikipedia.org/wiki/Sign_function) to generate a binary output of type zero/one.
The sign function introduce a relevant problem. Using a sign function it is **not possible to apply gradient descent** because this function is not differentiable. Another important detail I should tell you is that gradient based techniques (such as the backpropagation) were unknown at that time. 

**How did Rosenblatt train the Perceptron?** Here is the cool part, Rosenblatt used a form of [Hebbian learning](https://en.wikipedia.org/wiki/Hebbian_theory). The psychologist [Donald Hebb](https://en.wikipedia.org/wiki/Donald_O._Hebb) had published a few years before (1949) the work entitled *"The Organization of Behavior"* where he explained cognitive processes from the point of view of neural connectivity patterns. The principle is easy to grasp: **when two neurons fire at the same time their connection is strengthened**.
Ronsenblat was inspired by this work when he defined the update rule of the Perceptron. The update rule directly manages the three possible outcomes of the training phase and it is repeated for a certain number of epochs:

1. If the output $$y$$ is correct, leave the weights $$w$$ *unchanged*
2. If the output $$y$$ is incorrect (zero instead of one), *add* the input vector $$x$$ to the weight vector $$w$$
3. If the output $$y$$ is incorrect (one instead of zero), *subtract* the input vector $$x$$ from the weight vector $$w$$

The **geometric interpretation** of the update rule can help you understand what is going on. I said that $$w$$ is a vector, meaning that we can imagine this vector in a three-dimensional space. The starting point of the vector is at the origin of the axes, whereas the end of the vector is at the coordinates specified by its values. The set of all the input vectors can also be considered as a bunch of vectors that occupy part of the same three-dimensional space. What we are doing during the update is moving the weight vector changing its end-coordinates. The optimal position of the vector is inside a cone (with apex in the origin) where the first criterion of the learning rule is always satisfied. Said differently, when the weight vector reach this cone the second and third conditions of the update rule are no more triggered.

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
    #Dot product of input and weights (weighted sum)
    z = np.dot(x_array, w_array)
    #Applying the sign function
    if(z>0): y = 1
    else: y = 0
    #Returning the output
    return y

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

In the [previous post](https://mpatacchiola.github.io/blog/2017/12/11/dissecting-reinforcement-learning-7.html) I mentioned the fact that linear approximators are limited because they can only be used in linearly separable problems. The same considerations apply for the Perceptron. Historically, the research community was not aware of this problem and the work on the Perceptron continued for several years with alternating success. Interesting results were achieved by **Widrow and Hoff** in 1960 at Stanford with an evolution of the Perceptron called [ADALINE](https://en.wikipedia.org/wiki/ADALINE). The ADALINE network had multiple inputs and outputs, the activation function used on the outputs was a linear function and the update rule was the [Delta Rule](https://en.wikipedia.org/wiki/Delta_rule). The Delta Rule is a particular case of backpropagation based on a gradient descent procedure. At that time this was an important success but the main issues remained: ADALINE was still a linear model and the Delta Rule was not applicable to non-linear problems.

**There is a problem:** the publication of a book called [*"Perceptrons: an introduction to computational geometry"*](https://en.wikipedia.org/wiki/Perceptrons_(book)) in 1969 . In this book the authors, [Marvin Minsky](https://en.wikipedia.org/wiki/Marvin_Minsky) and [Seymour Papert](https://en.wikipedia.org/wiki/Seymour_Papert), mathematically proved that Perceptron-like models could only solve linearly separable problems and that they could not be applied to a non-linear dataset such as the XOR one. In the [previous post](https://mpatacchiola.github.io/blog/2017/12/11/dissecting-reinforcement-learning-7.html) I carefully chosen the **XOR gridworld** as an example of non-linear problem, showing you how a linear approximator was not able to describe the utility function of this world. This is what Minsky and Papert proved in their book. The so called **XOR affair** signed the end of the Perceptron era. The funding to artificial neural network projects gradually disappeared and just a few researchers continued to study these models.

Revival (Multi Layer Perceptron)
--------------------------------

After the winter the spring comes back. We must wait until 1985 to see the resurgence of neural networks. **Rumelhart, Hinton and Williams** published an article entitled *"Learning Internal Representations by Error Propagation"* on the use of a generalized delta rule for training a Perceptron having multiple layers. The **Multi Layer Perceptron (MLP)** is an extension of the classical Perceptron having one or more intermediate layers. The authors experimentally verified that using an additional layer (called hidden) and a new update rule, the network was able to solve the XOR problem. This result ignited again the research on neural networks. The authors stated:

*"In short, we believe that we have answered Minsky and Papert's challenge and have found a learning result sufficiently powerful to demonstrate that their pessimism about learning in multilayer machines was misplaced."*

The MLP in its classical form, is based on an input layer, an hidden layer and an output layer. The transfer function used between the layers is generally a [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function). The **error function** can be defined as the **mean squared error (MSE)** between the output and the labels. Each layer of the MLP can be represented as a multiplication between an input vector $$\boldsymbol{x}$$ and a weight matrix $$\boldsymbol{W}$$. The resulting value is added to a bias and passed to an activation function, generating an output vector $$\boldsymbol{y}$$. These operations are equivalent to the weighted sum of the input values used for the linear approximators. 
Similarly to the Perceptron we can represent an MLP as a **directed graph**. In the following image (that I suggest you to look very carefully) you can see how an MLP is assembled. The main difference with respect to the Perceptron is the hidden layer $$h$$ that is obtained through the multiplication of the input vector $$x$$ and the weight matrix $$W_{1}$$. After the weighted sum the array $$z$$ is passed through a Sigmoid activation function. You should notice how the bias term is embedded in both the input and hidden vectors. The same process is repeated another time. The output of the hidden layer is multiplied with the weight matrix $$W_{2}$$, passed through a Sigmoid, leading to the final output vector $$y$$. In the neural network lingo, the procedure that leads from the input $$x$$ to the output $$y$$ is called the **forward pass**.

![MLP Graph Chain]({{site.baseurl}}/images/reinforcement_learning_approximators_mlp_graph_chain.png){:class="img-responsive"}

**Why the MLP is not a linear model?** This is a crucial point. It can be easily proved that the MLP is a non-linear approximator only if we use a non-linear activation function. If we replace the Sigmoid with a linear activation then our MLP collapse into a Perceptron. Another interesting thing about the MLP is that given a sufficient amount of hidden neurons it can approximate any  continuous function. This is known as the [universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem) and it was proved by George Cybenko in 1989.


Backpropagation
-----------------

The main innovation behind the MLP is the **update rule** called **backpropagation**. The idea of backpropagation was not completely new at that time, but finding out that it was possible to apply it to feedforward networks took a while. As you remember from the [previous post](https://mpatacchiola.github.io/blog/2017/12/11/dissecting-reinforcement-learning-7.html) each time we want to train a function approximator we need two things:

1. An **error measure** that gives a distance between the output of the estimator and the target.
2. An **update rule** that set the parameters of the estimator in order to reduce the error.

Since neural networks are function approximators they also need these two components. The error measure is still given by the [Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) between the network output and the target value. The update rule is still an application of the [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) procedure, that in the neural network context is called **backpropagation**. If you do not want to get into the mathematical details of backpropagation this is enough, and you can skip this section. 

It is possible to think about backpropagation as an iterative application of the **chain rule** on the different layers of the network. If you want to understand the math behind backpropagation you should refresh your calculus. Here I can only give you a short explanation, because a wider **introduction to calculus is out of scope**. First of all, you must know what is a derivative and how you can calculate the derivative of the most important functions (e.g. sin, cos, exponential, polynomials). The chain rule is a simple iterative process that allows computing the derivative of a [composition of functions](https://en.wikipedia.org/wiki/Function_composition). The iterative process consists in moving into the composition and estimating the derivatives of all the functions encountered along the way. Here I will show you an **example of chain rule** applied to an equation having a single variable $$x$$:

![Backpropagation math]({{site.baseurl}}/images/reinforcement_learning_approximators_chain_rule.png){:class="img-responsive"}

As you can see the idea is to factor the starting equation in different parts (1), then estimate the derivative of each part (2), and finally multiply each part (3). Intuitively you can see backpropagation as the process of opening a set of black [Chinese boxes](https://en.wikipedia.org/wiki/Chinese_boxes) until a red box appear. In our example the red box is the variable $$x$$ and the black boxes to open are the various sub-equations we have to derive in order to reach it.

**How does the chain rule fit into neural networks?** This is something many people struggle with. Consider the neural network as a magic box where you can push an array, and get back another array. Push an input, get an output. Input and output can have different size, in our Perceptron for instance, the input array consisted of 400 values and the output was a single value. This is technically a [scalar field](https://en.wikipedia.org/wiki/Scalar_fields). In our examples of MLP we will feed an array representing the world state, and we will get an output representing the utility of that state (as a single value), or the utility of each possible action (as an array). Now, a neural network is not a magic box. The output $$y$$ is obtained through a series of operation. If you carefully look at the MLP scheme of the previous section you can go back from $$y$$ to $$h$$ and from $$h$$ to $$x$$. In the end, $$y$$ is obtained from applying a Sigmoid to the weighted sum of $$h$$ and a set of weights. Whereas $$h$$ is obtained in a similar way using the weighted sum of $$x$$ and the first matrix of weights. This long chain of operation is a **composition of multiple functions**, and by definition we can apply the chain rule to it.

**Which are the variables in a neural network?** In a neural network we do not have only a single variable to estimate, the **variables (or unknowns)** of our network are all the weights. Our goal is to find the set of weights that better represent the real utility function. Differently from a single variable equation (like the one used in the chain rule example above) a neural network is a [multivariable equation](https://en.wikipedia.org/wiki/Multivariable_calculus). For instance, you can think of the Rosenblatt's Perceptron as a multivariable equation having 400 unknowns. In a multivariable equation we do not think in terms of derivative but in terms of [gradient](https://en.wikipedia.org/wiki/Gradient). I already wrote about gradient in the [previous post](https://mpatacchiola.github.io/blog/2017/12/11/dissecting-reinforcement-learning-7.html). Dealing with a multivariable equation is not something that should scare you, since our chain rule still works. However, we now need to use [partial derivatives](https://en.wikipedia.org/wiki/Partial_derivative) instead of standard derivatives. 
Given the error function $$E$$ we have to look for the set of variables $$\boldsymbol{W}_{1}$$ and $$\boldsymbol{W}_{2}$$, meaning that we keep differentiating (opening the Chinese boxes) until we do not find them. In the image below you can find a mathematical description of this process for a MLP having a vector $$x$$ as input, a vector $$y$$ as output, two set of weights $$W_{1}$$ (between input and hidden layers) and $$W_{2}$$ (between hidden and output layers). the weighted sum of these components generate two arrays $$z_{1}$$ and $$z_{2}$$ and passed through two activation functions $$h = \phi(z_{1})$$ and $$y = \phi(z_{2})$$. Remember that the last link of the chain is the error function $$E$$, and for this reason $$E$$ is the starting point for the chain rule.

![Backpropagation math]({{site.baseurl}}/images/reinforcement_learning_approximators_backprop_math.png){:class="img-responsive"}

As you can notice the use of **backpropagation is very efficient** because it allows us to reuse part of the partial derivatives computed in later layers in order to estimate the derivatives of previous layers. Moreover all those operations can be formalised in matrix-vector form and can be easily parallelised on a GPU.

**What does backpropagation return?** The backpropagation finds the gradient vectors of the unknowns. In our MLP the backpropagation finds two vectors, one for the weights $$W_{1}$$ and the other for the weights $$W_{2}$$. Great, but how should we use those vectors? If you remember from the last post the gradient vector is the vector pointing up-hill on the error surface, meaning that it shows us where we should move in order to reach the top of the function. However here we are interested in going down-hill because we want to minimize an error (in the end we are doing gradient **descent**). This change in direction can be easily achieved changing the sign in front of the gradient. Now, knowing the direction in which to move the weights is not enough, we need to know **how much we have to move in that direction**. You must recall that the steepness of the function at a specific point is given by the **magnitude of the gradient vector**. From your linear algebra class you should know that multiplying a vector by a scalar changes the magnitude of the vector. 

This is something more complicated to estimate, 


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




Application: Multi Layer XOR
-------------------------------

Here I will reproduce the architecture used by Rumelhart and al. to solve the XOR problem. 



Policy gradient methods
-------------------------

Until now we used neural networks in order to approximate the utility (or state-action) function. However there is another widely adopted use of neural network: **the approximation of the policy**.
At the end of the [fourth post](https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html) I explained what is the difference between an actor-only and a critic-only algorithm. The policy gradient is an actor-only method that aims to approximate the policy. Before moving forward it may be useful to recap what is the difference between utility, state-action, and policy functions:

1. **Utility (or value) function**: $$y = U(s)$$ receives as input a state vector $$s$$ and return the utility $$y$$ of that state.
2. **State-action function**: $$y = Q(s, a)$$ receives as input a state $$s$$ and an action $$a$$, and returns the utility $$y$$ of that state-action pair.
3. **Policy function**: $$a = \pi(s)$$ receives as input a state $$s$$ and returns the action to perform in that state.

In policy gradient methods we use a neural network to approximate the policy function $$\pi(s)$$. As the name suggest we are using the **gradient**. At this point of the series you should know what is the **gradient vector** and how we are using it in gradient descent methods. Here, we want to estimate the gradient of our neural network, but we do not want to minimise an error anymore. Instead we want to **maximise the expected return** (long-term cumulative reward). As I told you before the gradient indicates the direction we should move the weights in order to maximise a function, and this is exactly what we want to achieve in policy gradient methods.

**Advantages**: the main advantage of policy gradient methods is that in some context it is much easier to estimate the policy function instead of the utility function. We know that the estimation of the utility function can be difficult due to the **credit assignment problem**. In policy gradients method we bypass this issue and we obtain a better convergence and an increased stability. Another advantage is that using policy gradient we can approximate a continuous action space. This overcomes an important limitation of **Q-Learning** where the $$max$$ operator only allow us to approximate discrete action spaces (see [third post](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html) for more details). 

**Disadvantages**: the main disadvantage of policy gradient methods is that they are generally slower and that they often converge to local optimum. Another problem is variance. Evaluating the policy with an high number of trajectories leads to an high variance in the approximation of the function.

REINFORCE
----------

In this post I will focus on REINFORCE, a policy gradient method based on a Monte Carlo approach.

The REINFORCE algorithm finds an unbiased estimate of the gradient without using any utility function. 

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

- Policy gradient methods on Scholarpedia by Prof. Jan Peters [[wiki]](http://www.scholarpedia.org/article/Policy_gradient_methods)

References
------------

Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. In Advances in neural information processing systems (pp. 1057-1063).


