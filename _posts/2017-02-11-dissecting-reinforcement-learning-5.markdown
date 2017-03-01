---
layout: post
title:  "Dissecting Reinforcement Learning-Part.5"
date:   2017-02-11 10:00:00 +0000
description: This blog series explains the main ideas and techniques used in reinforcement learning. In this post Genetic Algorithm for policy estimation. It includes complete Python code.
author: Massimiliano Patacchiola
comments: false
published: false
---

As I promised in this fifth episode of the "Dissecting Reinforcement Learning" series I will introduce evolutionary algorithms and in particular Genetic Algorithms (GAs). If you passed through the [fourth post](https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html) you know that GAs can be considered Actor-only algorithms, meaning that they search directly in the policy space without the need of a utility function. GAs are often considered separate from reinforcement learning. In fact GAs do not pay attention to the underlying [markov decision process](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) and to the actions the agent select during its lifetime. The use of this information can enable more efficient search but in some cases can be missleading. For example when states are misperceived or partially hidden the standard reinforcement learning algorithms may have problems. In the same scenario GAs play it safe, but there is a price to pay. The search space is larger and the convergence to a solution can be much slower.
Following this train of thoughts Sutton and Barto did not included GAs in the [holy book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html) of reinforcement learning. Our references for this post are two books. The first is *"Genetic Algorithms in Search, Optimization, and Machine Learning"* by Goldberg. The second is ["An Introduction to Genetic Algorithms"](https://mitpress.mit.edu/books/introduction-genetic-algorithms) by Melanie Mitchell. Moreover you may find useful the article ["Evolutionary Algorithms for Reinforcement Learning"](https://www.jair.org/media/613/live-613-1809-jair.pdf) which can give a rapid introduction in a few pages. 

![Books Genetic Algorithms]({{site.baseurl}}/images/books_genetic_algorithms_in_search_an_introduction_to_ga.png){:class="img-responsive"}

After my graduation I was fascinated by GAs. During that period I started a placement of one year at the [Laboratory of Autonomous Robotics and Artificial Life](http://laral.istc.cnr.it/), where I applied GAs to autonomous robots with the goal of investigating the emergence of complex behaviours from simple interactions. Take a bunch of simple organism controlled by a rudimentary neural network, then let them evolve in the same environment. What is going to happen? In most of the cases nothing special. The worst robots wonder around without goals, whereas the best individuals are extremely selfish. In some cases however it is possible to observe a group behaviour. The robots start to cooperate because they notice that it leads to an higher reward.  Complex behaviours arise through the interaction of simple entities. This is called [Emergence](https://en.wikipedia.org/wiki/Emergence) and it is a well know phenomenon in evolutionary robotics and [swarm robotics](https://en.wikipedia.org/wiki/Swarm_robotics).


![Genetic Algorithms in Evolutionary robotics]({{site.baseurl}}/images/reinforcement_learning_actor_only_genetic_algorithms_evolutionary_robotics.png){:class="img-responsive"}

In the same period I was studying the **decision making** strategies of single robots in [intertemporal choice](https://en.wikipedia.org/wiki/Intertemporal_choice) tasks.
Intertemporal choices concern options that can be obtained at different points in time: buying a luxury item today or saving the money to ensure a sizable pension in the future.
In those experiments I used simulated [e-puck robots](http://www.e-puck.org/) which were controlled by a simple neural network [see Figures (a) and (b)]. The sensors of the robot can localise the presence of two tokens: green and red. As you can see in figure (c) the sensors return a stronger signal when the token is close to the robot. The sensor signal is the input to the network, whereas the output is the speed of the two wheels of the robot.
In a **first phase** the neural networks controlling the robots were evolved following an evolutionary strategy. The environment could be rich of energetic tokens (green) or it can contains only few of those. In a **second phase** called the **test phase**, the robots had to choose between two tokens, one highly energetic and the other slightly energetic. The tokens are initialised at the same distance from the robot  and then one of those is moved far away.
When the tokens were at the same distance from the robot it chose the highly energetic token. However moving the highly energetic token far away increased the preference for the less energetic one. This behaviour has been observed in many animals and especially in rats.
If you are interested in this topic you can find more in the article I published with my colleagues: ["Investigating intertemporal choice through experimental evolutionary robotics"](http://www.sciencedirect.com/science/article/pii/S0376635715000595). This is one of the practical applications of evolutionary algorithms.

I will start this post with an intuitive introduction which should help the reader who does not have familiarity with evolutionary methods. After the introduction I will dissect the main concepts behind GAs and we will give a look to some Python code. I will introduce many new terms because GAs have a slang which is different from the one used in reinforcement learning.


Evolutionary Algorithms (and birds)
----------------------------------------
The 27th of December 1831 the ship HMS Beagle sailed from [Plymouth](https://en.wikipedia.org/wiki/Plymouth) in England. Onboard there was [Charles Darwin](https://en.wikipedia.org/wiki/Charles_Darwin), the father of the theory of natural selection. The expedition was planned to last only two years, but it lasted for five. The Beagle visited Brasil, Patagonia, Strait of Magellan, Chile, Galapagos, etc. During the travel Darwin took notes about biology, geology and antropology of those places. Later on he will publish those notes in a book called ["The voyage of the Beagle"](https://en.wikipedia.org/wiki/The_Voyage_of_the_Beagle). What did Darwin discovers during the travel? 


![Genetic Algorithms Darwin Finches]({{site.baseurl}}/images/reinforcement_learning_actor_only_genetic_algorithms_darwin_finches.png){:class="img-responsive"}

When visiting the Galapagos Darwin noticed the presence of some birds that appear to have evolved from a single ancestral flock.
Those birds shared common features but they were characterised by a remarkable diversity in beak form and function. Darwin hypothesised that the isolation of each species in a different small island was the cause of this high differentiation. In particular the shape of the beak evolved in accordance with the food availability on that island. Species with strong beak could eat hard seeds, whereas species with small beak preferred insects.

- **Geospiza magnirostris**: large size bird which can eat hard seeds.
- **Geospiza fortis**: medium size bird which prefer small and soft seeds.
- **Geospiza parvula**: small size bird which prefer medium size insects.
- **Geospiza olivacea**: small size bird which prefer to eat small insects.

Gradually Darwin found more and more proofs to this hypothesis and he eventually came out with the [theory of natural selection](https://en.wikipedia.org/wiki/Natural_selection). Today we call the Galapagos birds the [Darwin's_finches](https://en.wikipedia.org/wiki/Darwin's_finches). These birds are so isolated that it is possible to observe rapid evolution in a few generations when the environment change. For example, the Geospiza fortis generally prefer soft seeds. However in the Seventies a severe drought reduced the supply of seeds. This finch was forced to turn to harder seeds which led in two generations to a 10% change in the size of the beak. What's going on here? How can a population change so rapidly? **How natural selection operates?** 

![Evolutionary algorithms Intuition]({{site.baseurl}}/images/reinforcement_learning_actor_only_evolutionary_algorithms_intuition_drawing.png){:class="img-responsive"}

In the general population the individuals are the results of the **crossover** between two parents. The crossover is a form of **recombination** of the parents' genetic material. In isolated population redundancy of genetic material brings to very similar individuals. However it can happen that a random **mutation** modifies the newborn's genome generating organisms which carry on a new characteristic. For instance, in the Darwin's finches it was a stronger beak. The mutation can bring an advantage for the subject, leading to a longer life and an higher probability of **reproduction**. On the other hand if the mutation does not carry any advantage it is discarded during the selection.
That's it. Evolution selects individuals with the variations best suited to a specific environment. In this sense evolution is not "the strongest survives" but "the best suited survives".

Evolutionary algorithms uses operators inspired by natural selection (reproduction, mutation, recombination, and selection). They are optimisers, meaning that they search for solutions to a problem directly in the solution space. The candidate solutions are single individuals like the Galapagos birds. In the standard approach a sequence of steps is used in order to evaluate and selects the best solutions:

1. **Fitness function:** it evaluates the performance of each candidate 
2. **Selection:** it chooses the best individuals based on their fitness score
3. **Recombination:** it replicates and recombines the individuals

Evolutionary algorithms are part of a broader class called [evolutionary computation](https://en.wikipedia.org/wiki/Evolutionary_computation). The generic label *"evolutionary algorithm"* applies to many techniques, which differ in representation and implementation details. Here I will focus only on **genetic algorithms**, however you should be aware that there are many variations out there.

![Evolutionary algorithms Diagram]({{site.baseurl}}/images/reinforcement_learning_actor_only_evolutionary_algorithms_diagram.png){:class="img-responsive"}

In the next section I will introduce GAs which can be considered a special type of evolutionary algorithms. The metaphor used by GAs is the one of DNA, genotype, genes, etc. I will describe all these new words and I will compare them to the reinforcement learning terminology we encounter in the past episodes.


Genetic Algorithms
-------------------

Genetic Algorithms in Python
-----------------------------

Conclusions
-----------



Index
------

1. [[First Post]](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) Markov Decision Process, Bellman Equation, Value iteration and Policy Iteration algorithms.
2. [[Second Post]](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) Monte Carlo Intuition, Monte Carlo methods, Prediction and Control, Generalised Policy Iteration, Q-function. 
3. [[Third Post]](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html) Temporal Differencing intuition, Animal Learning, TD(0), TD(λ) and Eligibility Traces, SARSA, Q-learning.
4. [[Fourth Post]](https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html) Neurobiology behind Actor-Critic methods, computational Actor-Critic methods, Actor-only and Critic-only methods.
5. **[Fifth Post]** Evolutionary Algorithms introduction, Genetic Algorithms for policy selection.

Resources
----------

- The **complete code** for the Genetic Algorithm examples is available on the [dissecting-reinforcement-learning](https://github.com/mpatacchiola/dissecting-reinforcement-learning) official repository on GitHub.

- **Reinforcement learning: An introduction.** Sutton, R. S., & Barto, A. G. (1998). Cambridge: MIT press. [[html]](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)

- **Evolutionary Algorithms for Reinforcement Learning.** Moriarty, D. E., Schultz, A. C., & Grefenstette, J. J. (1999). [[pdf]](https://www.jair.org/media/613/live-613-1809-jair.pdf)

References
------------

Moriarty, D. E., Schultz, A. C., & Grefenstette, J. J. (1999). Evolutionary algorithms for reinforcement learning. J. Artif. Intell. Res.(JAIR), 11, 241-276.

