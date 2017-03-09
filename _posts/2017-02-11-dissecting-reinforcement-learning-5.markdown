---
layout: post
title:  "Dissecting Reinforcement Learning-Part.5"
date:   2017-02-11 10:00:00 +0000
description: This blog series explains the main ideas and techniques used in reinforcement learning. In this post Evolutionary aglorithms intuition and classification, Genetic Algorithms operators, Genetic Algorithm in reinforcement learning, Genetic Algorithms for policy estimation. It includes complete Python code.
author: Massimiliano Patacchiola
comments: false
published: false
---

As I promised in this fifth episode of the "Dissecting Reinforcement Learning" series I will introduce evolutionary algorithms and in particular Genetic Algorithms (GAs). If you passed through the [fourth post](https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html) you know that GAs can be considered Actor-only algorithms, meaning that they search directly in the policy space without the need of a utility function. GAs are often considered separate from reinforcement learning. In fact GAs do not pay attention to the underlying [markov decision process](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) and to the actions the agent select during its lifetime. The use of this information can enable more efficient search but in some cases can be missleading. For example when states are misperceived or partially hidden the standard reinforcement learning algorithms may have problems. In the same scenario GAs play it safe, but there is a price to pay. When the search space is large the convergence can be much slower.
Following this train of thoughts Sutton and Barto did not include GAs in the [holy book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html) of reinforcement learning. Our references for this post are two books. The first is *"Genetic Algorithms in Search, Optimization, and Machine Learning"* by Goldberg. The second is ["An Introduction to Genetic Algorithms"](https://mitpress.mit.edu/books/introduction-genetic-algorithms) by Melanie Mitchell. Moreover you may find useful the article ["Evolutionary Algorithms for Reinforcement Learning"](https://www.jair.org/media/613/live-613-1809-jair.pdf) which can give a rapid introduction in a few pages. 

![Books Genetic Algorithms]({{site.baseurl}}/images/books_genetic_algorithms_in_search_an_introduction_to_ga.png){:class="img-responsive"}

I am going to start this post with a short story about my past experience with GAs. After my graduation I was fascinated by GAs. During that period I started a placement of one year at the [Laboratory of Autonomous Robotics and Artificial Life](http://laral.istc.cnr.it/), where I applied GAs to autonomous robots with the goal of investigating the emergence of complex behaviours from simple interactions. Take a bunch of simple organism controlled by a rudimentary neural network, then let them evolve in the same environment. What is going to happen? In most of the cases nothing special. The worst robots wander around without goals, whereas the best individuals are extremely selfish. In some cases however it is possible to observe a group behaviour. The robots start to cooperate because they notice that it leads to an higher reward.  Complex behaviours arise through the interaction of simple entities. This is called [Emergence](https://en.wikipedia.org/wiki/Emergence) and it is a well know phenomenon in evolutionary robotics and [swarm robotics](https://en.wikipedia.org/wiki/Swarm_robotics).


![Genetic Algorithms in Evolutionary robotics]({{site.baseurl}}/images/reinforcement_learning_actor_only_genetic_algorithms_evolutionary_robotics.png){:class="img-responsive"}

In the same period I was studying the **decision making** strategies of single robots in [intertemporal choice](https://en.wikipedia.org/wiki/Intertemporal_choice) tasks.
Intertemporal choices concern options that can be obtained at different points in time: buying a luxury item today or saving the money to ensure a sizable pension in the future.
In those experiments I used simulated [e-puck robots](http://www.e-puck.org/) [see figure (a)] which were controlled by a simple neural network [see figure (b)]. The sensors of the robot can localise the presence of two tokens: green and red. As you can see in figure (c) the sensors return a stronger signal when the token is close to the robot. The sensor signal is the input to the network, whereas the output is the speed of the two wheels of the robot.
In a **first phase** the neural networks controlling the robots were evolved following an evolutionary strategy. The ratio of green (high energy) and red (low energy) tokens in the environment was manipulated, leading to rich and poor ecologies. In a **second phase** called the **test phase**, the robots had to choose between singles green or red tokens. The tokens were placed at the same distance from the robot but  one of those was moved far away at each new trial.
When the tokens were at the same distance from the robot it chose the highly energetic token. However moving the highly energetic token far away increased the preference for the less energetic one. The robot strategy changed based on the type of ecology used during the evolution. For example, poor ecologies increased the preference for the green token. This behaviour has been observed in many animals and especially in rats.
If you are interested in this topic you can find more in the article I published with my colleagues: ["Investigating intertemporal choice through experimental evolutionary robotics"](http://www.sciencedirect.com/science/article/pii/S0376635715000595). This is one of the practical applications of evolutionary algorithms.

I will start this post with an intuitive introduction which should help the reader who does not have familiarity with evolutionary methods. After the introduction I will dissect the main concepts behind GAs and we will give a look to some Python code. I will introduce many new terms because GAs have a slang which is different from the one used in reinforcement learning.


Evolutionary Algorithms (and birds)
----------------------------------------
The 27th of December 1831 the ship HMS Beagle sailed from [Plymouth](https://en.wikipedia.org/wiki/Plymouth) in England. Onboard there was [Charles Darwin](https://en.wikipedia.org/wiki/Charles_Darwin), the father of the theory of natural selection. The expedition was planned to last only two years, but it lasted for five. The Beagle visited Brasil, Patagonia, Strait of Magellan, Chile, Galapagos, etc. During the travel Darwin took notes about biology, geology and antropology of those places. Later on he published those notes in a book called ["The voyage of the Beagle"](https://en.wikipedia.org/wiki/The_Voyage_of_the_Beagle). What did Darwin discovers during the travel? 


![Genetic Algorithms Darwin Finches]({{site.baseurl}}/images/reinforcement_learning_actor_only_genetic_algorithms_darwin_finches.png){:class="img-responsive"}

When visiting the **Galapagos** Darwin noticed the presence of some **birds** that appeared to have evolved from a single ancestral flock.
Those birds shared common features but they were characterised by a remarkable diversity in beak form and function. Darwin hypothesised that the isolation of each species in a different small island was the cause of this high differentiation. In particular the **shape of the beak** evolved in accordance with the food availability on that island. Species with strong beak could eat hard seeds, whereas species with small beak preferred insects.

- **Geospiza magnirostris**: large size bird which can eat hard seeds.
- **Geospiza fortis**: medium size bird which prefer small and soft seeds.
- **Geospiza parvula**: small size bird which prefer medium size insects.
- **Geospiza olivacea**: small size bird which prefer to eat small insects.

Gradually Darwin found more and more proofs to this hypothesis and he eventually came out with the [theory of natural selection](https://en.wikipedia.org/wiki/Natural_selection). Today we call the Galapagos birds the [Darwin's finches](https://en.wikipedia.org/wiki/Darwin's_finches). These birds are so isolated that it is possible to observe rapid evolution in a few generations when the environment change. For example, the Geospiza fortis generally prefer soft seeds. However in the Seventies a severe drought reduced the supply of seeds. This finch was forced to turn to harder seeds which led in two generations to a 10% change in the size of the beak. What's going on here? How can a population change so rapidly? **How natural selection operates?** 

![Evolutionary algorithms Intuition]({{site.baseurl}}/images/reinforcement_learning_actor_only_evolutionary_algorithms_intuition_drawing.png){:class="img-responsive"}

In the general population the individuals are the results of the **crossover** between two parents. The crossover is a form of **recombination** of the parents' genetic material. In isolated population redundancy of genetic material brings to very similar individuals. However it can happen that a random **mutation** modifies the newborn's genotype generating organisms which carry on a new characteristic. For instance, in the Darwin's finches it was a stronger beak. The mutation can bring an advantage for the subject, leading to a longer life and an higher probability of **reproduction**. On the other hand if the mutation does not carry any advantage it is discarded during the selection.
That's it. Evolution selects individuals with the variations best suited to a specific environment. In this sense evolution is not "the strongest survives" but "the fittest survives".

![Evolutionary algorithms Operators Intuition]({{site.baseurl}}/images/reinforcement_learning_actor_only_evolutionary_algorithms_intuition_operators_drawing.png){:class="img-responsive"}


Evolutionary algorithms uses operators inspired by natural selection (reproduction, mutation, recombination, and selection). They are optimisers, meaning that they search for solutions to a problem directly in the solution space. The candidate solutions are single individuals like the Galapagos birds. In the standard approach a sequence of steps is used in order to evaluate and selects the best solutions:

1. **Fitness function:** it evaluates the performance of each candidate 
2. **Selection:** it chooses the best individuals based on their fitness score
3. **Recombination:** it replicates and recombines the individuals

Evolutionary algorithms are part of a broader class called [evolutionary computation](https://en.wikipedia.org/wiki/Evolutionary_computation). The generic label *"evolutionary algorithm"* applies to many techniques, which differ in representation and implementation details. Here I will focus only on **genetic algorithms**, however you should be aware that there are many variations out there.

![Evolutionary algorithms Diagram]({{site.baseurl}}/images/reinforcement_learning_actor_only_evolutionary_algorithms_diagram.png){:class="img-responsive"}

In the next section I will introduce GAs which can be considered a special type of evolutionary algorithms. I will describe all the new terms and I will compare them to the ones encountered in the past episodes.


Genetic Algorithms
-------------------

In the last section we saw how evolutionary algorithms works. In this section I will go deeper explaining how GAs work. As the name suggests GAs deals with genes. The metaphor of the selection continue to works here but it is applied to genetic material. The idea is to describe each individual of the population with a specific **genotype** (or chromosome) which is used to express the individual's features or **phenotype**. In biology the [genotype](https://en.wikipedia.org/wiki/Genotype) represents the hereditary information of an organism and the [phenotype](https://en.wikipedia.org/wiki/Phenotype) is the observed features and behaviour of that organism. The [distinction between genotype and phenotype](https://en.wikipedia.org/wiki/Genotype-phenotype_distinction) and their interaction can be easily misunderstood. The genotype is only one of the factor that can influence the phenotype. Non-inherited factor and acquired mutations are not part of the genotype but have an influence on the phenotype. 
In GAs the genotype is usually represented by **1-D array** where each value (sometimes called gene) represent a property. 

![Genetic Algorithms Genotype Representation]({{site.baseurl}}/images/reinforcement_learning_actor_only_genetic_algorithms_genotype_representation.png){:class="img-responsive"}

The genotype  can contain labels, float, integers, or bits. Type uniformity in the chromosome is generally necessary in order to use the genetic operators. Label-based genotypes can be used in **code-breaking**. GAs can search in large solution spaces in order to get correct decryption. The integer-based genotypes are used in **network topology** optimisation. A float-based genotype can contains the weights of a **neural network** and those weights can be adjusted using GAs instead of backpropagation. This kind of optimisation is the one I used to evolve e-puck robots (see introduction). Choosing the right chromosome representation is generally up to the designer.  
In the next sub-sections I will describe all the **operators** available in GAs and how they are used in the evolution process.

**Selection:** the selection operator is the core of GAs. Selection is a way to reduce the number of possible solutions choosing the one that leads to the best results. After fitness evaluation it is possible to order the chromosomes from best to worst. The ordered list is then used by the selection mechanism. There are different types of selection. **Truncated selection** is the simplest form of selection. Only a fraction of the total number of chromosomes is selected, the rest is rejected. 

![Genetic Algorithms Selection]({{site.baseurl}}/images/reinforcement_learning_actor_only_genetic_algorithms_operator_selection.png){:class="img-responsive"}

A more sophisticated form of selection is called **roulette wheel**. To each chromosomes is given a weight and this weight corresponds to a portion of a roulette wheel. Chromosomes with higher fitness will have higher weight and a larger portion of the wheel. Spinning the wheel for different times allows selecting the individuals for the next generation. The roulette wheel is nothing more than a weighted sort mechanism. You must notice that using the roulette wheel the same chromosome can be sorted multiple times and the probability of its presence in the next generation is higher. Another form of selection is called **tournament selection** and it consists in randomly selecting a group of individuals and taking only the one with the highest fitness.  
The selection process can mark a few individuals as **elite**, meaning that their genotype will be copied without variations in the new generation. This mechanism is called **elitism** and is used to keep the best solutions into the evolutionary loop.

**Mutation**: this operator is used in order to avoid uniformity in the population. The mutation randomly changes one value of the genotype. The mutation allows avoiding local minima in the search space, randomly changing the existing solutions. The simplest version of the mutation is called **point mutation** and is similar to the [biological](https://en.wikipedia.org/wiki/Point_mutation) mechanism that operates on DNA.

![Genetic Algorithms Single Point Mutation]({{site.baseurl}}/images/reinforcement_learning_actor_only_genetic_algorithms_single_point_mutation.png){:class="img-responsive"}


The probability of a mutation is generally described by a **mutation rate** which must be carefully selected. An high mutation rate can lead to the lost of the best individuals, whereas a low mutation rate leads to genotypes uniformity. For label and integer-based chromosomes, the mutation change a single gene to a value randomly picked from a predefined set. For bit-based chromosomes the value of the gene is switched to its opposite when the mutation happens. In float-based chromosomes the value can be sampled from a uniform distribution or from a Gaussian distribution centred on the gene value.

**Crossover**: this operator is similar to the [biological](https://en.wikipedia.org/wiki/Chromosomal_crossover) counterpart and it is used to recombine the genotype of the parents to generate offspring who have mixed features. The simplest form of crossover is called single-point. Given the two parents' genotypes and a random cutting point, the single-point crossover literally "cross" the two portions when generating the children.

![Genetic Algorithms Single Point Crossover]({{site.baseurl}}/images/reinforcement_learning_actor_only_genetic_algorithms_single_point_crossover.png){:class="img-responsive"}

Another version of the crossover is called multi-point crossover. The multi-point crossover selects multiple cutting points and recombines those parts in the children's genotype.

![Genetic Algorithms Multi Point Crossover]({{site.baseurl}}/images/reinforcement_learning_actor_only_genetic_algorithms_multi_point_crossover.png){:class="img-responsive"}

There are other crossover solutions that involve multiple parents, however in the next sections I will focus only on the single-point versions. 


GAs in reinforcement learning
-----------------------------
How can we use GAs in reinforcement learning? GAs can be used for **policy estimation**. In this case the 1-D genotype is a representation of the policy matrix. Here I will use again the cleaning robot example introduced in the [first post](](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html)). The reward for each state is -0.004 but for the charging station (+1.0) and the stairs (-1.0). The optimal policy for this environment is given. Here we want to see if the GAs can obtain the optimal policy through recombination of the population chromosomes. We can use an integer-based representation where to each **number** correspond an **action**: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT. For terminal states we do not really need an action, because at that state the episode ends and the action does not take place. For this reason we can represent the action taken in a terminal state using a random value.

![Genetic Algorithms Action to Index]({{site.baseurl}}/images/reinforcement_learning_actor_only_genetic_algorithms_cleaning_robot_action_index.png){:class="img-responsive"}

We can represent the **genotype** as the unrolled **policy matrix** which associate an action to each state. Starting from the state (1,1)  (columns x rows) and following the [Russel and Norvig](http://aima.cs.berkeley.edu/) convention for the grid world (the starting point is in the bottom-left corner) we associate to each action an index and we store each value in the chromosome.

![Genetic Algorithms Policy to Chromosome]({{site.baseurl}}/images/reinforcement_learning_actor_only_genetic_algorithms_cleaning_robot_policy_to_chromosome.png){:class="img-responsive"}


The starting population will have random initialised genotypes, meaning that the behaviour of each robot will be different. The **fitness** score allow us to discriminate between good and bad policies. Here the fitness value $$ f $$ is defined as the **sum of rewards** obtained at each time step $$ t $$ in a single episode:

$$ f = \sum_{t=0}^{N} r_{t} $$


The main concepts explained above can be summarised in two points:

- The **genotype** (or chromosome) represents the **policy matrix**
- The **fitness** score represents the cumulated **reward** in a single episode

The **limit** of the genetic approach is that it needs many episodes in order to converge. Moreover it cannot learn online as we saw in [temporal differencing learning](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html). For these reasons the use of GAs is generally coupled with simulated environments, where multiple instances of the world can run in parallel.

Python implementation
---------------------

The Python implementation is built upon different functions that represent the genetic operators described above. First of all we need a function to generate a **random population** of genotypes. Using the Numpy function `numpy.random.randint` it can be done in one line. 

```python
def return_random_population(chromosome_size, population_size):
    '''Returns a random initialised population

    @param chromosome_size
    @param population_size
    '''
    return np.random.randint(low=0, 
                             high=4, 
                             size=(population_size,chromosome_size))
```



The **mutation** operator iterates through all the values of all the chromosomes and for each value randomly mutate the content picking an integer from the set [0, 1, 2, 3]. This set represents the four actions available to the robot. We can iterate over Numpy array using the function `numpy.nditer()` and change the value of an element only if a random float picked from a uniform distribution is less than `mutation_rate`. 

```python
def return_mutated_population(population, mutation_rate):
    '''Returns a mutated population

    It applies the point-mutation mechanism to each value
    contained in the chromosomes.
    @param population list/array containing the chromosomes
    @mutation_rate a float repesenting the probaiblity of 
        mutation for each gene (e.g. 0.02=2%)
    '''
    for x in np.nditer(population, op_flags=['write']):
        if(np.random.uniform(0,1) < mutation_rate):
            x[...] = np.random.choice(4, 1)
    return population
```

The **roulette wheel** can be implemented easily using the Numpy function `numpy.random.choice` which takes as input an array representing the probabilities associated to each element of the input array. Because the probabilities must sum up to one we have to normalise the values contained in the fitness array using a [softmax function](https://en.wikipedia.org/wiki/Softmax_function).

```python
def return_roulette_selected_population(population, fitness_array):
  '''Returns a new population of individuals.

  Implementation of a roulette wheel mechanism. The population returned is
  obtained through a weighted sampling based on the fitness array.
  @param population
  @param fitness_array
  '''
  #Softmax to obtain a probability distribution from the fitness array.
  fitness_distribution = np.exp(fitness_array - np.max(fitness_array)) / 
                         np.sum(np.exp(fitness_array - np.max(fitness_array)))
  #Selecting the new population indeces through a weighted sampling
  pop_size = population.shape[0]
  pop_indeces = np.random.choice(pop_size, pop_size, p=fitness_distribution)

```




Conclusions
-----------

Acknowledgments
---------------

The e-puck picture is taken from the cyberbotics website [here](https://www.cyberbotics.com/features)


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

- List of genetic algorithm **applications** [[wiki]](https://en.wikipedia.org/wiki/List_of_genetic_algorithm_applications)

- **Evolutionary Algorithms for Reinforcement Learning.** Moriarty, D. E., Schultz, A. C., & Grefenstette, J. J. (1999). [[pdf]](https://www.jair.org/media/613/live-613-1809-jair.pdf)

- **Reinforcement learning: An introduction.** Sutton, R. S., & Barto, A. G. (1998). Cambridge: MIT press. [[html]](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)


References
------------

Moriarty, D. E., Schultz, A. C., & Grefenstette, J. J. (1999). Evolutionary algorithms for reinforcement learning. J. Artif. Intell. Res.(JAIR), 11, 241-276.

