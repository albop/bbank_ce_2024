---
title: Quick Introduction to deep learning
---

# Introduction

## Outline

- General discussion:
  - machine learning / artificial intelligence
  - technology / technology

- Introduction to deep learning
  - Stochastic Gradient Descent
  - Neural Networks

- Practical Application

- Application to an economic model

## We are in the big-data era

- Lots of data (consumers, regulators, flights, ...)

- Traditionnel models (econometrics for instance) fail
  - data is heterogenous
  - doesn't fit in the memory
  - doesn't match a model...

- Many machine learning algorithms

- Hardware developments: (GPU, online computing)
- Software stack: linux/git/python/scikit/tensorflow...


## What about deep learning ?

- Deep learning: hugely successful in many different areas:
  - a non-linear structure which self-organizes itself to reach a goal
  - compare with artificial intelligence (achieve a goal without being told how)

- Challenges: transferability, explainability


## Another step towards AI: reinforcement learning

- Reinforcement learning:
  - solve a dynamic problem by experimenting with it
  - actions have a cost: tradeoff (exploiting/exploring)
  - close to economic RE formulation: $\max \beta^t r_t$
- Successful applications to:
  - driving, trajectory optimizations
  - tic-tac-toe, backgammon
  - chess, go ! (combined with deep-learning)


# The basics of deep learning

## Optimization

Check http://ruder.io/optimizing-gradient-descent/ for more details

## Gradient descent

Consider the scalar function $\Xi(\theta)$ where $\theta$ is a vector.

Define $\nabla\_{\theta}=   \begin{bmatrix} \frac{\partial}{\partial \theta_1} \\\\...\\\\\frac{\partial}{\partial \theta_n} \end{bmatrix}$

- Gradient descent: go to steepest slope

  $\theta \leftarrow \theta - \gamma \nabla\_{\theta}\Xi(\theta)$

## Variants of Gradient Descent

- Momentum: (ball goes down the hill, $\gamma$ is air-resistence)

  $$v\_t = \gamma v\_{t-1} + \eta \nabla_{\theta} J(\theta)$$
  $$\theta \leftarrow \theta - v_t$$

- Nesterov Momentum: (slow down before going up...)

  $$v\_t = \gamma v\_{t-1} + \eta \nabla\_{\theta} J\left(\theta-\gamma v\_{t-1}\right)$$
  $$\theta \leftarrow \theta - v_t$$
----

## Variants of Gradient Descent (2)

- Learning rate annealing

  $$\eta_t = \eta_0 / ({1+\kappa t})$$

- Parameter specific updates (ADAM) 

  $$m_t = \beta\_1 m\_{t-1} + (1-\beta_1) g\_t$$
  $$v\_t = \beta\_2  v\_{t-1} + (1-\beta\_2) g\_t^2$$

  $$\theta_\{t+1} \leftarrow \theta_t-\frac{\eta}{\sqrt{\frac{v_t}{1-\beta_2^t}+\epsilon}}\frac{m_t}{1-\beta_1^t}$$

 - see AdaGrad, AdaMax, Rmsprop

## Problem 1: overshooting

![](contours_evaluation_optimizers.gif)


## Problem 2: saddle points

![](saddle_point_evaluation_optimizers.gif)


## Stochastic gradient descent

- Given  $\epsilon \sim \mathcal{D}$, minimize $$\Xi(\theta)=E J(\theta,\epsilon)$$

- Idea: draw a random $\epsilon_t$ at each step and do:
    $$\theta\leftarrow \theta + \gamma J(\theta,\epsilon_t)$$

- It works !
  - Reason: $\nabla\_{\theta}\Xi = E\left[ \nabla\_{\theta} J(\theta,\epsilon) \right]$
  -  $\gamma$ small, the cumulative last steps ($\sum\_k \gamma^k \nabla\_{\theta} J(\theta\_{t-k},\epsilon\_{t-k})$ ) are close to unbiased
  - logic extends to other GD algorithms


## Stochastic gradient descent (2)

- Can escape local minima (with annealing)

- Gradient is estimated from a random mini-batch $(\epsilon\_1, ... \epsilon\_{N_m})$

  $$\theta\leftarrow \theta + \gamma \sum_{i=1:N_m} \nabla J(\theta,\epsilon^i)$$

- Common case: dataset set finite $(\epsilon_1, ..., \epsilon_N)$

  - batch gradient: full dataset
  - mini-batch gradient: random (shuffled)  $(\epsilon\_i)\_{i \in 1:N_m}$
  - stochastic gradient: one point


## Neural networks
http://cs231n.github.io/neural-networks-1/

----

## Clarifications:
- neural networks and neurons in our brains are two distinct concepts
- brains do things in different ways (e.g. [long range connections]( http://mcgovern.mit.edu/news/videos/the-brains-long-range-connections/))
- NN are fast and their organization is not always inspired by the brain


## The Perceptron

First neural neuron, the perception, invented by a psychologist, Dr Rosenblatt (1952):

![](Perceptron.jpg)

## Modern neural model

Evolution of the model:
- arbitrary activation function $f$

<div id="left">
![](neuron.png)
<div>

<div id="right">
![](modern_neuron_model.jpeg)
</div>

## Activation functions

- heavyside function: $h(x)=1_{x \geq 0}$
- sigmoid (logistic): $\sigma(x)=\frac{1}{1+e^{-x}}$
    - cool fact: $\sigma^\prime(x)=\sigma(x)(1-\sigma(x))$
- arctan
- relu: $r(x)=x*1_{x \geq 0}$
- leaky relu: $r(x)=\kappa x 1\_{x < 0} + 1\_{x \geq 0}$


## Activation functions

![](activation_functions.png)


## Topologies (feed forward 1)

- Multilayer perceptron:

  $$A_0(\tau_1 (A_1 \tau_2 (A_2 (.... ) + B_2)) )+ B_1$$

where the $\tau_i$ are activation functions, A the weights, B the biases.

![](multilayer_perceptron.png)

## Topologies (feed forward 2)

- Convolutional networks:
    - recognize/classify images
    - used for features extraction in other algos
- Encoder

![](lenet5.png)

## Topologies (others)

- Recurrent neural network
  - have a memory
  - useful for sequential data (speech, text)

![](Recurrent_neural_network_unfold.svg)


## Training a neural network (1)

Two elements:
  - choose a network structure
  - specify an objective
  - optimize objective w.r.t. deep parameters (weights and biases)
  - test out of sample

## Training a neural network (2)

- Choose the right objective:
- regress data $y_i$ on input $x_i$: 
  $$\sum_i (f(x_i;\theta)-y_i)^2$$
   - possible regularization term to avoid overfitting
   - $y_i$ can be (0,1) for logistic regression
- encode data $\sum_i (x_i - f(x_i;\theta))^2$

## Training a neural network (3)

Practical challenges:
  - difficulties in computing derivatives in a numerically stable way
  - computational intensity grows fast with number of layers
  - finding good network structure and hyper-parameters is hard

## The deep learning technological stack

- automatic differentiation (reverse accumulation)
- vectorization (operates on small batches at a less than proportional cost)
  - processors: SSE, AVX, ...
  - GPUs: cuda (1920 cores on gtx 1070)
- parallelization
  - scales to very big architectures
- software (+ scientific python):
  - tensorflow, theano, ...
  - scikit.learn, keras


## Mental pause

“The Navy revealed the embryo of an electronic computer today that it expects will be able to walk, talk, see, write, reproduce itself an be conscious of its existence … Dr. Frank Rosenblatt, a research psychologist at the Cornell Aeronautical Laboratory, Buffalo, said Perceptrons might be fired to the planets as mechanical space explorers”

https://www.youtube.com/watch?time_continue=83&v=aygSMgK3BEM



## Mental Pause

Applications of deeplearning:

http://www.yaronhadad.com/deep-learning-most-amazing-applications/


## Mental Pause

Go out and play !



# Deep learning an economic model

based on *Deep Learning for Solving Dynamic Economic Models* with Lilia and Serguei Maliar.


## Motivation (1)

Meta-question:
- is RE (e.g. consumption savings) a complicated problem ?
    - must be simpler than go...
    - yet we (and computers) have a hard time solving RE problems
    - if a Neural network can solve it provided the right objective, it must be simple, right ?


## Motivation (2)

- using machine learning technology should provide fast solution
    - allows easy vectorization/parallelization

- algorithm *might* be rather model independent

- NL could reduce intrinsic dimension of high-dimensional problems and make them tractable


## The problem

#### Deterministic neo-classical model:

Controlled States:
- $$a\_t=\left(1-\rho\right)a\_{t-1}$$
- $$k\_t=\left(1-\delta\right)k\_{t-1} + i\_{t-1}$$

Objective:
- $$\max\_{i_t(a_t,k_t)} \sum\_t \beta^t U(c\_t)$$
- where $c_t = k\_t^{\theta} - i\_t$



## Euler equation

$$1 = \beta \frac{U^{\prime}(c\_{t+1})}{U^{\prime}(c\_{t})}\left[\left(1-\delta \right)+\theta k^{\theta-1}\right]$$

## Loss criterium

- Suppose:
  $$i(a,k) = \varphi(a,k;\theta)$$
- $\varphi$ can be any parameterized family of functions:
    - complete polynomials
    - neural networks
- Rewrite Euler equation as:
    $$R(a,k;\theta) = 0$$
- Loss criterium
    $$\Xi(\theta) = \int_{a,k\in G} R(a,k;\theta)^2$$


## Loss criterium 2

- For any random distribution of $a,k$ consider:

$$\Xi(\theta) = E\_{a,k} R(a,k;\theta)^2$$

- Given $N_b$ random values:

$$\Omega\_{N\_b}(\theta) = \frac{1}{N\_b} \sum\_{a_n,k_n} R(a,k;\theta)^2$$


## Loss criterium 3

- Using tensorflow we get easily:
$$\nabla\_\theta \Omega\_{N\_b}(\theta)$$

- The gradient is not biased:
  $$E\_{{a\_n},k\_{n}} \nabla\_\theta \Omega\_{N\_b}(\theta)= \nabla\_\theta \Xi(\theta)$$

## Batch vs Mini-batch
![](samples.png)


## Training

![](capital_results.png)
- Algorithm is variant of SGD: ADAM

## Conclusion

- Convergence obtained relatively *fast*:
  - 1000 iterations but $1000 N\_m$ points visited vs $1000 G$ for time_iteration

- Extends to any number of dimensions
- Neural network can (in principle) account for non- differentiabilities

- Can be extended to stochastic case (see paper)
