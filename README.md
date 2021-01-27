# Nghiên Cứu Khoa Học

# Understanding Machine Learning
## 1. Overview:
* What machine learning is?
* Machine learning process
* Basic concepts & terminology

## 2. What is machine learning?
What machine learning does?

* Finds patterns in data
* Uses those patterns to predict the future
* Examples:
	* Detecting credit card fraud
	* Determining whether a customer is likely to switch to a competitor
	* Deciding when to do preventive maintenance on a factory robot

What does it mean to learn?

Learning requires:
* Identifying patterns
* Recognizing those patterns when you see them again

This is what machine learning does

## 3. Machine learning in a nutshell
- Data: contains pattern  
-> Feed data into a machine learning algorithm  
- Machine learning algorithm: finds patterns  
-> The algorithm generates something called a model  
- Model: recognizes patterns  
-> Application supplies data to the model  
- Application: Supplies new data to see if it matches known pattern

## 4. Why is machine learning is so hot right now?
Doing machine learning requires:
* Lots of data: (Big data)
* Lots of computer power: (Cloud computing)
* Effective machine learning algorithms  

=> All of those things are now more available than ever

## 5. Who's interested in Machine learning?
1. Business leaders: want solutions to bussiness problems
2. Software developers: want to create better applications
3. Data scientists: want to powerful, easy-to-use tools

## 6. The ethics of Machine learning?
- What if the data is biased?
- How can you explain a model's decision?

## 7. The main points
- Machine learning lets us find patterns in existing data, then create and use a model that recognizes those patterns in new data
- Machine learning has gone mainstream
	- Although it raises ethical concerns
- Machine learning can probably help your organization

## 8. The machine leaning process
* **Iterative**: In both big and small ways
* **Challenging**: It's rarely easy
* **Often rewarding**: But not always

## 9. The first problem: asking the right question
Choosing what question to ask is the most important of the process
Ask yourself: do you have the right data to answer this question?
Ask yourself: do you know how you'll measure success?

![The machine learning process](assets/process.png)

## 10. A closer look at Machine learning
- Training data
- Supervised and unsupervised learning
- Classifying machine learning problems and algorithms
- Training a model
- Testing a model
- Using a model 

### Some terminology
- Training data: the prepared data used to create a model
- Creating a model is called training a model
- Supervised learning: the value you want to predict is in the training data. The data is labeled
- Unsupervised learning: the value you want to predict is not in the training data. The data is unlabeled

### 10.1 Data preprocessing with supervised learning
![Supervised learning](assets/supervised-learning.png)

### 10.2 Categorizing machine learning problems: 
#### 1. Regression:
The problem here is that we have data, and we'd like to find a line or curve that best fits that data. Regression problems are typically supervised learning scenarios.
An example question would be something like: how many units of this product will we sell next month?
![Regression machine learning](assets/regression.png)

#### 2. Classification
Here, we have data that we want to group into classes, at least two, can be more than two classes. When new data comes in, we want to determine which class that data belongs to. This is commonly used with supervised learning and an example question would be something like: Is this credit card transaction fraudulent? When a new transaction comes in, we want to predict which class it's in, fraudulent or not fraudulent. And often, what you'll get back is not yes or no, but a probability of which class this new transaction might be in.
![Classification machine learning](assets/classification.png)

#### 3. Clustering
Here, we have data. We want to find clusters in that data. This is a good example of when we're going to use unsupervised learning because we don't have labeled data. We don't know necessarily what we're looking for. An example question here is something like what are our customer segments? We might not know these things up front, but we can use machine learning, unsupervised machine learning to help use figure that out.
![Clustering machine learning](assets/clustering.png)

### 10.3 Styles of machine learning algorithms
![Styles of machine learning algorithms](assets/algorithms.png)

### 10.4 Training a model with supervised learning
**Traing a model with supervised learning**
![Training a model](assets/training-model.png)

**Testing a model**
![Testing a model](assets/testing-model.png)

**Improving a model**
![Improve a model](assets/improve-model.png)

### 10.5 Using a model
![Using a model](assets/using-model.png)

### 10.6 Implementing machine learning: Example technologies
![Machine learning technologies](assets/technologies.png)

# Understanding Machine Learning with Python
## 1. Getting started in Machine learning
### 1.1: What is machine learning?
Machine learning in action:
* Is this email spam?
* How can cars drive themselves?
* What will people buy?
	
> Machine learning
> Build a model from example inputs to make data-driven predictions vs. following strictly static program instructions

### 1.2: Types of Machine learning?
1. Supervised:
- Value prediction
- Needs training data containing value being predicted
- Training model predicts value in new data

2. Unsupervised:
- Identify clusters of like data
- Data does not contain cluster membership
- Model provides access to data by cluster

## 2. Understanding the machine learning workflow
Machine learning workflow
> An orchestrated and repeatable pattern which systematically transforms and processes information to create prediction solutions.

Workflow:
1. Asking the right question
2. Preparing data
3. Selecting the algorithm
4. Training the model
5. Testing the model

### Machine learning workflow guidelines
- Early steps are most important: Each step depends on previous steps
- Expect to go backwards: later knowledges effects previous steps
- Data is never as you need it: data will have to be altered
- More data is better: more data => Best results
- Don't pursue a bad solution: reevalute, fix or quit

### 2.1: Asking the right question
- Deine scope (including data sources)
- Define target performance
- Define context for usage
- Define how solution will be created

> Use the Machine learning workflow to process and transform Pima Indian data to create a prediction model. This model must predict which people are likely to develop diabetes with 70% or greater accuracy.

### 2.2: Preparing your data
- Find the data we need
- Inspect and clean the data
- Explore the data
- Mold the data to Tidy data

#### Tidy Data
> Tidy datasets are easy to manipulate, model and visualize, and have a specific structure:
- each variable is a column,
- each observation is a row,
- each type of observational unit is a table

50-80% of a Machine Learning project is spent getting, cleaning, and organizing data

#### Getting data
* Google
* Government databases
* Professional or company data sources
* Your company
* Your department
* All of the above

##### Data Rule #1
Closer the data is to what you are predicting, the better

##### Data Rule #2
Data will never be in the format you need (pandas DataFrames)

##### Data Rule #3
Accurately predicting rare events is difficult

#### Data Rule #4
Track how you minipulate data

### 2.3: Selecting Your Algorithm
Algorithm Selection:
* Compare factors
* Difference of opinions about which factors are important
* You will develop your own factors

Algorithm Decision Factors:
* Learning type
* Result
* Complexity
* Basic vs enhanced

### 2.4: Training The Model
> Machine Learning Training
> Letting specific data teach a Machine Learning algorithm to create a specific prediction model

# Foundations of PyTorch

## 1. Course outline
### 1.1 Getting started with PyTorch
* Introducing neural networks and PyTorch
* Tensor operations and CUDA support

### 1.2 Gradients and Autograd library
* Gradient descent to train NNs
* Working with gradients in PyTorch

### 1.3 Dynamic computations graphs
* Pros and cons of working with each 
* STatic graph in TF vs. dynamic graph in PyTorch

## 2. Getting started with Pytorch for machine learning

> The PyTorch framework is used to build neural networks

### 2.1 Introducing Neural Networks
Machine Learning base classifier 
* **Training**: Feed in a large corpus of data classified correctly
* **Prediction**: Use it to classify new instances which it has not seen before

> "Traditional" ML-based systems rely on experts to decide what features to pay attention to

> "Representation" ML-based systems figure out by themselves what features to pay attention to.  
Neural networks are examples of such systems

### 2.2 What is a Neural Network?
* **Deep learning** Algorithms that learn what features matter. The performance of these algorithms get better as you feed them more data.
* **Neural networks** The most common class of deep learning algorithms. One reason neural networks are called deep learning because they use many layers in order to learn from input data.
* **Neurons** The fundamental building block of all neural networks. Simple building blocks that actually "learn". Neurons are the active learning units in the neural networks that actually learn from the training data that you feed in.

![Neural network](assets/neural-network.png)
Within the neural network, there are different layers and all of these layers share responsibility in understanding patterns and data.

Different layers are responsible for understanding different details in the data that is fed in.

Each layer has its own set of patterns to identify and piece together.

If you're thinking of an image classification system:
* The first layer, say layer 1, will be responsible for looking at pixels.
* Layer 2 might piece pixels together to find edges and corners.
* Layer 3 might piece edges and corners together to identify objects.
* Layer 4 might piece objects together to identify features, such as a nose or a face.

**The layers with which we interact directly are called visible layers of a neural network.**
* This is the input layer where we feed in the training data and instances for prediction.
* The output layer, which outputs the prediction of the neural network. This can be a classification label or a regression prediction.

![Visible layer in neural network](assets/visible-layer-neural-network.png)

**The layers that we as developers don't interact with directly are termed as hidden layers on the neural network.**

These are the layers that are responsible for extracting granular detail from the input, finding patterns, and piecing these patterns into higher level abstractions.

**Every layer of a neural network is composed of the active learning unit, that is a neuron**

So each layer consists of individual interconnected neurons.

Neurons receive input from neurons in the previous layer and pass the output to neurons in the next layer.

![Neurons Interconnected](assets/neurons-interconnected.png)

### 2.3 Neurons and Activation Functions
* Operation of a single neuron: 
	* A neuron is just a mathematical function that is applied to the inputs that are fed into a neuron.
	* A single neuron can have any number of inputs, and these inputs are typically represented as X. These are also the X variables, or the features, that you use to train your model.
	* The input to a neuron is typically a vector X variables or X values. The neuron then applies a mathematical function to this input and produces a scalar output Y.
	* **For an active neuron a change in input should trigger a corresponding change in the output**

	![Operation of a single neuron](assets/operation-of-single-neuron.png)

* A single neuron can be connected to any number of neurons in the next layer.
* In the below figure, the output of the first neuron is fed to the input of the second neuron.
* Every interconnection between two neurons is associated with the *weight* that is typically represented by the letter W. This *weight* denotes the significance of this particular input for the second neuron.
* **If the second neuron is sensitive to the output of the first neuron, the connection between them gets stronger**
* **Neurons that fire together are wired together**

![Operation of a single neuron - Weight](assets/operation-of-single-neuron-weight.png)

* A term that you'll hear used for this W is model parameters. All of the weights associated with all of the interconnections in your neural network are the parameters that make up your machine learning model.
* The above figure is a zoom in view of a single interconnection between two neurons. But all of the neurons in all of the layers of a neural network are connected together in exactly the same way.
* **So this entire thing is a very large computation graph** and the nodes in this enormous computation graph are just neurons. 

![The computation graph](assets/computation-graph.png)

* The interconnections between neurons are data that is the output of a neuron fed into another neuron. These are the edges in the computation graph and often referred to as answers, that is **multidimensional** data.

![The computation graph tensors](assets/computation-graph-tensor.png)

* It's easy to imagine a weight and a single connection between two neurons. Now just extend this to thousands of connections, which means thousands of neurons and you're got your neural network.

![The computation graph trained](assets/computation-graph-trained.png) 

* Each neuron only applies two simple functions to its inputs
![Two simple function](assets/two-simple-functions.png)

* Affine Transformation

The first of these functions is called the affine transformation, and it's just a linear function.

![Affine Transformation](assets/affine-transformation.png)

**The affine transformation** is only capable f learning linear relationships between the inputs and the output of the form W multiple by X plus B.
	* So, you have the X vector, that is the input to a neuron, and every X value is associated with a corresponding weight and there is a *bias* value that is also passed into the affine transformation
	* The affine transformation is just a weighted sum with a bias added of the form W1X1 + W2X2, all the way up to WnXn + b

![Affine transformation detail](assets/affine-transformation-detail.png)

* Neural network are capable of modeling very complex non-linear relationships as well. And this is made possible due to the other function that a neuron applies to its inputs, that is the activation function.
* The output of the affine transformation is fed into this **activation function**, and this is the function that helps the neuron discover non-linear relationships and data.
* There are a number of different activation functions that you can use when you're building up the layers of a neural network, and the choice of activation function is a part of designing a neural network.

![Activation function](assets/activation-function.png)

* When the activation function is the identify function. It takes the output of the affine transformation and simply passes the same output through as the output of the neuron.
* This neuron is referred to as a **linear neuron**. Such a neuron is only capable of learning linear relationships between the inputs and the output

![The identify function](assets/activation-function-identify.png)

### 2.4 Introducing PyTorch
> A deep learning framework for fast, flexible experimentation.

## 3. Working PyTorch Tensors
> Tensor: The central unit of data in PyTorch. A tensor consists of a set of primitive values shaped into an array of any number of dimensions.

* **Data is represented as Tensors**

* Scalars are essentially **0-D** tensors

* Vectors are **1-D** tensors
![1-D tensors](assets/1-D-tensors.png)

* Matrices are **2-D** tensors
![2-D tensors](assets/2-D-tensors.png)

* N-Dimensional matrices are **N-D** tensors
![N-D tensors](assets/n-D-tensors.png)

> PyTorch Tensors have been architected to make optimal use of GPUs for massively parallel computations






















