# Nghiên Cứu Khoa Học

# Understanding Machine Learning
## 1. Overview:
- What machine learning is?
- Machine learning process
- Basic concepts & terminology

## 2. What is machine learning?
<p>What machine learning does?</p>
- Finds patterns in data
- Uses those patterns to predict the future
- Examples:
	- Detecting credit card fraud
	- Determining whether a customer is likely to switch to a competitor
	- Deciding when to do preventive maintenance on a factory robot
<p>What does it mean to learn?</p>
<p>How did you learn to read?</p>
<p>Learning requires:</p>
- Identifying patterns
- Recognizing those patterns when you see them again
<P>This is what machine learning does</P>

## 3. Machine learning in a nutshell
- Data: contains pattern
-> Feed data into a machine learning algorithm
- Machine learning algorithm: finds patterns
-> The algorithm generates something called a model
- Model: recognizes patterns
-> Application supplies data to the model
- Application: Supplies new data to see if it matches known pattern

## 4. Why is machine learning is so hot right now?
<p>Doing machine learning requires:</p>
- Lots of data: (Big data)
- Lots of computer power: (Cloud computing)
- Effective machine learning algorithms
<p>=> All of those things are now more available than ever</p>

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
<p>Iterative: In both big and small ways</p>
<p>Challenging: It's rarely easy</p>
<p>Often rewarding: But not always</p>

## 9. The first problem: asking the right question
<p>Choosing what question to ask is the most important of the process</p>
<p>Ask yourself: do you have the right data to answer this question?</p>
<p>Ask yourself: do you know how you'll measure success?</p>

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
<p>Traing a model with supervised learning</p>
![Training a model](assets/training-model.png)

<p>Testing a model</p>
![Testing a model](assets/testing-model.png)

<p>Improving a model</p>
![Improve a model](assets/improve-model.png)

### 10.5 Using a model
![Using a model](assets/using-model.png)

### 10.6 Implementing machine learning: Example technologies
![Machine learning technologies](assets/technologies.png)

# Foundations of PyTorch

## 1. Getting started with Pytorch for machine learning
### Overiew:
- Deep learning using neural networks
- Neurons and activation functions
- Introducing PyTorch to build neural networks
- Understanding the differences between PyTorch and TensorFlow
### Prerequisites and Course outline:
- Working with Python and Python libraries
- Basic understanding of machine learning algorithms
- No prior experience with neural networks necessary
