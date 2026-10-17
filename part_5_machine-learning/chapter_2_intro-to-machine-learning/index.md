---
title: "Chapter 2: Introduction to Machine Learning"
description:  Introduction to Machine Learning? Types of Machine Learning
date: 2026-10-04
numbering:
  enumerator: 2.%s
---


# What is Machine Learning
- Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed. - Arthur Samuel (1959)

:::{note}
**Arthur Samuel and His Checker Program**

Samuel wrote a checkers playing program. he programmed the computer to play maybe tens of thousands of games against itself. By watching what social support positions tend to lead to wins and what positions tend to lead to losses the checkers plane program learned over time what are good or bad suport positions by trying to get a good and avoid bad positions, this program learned to get better and better at playing checkers because the computer had the patience to play tens of thousands of games against itself. It was able to get so much checkers playing experience that eventually it became a better checkers player than also, Samuel himself.
:::

:::{exercise}
:label: pt3-ch2-ex-1

If Arthur Samuel's checkers-playing program had been allowed to play only 10 games against itself, how would this have affected its performance compared to when it was allowed to play over 10,000 games?

a. Would have made it better

b. Would have made it worse
:::

:::{solution} pt3-ch2-ex-1
:class: dropdown

b. Would have made it worse

In general, the more opportunities you give a learning algorithm to learn, the better it will perform.
:::

***In general, the more opportunities you give a learning algorithm to learn, the better it will perform.***


# Types of Machine Learning
- There are many types of Machine Learning:
	1. Supervised Learning (most used)
	2. Unsupervised Learning
	3. Recommender System
	4. Reinforcement Learning

# Supervised "Machine" Learning
- Supervised Learning refers to algorithms that learn `x (input)` to `y (output)` mappings.
- The key characteristic of supervised learning is that you give your learning algorithm examples to learn from.That includes the right answers, whereby right answer, I mean, the correct label y for a given input x, and is by seeing correct pairs of input x and desired output label y that the learning algorithm eventually learns to take just the input alone without the output label and gives a reasonably accurate prediction or guess of the output.
- Applications of Supervised Learning   

:::{figure}
:label: ch02-applications-of-machine-learning-figure
:no-subfigures:

```{image} diagrams/0201-light.svg
:alt: Applications of Machine Learning
:class: dark:hidden
```

```{image} diagrams/0201-dark.svg
:alt: Applications of Machine Learning
:class: hidden dark:block
```

Applications of Machine Learning
:::

- To reiterate: In all of these applications, you will first train your model with examples of inputs x and the right answers, that is the labels y. After the model has learned from these input, output, or x and y pairs, they can then take a brand new input x, something it has never seen before, and try to produce the appropriate corresponding output y.
- There are two types of Supervised Learning Problems:
	1. Regression
	2. Classification

## Regression
- Regression model predicts numbers.
	- Infinite many possible numbers.
	- Example:
		- Predicting a Housing Price from size of the house
		- Predicting marks based on factors like sleeping hours, studying hours, watching cartoons, etc.
- For Example: Housing Price Prediction
:::{figure}
:label: ch02-house-price-prediction-figure
:no-subfigures:

```{image} diagrams/0202-light.svg
:alt: House Price Prediction
:class: dark:hidden
```

```{image} diagrams/0202-dark.svg
:alt: House Price Prediction
:class: hidden dark:block
```

Example of Regression Model
:::
- What you've seen is the example of supervised learning. Because we gave the algorithm a dataset in which the so-called right answer, that is the label or the correct price y is given for every house on the plot. The task of the learning algorithm is to produce more of these right answers, specifically predicting what is the likely price for other houses like your friend's house. That's why this is supervised learning. To define a little bit more terminology, this housing price prediction is the particular type of supervised learning called **regression**.
- ***Regression** means we're trying to predict a number from infinitely many possible numbers (such as house price)*.

## Classification
- Classification model predicts categories. 
	- Small number of possible outcomes.
	- Example:
		- Predicting cats vs dogs
		- recognizing of any of 10 possible medical conditions in a patient
- For Example: Breast Cancer Detection
:::{figure}
:label: ch02-breast-cancer-detection-figure
:no-subfigures:

```{image} diagrams/0203-light.svg
:alt: Breast Cancer Prediction
:class: dark:hidden
```

```{image} diagrams/0203-dark.svg
:alt: Breast Cancer Prediction
:class: hidden dark:block
```

Example of Classification Model
:::     

- Classification algorithms predict categories. Categories don't have to be numbers.
	- It can predict if picture is dog or cat?
	- It can also predict if a tumor is benign or malignant?
	- Categories can be numbers like 0, 1 or 1, 2, 3, ...
- But what makes classification different from regression is that classification predicts a small finite limited set of possible output categories such as 0, 1 and 2 but not all possible numbers in between like 0.5 or 1.7 (which regression can find).
- In classification, the terms output classes and output categories are often used interchangeably. 
## Summarize
- Supervised learning maps input x to output y, where the learning algorithm learns from the quote right answers.
- Two major types of supervised learning are
	 1.  Regression - Algorithm has to predict numbers from infinitely many possible output numbers
	 2. Classification - Algorithm has to make a prediction of a category (small set of possible outputs)


:::{exercise}
:label: pt3-ch2-ex-2

Supervised learning is when we give our learning algorithm the right answer *y* for each example to learn from.  Which is an example of supervised learning?

a. Calculating the average age of a group of customers.

b. Spam filtering.
:::

:::{solution} pt3-ch2-ex-2
:class: dropdown

***b. Spam filtering.***

Explanation: For instance, emails labeled as "spam" or "not spam" are examples used for training a supervised learning algorithm. The trained algorithm will then be able to predict with some degree of accuracy whether an unseen email is spam or not.
:::


# Unsupervised Learning
- In Unsupervised Learning the data only comes with inputs `X`, but not output labels `y`. The algorithm has to find some structure or some pattern or something interesting in the data.
:::{figure}
:label: ch02-supervised-vs-unsupervised-figure
:no-subfigures:

```{image} diagrams/0204-light.svg
:alt: Supervised Learning vs Unsupervised Learning
:class: dark:hidden
```

```{image} diagrams/0204-dark.svg
:alt: Supervised Learning vs Unsupervised Learning
:class: hidden dark:block
```

Supervised Learning vs Unsupervised Learning
:::    
- In Unsupervised Learning, We are given data that isn't associated with any output labels y, say you're given data on patients and their tumor size and the patient's age. But not whether the tumor was benign or malignant. We're not asked to diagnose whether the tumor is benign or malignant, because we're not given any labels. Why in the dataset, instead, our job is to find some structure or some pattern or just find something interesting in the data. This is Unsupervised Learning.
- we call it unsupervised because we're not trying to supervise the algorithm. 
- There are types of Unsupervised Learning Problems
	1.  Clustering Problem 
	2. Anomaly Detection - Used to detect unusual events.
	3. Dimensionality Reduction

:::{exercise}
:label: pt3-ch2-ex-3

Of the following examples, which would you address using an unsupervised learning algorithm?  (Check all that apply.)

a. Given a set of news articles found on the web, group them into sets of articles about the same stories.

b. Given email labeled as spam/not spam, learn a spam filter.

c. Given a database of customer data, automatically discover market segments and group customers into different market segments.

d. Given a dataset of patients diagnosed as either having diabetes or not, learn to classify new patients as having diabetes or not.
:::

:::{solution} pt3-ch2-ex-3
:class: dropdown

***a. Given a set of news articles found on the web, group them into sets of articles about the same stories.***

***c. Given a database of customer data, automatically discover market segments and group customers into different market segments.***
:::


## Clustering Problem
- Unsupervised Clustering Learning groups similar data points together.
- Clustering algorithm places the unlabeled data, into different clusters or groups.
- Clustering is used in
	- Google News: Google news does is every day it goes. It looks at hundreds of thousands of news articles on the internet, and groups related stories together.

    ```{image} Images/IM_0201.png
    :alt: Example of Clustering from Google News 
    :width: 70%
    :align: center
    ```
	- Clustering algorithm figures out on his own which words suggest, that certain articles are in the same group.
	- Clustering Genetics or DNA data

	:::{figure}
	:label: ch02-dna-clustering-figure
	:no-subfigures:

	```{image} diagrams/0205-light.svg
	:alt: DNA Clustering Example
	:class: dark:hidden
	```

	```{image} diagrams/0205-dark.svg
	:alt: DNA Clustering Example
	:class: hidden dark:block
	```

	Example of Clustering in Unsupervised Learning
	:::  

	- A clustering algorithm, which is a type of unsupervised learning algorithm, takes data without labels and tries to automatically group them into clusters.

## Anomaly Detection
- Anomaly Detection is used to detect unusual events. 
- Unsupervised Anomaly Detection finds unusual data points.
- This turns out to be really important for fraud detection in the financial system, where unusual events, unusual transactions could be signs of fraud and for many other applications.

## Dimensionality Reduction
- Dimensionality Reduction takes big dataset and compress it to a much smaller dataset while losing as little information as possible.