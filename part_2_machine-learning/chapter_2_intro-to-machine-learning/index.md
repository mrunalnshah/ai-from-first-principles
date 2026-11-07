---
title: "Chapter 2: Introduction to Machine Learning"
description: An introduction to machine learning and its main types
date: 2026-10-04
numbering:
  enumerator: 2.%s
---

# What is Machine Learning?

One of the best-known definitions of machine learning is credited to Arthur Samuel:

> Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed.
>
> — Arthur Samuel (1959)

:::{note} Arthur Samuel and His Checkers Program
Samuel wrote a program to play checkers and set it to play against itself, tens of thousands of times. Each game was a lesson. By noticing which board positions usually came before a win and which usually came before a loss, the program slowly built up a sense of what a strong position looks like. It then aimed for strong positions and steered clear of weak ones, improving with every batch of games.

No person could sit through that many games, but a computer never runs out of patience. With so much experience behind it, the program eventually played checkers better than Samuel, its own creator.
:::

:::{exercise}
:label: pt3-ch2-ex-1

Imagine Samuel had stopped his program after just 10 practice games against itself, rather than letting it play more than 10,000. Compared with the real program, how well would it play?

**a.** It would play better.

**b.** It would play worse.
:::

:::{solution} pt3-ch2-ex-1
:class: dropdown

**Answer: b.** It would play worse.

With only 10 games, the program would have had very little experience to learn from.
:::

:::{important} Key idea
As a rule, a learning algorithm gets better the more opportunities it has to learn.
:::

# Types of Machine Learning

Machine learning comes in several types:

1. Supervised learning, the most widely used type
2. Unsupervised learning
3. Recommender systems
4. Reinforcement learning

This chapter focuses on the first two.

# Supervised "Machine" Learning

:::{prf:definition} Supervised Learning
:label: def-supervised-learning

In **supervised learning**, an algorithm learns a mapping from inputs $x$ to outputs $y$, using examples for which the correct output is already known.
:::

Those known outputs are the "right answers." Each training example pairs an input $x$ with its correct output, which we call the **label** $y$. After learning from enough of these $(x, y)$ pairs, the algorithm can be given an input on its own, without a label, and still predict the output fairly accurately.

[](#ch02-applications-of-machine-learning-figure) shows some common applications of supervised learning.

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

All of these applications work in the same two steps:

1. **Training.** We show the model many examples of inputs $x$ together with their right answers, the labels $y$.
2. **Prediction.** Once the model has learned from those $(x, y)$ pairs, we give it a new input $x$ that wasn't among the training examples, and it produces its best guess for the matching output $y$.

Supervised learning problems come in two types: **regression** and **classification**.

## Regression

A regression model predicts a number, and there are infinitely many numbers it could output. For example, a regression model could:

- predict the price of a house from its size
- predict a student's marks from factors such as hours of sleep, hours of study, time spent watching cartoons, and so on

[](#ch02-house-price-prediction-figure) shows the first example, predicting house prices.

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

Why does this count as supervised learning? Look at the data: for every house in the plot, we already know the correct answer, its actual price, which serves as the label $y$. What we want from the algorithm is more answers of the same kind, such as an estimate of what a house your friend wants to sell is likely to be worth. Because the thing being predicted is a number, this type of supervised learning is called **regression**.

:::{prf:definition} Regression
:label: def-regression

**Regression** is supervised learning in which the output is a number that can take any of infinitely many values, such as the price of a house.
:::

## Classification

A classification model predicts a category, and there are only a few possible outcomes. For example, a classification model could:

- tell whether a picture shows a cat or a dog
- recognize which of 10 possible medical conditions a patient has

[](#ch02-breast-cancer-detection-figure) shows another example, detecting breast cancer.

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

A category doesn't have to be a number. A classifier can decide whether a picture shows a dog or a cat, or whether a tumor is benign or malignant. Categories can also be numbers, such as 0 and 1, or 1, 2, 3, and so on.

The real difference from regression lies in the outputs. A classifier must pick one answer from a short, fixed list, say 0, 1, or 2, whereas a regression model is free to output values that fall between them, like 0.5 or 1.7.

The possible outputs of a classifier are called **classes** or **categories**. The two words are often used interchangeably.

:::{prf:definition} Classification
:label: def-classification

**Classification** is supervised learning in which the output is a category chosen from a small, limited set of possibilities.
:::

## Summary

Supervised learning maps an input $x$ to an output $y$, and the algorithm learns from examples that include the "right answers." [](#ch02-regression-vs-classification-table) compares its two main types.

:::{table} Regression vs. classification
:label: ch02-regression-vs-classification-table

| | Regression | Classification |
| --- | --- | --- |
| What it predicts | A number | A category |
| Possible outputs | Infinitely many | A small, limited set |
| Examples | House prices, exam marks | Cat or dog, benign or malignant tumor |
:::

:::{exercise}
:label: pt3-ch2-ex-2

In supervised learning, the algorithm learns from examples that each come with the right answer $y$. Which of these is an example of supervised learning?

**a.** Working out the average age of a group of customers.

**b.** Filtering spam emails.
:::

:::{solution} pt3-ch2-ex-2
:class: dropdown

**Answer: b.** Spam filtering.

To train a spam filter, we give the algorithm emails that are already labeled "spam" or "not spam." It can then judge, fairly reliably, whether a new email is spam. Option a involves no learning at all: it's a simple calculation, with no right answers to learn from.
:::

# Unsupervised Learning

:::{prf:definition} Unsupervised Learning
:label: def-unsupervised-learning

In **unsupervised learning**, the data contains only inputs $x$, with no output labels $y$. With no right answers to learn from, the algorithm has to discover structure, patterns, or anything else interesting in the data on its own.
:::

[](#ch02-supervised-vs-unsupervised-figure) shows the difference between the two approaches.

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

Take the tumor example again. This time, we have data about patients, such as the size of each tumor and each patient's age, but nothing tells us whether a tumor is benign or malignant. Since there are no labels, we can't ask the algorithm to make a diagnosis. Instead, it has to explore the data and find whatever structure or patterns are hidden in it.

We call this *unsupervised* learning because we aren't supervising the algorithm: we never show it a right answer to aim for.

There are three main types of unsupervised learning problems:

1. **Clustering**: grouping similar data points together
2. **Anomaly detection**: spotting unusual events
3. **Dimensionality reduction**: compressing data into a smaller form

:::{exercise}
:label: pt3-ch2-ex-3

Which of these problems would you solve with an unsupervised learning algorithm? Choose all that apply.

**a.** You have a collection of news articles from the web, and you want to sort them into groups that each cover the same story.

**b.** You have emails labeled as spam or not spam, and you want to build a spam filter.

**c.** You have a database of customer data, and you want to discover market segments automatically and place each customer in the right one.

**d.** You have a dataset of patients, each diagnosed as either having diabetes or not, and you want to predict whether new patients have diabetes.
:::

:::{solution} pt3-ch2-ex-3
:class: dropdown

**Answers: a and c.**

- **a.** Sorting news articles into groups about the same story.
- **c.** Discovering market segments and grouping customers into them.

In both cases, the data has no labels, so the algorithm has to find the groups by itself. Options b and d are supervised learning, because every example comes with the right answer: spam or not spam, diabetes or no diabetes.
:::

## Clustering Problem

**Clustering** gathers similar data points into groups. Given unlabeled data, a clustering algorithm sorts it into clusters of items that resemble each other.

Here are two places where clustering is used.

**Google News.** Each day, Google News sorts through hundreds of thousands of articles published online and bundles stories about the same event together ([](#ch02-google-news-clustering-figure)).

:::{figure} Images/IM_0201.png
:label: ch02-google-news-clustering-figure
:alt: Example of Clustering from Google News
:width: 70%
:align: center

Example of Clustering from Google News
:::

No one gives the algorithm a list of keywords. It discovers by itself which words hint that two articles are about the same thing.

**Genetic (DNA) data.** Clustering is also used to find groups in genetic, or DNA, data ([](#ch02-dna-clustering-figure)).

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

That's the essence of clustering: as an unsupervised method, it receives data with no labels at all and works out the groups automatically.

## Anomaly Detection

**Anomaly detection** is about spotting unusual events. The algorithm scans unlabeled data and flags the data points that stand out from the rest.

In finance, for example, anomaly detection is a key tool for catching fraud, because an unusual transaction can be a sign that something is wrong. It is useful in many other applications as well.

## Dimensionality Reduction

**Dimensionality reduction** compresses a large dataset into a much smaller one while keeping as much of the important information as possible.
