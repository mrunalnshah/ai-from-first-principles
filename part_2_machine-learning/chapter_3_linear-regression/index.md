---
title: "Chapter 3: Linear Regression"
description: Implement Linear Regression from scratch
date: 2026-10-11
numbering:
  enumerator: 3.%s
---

In this chapter, we build linear regression from scratch: first with a single feature, then with multiple features. Along the way, we will see how the model's parameters, the cost function, and gradient descent work together, and develop the intuition behind each one before turning it into code.

A regression model predicts a number. Types of regression models include:

- Linear regression model
- Polynomial regression model

# Architecture

:::{figure}
:label: ch03-architecture-figure
:no-subfigures:

```{image} diagrams/0300-light.svg
:alt: architecture of Linear Regression
:class: dark:hidden
```

```{image} diagrams/0300-dark.svg
:alt: architecture of Linear Regression
:class: hidden dark:block
```
:::

**Linear regression model**

$$
\hat{y} = f_{\vec{w}, b} (\vec{X}) = \vec{w} \cdot \vec{X} + b
$$

**Cost function**

$$
J(\vec{w}, b) = \sum_{i=1}^{m} L(f(\vec{w}, b)(\vec{X}), y) + R_{w} + R_{b}
$$

Here, $L$ is the loss function. I have kept the $\frac{1}{m}$ inside the loss function, but to generalize, we can also write it outside the summation. So, in general, the equation above can also be written as:

$$
J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} L(f(\vec{w}, b)(\vec{X}), y) + R_{w} + R_{b}
$$

**Gradient descent**

\begin{align}

& \text{repeat until convergence \{} \\
& w_1 = w_1 - \alpha \frac{\partial}{\partial w_{1}} J(\vec{w}, b) \\

& w_2 = w_2 - \alpha \frac{\partial}{\partial w_{2}} J(\vec{w}, b) \\

& . \\
& . \\
& . \\

& w_n = w_n - \alpha \frac{\partial}{\partial w_{n}} J(\vec{w}, b) \\ \\ \\

& b = b - \alpha \frac{\partial}{\partial b} J(\vec{w}, b) \\ \\

& \text{\}} \text{  simultaneously update} \\ \\ \\

\end{align}

In these equations:

:::{table} Symbols used in this chapter
:label: ch03-architecture-symbols-table

| Symbol | Meaning |
| --- | --- |
| $\vec{w}$ | Weights |
| $b$ | Bias |
| $\vec{X}$ | Features |
| $y$ | Target (output) |
| $\hat{y}$ | Prediction |
| $L$ | Loss function |
| $J$ | Cost function |
| $R_x$ | Regularization term for parameter $x$ (here, $R_w$ and $R_b$) |
:::

# Linear Regression Model with Single Feature

A linear regression model fits a straight line to the data.

As an example problem, suppose we want to predict the price of a house from its size. The following graph shows a dataset of houses, with size in square feet on the horizontal axis and price in thousands of dollars on the vertical axis.

:::{figure}
:label: ch03-01
:no-subfigures:

```{image} diagrams/0301-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0301-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

Picture yourself as a real estate agent. A client wants to sell her house and asks what price she can expect. This dataset can help you make an estimate. You measure her house: it's 1,250 square feet. What could a house that size sell for?

**To answer that, you can build a linear regression model from this dataset.** The model fits a straight line through the data, something like this:

:::{figure}
:label: ch03-02
:no-subfigures:

```{image} diagrams/0302-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0302-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

Start at 1,250 square feet on the horizontal axis, move up until you reach the best-fit line, and then move across to the vertical axis on the left. The price there is about \$220,000.

This is an example of a regression problem in supervised learning.

Besides the plot, there's one other useful way to look at the data. In the next figure, the plot on the left shows the data visually, and the table on the right shows the same data as numbers.

:::{figure}
:label: ch03-03
:no-subfigures:

```{image} diagrams/0303-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0303-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

The data consists of inputs, the size of each house, and outputs, the price of each house. Each row of the table is one diamond in the plot, so a table with 47 rows means 47 diamonds in the plot.

## Process

:::{figure}
:label: ch03-04
:no-subfigures:

```{image} diagrams/0304-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0304-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

Everything starts with the **training set**. For each example, it contains an input feature (here, a house's size) and the matching output target (that house's price). These targets are the right answers the model learns from.

Training works like this: the learning algorithm receives the entire training set, inputs and targets together, and from it produces a function. We call this function $f$, short for "function." In older texts, you'll also see it called a *hypothesis*.

Given a new input $X$, the function $f$ produces an estimate called $\hat{y}$ (pronounced "y-hat"). Machine learning uses the hat to mark predictions: $\hat{y}$ is always the model's guess for $y$. In this vocabulary, $f$ is the **model**, $X$ is the **input feature** (or simply the input), and $\hat{y}$ is the **prediction**, the model's estimate of $y$.

Without the hat, $y$ means the **target**: the true value recorded in the training set. A prediction $\hat{y}$, on the other hand, is only an estimate, and it may or may not equal the true value.

Think back to your client: nobody knows the true price of her house until it actually sells. What your model $f$ gives you, based on the size, is an estimate, a prediction of what that true price will turn out to be.

This raises a key design question for any learning algorithm: how should we represent $f$? Put differently, which math formula will we use to compute it?

For now, we'll make $f$ a straight line. The function can then be written as:

$$
f_{w,b}(X) = wX + b
$$

where $w$ and $b$ are numbers called the **weight** and the **bias**. Together, they determine the prediction $\hat{y}$ for a given input feature $X$.

The subscripts in $f_{w,b}(X)$ tell us that $f$ takes $X$ as its input, and that its output, the prediction $\hat{y}$, depends on the values of $w$ and $b$. For brevity, we sometimes leave out the subscripts and write $f(X)$ instead:

$$
f(X) = wX + b
$$

The next figure plots the training set, with the input feature $X$ on the horizontal axis and the output target $y$ on the vertical axis. Remember, the algorithm learns from these points and produces the best-fit line.

:::{figure}
:label: ch03-05
:no-subfigures:

```{image} diagrams/0305-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0305-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

In other words, $f(X) = wX + b$ predicts $y$ with a straight-line function of $X$.

:::{note} Why a straight line?
You might wonder why we use a linear function (linear is just a fancy word for a straight line) instead of a non-linear function such as a curve or a parabola. Sometimes you do want to fit more complex, non-linear functions, such as a curve. But a straight line is simple and easy to work with, which makes it a good foundation before we move on to more complex models.
:::

This model is called **linear regression**. Since it has only one input variable, the size of the house, it's more specifically **linear regression with one variable**. You'll also hear it called **univariate linear regression**: in Latin, *uni* means "one," and *variate* means "variable."

:::{prf:definition} Univariate Linear Regression
:label: def-univariate-linear-regression

**Univariate linear regression**, or linear regression with one variable, predicts the target as a straight-line function of a single input feature $X$: $f_{w,b}(X) = wX + b$.
:::

## Cost Function

To improve a model, we first need a way to measure how well it's doing. That's the job of the **cost function**.

Each choice of $w$ and $b$ gives a different function $f_{w,b}(X)$, or $f(X)$ for short, and therefore a different line on the graph:

:::{figure}
:label: ch03-06
:no-subfigures:

```{image} diagrams/0306-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0306-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

Here's our training set again:

:::{figure}
:label: ch03-07
:no-subfigures:

```{image} diagrams/0305-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0305-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

In linear regression, we want to pick values for the parameters $w$ and $b$ so that the line defined by $f$ fits the data well, like the line in the next figure. Visually, a line "fits well" when it passes through, or close to, the training examples, compared with other lines that stay farther away from them.

:::{figure}
:label: ch03-08
:no-subfigures:

```{image} diagrams/0307-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0307-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

So how do we find values of $w$ and $b$ that make the prediction $\hat{y}^{(i)}$ close to the true target $y^{(i)}$ for many, or ideally all, of the training examples $(X^{(i)}, y^{(i)})$?

**Goal:** find $w, b$ such that $\hat{y}^{(i)}$ is close to $y^{(i)}$ for all $(x^{(i)}, y^{(i)})$.

### Building the Cost Function

To reach that goal, we first need a way to measure how well a line fits the training data. We build this measure, the cost function, in five steps:

1. **Compute the error.** For a training example $i$, subtract the target from the prediction: $\hat{y}^{(i)} - y^{(i)}$. This difference is called the **error**. It measures how far off the prediction is.
2. **Square the error.** We compute this squared error term for every training example $i$ in the training set.
3. **Add up the squared errors.** To measure the error across the entire training set, we sum the squared errors from $i = 1$ up to $m$, where $m$ is the number of training examples (47 in our dataset).
4. **Take the average.** A larger training set means more terms in the sum, so the total grows just because there's more data. To build a cost function that doesn't grow with the size of the training set, we divide by $m$, turning the total squared error into the average squared error.
5. **Divide by 2.** By convention, machine learning practitioners divide by $2m$ rather than $m$. The extra division by 2 only makes some of our later calculations neater; the cost function works whether you include it or not.

Mathematically:

$$
\frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
$$

where $m$ is the number of training examples.

This expression is the cost function, which we write as $J(w,b)$:

:::{prf:definition} Squared Error Cost Function
:label: def-squared-error-cost-function

$$
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
$$

This is called the **squared error cost function** because it squares the error terms.
:::

Since the prediction $\hat{y}^{(i)}$ is the output of the model, $f_{w,b}(X^{(i)})$, we can also write $J(w,b)$ as:

$$
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w, b}(X^{(i)}) - y^{(i)})^2
$$

Many other cost functions exist, and different applications call for different ones. For linear regression, though, and for regression problems more broadly, the squared error cost function is by far the most common choice, since it works well in so many applications.

### Intuition

What does it mean when $J(w,b)$ is large, and what does it mean when it's small? How does the cost function affect the model?

:::{figure}
:label: ch03-09
:no-subfigures:

```{image} diagrams/0308-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0308-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

### Simplified (bias = 0)

To make $J$ easier to visualize, let's work with a simpler version of the linear regression model, with the bias $b$ set to 0.

:::{figure}
:label: ch03-10
:no-subfigures:

```{image} diagrams/0309-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0309-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

Let's see how the cost changes as we choose different values for the parameter $w$. To do that, we'll look at graphs of the model $f(X)$ and of the cost function $J(w)$ side by side.

Notice that the two graphs plot different things. On the left, $w$ is held at a constant value, so $f_w$ varies only with the input $X$: change $X$, and the estimated value of $y$ changes. On the right, the cost $J$ varies with $w$, the parameter that sets the slope of the line $f_w$. In short:

- $f_w(X)$ is a function of $X$ (for a fixed $w$); $X$ is the input.
- $J(w)$ is a function of $w$; $w$ is the parameter.

The next figure shows how different values of $w$ affect the cost.

:::{figure}
:label: ch03-11
:no-subfigures:

```{image} diagrams/0310-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0310-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

Every value of $w$ gives a different straight line $f(X)$ on the left-hand graph, and each of those lines has its own cost $J(w)$. Plotting these costs point by point traces out the curve on the right.

So how do we pick the value of $w$ that makes $f$ fit the data well? A good bet is the value of $w$ that makes $J(w)$ as small as possible. **Since $J$ measures how large the squared errors are, the value of $w$ that minimizes $J$ makes those errors as small as possible, and that gives us a good model.** In this example, $w = 1$ is a great choice.

In the general case, with both parameters $w$ and $b$, we look for the values of $w$ and $b$ that minimize $J$.

:::{figure}
:label: ch03-12
:no-subfigures:

```{image} diagrams/0311-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0311-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

:::{exercise}
:label: pt3-ch3-ex-1

Which of these tells you that your choice of the parameter $w$ fits the data relatively well, compared with other choices of $w$?

**a.** The input $X$ is at or near its minimum value.

**b.** The prediction $f_w(X)$ is at or near its minimum for every value of $X$ in the training set.

**c.** The parameter $w$ is close to zero.

**d.** The cost $J$ is at or near its minimum.
:::

:::{solution} pt3-ch3-ex-1
:class: dropdown

**Answer: d.** The cost $J$ is at or near its minimum.

Linear regression looks for the parameters ($w$, or $w$ and $b$) that make the cost $J$ as small as possible, so a value of $w$ near the minimum of $J$ gives a good fit.
:::

:::{important} Key idea
Training a linear regression model means searching for the parameters, $w$ or $w$ and $b$, that give the lowest possible cost $J$.
:::

### Full Version

:::{figure}
:label: ch03-13
:no-subfigures:

```{image} diagrams/0308-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0308-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

In the simplified version, we set $b = 0$. Now let's return to the original model, with both parameters $w$ and $b$, and no longer force $b$ to be 0. Here is the training dataset:

:::{figure}
:label: ch03-14
:no-subfigures:

```{image} diagrams/0312-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0312-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

Earlier, with just $w$, the cost was a U-shaped curve, a bit like the profile of a soup bowl. Adding $b$ as a second parameter makes the picture a little more complex: the cost becomes a surface over all combinations of $w$ and $b$, and that surface is shaped like a whole soup bowl, in three dimensions instead of two.

:::{figure}
:label: ch03-15
:no-subfigures:

```{image} diagrams/0313-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0313-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

#### Contour Plots

3D surface plots can be hard to read, so here's another way to draw $J$ that's useful for visualization: a **contour plot**.

You've probably seen the idea on a hiking map. The lines on a topographical map each trace one height on the landscape, as if someone had sliced the mountain horizontally at that level. The next figure shows such a contour map of Mount Fuji.

:::{figure}
:label: ch03-16
:no-subfigures:

```{image} diagrams/0314-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0314-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

In the next figure, the upper-right panel is a contour plot of the very same cost function that's drawn as a 3D surface at the bottom, with $w$ on the horizontal axis and $b$ on the vertical axis. Each oval, or ellipse, marks a set of points on the 3D surface that are all at exactly the same height.

:::{figure}
:label: ch03-17
:no-subfigures:

```{image} diagrams/0315-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0315-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

Put differently, all the points on one ellipse share the same value of $J$. To make a contour plot, imagine taking a knife to the 3D surface and slicing it horizontally. Each slice gathers all the points at one height, and that slice appears on the contour plot as one ellipse. So the three points marked on the same ellipse all have the same cost $J$, despite having different combinations of $w$ and $b$.

The upper-left panel shows that these three points also correspond to three different functions $f$, and all three are actually pretty poor at predicting housing prices. The bottom of the bowl, where $J$ is at its minimum, sits at the center of the concentric ellipses.

Contour plots, then, are a convenient way to show the 3D cost function $J$ in just two dimensions. **Remember that the minimum is at the center of the smallest ellipse.**

#### Trying Different Parameter Values

Let's look at a few important visualizations on our house price dataset.

**For $w = -0.15$ and $b = 800$:**

:::{figure} Images/IM_0401.png
:label: ch03-contour-fit-1
:alt: Fitted line and contour plot for w = -0.15 and b = 800
:width: 70%
:align: center
:::

**For $w = 0$ and $b = 360$:**

:::{figure} Images/IM_0402.png
:label: ch03-contour-fit-2
:alt: Fitted line and contour plot for w = 0 and b = 360
:width: 70%
:align: center
:::

**For $w = 0.13$ and $b = 71$:**

:::{figure} Images/IM_0403.png
:label: ch03-contour-fit-3
:alt: Fitted line and contour plot for w = 0.13 and b = 71
:width: 70%
:align: center
:::

This last line fits the training set pretty well. On the contour plot, the point representing its cost is very close to the center of the smallest ellipse. It's not exactly at the minimum, but it's close.

I hope these figures give you a feel for how each choice of parameters changes the line $f(X)$ and its cost $J$, and how better-fitting lines sit at points on the graph of $J$ that are closer to the minimum of $J(w,b)$.

Reading a contour plot by hand to find the best $w$ and $b$ isn't a practical method, and it won't work at all once we get to more complex machine learning models. What we really need is an efficient algorithm, written in code, that automatically finds the values of $w$ and $b$ that give the best-fit line, that is, the values that minimize $J$. That algorithm is called **gradient descent**.

Because the cost function squares the errors, its "error surface" is convex, like a soup bowl. That means it always has a minimum, and we can reach it by following the gradient in every dimension.

:::{figure} Images/MD - 04 - 09.png
:label: ch03-convex-error-surface
:alt: Convex, bowl-shaped error surface of the squared error cost function
:width: 70%
:align: center
:::

## Gradient Descent

Few algorithms matter more in machine learning than gradient descent. Versions of it are used to train not only linear regression, but also some of the biggest and most complex models in all of AI.

We want a systematic way to find the values of $w$ and $b$ that give the smallest possible cost $J(w,b)$, and gradient descent gives us exactly that.

**Overview**

- You have a cost function $J(w,b)$ that you want to minimize.
- Gradient descent isn't limited to linear regression: it can try to minimize any function. For example, if the cost $J$ depends on $w_1, w_2, \dots, w_n$ and $b$, the objective is to choose values for $w_1$ through $w_n$ and $b$ that make $J$ as small as possible, and gradient descent can do that too.
- Begin with a guess for $w$ and $b$. For linear regression, the starting point hardly matters, so setting both to 0 is a common choice.
- From there, nudge $w$ and $b$ step by step in whatever way lowers $J(w, b)$, and stop once $J$ reaches, or comes close to, a minimum.

:::{prf:definition} Gradient Descent
:label: def-gradient-descent

**Gradient descent** is an algorithm that minimizes a function, such as the cost $J(w,b)$, by starting from an initial guess and repeatedly changing the parameters a little at a time in the direction that reduces the function's value.
:::

:::{note}
Not every cost function is shaped like a bowl or a hammock. When it isn't, it can have more than one minimum.
:::

The next figure gives an overview of the process.

:::{figure}
:label: ch03-18
:no-subfigures:

```{image} diagrams/0316-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0316-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

### Example of Gradient Descent - Moving Down the Hill

To see what gradient descent is doing, let's use a more complex surface plot of $J$.

:::{figure} Images/MD - 04 - 04.png
:label: ch03-hilly-cost-surface
:alt: Hilly cost surface J(w, b) that is not a squared error cost function
:width: 70%
:align: center
:::

Note that this is *not* a squared error cost function. For linear regression with the squared error cost function, the surface is always shaped like a bowl or a hammock. Surfaces like this one show up when training neural networks.

The two horizontal axes are $w$ and $b$. Pick any pair of values, and the height of the surface above that spot is the cost $J(w,b)$: the higher the surface, the larger the cost.

Now picture this surface as a gently hilly park or golf course: the high points are hills, and the low points are valleys. Imagine you're standing somewhere up on one of the hills. (If it helps you relax, add soft green grass, flowers, and a few butterflies.)

- Your goal is to get from where you're standing down to the bottom of one of the valleys as efficiently as possible.
- Gradient descent works like this: you turn all the way around, 360 degrees, and ask yourself, "If I take one tiny baby step, which direction gets me downhill the fastest?"
- Looking around, you find the direction that leads downhill most steeply from where you stand.
- Mathematically, this is the **direction of steepest descent**: a tiny step this way takes you down faster than a tiny step in any other direction. You take that step and arrive at a new, lower point on the hill.
- Then you repeat the process. From the new spot, you look all around again, pick the steepest way down, and take another small step. Step after step, you keep descending until you reach the bottom of the valley, a **local minimum**.
- What you just did was run several steps of gradient descent. And gradient descent has an interesting property. Recall that your starting point on the surface depends on the starting values you pick for $w$ and $b$.

:::{figure} Images/MD - 04 - 05.png
:label: ch03-gd-first-run
:alt: Gradient descent steps from a starting point down into a valley
:width: 70%
:align: center
:::

Now suppose you run gradient descent again, but this time you start just a couple of steps to the right of your first starting point. You follow the same process: look around, step in the direction of steepest descent, look around again, and so on. This time, though, you end up in a completely different valley, at a different minimum on the right.

:::{figure} Images/MD - 04 - 06.png
:label: ch03-gd-second-run
:alt: Gradient descent from a nearby starting point ending in a different local minimum
:width: 70%
:align: center
:::

The bottoms of both valleys are called **local minima**. Once you start descending into one valley, gradient descent keeps you there: it won't carry you over into the other valley. Where you end up depends on where you start, which is an interesting property of gradient descent.

### Gradient Descent Algorithm & Its Implementation

Let's look at the math you can implement to make gradient descent work.

**Algorithm**

$$
\begin{aligned}
repeat \ until \ convergence \  \{ \\
w = w - \alpha \frac{\partial}{\partial w}J(w, b) \\
b = b - \alpha \frac{\partial}{\partial b}J(w, b) \\
\}
\end{aligned}
$$

Solving the derivatives with calculus gives these equations:

$$
\begin{aligned}
\text{repeat until convergence} \{ \\
w = w - \alpha \frac{1}{m}\sum_{i=1}^{m} (f_{w,b}(X^{(i)}) - y^{(i)}) X^{(i)} \\
b = b - \alpha \frac{1}{m}\sum_{i=1}^{m} (f_{w,b}(X^{(i)}) - y^{(i)}) \\
\}
\end{aligned}
$$

:::{figure}
:label: ch03-19
:no-subfigures:

```{image} diagrams/0317-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0317-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

:::{figure}
:label: ch03-20
:no-subfigures:

```{image} diagrams/0318-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0318-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

Here's what each part of the update means:

- **Equals sign ($=$):** It sets the left-hand side to the value of the right-hand side. It's an assignment, not a comparison.
- **Learning rate ($\alpha$):** A small positive number, usually between 0 and 1, such as 0.01. It sets how big a step you take each time you update the parameters $w$ and $b$. A large $\alpha$ makes gradient descent bold, taking big strides downhill; a small $\alpha$ makes it cautious, inching down in tiny baby steps.
- **Derivative term ($\frac{\partial}{\partial w} J(w, b)$ or $\frac{\partial}{\partial b} J(w, b)$):** It tells you which direction to take your baby step in.
- **Step ($\alpha \times \frac{\partial}{\partial w} J(w, b)$ or $\alpha \times \frac{\partial}{\partial b} J(w, b)$):** Together with the learning rate, the derivative also determines how big a step you take downhill.

:::{prf:definition} Learning Rate
:label: def-learning-rate

The **learning rate** $\alpha$ is a small positive number that controls how big a step gradient descent takes each time it updates the parameters.
:::

Just like taking baby steps down the hill until you reach the valley floor, gradient descent repeats these two update steps until the algorithm **converges**. By "converges," I mean that it reaches a local minimum, where $w$ and $b$ barely change with each additional step.

**Implementing the update correctly**

The two parameters must be updated *simultaneously*. [](#ch03-simultaneous-update-table) shows the correct way and a common mistake.

:::{table} Simultaneous update (correct) vs. sequential update (incorrect)
:label: ch03-simultaneous-update-table

| Correct: simultaneous update | Incorrect |
| --- | --- |
| $temp\_w = w - \alpha \frac{\partial}{\partial w}J(w,b)$ | $temp\_w = w - \alpha \frac{\partial}{\partial w}J(w,b)$ |
| $temp\_b = b - \alpha \frac{\partial}{\partial b}J(w,b)$ | $w = temp\_w$ |
| $w = temp\_w$ | $temp\_b = b - \alpha \frac{\partial}{\partial b}J(w,b)$ |
| $b = temp\_b$ | $b = temp\_b$ |
:::

In the incorrect version, $w$ is changed first, and this changed $w$ is then used to update the bias. That's wrong: we need to update the bias using the old weight, not the new one.

### Intuition

#### Derivative Intuition

To build intuition for what gradient descent does, so that you can implement and apply it yourself, let's take a closer look at the derivative term.

We'll use a simpler example where we minimize just one parameter. Suppose the cost $J$ depends only on $w$, which is a number. Then gradient descent becomes:

$$
w = w - \alpha \frac{\partial}{\partial w}J(w)
$$

The goal is to minimize the cost by adjusting the parameter $w$. Just like in the earlier example where we temporarily set $b = 0$, having one parameter instead of two means we can look at 2D graphs of $J$ instead of 3D ones.

One way to think about the derivative at a point on the curve is to draw a **tangent line**, a straight line that just touches the curve at that point. In math, the slope of this tangent line is the derivative of $J$ at that point. To find the slope, draw a small triangle under the line and divide its height by its width.

:::{figure}
:label: ch03-21
:no-subfigures:

```{image} diagrams/0319-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0319-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

The figure shows two possible cases:

**Example 1 (positive slope):** The update subtracts the learning rate times a positive number. The learning rate is always positive, so subtracting a positive number makes $w$ smaller, which moves you to the left on the graph. That's exactly right if your goal is to decrease the cost $J$: moving left along this curve lowers $J$ and brings you closer to its minimum.

$w = w - \alpha \cdot \text{(positive number)}$ ; this will decrease $w$

**Example 2 (negative slope):** The update subtracts the learning rate times a negative number. Subtracting a negative number makes $w$ bigger, which moves you to the right on the graph. Again, that's what we want: moving right along this curve lowers the cost $J$ and brings you closer to its minimum.

$w = w - \alpha \cdot \text{(negative number)}$ ; this will increase $w$

#### Learning Rate Intuition

The other key ingredient in gradient descent is the learning rate $\alpha$. How do you choose it? What happens if it's too small, or too big?

The choice of $\alpha$ has a huge impact on how efficiently your implementation of gradient descent runs. If $\alpha$ is chosen poorly, gradient descent may not work at all. There are three possibilities:

:::{figure}
:label: ch03-22
:no-subfigures:

```{image} diagrams/0320-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0320-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

1. **If $\alpha$ is too small (surely but slowly):** Gradient descent still works, but it's slow. Because it takes tiny baby steps, it needs a great many of them to get anywhere close to the minimum.
2. **If $\alpha$ is too big (may not converge, and even diverge):** A single update can throw $w$ far past the minimum, all the way to the other side of the curve, where the cost is actually *higher* than before. At the new point, the derivative correctly says to move back, but the step is again far too big, so $w$ overshoots the minimum once more. Each jump lands farther from the minimum than the last one. In other words, gradient descent may overshoot and never reach the minimum: it may fail to converge, and it may even diverge.
3. **If $\alpha$ is perfect:** A perfect $\alpha$ is truly a work of art. It's an active research topic.

Here's a tricky question: suppose $w$ is already at a local minimum, so the cost $J$ is already at a local minimum too. What will one more step of gradient descent do?

:::{figure}
:label: ch03-23
:no-subfigures:

```{image} diagrams/0321-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0321-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

The cost function in this figure isn't a squared error cost function, and it has two local minima, one for each valley. Suppose that after some number of steps, gradient descent has brought $w$ to 5, and that this happens to be one of the local minima. If you draw the tangent line to the function at this point, it's perfectly flat.

- A flat line has a slope of zero, so the derivative term is zero for the current value of $w$. The update then sets $w$ to $w$ minus the learning rate times zero, which is just $w$. So if you're already at a local minimum, gradient descent leaves $w$ unchanged: the new value is exactly the same as the old one.
- For example, with $w = 5$ and $\alpha = 0.1$, one iteration computes $5 - 0.1 \times 0 = 5$. Once your parameters have brought you to a local minimum, further gradient descent steps do absolutely nothing.
- That's exactly what you want, because it keeps the solution at that local minimum.

**This also explains why gradient descent can reach a local minimum, even with a fixed learning rate $\alpha$.**

**Can gradient descent reach a local minimum with a fixed learning rate?**

:::{figure}
:label: ch03-24
:no-subfigures:

```{image} diagrams/0322-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0322-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

Yes. Far from the minimum, the derivative is fairly large, so the first update takes a relatively big step. At the second point, the curve is less steep, so the derivative is smaller, and so is the next step. At the third point, the derivative is smaller still, and the step shrinks again. As we approach the minimum, the derivative gets closer and closer to zero, so the steps become very small, until we finally reach the local minimum.

:::{important} Key idea
Near a local minimum, the slope flattens out, so the derivative, and with it each update step, shrinks by itself. Gradient descent naturally slows down as it arrives at the minimum, even when the learning rate $\alpha$ stays fixed.
:::

### Gradient Descent for Linear Regression Model

Now let's put everything together: the linear regression model, the squared error cost function, and gradient descent. This lets us train the linear regression model to fit a straight line to the training data.

**Linear regression model**

$$
f_{w, b}(X) = wx + b
$$

**Cost function**

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w, b}(X^{(i)}) - y^{(i)})^2
$$

**Gradient descent algorithm**

$$
\begin{aligned}
\text{repeat until convergence} \{ \\
w = w - \alpha \frac{\partial}{\partial w}J(w, b) \\
b = b - \alpha \frac{\partial}{\partial b}J(w, b) \\
\}
\end{aligned}
$$

If we calculate the derivatives, we get:

$$
\begin{aligned}
\text{repeat until convergence} \{ \\
w = w - \alpha \frac{1}{m}\sum_{i=1}^{m} (f_{w,b}(X^{(i)}) - y^{(i)}) X^{(i)} \\
b = b - \alpha \frac{1}{m}\sum_{i=1}^{m} (f_{w,b}(X^{(i)}) - y^{(i)}) \\
\}
\end{aligned}
$$

These equations come from calculus. The next figure shows how they're derived.

:::{figure}
:label: ch03-25
:no-subfigures:

```{image} diagrams/0323-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0323-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

Putting it together, here's the gradient descent algorithm for linear regression. You repeat these updates to $w$ and $b$ until convergence:

$$
\begin{aligned}
\text{repeat until convergence} \{ \\
w = w - \alpha \frac{1}{m}\sum_{i=1}^{m} (f_{w,b}(X^{(i)}) - y^{(i)}) X^{(i)} \\
b = b - \alpha \frac{1}{m}\sum_{i=1}^{m} (f_{w,b}(X^{(i)}) - y^{(i)}) \\
\}
\end{aligned}
$$

#### Local and Global Minima

Earlier, we saw that gradient descent can end up at a local minimum instead of the **global minimum**, which is the point with the lowest possible value of the cost function $J$ among all points.

:::{figure} Images/MD - 04 - 07.png
:label: ch03-multiple-local-minima
:alt: Cost surface with more than one local minimum
:width: 70%
:align: center
:::

Remember the surface plot that looks like an outdoor park with a few hills? That function has more than one local minimum, so depending on where you initialize the parameters $w$ and $b$, you can end up in different valleys.

With linear regression and the squared error cost function, however, this never happens. Because of its bowl shape, this cost function has no extra local minima, only a single global minimum. The technical term for this is a **convex function**.

:::{figure} Images/MD - 04 - 08.png
:label: ch03-convex-cost-function
:alt: Bowl-shaped convex cost function with a single global minimum
:width: 70%
:align: center
:::

:::{prf:definition} Convex Function
:label: def-convex-function

Informally, a **convex function** is bowl-shaped: its only minimum is a single global minimum, with no other local minima.
:::

A nice consequence is that when you run gradient descent on a convex function, it always converges to the global minimum, as long as the learning rate is chosen appropriately.

#### Action

:::{figure}
:label: ch03-26
:no-subfigures:

```{image} diagrams/0324-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0324-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

More precisely, the version of gradient descent we've been using is called **batch gradient descent**.

:::{prf:definition} Batch Gradient Descent
:label: def-batch-gradient-descent

**Batch gradient descent** is gradient descent in which every step looks at all of the training examples, rather than just a subset of the training data.
:::

## Implementation Code

See [Lab: Linear Regression with One Feature](lab/linear-regression-with-one-feature.ipynb).

# Linear Regression Model with Multiple Features

Let's extend linear regression to handle multiple input features.

This is called **multiple linear regression**! (Not multivariate regression; that's something else.)

## Multiple Features

So far, we've used a single feature $X$, the size of the house, to predict $y$, the price of the house. But what if, besides the size, we also knew each home's number of bedrooms, number of floors, and age in years? [](#ch03-housing-features-table) shows such a dataset.

:::{table} Housing data with four features
:label: ch03-housing-features-table

| Size in $\text{feet}^2$ ($X_1$) | Number of Bedrooms ($X_2$) | Number of Floors ($X_3$) | Age of home in years ($X_4$) | Price in \$1000's ($y$) |
| --- | --- | --- | --- | --- |
| 2104 | 5 | 1 | 45 | 460 |
| 1416 | 3 | 2 | 40 | 232 |
| 1532 | 3 | 2 | 30 | 315 |
| 852 | 2 | 1 | 36 | 178 |
| ... | ... | ... | ... | ... |
:::

We'll use the variables $X_1$, $X_2$, $X_3$, and $X_4$ to denote the four features.

To refer to a specific value, we use two indices, $x_{j}^{(i)}$, where $i$ denotes the $i^{th}$ training example and $j$ denotes the feature. For example:

$$
\vec{X}^{(2)} = \begin{bmatrix} 1416 & 3 & 2 & 40\end{bmatrix}
$$

$$
x^{(2)}_{3} = 2
$$

In general, with $n$ features, the model looks like this:

$$
f_{\vec{w}, b} (\vec{X}) = w_1 X_1 + w_2 X_2 + w_3 X_3 + w_4 X_4 + ... + w_n X_n + b
$$

where:

- $\vec{w} = \begin{bmatrix} w_1 & w_2 & w_3 & w_4 & ... & w_n \end{bmatrix}$ is a **vector**, which in math simply means a list of numbers.
- $b$ is a single number, not a vector.
- $\vec{w}$ and $b$ are the **parameters** of the model.
- $\vec{X} = \begin{bmatrix} X_1 & X_2 & X_3 & X_4 & ... & X_n \end{bmatrix}$ is also a vector.

Therefore, we can rewrite the model as:

$$
f_{\vec{w}, b} (\vec{X}) = \vec{w} \cdot \vec{X} + b
$$

$$
f_{\vec{w}, b} (\vec{X}) = \vec{w} \cdot \vec{X} + b = w_1 X_1 + w_2 X_2 + w_3 X_3 + w_4 X_4 + ... + w_n X_n + b
$$

What is this dot product? The **dot product** of two vectors, two lists of numbers $\vec{w}$ and $\vec{X}$, is found by multiplying their matching entries, $w_1$ with $X_1$, $w_2$ with $X_2$, $w_3$ with $X_3$, and so on up to $w_n$ with $X_n$, and then adding up all of those products. Dot product notation lets us write the model in a more compact form, with fewer symbols.

A linear regression model with multiple input features is called **multiple linear regression**, as opposed to univariate regression, which has just one feature. (You might expect it to be called *multivariate* regression, but that term actually means something else, which we won't be using here.)

:::{prf:definition} Multiple Linear Regression
:label: def-multiple-linear-regression

**Multiple linear regression** is linear regression with $n$ input features: $f_{\vec{w}, b} (\vec{X}) = \vec{w} \cdot \vec{X} + b$.
:::

To implement it, we'll use a really neat trick called **vectorization**, which makes this and many other learning algorithms much simpler to implement.

## Vectorization

Vectorization makes your code shorter, and it also makes it run much faster. Consider an example with three features:

$$
\vec{w} = \begin{bmatrix} w_1 & w_2 & w_3 \end{bmatrix}
$$
$$
b \text{ is a  number}
$$
$$
\vec{X} = \begin{bmatrix} X_1 & X_2 & X_3 \end{bmatrix}
$$
$$
f_{\vec{w}, b} (\vec{X}) = w_1 X_1 + w_2 X_2 + w_3 X_3 + b
$$

In Python with NumPy, we can write this as:

``` python
import numpy as np

w = np.array([1.0, 2.5, -3.5])
b = 4
X = np.array([10, 20, 30])
```

In Python, indexing starts at 0, so the elements are numbered $X_0, X_1, X_2$.

### Without Vectorization

$$
f_{\vec{w}, b} (\vec{X}) = w_1 X_1 + w_2 X_2 + w_3 X_3 + b
$$

``` python
f = w[0] * X[0] + w[1] * X[1] + w[2] * X[2] + b
```

The math uses $x_1, x_2, x_3$, but Python indexes from 0, so the code uses `X[0]`, `X[1]`, `X[2]`.

What if the number of inputs $n$ is 100, or 1,000? We can use a loop!

In math, we can use a summation to add up all the products of $w_j$ and $X_j$:

$$
f_{\vec{w}, b} (\vec{X}) = \sum_{j=1}^{n} w_j X_j + b
$$

In code:

``` python
f = 0
for j in range(0, n, 1):
	f = f + w[j] * X[j]
	
f = f + b
```

### With Vectorization

Here is the math expression for the function $f$:

$$
f_{\vec{w}, b} (\vec{X}) = \vec{w} \cdot \vec{X} + b
$$

And in code:

``` python
import numpy as np

f = np.dot(w, X) + b
```

NumPy's `dot` function is a vectorized implementation of the dot product of two vectors. Especially when $n$ is large, it runs much faster than the two previous code examples.

**What are the benefits?**

- **Shorter code.** The whole calculation is now just one line. Pretty cool, right?
- **Faster code.** It also runs much faster than either of the two previous implementations, which didn't use vectorization.
  - The speedup happens behind the scenes: NumPy's `dot` function can use the parallel hardware in your computer. That's true whether you're running on an ordinary CPU or on a GPU (graphics processing unit), which is often used to accelerate machine learning jobs. Using parallel hardware makes it much more efficient than the `for` loop, or the one-after-another calculation we saw earlier.
  - Also, its core is written in optimized C, and C is fast! Add all those optimizations on top. Hehe!

### Example

:::{figure}
:label: ch03-27
:no-subfigures:

```{image} diagrams/0325-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0325-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

:::{figure}
:label: ch03-28
:no-subfigures:

```{image} diagrams/0326-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0326-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

A vectorized implementation makes linear regression much more efficient. With 16 features, the speedup might not be huge. But with thousands of features and perhaps very large training sets, vectorization can make an enormous difference to your algorithm's running time: your code might finish in a minute or two instead of taking many hours.

In short, vectorization gives you shorter code, which is hopefully easier for you and others to write and read, and it makes that code run much faster.

## Gradient Descent for Multiple Linear Regression

[](#ch03-vector-notation-table) compares the previous notation with vector notation.

:::{table} Previous notation vs. vector notation
:label: ch03-vector-notation-table

|                  | Previous Notation                                       | Vector Notation                                                                   |
| ---------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Parameters       | $w_1, ..., w_n$ and b                                   | $\vec{w} = \begin{bmatrix} w_1 & ... & w_n \end{bmatrix}$ and b is still a number |
| Model            | $f_{\vec{w},b} (\vec{X}) = w_1 X_1 + ... + w_n X_n + b$ | $f_{\vec{w}, b} (\vec{X}) = \vec{w} \cdot \vec{X} + b$                            |
| Cost Function    | $J(w_1, ..., w_n, b)$                                   | $J(\vec{w}, b)$                                                                   |
| Gradient Descent | See (1) below                                          | See (2) below                                                                    |
:::

**(1) Gradient Descent in previous notation:**

\begin{align}

\text{repeat\{} \\
w_j = w_j - \alpha \frac{\partial}{\partial w_j} J(w_1, ..., w_n, b) \\
b = b - \alpha \frac{\partial}{\partial b} J(w_1, ..., w_n, b) \\
\text{\}}

\end{align}

**(2) Gradient Descent in Vector Notation:**

\begin{align}

\text{repeat\{} \\
w_j = w_j - \alpha \frac{\partial}{\partial w_j} J(\vec{w}, b) \\
b = b - \alpha \frac{\partial}{\partial b} J(\vec{w}, b) \\
\text{\}}

\end{align}

### Let's look at how gradient descent works for multiple features

We already know how it works for a single feature:

\begin{align}

\text{repeat\{} \\
w = w - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{w,b} (X^{(i)}) - y^{(i)}) X^{(i)} \\

b = b - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{w,b} (X^{(i)}) - y^{(i)}) \\ \\

\text{\}} \text{ simultaneously update w, b} \\ 

\end{align}

Now, here is the new notation when we have $n$ features.

\begin{align}

& \text{repeat\{} \\
& w_1 = w_1 - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b} (\vec{X}^{(i)}) - y^{(i)}) x_1^{(i)} \\

& w_2 = w_2 - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b} (\vec{X}^{(i)}) - y^{(i)}) x_2^{(i)} \\

& . \\
& . \\
& . \\

& w_n = w_n - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b} (\vec{X}^{(i)}) - y^{(i)}) x_n^{(i)} \\ \\ \\ \\

& b = b - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b} (\vec{X}^{(i)}) - y^{(i)}) \\ \\

& \text{\}} \text{simultaneously update} \\ \\ \\

\end{align}

Implementing these updates gives you gradient descent for multiple linear regression.

## Alternative to Gradient Descent - Normal Equation

For linear regression, there's another way to find $w$ and $b$, called the **normal equation**.

With the normal equation method, an advanced linear algebra library solves for $w$ and $b$ directly, all in one go, without iterations. **However, it only works for linear regression.**

**Normal equation:**

- Only for linear regression
- Solves for $w, b$ without iterations

**Disadvantages:**

- It doesn't generalize to other learning algorithms. Unlike gradient descent, it can't be used for algorithms such as logistic regression.
- It's slow when the number of features is large (> 10,000).

## Feature Scaling

A few tricks, such as choosing and scaling features appropriately and picking a good learning rate $\alpha$, can make gradient descent work much better.

**Feature scaling**, in particular, lets gradient descent run much faster.

Let's start by looking at how the size of a feature, meaning how big its values are, relates to the size of its parameter.

:::{figure}
:label: ch03-29
:no-subfigures:

```{image} diagrams/0327-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0327-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

So how does this relate to gradient descent?

:::{figure}
:label: ch03-30
:no-subfigures:

```{image} diagrams/0328-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0328-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

In short, when features take on very different ranges of values, gradient descent can run slowly. Rescaling the features so they all have comparable ranges can speed it up significantly.

### How to do Feature Scaling?

:::{prf:definition} Feature Scaling
:label: def-feature-scaling

**Feature scaling** transforms features that take on very different ranges of values so that they all have comparable ranges.
:::

#### Method 1: Divide by Max Value

Divide each value of the feature by the feature's maximum value:

$$
n_1 \le X_n \le n_2
$$
$$
X_{n, scaled} = \frac{X_n}{n_2}
$$

:::{figure}
:label: ch03-31
:no-subfigures:

```{image} diagrams/0329-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0329-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

#### Method 2: Mean Normalization

Mean normalization rescales the features so that they're centered around zero. Where the original features had only values greater than zero, the rescaled features have both negative and positive values, usually between $-1$ and $+1$.

To apply mean normalization:

1. Find the average, also called the mean, of $X_n$. Let's call it $\mu_n$.
2. Note the range of the feature: $n_1 \le X_n \le n_2$.
3. Apply feature scaling:

$$
X_{n, scaled} = \frac{X_n - \mu_n}{n_2 - n_1}
$$

:::{figure}
:label: ch03-32
:no-subfigures:

```{image} diagrams/0330-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0330-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

#### Method 3: Z-Score Normalization

To apply z-score normalization:

1. Calculate the mean $\mu$ and the standard deviation $\sigma$ of each feature.
2. Apply feature scaling:

$$
X_{n, scaled} = \frac{X_n - \mu_n}{\sigma_n}
$$

:::{figure}
:label: ch03-33
:no-subfigures:

```{image} diagrams/0331-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0331-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

### Rule of Thumb

When scaling features, aim for each feature $X_j$ to range from roughly $-1$ to $+1$, that is, about $-1 \le X_j \le +1$. But these limits can be ***a little bit loose***, as [](#ch03-feature-ranges-table) shows.

:::{table} Rule of thumb for feature ranges
:label: ch03-feature-ranges-table

| Feature range | Should you rescale? |
| --- | --- |
| $-3 \le X_j \le +3$ | No, completely acceptable |
| $-0.3 \le X_j \le 0.3$ | No, completely acceptable |
| $0 \le X_j \le 3$ | Optional; it should work okay either way (no harm) |
| $-2 \le X_j \le 0.5$ | Optional; it should work okay either way (no harm) |
| $-100 \le X_j \le 100$ | Yes, too large |
| $-0.001 \le X_j \le 0.001$ | Yes, too small |
| $98.6 \le X_j \le 105$ | Yes, the values are too large (around 100) |
:::

For a range that's too large or too small, you're probably better off rescaling the feature so that it ranges from something closer to $-1$ to $+1$.

The last row is a hospital patient's temperature, a feature $X_5$ that ranges from 98.6 to 105 degrees Fahrenheit. These values are around 100, which is large compared with the other, scaled features, and that will make gradient descent run more slowly. Here, feature rescaling will likely help.

:::{tip}
Rescaling features almost never hurts. When in doubt, go ahead and rescale.
:::

And that's feature scaling! This little technique can often make gradient descent run much faster.

## Is Gradient Descent Converging?

With or without feature scaling, how can you check that gradient descent is really working, that is, whether it's converging toward parameters at or near the global minimum of the cost function? Let's look at how to recognize when gradient descent is converging.

**Objective of gradient descent:**

$$
\min_{\vec{w}, b} \; J(\vec{w}, b)
$$

:::{figure}
:label: ch03-34
:no-subfigures:

```{image} diagrams/0332-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0332-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

When gradient descent is working properly, the cost $J$ decreases after every single iteration. If $J$ ever increases after an iteration, either $\alpha$ was chosen poorly (usually it's too large) or there's a bug in the code.

This plot tells you something else, too. By around 300 iterations, the cost $J$ is leveling off and no longer decreasing much, and by 400 iterations, the curve has flattened out. A flat curve means gradient descent has more or less converged.

A plot like this, showing the cost after each iteration, is called a **learning curve**, and it lets you spot whether gradient descent is converging. The number of iterations gradient descent needs can vary a lot between applications: one might converge after just 30 iterations, while another takes 1,000 or even 100,000. Since it's very hard to know in advance, plotting a learning curve is the practical way to find out.

### Ways to Tell If the Model Is Done Training

1. Plot a learning curve.
2. Use an automatic convergence test.

#### Automatic Convergence Test

Let $\epsilon$ ("epsilon") be $10^{-3}$ (or 0.001).

If the cost $J(\vec{w}, b)$ decreases by $\le \epsilon$ in one iteration, declare **convergence**.

:::{tip} In practice
Choosing the right threshold $\epsilon$ is difficult. Looking at the learning curve is usually more reliable than relying on an automatic convergence test.
:::

## Choosing the Learning Rate

A well-chosen learning rate makes your learning algorithm run much better. If it's too small, the algorithm runs very slowly; if it's too large, it may not converge at all.

If you plot the cost over a number of iterations and notice that it sometimes goes up and sometimes goes down, take that as a clear sign that gradient descent isn't working properly. The cause could be:

- a bug in the code, or
- a learning rate that's too large.

:::{figure}
:label: ch03-35
:no-subfigures:

```{image} diagrams/0333-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0333-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

### A Common Bug: Wrong Sign

In your code, you might be doing:

$$
w_1 = w_1 + \alpha d_1
$$

when you should use a minus sign:

$$
w_1 = w_1 - \alpha d_1
$$

With a plus sign, each update moves the cost $J$ further from the global minimum instead of closer to it.

:::{tip} Choosing Learning Rate $\alpha$
A useful debugging tip for gradient descent is that, with a sufficiently small learning rate, the cost function should generally decrease with each iteration. If gradient descent is not working as expected, try setting the learning rate to a very small value and checking whether the cost decreases at every iteration. If the cost still increases during some iterations, even with a very small learning rate, this may indicate a bug in the implementation.

Keep in mind that using a very small learning rate is primarily a debugging technique. It can make the learning process extremely slow, so it may not be an efficient choice for training the model.
:::

**Keep the trade-off in mind: a learning rate that's too small can make gradient descent take a very large number of iterations to converge.**

### Choosing the right value!

:::{figure}
:label: ch03-36
:no-subfigures:

```{image} diagrams/0334-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0334-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

# Feature Engineering

The features you choose can have a huge impact on how well your learning algorithm performs. In fact, in many practical applications, choosing or designing the right features is a critical step in making the algorithm work well.

:::{figure}
:label: ch03-37
:no-subfigures:

```{image} diagrams/0335-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0335-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

Suppose we have two features for each house:

- $X_1$ is the width of the plot of land the house is built on. In real estate, this is called the **frontage** of the lot.
- $X_2$ is the depth of the lot.

A model using these features would be $f_{\vec{w}, b} (\vec{X}) = w_1 X_1 + w_2 X_2 +b$, and it might work okay.

But there may be an even more effective way to use these features in the model. The area of the land can be calculated as the frontage (width) times the depth:

$$
X_3 = X_1 \times X_2
$$

and you might have an intuition that the area of the land predicts the price better than the frontage and depth do as separate features. So we add the area as a new feature.

**New model:**

$$
f_{\vec{w}, b}(\vec{X}) = w_1 X_1 + w_2 X_2 + w_3 X_3 + b
$$

Creating a new feature like this is an example of **feature engineering**.

:::{prf:definition} Feature Engineering
:label: def-feature-engineering

**Feature engineering** is creating new features from the original ones, usually by transforming or combining them, based on your knowledge or intuition about the problem, so that the learning algorithm can make accurate predictions more easily.
:::

# Full Implementation

See [Lab: Linear Regression with Multiple Features](lab/linear-regression-with-multiple-feature.ipynb).

# Summary

Linear regression is one of the most widely used learning algorithms in the world, and quite possibly the most widely used of all.

:::{exercise}
:label: pt3-ch3-ex-2

True or false: doubling the learning rate $\alpha$ almost always makes gradient descent converge about twice as fast.

**a.** True

**b.** False
:::

:::{solution} pt3-ch3-ex-2
:class: dropdown

**Answer: b.** False.

A doubled learning rate may be too large. Gradient descent can then overshoot and fail to find the best values for the parameters $w$ and $b$.
:::
