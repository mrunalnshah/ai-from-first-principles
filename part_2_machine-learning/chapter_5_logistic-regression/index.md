---
title: "Chapter 5: Logistic Regression"
description: Predicting categories with logistic regression
date: 2026-10-17
numbering:
  enumerator: 5.%s
---

In this chapter, we will learn how to predict categories using the logistic regression model.

In classification, which is what logistic regression is for, the output variable $y$ can take only one of a small handful of possible values, rather than any number from an infinite range. That's why linear regression turns out to be a poor algorithm for classification problems, as we'll see.

# Classification Problem

Here are a few examples of classification problems:

:::{table} Examples of binary classification problems
:label: ch05-classification-examples-table

| Question | Answer ($y$) |
| --- | --- |
| Is this email spam? | no or yes |
| Is this transaction fraudulent? | no or yes |
| Is the tumor malignant? | no or yes |
:::

Each of these problems has only two possible outputs. A classification problem like this is called **binary classification**, where "binary" means there are only two possible classes, or categories. In these problems, the words *class* and *category* are used more or less interchangeably.

:::{prf:definition} Binary Classification
:label: def-binary-classification

**Binary classification** is a classification problem in which the target $y$ can take only one of two possible values, or classes.
:::

## Binary Classification

- The **target output $y$** can be only one of two values, which is why this is called "binary" classification.
- The two values can be written as, for example:
  1. no or yes
  2. false or true
  3. 0 or 1

:::{note}
By convention in computing, 0 means false and 1 means true.
:::

**We use the numbers 0 and 1 to represent the answer $y$ (the target output),** because numbers are what our learning algorithms can work with most easily.

- 0 is the **negative class** (not "bad"); it's more like "absence."
- 1 is the **positive class** (not "good"); it's more like "presence."
- Which is which depends on the question. For example, for "Is the email spam?":
  - 0 means no, it's not spam.
  - 1 means yes, it's spam.

### Example of classifying if a tumor is malignant

Here's an example of a training set for classifying whether a tumor is malignant.

:::{figure}
:label: 501
:no-subfigures:

```{image} diagrams/01-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/01-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

# Classification with Linear Regression!

What if we simply reuse the algorithm we already know? We could run linear regression on this training set and fit a straight line to the data.

:::{figure}
:label: 502
:no-subfigures:

```{image} diagrams/01-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/01-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

The result might be a line like this:

:::{figure}
:label: 503
:no-subfigures:

```{image} diagrams/02-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/02-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

- A straight line doesn't stop at 0 and 1. Linear regression outputs every value in between, and values below 0 or above 1 as well.
- **But here we want to predict categories.**
  - We could turn the line's output into a category with a threshold, such as 0.5. Outputs below 0.5 mean $y = 0$ (not malignant), and outputs of 0.5 or more mean $y = 1$ (malignant).
  - Where the line reaches 0.5, draw a vertical line. Tumors to its left are classified as $y = 0$, and tumors to its right as $y = 1$.

:::{figure}
:label: 504
:no-subfigures:

```{image} diagrams/03-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/03-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

Now suppose the dataset gets one more training example, far out on the right.

:::{figure}
:label: 505
:no-subfigures:

```{image} diagrams/04-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/04-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

- This new tumor is large, so we clearly want the algorithm to classify it as malignant, and it shouldn't change how the other tumors are classified. Yet adding this one example on the right shifts linear regression's best-fit line, and with it the dividing line, also called the **decision boundary**, moves to the right.
- That's why it's better to use **logistic regression**, whose output always stays between 0 and 1. This avoids these problems.
- Here's another example:

:::{figure} Images/IM_01.png
:label: ch05-linreg-another-example
:alt: A single artificial neuron
:width: 100%
:align: center
:::

# Logistic Regression

- One confusing thing about the name: even though *logistic regression* has the word "regression" in it, it's actually used for classification.
- Specifically, it solves binary classification problems, where the output label $y$ is either 0 or 1.

:::{prf:definition} Logistic Regression
:label: def-logistic-regression

**Logistic regression** is an algorithm for binary classification, where the label $y$ is either 0 or 1. Despite its name, it's used for classification, not regression, and its output is always a number between 0 and 1.
:::

:::{figure}
:label: 506
:no-subfigures:

```{image} diagrams/05-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/05-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

- To classify whether a tumor is malignant, logistic regression fits an S-shaped curve like this one to the dataset.
- Suppose a patient comes in with a tumor of size 'z' on the horizontal axis. For this tumor, the algorithm outputs 0.7, which suggests the tumor is more likely to be malignant than benign.
  - But the output label $y$ is never 0.7.
    - ***It's only ever 0 or 1.***
- To build logistic regression, we need an important mathematical function called the **sigmoid function**, also known as the **logistic function**.

## Sigmoid Function (Logistic Function)

The next figure shows the sigmoid function. Notice that the horizontal axes of the graphs on the left and right are different. On the left, the horizontal axis is the tumor size, so it has only positive numbers. On the right, the horizontal axis is $z$, which can be negative or positive.

:::{figure}
:label: 507
:no-subfigures:

```{image} diagrams/06-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/06-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

- If $z = 100$, then $g(z)$ is very close to 1.
- If $z = -100$, then $g(z)$ is very close to 0.
- If $z = 0$, then $g(z)$ is exactly 0.5. (That's why the curve crosses the vertical axis at 0.5.)

That's why the sigmoid function has its S shape: it starts very close to 0 and gradually rises toward 1.

:::{prf:definition} Sigmoid Function
:label: def-sigmoid-function

The **sigmoid function** $g(z)$, also called the **logistic function**, takes any number $z$ and outputs a value between 0 and 1, following an S-shaped curve.
:::

## Let's build: Logistic Regression

We'll build it in two steps.

**Step 1: compute $z$.** Start with the straight-line function from linear regression:

$$
f_{\vec{w}, b} (\vec{X}) = \vec{w} \cdot \vec{X} + b
$$

and call its output $z$:

$$
z = \vec{w} \cdot \vec{X} + b
$$

**Step 2: apply the sigmoid.** Then feed $z$ into the sigmoid (logistic) function $g$:

$$
g(z) = \frac{1}{1 + e^{-z}} ; 0 \lt g(z) \lt 1
$$

**Putting it together.** Substituting the first equation into the second gives the logistic regression model:

$$
f_{\vec{w}, b}(\vec{X}) = g (\vec{w} \cdot \vec{X} + b) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{X} + b)}}
$$

This model takes a single feature, or a set of features $\vec{X}$, and returns a number between 0 and 1.

## Interpretation of Logistic Regression Output

The model's output has a useful interpretation: it's the probability that the class, or label, $y$ is 1, given the input $\vec{X}$.

$$
f_{\vec{w}, b}(\vec{X}) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{X} + b)}}
$$

In other words, the output is the "probability" that the class is 1.

### Example

- $x$ is the tumor size.
- $y$ is 0 (not malignant) or 1 (malignant).

Suppose that for one patient, the model outputs $f_{\vec{w},b}(\vec{X}) = 0.7$.

- This means the model thinks there's a 70% chance that the true label $y$ is 1 for this patient. In other words, it estimates a 70% chance that the tumor will turn out to be malignant.
- So what's the chance that $y$ is 0? Since $y$ must be either 0 or 1, the two probabilities have to add up to 1, or 100%. So if there's a 70% (0.7) chance that $y$ is 1, there must be a 30% (0.3) chance that it's 0.

### Adds to 1

$P(y = 0) + P (y = 1) = 1$

In general, the probabilities of all the possible classes add up to 1:
$P(y=0) + P(y = 1) + P(y = 2) + ... + P(y = n) = 1$

### Logistic Notation

$$
f_{\vec{w}, b} (\vec{X}) = P(y = 1 | \vec{X}; \vec{w}, b)
$$

The semicolon shows that $\vec{w}$ and $b$ are parameters: they affect how the model computes the probability that $y$ equals 1, given the input features $\vec{X}$.

## Real-world usage

For years, a lightly modified version of logistic regression powered a large share of online advertising. On some major websites, it was essentially the algorithm that chose which ad to show you and many other visitors, and it earned some large companies a great deal of money.

# Decision Boundary

To get a better sense of how logistic regression computes its predictions, let's look at the **decision boundary**.

## Recap & Intuition

To recap:

$$
f_{\vec{w}, b} (\vec{X}) = g(z) = g(\vec{w} \cdot \vec{X} + b) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{X} + b)}} = P(y =1|\vec{X};\vec{w}, b)
$$

where:

$$
z = \vec{w} \cdot \vec{X} + b
$$

The output is a number such as 0.7 or 0.3.

What if you want the algorithm to predict whether $y$ is 0 or 1? You can set a **threshold**: above it, predict $y = 1$, and below it, predict $y = 0$. So:

:::{figure}
:label: 508
:no-subfigures:

```{image} diagrams/07-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/07-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

- Is $f_{\vec{w}, b}(\vec{X}) \ge 0.5$?
  - Yes: $\hat{y} = 1$
  - No: $\hat{y} = 0$

### When is $f_{\vec{w}, b}(\vec{X}) \ge 0.5$?

Recall:

$$
f_{\vec{w}, b} (\vec{X}) = g(z) = g(\vec{w} \cdot \vec{X} + b) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{X} + b)}} = P(y =1|\vec{X};\vec{w}, b)
$$

So:

1. $f_{\vec{w}, b} (\vec{X}) \ge 0.5$ when $g(z) \ge 0.5$.
2. When is $g(z) \ge 0.5$? From the sigmoid graph, when $z \ge 0$.
3. When is $z \ge 0$? Since $z = \vec{w} \cdot \vec{X} + b$, $z \ge 0$ whenever $\vec{w} \cdot \vec{X} + b \ge 0$.
   - Whenever $\vec{w} \cdot \vec{X} + b \gt 0$, $\hat{y} = 1$.
   - Whenever $\vec{w} \cdot \vec{X} + b \lt 0$, $\hat{y} = 0$.
   - What about $\vec{w} \cdot \vec{X} + b = 0$? Well, that depends on the problem we're solving!

## Visualize how the model makes predictions

### Problem 1 - Linear Decision Boundary

:::{figure}
:label: 509
:no-subfigures:

```{image} diagrams/08-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/08-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

Let's take a classification problem with two features, $x_1$ and $x_2$, instead of just one. In this training set, the red diamonds are the positive examples and the blue circles are the negative examples.

- The red diamonds correspond to $y = 1$, and the blue circles correspond to $y = 0$.

Since we have two features, $x_1$ and $x_2$, the logistic regression model makes predictions using this function:

$$
f_{\vec{w}, b} (\vec{X}) = g(z) = g(w_1x_1 + w_2x_2 + b)
$$

Now let's see how logistic regression makes predictions. In particular, let's work out when $\vec{w} \cdot \vec{X} + b \ge 0$ and when $\vec{w} \cdot \vec{X} + b \lt 0$.

To figure that out, it helps to look at one special line: the line where $\vec{w} \cdot \vec{X} + b = 0$. This line is called the ***decision boundary***, because on it the model is perfectly neutral about whether $y$ is 0 or 1.

:::{prf:definition} Decision Boundary
:label: def-decision-boundary

The **decision boundary** is the set of points where $z = 0$, which for the basic model means $\vec{w} \cdot \vec{X} + b = 0$. On one side of it, the model predicts $\hat{y} = 1$, and on the other side, $\hat{y} = 0$.
:::

:::{figure}
:label: 510
:no-subfigures:

```{image} diagrams/09-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/09-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

- The figure shows the decision boundary when the parameters $w_1$, $w_2$, and $b$ are 1, 1, and $-3$. In that case, $z = x_1 + x_2 - 3$, so the decision boundary is the line $x_1 + x_2 = 3$.
- With a different choice of parameters, the decision boundary would be a different line.

### Problem 2 - Non-Linear Decision Boundary

Now let's look at a more complex example, where the decision boundary is no longer a straight line.

:::{figure}
:label: 511
:no-subfigures:

```{image} diagrams/10-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/10-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

As before, the diamonds are the class $y = 1$, and the circles are the class $y = 0$.

We saw how to use polynomials in linear regression, and we can do the same in logistic regression. Let's use this function:

$$
z = w_{1}x_{1}^2 + w_{2}x_{2}^2 + b
$$

so that:

$$
f_{\vec{w}, b}(\vec{X}) = g(z) = g(w_{1}x_{1}^2 + w_{2}x_{2}^2 + b)
$$

Suppose we end up with $w_1 = 1, w_2 = 1$ and $b = -1$. Then:

$$
z = x_{1}^2 + x_{2}^2 - 1
$$

As before, the decision boundary is where $z = 0$:

$$
z = x_{1}^2 + x_{2}^2 - 1 = 0
$$

$$
x_{1}^2 + x_{2}^2 = 1
$$

That's the decision boundary, and it looks like this:

:::{figure}
:label: 512
:no-subfigures:

```{image} diagrams/11-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/11-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

- When $x_{1}^2 + x_{2}^2 \geq 1$, you're in the area outside the circle, and the model predicts $y = 1$.
- When $x_{1}^2 + x_{2}^2 < 1$, you're in the area inside the circle, and the model predicts $y = 0$.

### Problem X - More Complex?

Can we get even more complex decision boundaries than these? Yes! We can, by using even higher-order polynomial terms.

#### Ellipse

:::{figure}
:label: 513
:no-subfigures:

```{image} diagrams/12-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/12-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

#### More Complex?

Yes!

:::{figure}
:label: 514
:no-subfigures:

```{image} diagrams/13-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/13-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

You can get even more complex decision boundaries, shaped like the one in this figure, and even more complex ones than that! Here, logistic regression predicts $y = 1$ inside the shape and $y = 0$ outside it.

With polynomial features like these, decision boundaries can become very complex. In other words, logistic regression can learn to fit fairly complex data.

:::{note}
If you don't include any higher-order polynomial terms, so that the only features are $x_1, x_2, x_3$, and so on, the decision boundary will always be linear.
:::

# Cost Function

The cost function measures how well a particular set of parameters fits the training data, which gives us a way to look for better parameters.

Logistic regression is defined by this function:

$$
f_{\vec{w}, b} (\vec{X}) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{X} + b)}}
$$

The question we want to answer is: given a training set, how do we choose the parameters $\vec{w}$ and $b$? Let's start by recalling linear regression.

## Why can't we use mean squared error for classification?

For linear regression, we used the squared error cost:

$$
J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} \frac{1}{2} (f_{\vec{w},b}(\vec{X}^{(i)}) - y^{(i)})^2
$$

:::{figure}
:label: 515
:no-subfigures:

```{image} diagrams/14-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/14-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

- The logistic model is non-linear, so the squared error cost becomes bumpy ("wiggly") and non-convex, with lots of local minima where gradient descent could get stuck.
- That makes it much harder for gradient descent to find the best values of $\vec{w}$ and $b$.

## Different cost function!

So we'll switch to a different cost function. With it, the cost is convex again, and gradient descent can be relied on to reach the global minimum.

:::{note}
Compared with the version in Chapter 3, the squared error cost above moves the $\frac{1}{2}$ inside the summation. This doesn't change its value; it just makes the math a little simpler.
:::

### Generalized Cost Function

First, let's generalize the cost function:

$$
J(\vec{w}, b) = \frac{1}{m} \sum_{i = 1}^{m} L(f_{\vec{w}, b} (\vec{X}^{(i)}), y^{(i)})
$$

where $L$ is the loss function.

### Logistic Loss Function

$$
L(f_{\vec{w}, b}(\vec{X}^{(i)}), y^{(i)}) =
\begin{cases}
-\log\left(f_{\vec{w}, b}(\vec{X}^{(i)})\right), & \text{if } y^{(i)} = 1, \\
-\log\left(1 - f_{\vec{w}, b}(\vec{X}^{(i)})\right), & \text{if } y^{(i)} = 0.
\end{cases}
$$

Adding up the losses over all the training examples gives the cost function, which measures how well the model is doing on the entire training set.

#### y = 1

Let's start with the case $y = 1$, and plot this part of the loss function to get some intuition about what it does.

:::{figure}
:label: 516
:no-subfigures:

```{image} diagrams/15-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/15-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

#### y = 0

Now let's look at the second part of the loss function, for when $y = 0$.

:::{figure}
:label: 517
:no-subfigures:

```{image} diagrams/16-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/16-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

#### Coming back to our tumor dataset

Going back to the tumor example: if the model predicts a 99.9% chance of malignancy, but it turns out not to be malignant ($y = 0$), then we penalize the model with a very high loss.

For $y = 0$, just as for $y = 1$, the further the prediction $f_{\vec{w}, b}(\vec{X})$ is from the true value of $y$, the higher the loss.

:::{important} Key idea
The further the prediction $f_{\vec{w}, b} (\vec{X}^{(i)})$ is from the target $y^{(i)}$, the higher the loss.
:::

In fact, as the prediction gets close to the wrong answer, the loss grows very large and approaches infinity.

Likewise, when the true label is 1, the algorithm is strongly encouraged not to predict values too close to 0.

### Convex is better

With this loss, the overall cost function is convex, so you can reliably use gradient descent to reach the global minimum.

***Learn to prove whether the cost function is convex***

$$
L(f_{\vec{w}, b}(\vec{X}^{(i)}), y^{(i)}) =
\begin{cases}
-\log\left(f_{\vec{w}, b}(\vec{X}^{(i)})\right), & \text{if } y^{(i)} = 1, \\
-\log\left(1 - f_{\vec{w}, b}(\vec{X}^{(i)})\right), & \text{if } y^{(i)} = 0.
\end{cases}
$$

Find $\vec{w}$ and $b$ that minimize the cost $J$.

## Simplified Cost Function for Logistic Regression

Here is the loss function for logistic regression:

$$
L(f_{\vec{w}, b}(\vec{X}^{(i)}), y^{(i)}) =
\begin{cases}
-\log\left(f_{\vec{w}, b}(\vec{X}^{(i)})\right), & \text{if } y^{(i)} = 1, \\
-\log\left(1 - f_{\vec{w}, b}(\vec{X}^{(i)})\right), & \text{if } y^{(i)} = 0.
\end{cases}
$$

We're still working on a binary classification problem, so $y$ can only be 0 or 1. Because $y$ can't take any other value, there's a simpler way to write this loss function:

$$
L(f_{\vec{w}, b} (\vec{X}^{(i)}),y^{(i)}) = -y^{(i)} \log(f_{\vec{w}, b}(\vec{X}^{(i)})) - (1-y^{(i)}) \log (1 - f_{\vec{w}, b} (\vec{X}^{(i)}))
$$

### Test for y = 1

If $y^{(i)} = 1$:

$$
L(f_{\vec{w}, b} (\vec{X}^{(i)}),y^{(i)}) = -y^{(i)} \log(f_{\vec{w}, b}(\vec{X}^{(i)}))
$$

### Test for y = 0

If $y^{(i)} = 0$:

$$
L(f_{\vec{w}, b} (\vec{X}^{(i)}),y^{(i)}) = - (1-y^{(i)}) \log (1 - f_{\vec{w}, b} (\vec{X}^{(i)}))
$$

Whether $y$ is 1 or 0, this single expression gives the same result as the two-case version above. So it's a simpler way to write the loss: one equation, with no need to separate the two cases.

## Let's Write the Cost Function

**Loss function:**

$$
L(f_{\vec{w}, b} (\vec{X}^{(i)}),y^{(i)}) = -y^{(i)} \log(f_{\vec{w}, b}(\vec{X}^{(i)})) - (1-y^{(i)}) \log (1 - f_{\vec{w}, b} (\vec{X}^{(i)}))
$$

**Cost function:**

$$
J(\vec{w}, b) = \frac{1}{m} \sum_{i = 1}^{m} [L(f_{\vec{w}, b} (\vec{X}^{(i)}), y^{(i)})]
$$

Therefore, the cost function for logistic regression is:

$$
J(\vec{w}, b) = -\frac{1}{m} \sum_{i = 1}^{m} y^{(i)} \log(f_{\vec{w}, b}(\vec{X}^{(i)})) + (1-y^{(i)}) \log (1 - f_{\vec{w}, b} (\vec{X}^{(i)}))
$$

This is the cost function that almost everyone uses for logistic regression.

Where does it come from? It's derived using **maximum likelihood estimation**, a statistical method for finding the parameters of a model efficiently.

:::{important} Loss vs. cost
1. Loss is calculated per training example.
2. Cost is calculated over all training examples.
:::

# Gradient Descent for Logistic Regression

Training logistic regression means finding the values of $\vec{w}$ and $b$ with the lowest cost $J(\vec{w}, b)$, and for that we'll turn to gradient descent once again.

Here is the cost function:

$$
J(\vec{w}, b) = -\frac{1}{m} \sum_{i = 1}^{m} y^{(i)} \log(f_{\vec{w}, b}(\vec{X}^{(i)})) + (1-y^{(i)}) \log (1 - f_{\vec{w}, b} (\vec{X}^{(i)}))
$$

To minimize the cost function, we use gradient descent:

$$
\begin{aligned}
&\text{repeat \{}  \\
&w_{j} = w_{j} - \alpha \frac{\partial}{\partial w_{j}} J(\vec{w}, b) \\
&b = b - \alpha \frac{\partial}{\partial b} J(\vec{w}, b) \\
&\} \text{ Simultaneous Updates}
\end{aligned}
$$

for $j = 1 ... n$, where $n$ is the number of features.

If we work out the partial derivatives:

$$
\frac{\partial}{\partial w_{j}} J(\vec{w}, b) = \frac{1}{m} \sum_{i = 1}^{m} (f_{\vec{w}, b} (\vec{X}^{(i)}) -y^{(i)}) x_{j}^{(i)}
$$

and

$$
\frac{\partial}{\partial b} J(\vec{w}, b) = \frac{1}{m} \sum_{i = 1}^{m} (f_{\vec{w}, b} (\vec{X}^{(i)}) -y^{(i)})
$$

:::{note}
Update all the parameters simultaneously: calculate every right-hand side using the current values first, and only then assign the new values to $w_j$ and $b$.
:::

## Gradient Descent For Logistic Regression

$$
\begin{aligned}
&\text{repeat \{}  \\
&w_{j} = w_{j} - \alpha \left[ \frac{1}{m} \sum_{i = 1}^{m} (f_{\vec{w}, b} (\vec{X}^{(i)}) - y^{(i)}) x_{j}^{(i)} \right] \\
&b = b - \alpha \left[ \frac{1}{m} \sum_{i = 1}^{m} (f_{\vec{w}, b} (\vec{X}^{(i)}) - y^{(i)}) \right] \\
&\} \text{ Simultaneous Updates}
\end{aligned}
$$

These two equations might look strange: they're exactly the same as the ones we derived for linear regression. Is linear regression secretly the same as logistic regression?

- No. Even though the equations look the same, the definition of the function $f_{\vec{w}, b}(\vec{X})$ has changed:
  - For linear regression: $f_{\vec{w}, b} (\vec{X}) = \vec{w} \cdot \vec{X} + b$
  - For logistic regression: $f_{\vec{w}, b} (\vec{X}) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{X} + b)}}$
- So although the algorithm looks the same on paper, linear regression and logistic regression are two very different algorithms, because $f$ is defined differently.

## Same Concepts as Linear Regression

These ideas from linear regression apply to logistic regression too:

1. Monitoring gradient descent (learning curve)
2. Vectorized implementation
3. Feature scaling
   - Scaling the features so they take on similar ranges of values speeds up gradient descent for logistic regression, just as it does for linear regression.
