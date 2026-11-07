---
title: "Chapter 6: Fit the Data Well"
description: Overfitting, underfitting, and how regularization helps a model fit the data well
date: 2026-10-17
numbering:
  enumerator: 6.%s
---

A learning algorithm can sometimes perform poorly because of a problem called **overfitting**. This chapter looks at overfitting, its opposite, **underfitting**, and what to do about them.

What we really want is a learning algorithm that **generalizes** well: one that makes good predictions even on brand-new examples it has never seen before.

:::{prf:definition} Generalization
:label: def-generalization

A model **generalizes** well when it makes good predictions on new examples it has never seen before, not just on its training data.
:::

You could say the goal of machine learning is to find a model that neither underfits nor overfits. In other words, we want a model that has neither high bias nor high variance.

# Overfitting vs Underfitting

## Linear Regression

Let's go back to our original example: using linear regression to predict the price of a house from its size.

:::{figure}
:label: 01
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

- **High variance:** Fit this fourth-order polynomial to two datasets that differ only slightly, and two different machine learning engineers could end up with wildly different predictions. Because the predictions vary so much with small changes in the data, we say the algorithm has high variance.

:::{prf:definition} High Variance
:label: def-high-variance

An algorithm has **high variance** when fitting it to slightly different training sets can produce very different predictions. High variance is another way of saying that the model overfits.
:::

:::{note} Story: Goldilocks and the Three Bears
Underfitting and overfitting, high bias and high variance, sometimes remind me of the children's story of Goldilocks and the Three Bears. Goldilocks, a little girl, wanders into the home of a family of bears and finds three bowls of porridge. The first is too cold to taste, so it's no good. The second is too hot to eat, which is no good either. The third is neither too cold nor too hot: its temperature is in the middle, just right to eat.

A good model is like that third bowl: neither underfitting nor overfitting, but just right.
:::

## Logistic Regression

:::{figure}
:label: 02
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

- With all these higher-order polynomial features, the algorithm is free to choose an overly complex decision boundary.

You've now seen both ways an algorithm can go wrong: it can underfit, with high bias, or overfit, with high variance.

# Addressing Overfitting

So what can you do once you suspect overfitting? Suppose you fit a model and it has high variance, meaning it's overfit.

:::{figure}
:label: 03
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

## Method 1: Collect more training data

- More data helps. With additional training examples of house sizes and prices, the learning algorithm learns a smoother, less wiggly function.

:::{figure}
:label: 04
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

- You can even keep the high-order polynomial, or another function with lots of features. Given enough training examples, it will still do fine.

:::{important} Key idea
Getting more training data is the single most effective way to fight overfitting.
:::

## Method 2: Feature Selection

- Select which features to include and which to exclude.
- A second way to address overfitting is to see whether you can use fewer features.
  - For example: $f(x) = w_1 x_1 + w_2 x_2^2 + w_3 x_3^3 + w_4 x_4 ^ 4 + b$
- That's a lot of polynomial features. Not using so many of them is one way to reduce overfitting.
- The same applies to ordinary features. To predict a house's price, you might have many of them: the size, number of bedrooms, number of floors, age, average income of the neighborhood, total distance to the nearest coffee shop, and so on.
  - With lots of features like these but not enough training data, the learning algorithm may also overfit the training set.
- Out of, say, 100 features, you could keep just a subset of the most useful ones, such as the size, the number of bedrooms, and the age of the house.
  - If those really are the most relevant features, a model that uses only this small subset may no longer overfit as badly.

:::{prf:definition} Feature Selection
:label: def-feature-selection

**Feature selection** means deciding which subset of the available features a model should use. A simple approach is to rely on your intuition about which features matter most.
:::

:::{note}
Feature selection has a downside: every feature you drop takes some information with it. If all 100 features are actually useful for predicting a house's price, discarding some of them means losing useful information.
:::

To summarize:

1. All features + insufficient data → overfitting
2. Selected features → just right
   - Disadvantage: useful features could be lost

## Method 3: Regularization

- Reduce the size of the parameters.
- Look again at an overfit model that uses polynomial features: $x, x^2, x^3$, and so on.

:::{figure}
:label: 05
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

:::{note}
Setting a parameter to 0 is equivalent to eliminating a feature.
:::

- Regularization takes a softer approach: instead of removing features entirely, it reduces their impact.
- It pushes the learning algorithm toward smaller parameter values, without requiring any parameter to be exactly 0.
- So even with a higher-order polynomial like this one, if the algorithm keeps $w_1, w_2, w_3, w_4$ small, the resulting curve fits the data much better.
- In other words, you get to keep all of your features, but none of them can have an overly large effect, which is what can sometimes cause overfitting.
- By convention, we shrink only the $w_j$ parameters, $w_1$ through $w_n$. Regularizing $b$ as well makes very little difference in practice, so it's optional. I usually leave $b$ alone and regularize only $w_1, w_2, \dots, w_n$, which works just fine.

## Recap

There are three ways to address overfitting:

1. Collect more data
2. Select features (feature selection)
3. Reduce the size of the parameters using regularization

# Regularization

Regularization reduces the size of the parameters.

## Cost Function with Regularization

Regularization works by keeping the parameters $w_1$ through $w_n$ small to reduce overfitting. To put that idea into practice, we'll build a modified cost function that the learning algorithm can minimize.

### Intuition

:::{figure}
:label: 06
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

**Goal:** make $w_3$ and $w_4$ really small ($\approx 0$).

Here's a trick: take the usual linear regression cost function and add two extra terms, $1000 w_3^2$ and $1000 w_4^2$. (There's nothing special about 1000; any really large number would do.) Now the cost goes up sharply whenever $w_3$ or $w_4$ is large, so the model is penalized for large values:

$$
\underset{\vec{w}, b}{\text{min}} \frac{1}{2m} \sum_{i = 1}^{m} (f_{\vec{w}, b} (\vec{X}^{(i)}) - y^{(i)})^2 + 1000w_{3}^2 + 1000 w_{4}^2
$$

The only way to keep this new cost low is to make $w_3$ and $w_4$ small. If either one is large, its term, multiplied by 1000, dominates the cost. So minimizing this function drives $w_3$ and $w_4$ close to 0.

With $w_3$ and $w_4$ near 0, the features $x^3$ and $x^4$ barely contribute, which is almost like deleting those two terms. The fitted curve ends up much closer to a quadratic function, with perhaps just tiny contributions from $x^3$ and $x^4$.

That's a much better fit to the data than the wiggly curve you get when all the parameters are free to be large.

### Regularization

More generally, here's the idea behind regularization.

Keeping the parameters small has an effect similar to using a simpler model, such as one with fewer features, and simpler models are less prone to overfitting. In the previous example, we penalized, or regularized, only $w_3$ and $w_4$. But with many features, say 100, you usually don't know which ones matter most, or which ones to penalize.

So in practice, regularization penalizes all of the features, or more precisely, all of the $w_j$ parameters. It's possible to show that this usually leads to a smoother, simpler, less wiggly function that is less prone to overfitting.

With 100 features per house, deciding in advance which ones to keep and which to drop would be hard. So instead, let's build a model that uses all $n$ features:

$$
J(\vec{w}, b) = \frac{1}{2m} \sum_{i = 1}^{m} (f_{\vec{w}, b} (\vec{X}^{(i)}) - y^{(i)}) ^ 2 + \frac{\lambda}{2m} \sum_{j = 1}^{n} w_{j}^2 + \frac{\lambda}{2m}b^2
$$

- **$\lambda$** (lambda) is called the **regularization parameter**. Like the learning rate $\alpha$, it's a number you have to choose, with $\lambda > 0$.
- Dividing $\lambda$ by $2m$ puts both terms of the cost on the same $\frac{1}{2m}$ scale. That makes a good value of $\lambda$ easier to find. It also means that a value that works now is more likely to keep working if you later add training examples and $m$ gets bigger.
- The last term, which penalizes $b$, is optional. By convention, we don't penalize $b$ for being large: it makes very little difference in practice, and the more common convention, which this book follows, is to regularize only the parameters $w$, not $b$.

:::{figure}
:label: 07
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

To summarize, the modified cost has two parts: the original mean squared error cost, plus a second part called the **regularization term**. Minimizing their sum means balancing two goals:

1. **Fit the training data well.** The first term is small when the predictions are close to the actual values.
2. **Keep the parameters small.** The second term is small when the parameters $w_j$ are small, which tends to reduce overfitting.

The value of $\lambda$ you choose decides how much weight each goal gets, that is, how you trade one off against the other.

:::{prf:definition} Regularization
:label: def-regularization

**Regularization** adds a term to the cost function that penalizes large parameter values, $\frac{\lambda}{2m} \sum_{j = 1}^{n} w_{j}^2$. The **regularization parameter** $\lambda$ controls how strongly the parameters are pushed toward small values, which tends to reduce overfitting.
:::

#### Choosing Lambda

- **$\lambda = 0$ → overfits.** Setting $\lambda$ to 0 multiplies the regularization term by 0, so it's as if you weren't regularizing at all.
- **$\lambda$ very large → underfits.** With a huge value, say $\lambda = 10^{10}$, the regularization term dominates the cost. The only way to minimize it is to make all the parameters $w_j$, including $w_1$ through $w_4$, extremely close to 0. Then $f(x)$ is basically just $b$: the learning algorithm fits a flat, horizontal line and underfits.
- **$\lambda$ in between → balanced.** The best choice lies somewhere in the middle, balancing a low mean squared error against small parameters.

:::{figure}
:label: 08
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

#### Regularized Linear Regression

Here is the cost function for regularized linear regression:

$$
J(\vec{w}, b) = \frac{1}{2m} \sum_{i = 1}^{m} (f_{\vec{w}, b} (\vec{X}^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m}\sum_{j = 1}^{n} w_{j}^2
$$

Our task is to:

$$
\underset{\vec{w}, b}{\text{min }} J(\vec{w}, b) = \underset{\vec{w}, b}{\text{min }} \frac{1}{2m} \sum_{i = 1}^{m} (f_{\vec{w}, b} (\vec{X}^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m}\sum_{j = 1}^{n} w_{j}^2
$$

Gradient descent is:

$$
\begin{aligned}
&repeat \ until \ convergence \  \{ \\
&w = w - \alpha \frac{d}{dw}J(w, b) \\
&b = b - \alpha \frac{d}{db}J(w, b) \\
&\} \text{ simultaneous update}
\end{aligned}
$$

Previously, the derivatives were:

$$
\frac{\partial}{\partial w_{j}} J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w}, b} (\vec{X}^{(i)}) - y^{(i)}) x_{j}^{(i)}
$$

and

$$
\frac{\partial}{\partial b} J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w}, b} (\vec{X}^{(i)}) - y^{(i)})
$$

Now that we've added the regularization term, the only thing that changes is the derivative with respect to $w_j$:

$$
\frac{\partial}{\partial w_{j}} J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w}, b} (\vec{X}^{(i)}) - y^{(i)}) x_{j}^{(i)} + \frac{\lambda}{m}w_{j}
$$

We're not regularizing $b$, so its derivative stays the same:

$$
\frac{\partial}{\partial b} J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w}, b} (\vec{X}^{(i)}) - y^{(i)})
$$

Substituting these derivatives into the gradient descent updates gives the algorithm for regularized linear regression:

$$
\begin{aligned}
&\text{repeat until convergence} \{ \\
&	w_{j} = w_{j} - \alpha \big[ \frac{1}{m}\sum_{i=1}^{m} \big[(f_{\vec{w},b}(\vec{X}^{(i)}) - y^{(i)}) x_{j}^{(i)}\big] + \frac{\lambda}{m}w_{j} \big] \\
&b = b - \alpha \frac{1}{m}\sum_{i=1}^{m} (f_{\vec{w},b}(\vec{X}^{(i)}) - y^{(i)}) \\
&		\} \text{ simultaneous update}
\end{aligned}
$$

##### Deeper Intuition (Math)

$$
w_{j} = w_{j} - \alpha \big[ \frac{1}{m}\sum_{i=1}^{m} \big[(f_{\vec{w},b}(\vec{X}^{(i)}) - y^{(i)}) x_{j}^{(i)}\big] + \frac{\lambda}{m}w_{j} \big]
$$

Rewrite it as:

$$
w_{j} = 1 w_{j} - \alpha \frac{\lambda}{m} w_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w}, b} (\vec{X}^{(i)}) - y^{(i)}) x_{j}^{(i)}
$$

Simplifying:

$$
w_{j} = w_{j} \left( 1 - \alpha \frac{\lambda}{m} \right) - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w}, b} (\vec{X}^{(i)}) - y^{(i)}) x_{j}^{(i)}
$$

You might recognize the second term: it's the usual gradient descent update for linear regression without regularization.

**Intuition:** Let's plug in some numbers: $\lambda = 1$, a training set of $m = 50$ examples, and a learning rate $\alpha = 0.01$. Then $\alpha \frac{\lambda}{m} = 0.01 \times 1 / 50 = 0.0002$, a small positive number, so the factor $1 - \alpha \frac{\lambda}{m}$ equals 0.9998, just under 1.

- On every iteration, before the usual update, $w_j$ gets multiplied by this factor of 0.9998.
- Multiplying by a number just below 1 shrinks $w_j$ a little bit every single time. That's another way to see what regularization does: it gently shrinks each $w_j$ on every iteration.

##### How we get the derivative term

$$
\frac{\partial}{\partial w_{j}} J(\vec{w}, b) = \frac{\partial}{\partial w_{j}} \big[\frac{1}{2m} \sum_{i=1}^{m} (f(\vec{X}^{(i)})- y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_{j}^2 \big]
$$

$$
= \frac{1}{2m} \sum_{i=1}^{m} \big[ (\vec{w} \cdot \vec{X}^{(i)} + b - y^{(i)}) 2x_{j}^{(i)} \big] + \frac{\lambda}{2m} 2 w_{j}
$$

$$
= \frac{1}{m} \sum_{i=1}^{m} \big[ (\vec{w} \cdot \vec{X}^{(i)} + b - y^{(i)}) x_{j}^{(i)} \big] + \frac{\lambda}{m}  w_{j}
$$

That's everything you need to implement regularized linear regression.

:::{important} Key idea
Regularization is most valuable when you have many features but a relatively small training set. In that situation, it can make linear regression work much better on many problems.
:::

#### Regularized Logistic Regression

The cost function for logistic regression is:

$$
J(\vec{w}, b) = -\frac{1}{m} \sum_{i = 1}^{m} \big[ y^{(i)} \log(f_{\vec{w}, b}(\vec{X}^{(i)})) + (1-y^{(i)}) \log (1 - f_{\vec{w}, b} (\vec{X}^{(i)})) \big]
$$

To add regularization, we just add the regularization term:

$$
J(\vec{w}, b) = -\frac{1}{m} \sum_{i = 1}^{m} \big[ y^{(i)} \log(f_{\vec{w}, b}(\vec{X}^{(i)})) + (1-y^{(i)}) \log (1 - f_{\vec{w}, b} (\vec{X}^{(i)})) \big] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_{j}^2
$$

The extra term means that, as gradient descent minimizes the cost with respect to $\vec{w}$ and $b$, large values of $w_1$ through $w_n$ are penalized, which keeps the parameters from growing too big:

$$
\underset{\vec{w}, b}{\text{min }} J(\vec{w}, b) = \underset{\vec{w}, b}{\text{min }} -\frac{1}{m} \sum_{i = 1}^{m} \big[ y^{(i)} \log(f_{\vec{w}, b}(\vec{X}^{(i)})) + (1-y^{(i)}) \log (1 - f_{\vec{w}, b} (\vec{X}^{(i)})) \big] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_{j}^2
$$

Here's gradient descent with regularization:

$$
\begin{aligned}
&\text{repeat \{}  \\
&w_{j} = w_{j} - \alpha \frac{\partial}{\partial w_{j}} J(\vec{w}, b) \\
&b = b - \alpha \frac{\partial}{\partial b} J(\vec{w}, b) \\
&\} \text{ Simultaneous Updates}
\end{aligned}
$$

$$
\begin{aligned}
&\text{repeat \{}  \\
&w_{j} = w_{j} - \alpha \left[ \frac{1}{m} \sum_{i = 1}^{m} (f_{\vec{w}, b} (\vec{X}^{(i)}) - y^{(i)}) x_{j}^{(i)} \right] + \frac{\lambda}{m} w_{j}\\
&b = b - \alpha \left[ \frac{1}{m} \sum_{i = 1}^{m} (f_{\vec{w}, b} (\vec{X}^{(i)}) - y^{(i)}) \right] \\
&\} \text{ Simultaneous Updates}
\end{aligned}
$$

Since $b$ is not regularized, its update doesn't change.

# Summary

To fit the model well:

1. Collect more data
2. Feature selection
3. Regularization
