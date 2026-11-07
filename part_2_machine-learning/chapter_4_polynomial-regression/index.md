---
title: "Chapter 4: Polynomial Regression"
description: Implement Polynomial Regression from scratch
date: 2026-10-12
numbering:
  enumerator: 4.%s
---

# Polynomial Regression

:::{prf:definition} Polynomial Regression
:label: def-polynomial-regression

**Polynomial regression** is a form of regression that can fit curved, non-linear functions to data, rather than only straight lines.
:::

:::{figure}
:label: ch04-01
:no-subfigures:

```{image} diagrams/0401-light.svg
:alt: FIG
:class: dark:hidden
```

```{image} diagrams/0401-dark.svg
:alt: FIG
:class: hidden dark:block
```
:::

See [Lab: Polynomial Regression](lab/polynomial-regression.ipynb).

## In Real Life

In real life, intuition suggests candidate equations, and running the models decides between them. In our lab notebook, I knew the perfect model only because I generated the data from it. Real data rarely has a hidden simple formula, so in practice, you stop when the model is as good as the noise allows.

Ideas for candidate equations usually come from two places:

1. **Domain knowledge.** If something has diminishing returns (each extra dollar of ads helps a little less), $\ln(x)$ is a natural guess. If something grows with an area, $x^2$ is.
2. **Plots.** The most useful plot isn't the raw data. It's the **residuals**: fit a simple model, then plot what it failed to explain, $y - \hat{y}$ (the actual value minus the prediction), against $x$.

:::{prf:definition} Residual
:label: def-residual

A **residual** is the part of a target value that a model fails to explain: the actual value minus the prediction, $y - \hat{y}$.
:::
