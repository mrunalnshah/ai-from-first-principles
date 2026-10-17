---
title: "Chapter 7: Neural Network"
description: 
date: 2026-10-15
numbering:
  enumerator: 7.%s
---
In this chapter, you will learn about neural networks, also called deep learning algorithms. These are some of the most powerful and widely used machine learning algorithms and you'd get to implement them and get them to work for yourself.

we'll go over neural networks and how to carry out inference or prediction. If you were to go to the Internet and download the parameters of a neural network that someone else had trained and whose parameters that posted on the Internet, then to use that neural network to make predictions would be called inference. Then, you'll learn how to train your own neural network. 

# Origins
- Work in neural networks had started back in the 1950s, and then it fell out of favor for a while.
- Then in the 1980s and early 1990s, they gained in popularity again and showed tremendous traction in some applications like handwritten digit recognition, which were used even backed then to read postal codes for writing mail and for reading dollar figures in handwritten checks.
- But then it fell out of favor again in the late 1990s.
- It was from about 2005 that it enjoyed a resurgence and also became re-branded little bit with deep learning. 
	- deep learning and neural networks meant very similar things.
- Since then, neural networks have revolutionized application area after application area.
- I think the first application area that modern neural networks or deep learning, had a huge impact on was probably speech recognition, where we started to see much better speech recognition systems due to modern deep learning and authors such as Geoff Hinton were instrumental to this, and then it started to make inroads into computer vision. 
- Sometimes people still speak of the ImageNet moments in 2012, and that was maybe a bigger splash where then caught broder imagination and had a big impact on computer vision.
- Then the next few years, it made us inroads into texts or into natural language processing, and so on and so forth.
- Now, neural networks are used in everything from climate change to medical imaging to online advertising to prouduct recommendations and really lots of application areas of machine learning now use neural networks.

# Neurons, The Brain and Why now?

## Human Brain vs Artificial Brain
Even though today's neural networks have almost nothing to do with how the brain learns, there was the early motivation of trying to build software to mimic the brain.  

:::{note}
When neural networks were first invented many decades ago, the original motivation was to write software that could mimic how the human brain or how the biological brain learns and thinks. Even though today, neural networks, sometimes also called artificial neural networks, have become very different than how any of us might think about how the brain actually works and learns.
:::

**Artificial Neural Networks are algorithms that try to mimic the brain.**

So how does the brain work? 

:::{figure}
:label: ch07-01
:no-subfigures:

```{image} diagrams/01-light.svg
:alt: Illustration by Christy Krames, MA, CMI
:class: dark:hidden
```

```{image} diagrams/01-dark.svg
:alt: Illustration by Christy Krames, MA, CMI
:class: hidden dark:block
```

Illustration by Christy Krames, MA, CMI @kramesnia
::: 

Above is a diagram illustrating what neurons in a brain look like. 

All of human thought is from neurons like this in your brain and mine, sending electrical impulses and sometimes forming new connections of other neurons.

Given a neuron like this one, it has a number of inputs where it receives electrical impulses from other neurons, and then this neuron will then send this outputs to other neurons by electrical impulses, and upper neuron's output becomes the input to this neuron down below, which again aggregates inputs from multiple other neurons to then maybe send its own output, to yet other neurons, and this is the stuff of which human thought is made.

:::{figure}
:label: ch07-02
:no-subfigures:

```{image} diagrams/02-light.svg
:alt: Simplified diagram of neuron
:class: dark:hidden
```

```{image} diagrams/02-dark.svg
:alt: Simplified diagram of neuron
:class: hidden dark:block
```

Simplified diagram of neuron
::: 

Here's a simplified diagram of a biological neuron. A neuron comprises a cell body, nucleus, dendrites, axon. You don't need to remember this, just keep the intuition of how you think in real. (Though we still don't know how brain works!)

| Part | Function |
|:------|-----------|
| Nucleus   | Contains DNA and controls the neuron's activities. |
| Cell body (soma)   | Maintains the neuron and integrates incoming signals. |
| Dendrites   | Receive signals from other neurons. |
| Axon   | Carries electrical signals away from the cell body to other cells. |


the artificial neural network uses a very simplified Mathematical model of what a biological neuron does.

:::{figure}
:label: ch07-03
:no-subfigures:

```{image} diagrams/03-light.svg
:alt: Simplified diagram of artificial neuron 
:class: dark:hidden
```

```{image} diagrams/03-dark.svg
:alt: Simplified diagram of artificial neuron 
:class: hidden dark:block
```

Simplified diagram of artificial neuron 
::: 

What a neuron does is it takes some inputs, one or more inputs, which are just numbers. It does some computation and it outputs some other number, which then could be an input to a second neuron. When you're building an artificial neural network or deep learning algorithm, rather than building one neuron at a time, you often want to simulate many such neurons at the same time. What these neurons do collectively is input a few numbers, carry out some computation, and output some other numbers. 

even with these extremely simplified models of a neuron, we'll be able to build really powerful deep learning algorithms.

## Why Neural Networks and Why Now?
Why is it that only in the last handful of years that neural networks have really taken off?"

Over the last couple of decades, with the rise of the Internet, the rise of mobile phones, the digitalization of our society, the amount of data we have for a lot of applications has been rising. Every data is being digitalized like your health records, your bills, etc...

So in many application areas, the amount of digital data has exploded. 

### What happened to Traditional Learning Algorithms?
What we saw was with traditional machine-learning algorithms, such as logistic regression and linear regression, even as you fed those algorithms more data, it was very difficult to get the performance to keep on going up. So it was as if the traditional learning algorithms like linear regression and logistic regression, they just weren't able to scale with the amount of data we could now feed it and they weren't able to take effective advantage of all this data we had for different applications.

:::{figure}
:label: ch07-04
:no-subfigures:

```{image} diagrams/04-light.svg
:alt: Why Now? 
:class: dark:hidden
```

```{image} diagrams/04-dark.svg
:alt: Why Now?
:class: hidden dark:block
```

Why Now? 
::: 

Let's now dive more deeply into the details of how neural network actually works.

# Demand Prediction

