---
title: "Chapter 7: Neural Network"
description: 
date: 2026-10-15
numbering:
  enumerator: 7.%s
---
In this chapter, you will learn about neural networks, also called deep learning algorithms. They are some of the most powerful and widely used machine learning algorithms, and you'll get to implement them and make them work yourself.

We'll start with how neural networks carry out inference, or prediction. If you downloaded the parameters of a neural network that someone else had trained and posted online, using that network to make predictions would be called inference. Then, you'll learn how to train your own neural network.

# Origins

- Work on neural networks started in the 1950s, then fell out of favor for a while.
- In the 1980s and early 1990s, they gained popularity again and showed tremendous traction in applications like handwritten digit recognition, which was used even then to read postal codes for routing mail and dollar amounts on handwritten checks.
- They fell out of favor again in the late 1990s.
- From about 2005, they enjoyed a resurgence and were rebranded somewhat as deep learning.
  - Deep learning and neural networks mean very similar things.
- Since then, neural networks have revolutionized application area after application area.
- The first area where modern deep learning had a huge impact was probably speech recognition, where it led to much better systems, with researchers such as Geoff Hinton playing an instrumental role. It then started to make inroads into computer vision.
- People still speak of the ImageNet moment of 2012, which was maybe an even bigger splash: it captured the broader imagination and had a big impact on computer vision.
- Over the next few years, it made inroads into text (natural language processing).
- Today, neural networks are used in everything from climate change and medical imaging to online advertising and product recommendations, and many areas of machine learning now rely on them.

# Neurons, the Brain, and Why Now?

## Human Brain vs Artificial Brain

When neural networks were first invented many decades ago, the original motivation was to write software that could mimic how the biological brain learns and thinks.

:::{note}
Today's neural networks, also called artificial neural networks, have almost nothing to do with how the brain actually works and learns.
:::

**Artificial neural networks are algorithms that try to mimic the brain.**

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

The diagram above shows what neurons in the brain look like.

All human thought comes from neurons like these, sending electrical impulses and sometimes forming new connections with other neurons.

Each neuron has a number of inputs through which it receives electrical impulses from other neurons, and it sends its own output to other neurons as electrical impulses. For example, the upper neuron's output becomes an input to the neuron below, which aggregates inputs from multiple neurons and may then send its own output to yet other neurons.

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

The figure above is a simplified diagram of a biological neuron. It consists of a cell body, a nucleus, dendrites, and an axon. You don't need to memorize these parts; just keep an intuition for how real neurons work. (We still don't fully understand how the brain works!)

| Part | Function |
|:-----|----------|
| Nucleus | Contains DNA and controls the neuron's activities. |
| Cell body (soma) | Maintains the neuron and integrates incoming signals. |
| Dendrites | Receive signals from other neurons. |
| Axon | Carries electrical signals away from the cell body to other cells. |

An artificial neural network uses a very simplified mathematical model of what a biological neuron does.

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

An artificial neuron takes one or more inputs (just numbers), does some computation, and outputs another number, which can then be the input to a second neuron. When building an artificial neural network or deep learning algorithm, rather than building one neuron at a time, you often simulate many neurons at once. Collectively, these neurons take a few numbers as input, carry out some computation, and output some other numbers.

Even with these extremely simplified models of a neuron, we can build really powerful deep learning algorithms.

## Why Neural Networks and Why Now?

Why have neural networks only really taken off in the last handful of years?

Over the last couple of decades, the rise of the internet and mobile phones, along with the digitalization of society, has steadily increased the amount of data available for many applications. More and more information, such as health records and bills, is now digital.

So in many application areas, the amount of digital data has exploded.

### What Happened to Traditional Learning Algorithms?

With traditional machine learning algorithms such as linear regression and logistic regression, it was very difficult to keep improving performance even as you fed them more data. They just weren't able to scale with the amount of data available or take effective advantage of it.

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

Let's now dive into the details of how neural networks actually work.

# Neural Network Example: Demand Prediction

Consider an example from demand prediction: you look at a product and try to predict whether it will be a top seller.

## Simple Form

In this example, you're selling T-shirts and want to know whether a particular T-shirt will be a top seller. You've collected data on different T-shirts sold at different prices, along with which ones became top sellers.

**Retailers use this type of application today to better plan inventory levels and marketing campaigns.**

:::{figure}
:label: ch07-05
:no-subfigures:

```{image} diagrams/05-light.svg
:alt: One-feature Neuron
:class: dark:hidden
```

```{image} diagrams/05-dark.svg
:alt: One-feature Neuron
:class: hidden dark:block
```
::: 

The single neuron above is just a logistic regression unit, which is much simpler than what any biological neuron in your brain does. That's why an artificial neural network is such a vastly oversimplified model of the human brain.

Given this description of a single neuron, building a neural network just requires taking a bunch of these neurons and wiring them together. Let's look at a more complex example of demand prediction.

## Complex Form

In this example, we use four features to predict whether a T-shirt is a top seller:

1. Price of the T-shirt
2. Shipping costs
3. Amount of marketing for that T-shirt
4. Material quality (is it high-quality, thick cotton or a lower-quality material?)

You might suspect that whether a T-shirt becomes a top seller actually depends on a few factors:

1. Affordability: how affordable the T-shirt is
2. Awareness: how aware potential buyers are of the T-shirt
3. Perceived quality: whether buyers or potential buyers think it's a high-quality T-shirt

We can derive these factors from the input features:

:::{figure}
:label: ch07-06
:no-subfigures:

```{image} diagrams/06-light.svg
:alt: Important Features from inputs
:class: dark:hidden
```

```{image} diagrams/06-dark.svg
:alt: Important Features from inputs
:class: hidden dark:block
```
::: 

Let's build our neural network!

:::{figure}
:label: ch07-07
:no-subfigures:

```{image} diagrams/07-light.svg
:alt: Important Features from inputs
:class: dark:hidden
```

```{image} diagrams/07-dark.svg
:alt: Important Features from inputs
:class: hidden dark:block
```
::: 

**The term *activation* comes from biological neurons. It refers to the degree to which a neuron is sending a high output value, or many electrical impulses, to the neurons downstream from it.**

This neural network carries out its computations as follows. It takes four numbers as input. The middle layer uses those four numbers to compute three new numbers, called activation values. Then the final layer, the output layer, uses those three numbers to compute one number.

- The list of four input numbers is also called the input layer.
- In a neural network, it's all numbers: from one layer, through activation functions, to the next layer, we're just computing numbers.

So far, we've gone through the neurons one at a time and decided which inputs each takes from the previous layer. For example, we said affordability is a function of just price and shipping costs, awareness is a function of just marketing, and so on. But in a large neural network, it would be a lot of work to manually decide which neurons should take which features as inputs.

In practice, each neuron in a layer (say, the middle layer) has access to every feature, that is, every value from the previous layer.

:::{figure}
:label: ch07-08
:no-subfigures:

```{image} diagrams/08-light.svg
:alt: Important Features from inputs
:class: dark:hidden
```

```{image} diagrams/08-dark.svg
:alt: Important Features from inputs
:class: hidden dark:block
```
::: 

For example, the neuron estimating affordability sees price, shipping cost, marketing, and material, but by setting its parameters appropriately, it can learn to ignore marketing and material and focus only on the features most relevant to affordability.

To simplify the notation, we write the four input features as a feature vector $\vec{x}$. This vector is fed to the middle layer, which computes three activation values. These three values form another vector that is fed to the output layer, which outputs the probability of the T-shirt being a top seller.

**That's all a neural network is: a few layers, where each layer takes a vector as input and outputs another vector of numbers.**

### Why Is the Middle Layer Called a "Hidden" Layer?

It's not the most intuitive name, but the term comes from the training set. In a training set, you observe both $x$ and $y$: the data tells you the correct inputs and the correct outputs. But it doesn't tell you the correct values for affordability, awareness, and perceived quality. Those values are hidden; you don't see them in the training set, which is why the middle layer is called a hidden layer.

### Think of a Neural Network as...

If we cover up the left half of the diagram, here's what we're left with:

:::{figure}
:label: ch07-09
:no-subfigures:

```{image} diagrams/09-light.svg
:alt: Important Features from inputs
:class: dark:hidden
```

```{image} diagrams/09-dark.svg
:alt: Important Features from inputs
:class: hidden dark:block
```
::: 

What's left is just logistic regression. It takes affordability, awareness, and perceived quality as input and uses these three features to estimate the probability of the T-shirt being a top seller. The cool thing is that instead of the original features (price, shipping cost, marketing, and so on), it uses a potentially better set of features that is hopefully more predictive of whether the T-shirt will be a top seller.

***One way to think of this neural network is as logistic regression, but a version that can learn its own features, making it easier to make accurate predictions.***

:::{tip} Manual Feature Engineering
Recall the housing example from the previous chapter: to predict the price of a house, we multiplied the frontage (width) of the lot by its depth to construct a more complex feature, $x_1 \times x_2$, the area of the lot. That was manual feature engineering: we had to look at $x_1$ and $x_2$ and decide by hand how to combine them into better features. Instead, a neural network can learn its own features, as you'll see later, to make the learning problem easier for itself. This is what makes neural networks one of the most powerful learning algorithms today.
:::

To summarize: the input layer has a vector of features (four numbers in this example), which is input to the hidden layer. The hidden layer outputs three numbers, which we denote as the activation vector $\vec{a}$. The output layer takes these three numbers as input and outputs one number: the final activation, or final prediction, of the neural network.

:::{note}
One really nice property of a neural network is that when you train it from data, you don't need to explicitly decide which features, such as affordability, it should compute. It figures out by itself which features to use in the hidden layer, which is what makes it such a powerful learning algorithm.
:::

Let's look at some other examples of neural networks.

## More Than One Hidden Layer

### Example 1

:::{figure}
:label: ch07-10
:no-subfigures:

```{image} diagrams/10-light.svg
:alt: Important Features from inputs
:class: dark:hidden
```

```{image} diagrams/10-dark.svg
:alt: Important Features from inputs
:class: hidden dark:block
```
::: 

This neural network has an input feature vector $\vec{x}$ that is fed to the first hidden layer. If this layer has three neurons, it outputs a vector of three activation values. These three numbers are then input to the second hidden layer. If the second hidden layer has two neurons, or logistic units, it outputs a vector of two activation values, which is fed to the output layer to produce the network's final prediction.

### Example 2

:::{figure}
:label: ch07-11
:no-subfigures:

```{image} diagrams/11-light.svg
:alt: Important Features from inputs
:class: dark:hidden
```

```{image} diagrams/11-dark.svg
:alt: Important Features from inputs
:class: hidden dark:block
```
::: 

In this neural network, the input goes to the first hidden layer, whose output goes to the second hidden layer, then to the third hidden layer, and finally to the output layer.

## Building Neural Networks

**When building your own neural network, one decision you need to make is how many hidden layers to use and how many neurons each hidden layer should have.**

These choices make up the *architecture* of the neural network. Choosing the right number of hidden layers and hidden units per layer can affect the performance of the learning algorithm, and later in this chapter you'll learn some tips for choosing a good architecture.

In some of the literature, a neural network with multiple layers is called a ***multilayer perceptron***. (If you see that term, it just refers to a neural network.)

# Neural Network Intuition: Image Recognition

## Face Recognition

If you're building a face recognition application, you might want to train a neural network that takes a picture as input and outputs the identity of the person in it.

:::{figure}
:label: ch07-12
:no-subfigures:

```{image} diagrams/12-light.svg
:alt: Image Debunk
:class: dark:hidden
```

```{image} diagrams/12-dark.svg
:alt: Image Debunk
:class: hidden dark:block
```
::: 

The picture is fed in as a feature vector of a million pixel brightness values.

This is how you might build a neural network for this task: the input image $\vec{x}$ is fed to the first hidden layer, which extracts some features. Its output is fed to a second hidden layer, then to a third, and finally to the output layer, which estimates, say, the probability that the picture shows a particular person.

:::{figure}
:label: ch07-13
:no-subfigures:

```{image} diagrams/13-light.svg
:alt: Face Classification 
:class: dark:hidden
```

```{image} diagrams/13-dark.svg
:alt: Face Classification
:class: hidden dark:block
```
::: 

It's interesting to take a neural network trained on many images of faces and visualize what its hidden layers are computing. If you peer at the different neurons in the hidden layers, this is what you might find:

:::{figure}
:label: ch07-14
:no-subfigures:

```{image} diagrams/14-light.svg
:alt: Face Classification (debunk) 
:class: dark:hidden
```

```{image} diagrams/14-dark.svg
:alt: Face Classification (debunk)
:class: hidden dark:block
```
:::
@lee2009convolutional

1. In the first hidden layer, neurons might look for very short lines or edges in the image.
2. In the second hidden layer, neurons might learn to group many short lines and edge segments together to look for parts of faces.
3. In the third hidden layer, the network aggregates different parts of faces to detect the presence or absence of larger, coarser face shapes.

Finally, detecting how much the face corresponds to different face shapes creates a rich set of features that helps the output layer determine the identity of the person in the picture.

**A remarkable thing about neural networks is that they can learn these feature detectors at the different hidden layers all by themselves.**

In this example, no one ever told the network to look for short edges in the first layer; eyes, noses, and other face parts in the second; and more complete face shapes in the third.

:::{note}
In this visualization, the neurons in the first hidden layer look at relatively small windows of the image to find edges. The second hidden layer looks at bigger windows, and the third looks at even bigger ones. These small neuron visualizations actually correspond to differently sized regions of the image.
:::

## Car Classification

Just for fun, let's see what happens if you train this neural network on a different dataset, say, lots of pictures of cars.

:::{figure}
:label: ch07-15
:no-subfigures:

```{image} diagrams/15-light.svg
:alt: Car Classification (debunk) 
:class: dark:hidden
```

```{image} diagrams/15-dark.svg
:alt: Car Classification (debunk)
:class: hidden dark:block
```
:::
@lee2009convolutional

Asked to detect cars, the same learning algorithm learns edges in the first layer, much like before, but then learns to detect parts of cars in the second hidden layer and more complete car shapes in the third.

Just by being fed different data, the neural network automatically learns to detect very different features, suited to whatever task it's trained on, whether car detection or person recognition. That's how a neural network works in computer vision applications.

# Neural Network Model
The fundamental building block of ​most modern neural networks is a layer of neurons.

## Decoding Neural Network Layer
:::{figure}
:label: ch07-16
:no-subfigures:

```{image} diagrams/16-light.svg
:alt: Neural Network with 2 layers
:class: dark:hidden
```

```{image} diagrams/16-dark.svg
:alt: Neural Network with 2 layers
:class: hidden dark:block
```
:::

In the demand prediction example, four input features are fed into a hidden layer of three neurons, which sends its output to an output layer with a single neuron.

Let's zoom into the hidden layer to look at its computations.

:::{figure}
:label: ch07-17
:no-subfigures:

```{image} diagrams/17-light.svg
:alt: Layer 1 of Neural Network with 2 layers
:class: dark:hidden
```

```{image} diagrams/17-dark.svg
:alt: Layer 1 of Neural Network with 2 layers
:class: hidden dark:block
```
:::

The hidden layer takes four numbers as input, and each of its three neurons receives all four. Each neuron is just a small logistic regression unit.

The first neuron has two parameters, $\vec{w}_1$ and $b_1$, where the subscript marks it as the first hidden unit. It outputs the activation value $a_1 = g(\vec{w}_1 \cdot \vec{x} + b_1)$. Here, $\vec{w}_1 \cdot \vec{x} + b_1$ is the familiar $z$ from logistic regression, and $g(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid (logistic) function. If $a_1 = 0.3$, there's a 0.3 chance that the t-shirt is highly affordable, based on the input features.

The second neuron has parameters $\vec{w}_2$ and $b_2$ and computes $a_2 = g(\vec{w}_2 \cdot \vec{x} + b_2)$. If $a_2 = 0.7$, there's a 0.7 chance that potential buyers are aware of the t-shirt. Similarly, the third neuron has parameters $\vec{w}_3$ and $b_3$ and computes $a_3 = g(\vec{w}_3 \cdot \vec{x} + b_3)$, which might be 0.2.

The three neurons output 0.3, 0.7, and 0.2. This vector of activation values, $\vec{a}$, is passed to the final output layer.

When a neural network has multiple layers, it's useful to number them. By convention, the hidden layer is layer 1 and the output layer is layer 2. The input layer is sometimes called layer 0. Today, some neural networks have dozens or even hundreds of layers.

To distinguish between layers, we use a superscript in square brackets. For example, $\vec{a}^{[1]}$ denotes the output of layer 1, $\vec{w}_1^{[1]}, b_1^{[1]}$ are the parameters of the first unit in layer 1, and $\vec{w}_2^{[1]}, b_2^{[1]}$ are the parameters of the second unit. Similarly, $a_1^{[1]}, a_2^{[1]}, a_3^{[1]}$ are the activation values of the hidden units in layer 1.

In short, a superscript $[1]$ refers to a quantity associated with layer 1, $[2]$ to layer 2, and so on for networks with more layers.

Now let's zoom into layer 2, which is also the output layer.

:::{figure}
:label: ch07-18
:no-subfigures:

```{image} diagrams/18-light.svg
:alt: Layer 2 of Neural Network with 2 layers
:class: dark:hidden
```

```{image} diagrams/18-dark.svg
:alt: Layer 2 of Neural Network with 2 layers
:class: hidden dark:block
```
:::

The input to layer 2 is the output of layer 1, $\vec{a}^{[1]} = [0.3, 0.7, 0.2]$. Following the same convention, quantities associated with layer 2 get the superscript $[2]$. The output layer has a single neuron, which computes $a_1^{[2]} = g(\vec{w}_1^{[2]} \cdot \vec{a}^{[1]} + b_1^{[2]})$, where $g$ is again the sigmoid function. If this gives, say, 0.84, that becomes the final output of the network, $a^{[2]}$. Since there's only one output neuron, this output is a scalar rather than a vector.

Once the network has computed $a^{[2]}$, there's one final, optional step: if you want a binary prediction (is this a top seller, yes or no?), you can threshold the output.

:::{figure}
:label: ch07-19
:no-subfigures:

```{image} diagrams/19-light.svg
:alt: Layer 2 output of Neural Network with 2 layers
:class: dark:hidden
```

```{image} diagrams/19-dark.svg
:alt: Layer 2 output of Neural Network with 2 layers
:class: hidden dark:block
```
:::

Threshold $a^{[2]} = 0.84$ at 0.5: if it's at least 0.5, predict $\hat{y} = 1$; otherwise, predict $\hat{y} = 0$. This is the same thresholding used in logistic regression, and it gives a final prediction of 1 or 0 instead of just the probability of being a top seller.

In summary, each layer takes a vector of numbers as input, applies a set of logistic regression units to it, and computes a new vector that is passed to the next layer. This continues until the output layer computes the network's prediction, which you can optionally threshold at 0.5.

## A More Complex Neural Network

You've learned how a neural network layer takes a vector of numbers as input and outputs another vector of numbers. Now let's use that layer to build a more complex neural network, which will also make the notation we're using clearer and more concrete. Below is an example of a neural network with four layers:

:::{figure}
:label: ch07-20
:no-subfigures:

```{image} diagrams/20-light.svg
:alt: More Complex Neural Network
:class: dark:hidden
```

```{image} diagrams/20-dark.svg
:alt: More Complex Neural Network
:class: hidden dark:block
```
:::

This network has four layers, not counting the input layer: layers 1, 2, and 3 are hidden layers, and layer 4 is the output layer. As usual, the input layer is called layer 0. By convention, when we say a neural network has four layers, we count all the hidden layers and the output layer, but not the input layer.

In general, the activation output of unit $j$ in an arbitrary layer $l$ is:

$$
a_j^{[l]} = g(\vec{w}_j^{[l]} \cdot \vec{a}^{[l-1]} + b_j^{[l]})
$$

:::{figure}
:label: ch07-21
:no-subfigures:

```{image} diagrams/21-light.svg
:alt: notation of neural network
:class: dark:hidden
```

```{image} diagrams/21-dark.svg
:alt: notation of neural network
:class: hidden dark:block
```
:::

Here, $\vec{w}_j^{[l]}$ and $b_j^{[l]}$ are the weight vector and bias of unit $j$ in layer $l$: the superscript $[l]$ denotes the layer, and the subscript $j$ denotes the unit. Notice that the dot product is taken with $\vec{a}^{[l-1]}$, not $\vec{a}^{[l]}$, because each layer takes the output of the previous layer as its input. For example, the activation of unit 2 in layer 3 is $a_2^{[3]} = g(\vec{w}_2^{[3]} \cdot \vec{a}^{[2]} + b_2^{[3]})$.

Unit $j$ is the $j$th neuron in the layer; the terms *unit* and *neuron* are used interchangeably.

The function $g$ is the sigmoid function. In the context of neural networks, $g$ is also called the *activation function*, because it outputs the activation values. So far, the only activation function you've seen is the sigmoid function.

One last piece of notation: to keep things consistent, we also denote the input vector $\vec{x}$ as $\vec{a}^{[0]}$. This way, the same equation works for the first layer too. When $l = 1$, the activations of the first layer are $a_j^{[1]} = g(\vec{w}_j^{[1]} \cdot \vec{a}^{[0]} + b_j^{[1]})$, where $\vec{a}^{[0]}$ is just the input feature vector $\vec{x}$.

With this notation, you can compute the activation values of any layer in a neural network as a function of its parameters and the activations of the previous layer.

Next, let's put this into an inference algorithm, that is, an algorithm for getting a neural network to make predictions.

## Inference: Making Predictions

The algorithm a neural network uses to make an inference, or prediction, is called **forward propagation**.

Let's take handwritten digit recognition as an example. For simplicity, we'll just distinguish between the handwritten digits 0 and 1, so it's a binary classification problem: given an input image, is it the digit 0 or the digit 1?

:::{figure}
:label: ch07-22
:no-subfigures:

```{image} diagrams/22-light.svg
:alt: Handwritten Digit Binary Classification
:class: dark:hidden
```

```{image} diagrams/22-dark.svg
:alt: Handwritten Digit Binary Classification
:class: hidden dark:block
```
:::

In this example, the image of a 1 is an $8 \times 8$ grid, or matrix, of 64 pixel intensity values, where 255 denotes a bright white pixel, 0 denotes a black pixel, and the numbers in between are different shades of gray. Given these 64 input features, we'll use a neural network with two hidden layers.

:::{figure}
:label: ch07-23
:no-subfigures:

```{image} diagrams/23-light.svg
:alt: Handwritten Neural Network
:class: dark:hidden
```

```{image} diagrams/23-dark.svg
:alt: Handwritten Neural Network
:class: hidden dark:block
```
:::

The first hidden layer has 25 neurons, or units, and the second hidden layer has 15. The output layer has a single unit, which outputs the chance that the image is a 1 rather than a 0.

Let's step through the sequence of computations the neural network needs to make to go from the input $\vec{x}$ (the $8 \times 8$, or 64, numbers) to the predicted probability $a^{[3]}$.

The first computation goes from $\vec{x}$ to $\vec{a}^{[1]}$, which is what the first hidden layer does. It computes $\vec{a}^{[1]}$ using the formula shown below. Notice that $\vec{a}^{[1]}$ has 25 numbers because this hidden layer has 25 units, which is why the parameters run from $\vec{w}_1^{[1]}$ through $\vec{w}_{25}^{[1]}$ and $b_1^{[1]}$ through $b_{25}^{[1]}$. The formula uses $\vec{x}$, but it could also use $\vec{a}^{[0]}$, because by convention the activation of layer 0, $\vec{a}^{[0]}$, is equal to the input feature vector $\vec{x}$.

:::{figure}
:label: ch07-24
:no-subfigures:

```{image} diagrams/24-light.svg
:alt: Step 1
:class: dark:hidden
```

```{image} diagrams/24-dark.svg
:alt: Step 1
:class: hidden dark:block
```
:::

The next step is to compute $\vec{a}^{[2]}$. The second hidden layer computes $\vec{a}^{[2]}$ as a function of $\vec{a}^{[1]}$: each unit $j$ outputs $g(\vec{w}_j^{[2]} \cdot \vec{a}^{[1]} + b_j^{[2]})$, where $g$ is the sigmoid activation function. Notice that layer 2 has 15 neurons, or units, which is why the parameters here run from $\vec{w}_1^{[2]}$ through $\vec{w}_{15}^{[2]}$ and $b_1^{[2]}$ through $b_{15}^{[2]}$.

:::{figure}
:label: ch07-25
:no-subfigures:

```{image} diagrams/25-light.svg
:alt: Step 2
:class: dark:hidden
```

```{image} diagrams/25-dark.svg
:alt: Step 2
:class: hidden dark:block
```
:::

The final step is to compute $a^{[3]}$ using a very similar computation. The only difference is that the third layer, the output layer, has just one unit, so there's just one output: $a^{[3]}$ is a scalar.

:::{figure}
:label: ch07-26
:no-subfigures:

```{image} diagrams/26-light.svg
:alt: Step 3
:class: dark:hidden
```

```{image} diagrams/26-dark.svg
:alt: Step 3
:class: hidden dark:block
```
:::

Optionally, you can then threshold $a_1^{[3]}$ at 0.5 to get a binary classification label: is this the digit 1, yes or no?

:::{figure}
:label: ch07-27
:no-subfigures:

```{image} diagrams/27-light.svg
:alt: Step 4 - Threshold
:class: dark:hidden
```

```{image} diagrams/27-dark.svg
:alt: Step 4 - Threshold
:class: hidden dark:block
```
:::

In short, the sequence of computations starts with $\vec{x}$ and then computes $\vec{a}^{[1]}$, $\vec{a}^{[2]}$, and $a^{[3]}$, which is also the output of the neural network.

You can also write this output as $f(\vec{x})$. Recall that in linear regression and logistic regression, we used $f(\vec{x})$ to denote the model's output. Similarly, we can use $f(\vec{x})$ to denote the function computed by the neural network as a function of $\vec{x}$.

:::{figure}
:label: ch07-28
:no-subfigures:

```{image} diagrams/28-light.svg
:alt: a3 as f(x)
:class: dark:hidden
```

```{image} diagrams/28-dark.svg
:alt: a3 as f(x)
:class: hidden dark:block
```
:::

Because this computation goes from left to right, starting from $\vec{x}$ and computing $\vec{a}^{[1]}$, then $\vec{a}^{[2]}$, then $a^{[3]}$, this algorithm is called forward propagation: you're propagating the activations of the neurons in the forward direction.

This is in contrast to a different algorithm, called backward propagation or backpropagation, which is used for learning.

## TensorFlow Implementation (Forward Propagation)
TensorFlow is one of the leading frameworks for implementing deep learning algorithms; PyTorch is another popular one. To learn more, check out the TensorFlow tutorials. 

Let's look at how to write inference code in TensorFlow.

**One of the remarkable things about neural networks is that the same algorithm can be applied to so many different applications.**

### Example - Coffee Roasting
Can a learning algorithm help optimize the quality of the beans you get from a roasting process? When roasting coffee, you control two parameters: the temperature at which you heat the raw beans, and the duration, or how long you roast them.

In this slightly simplified example, we've created a dataset of different temperatures and durations, along with labels showing whether the roasted coffee tasted good. In the figure below, red crosses ($y = 1$) mark good coffee and blue circles ($y = 0$) mark bad coffee.

:::{figure}
:label: ch07-29
:no-subfigures:

```{image} diagrams/29-light.svg
:alt: coffee-roasting-data
:class: dark:hidden
```

```{image} diagrams/29-dark.svg
:alt: coffee-roasting-data
:class: hidden dark:block
```
:::

Here's a reasonable way to think about this dataset:
- If you roast the beans at too low a temperature, they don't roast properly and end up undercooked.
- If you don't roast them long enough (the duration is too short), they also don't come out nicely roasted.
- If you roast them for too long or at too high a temperature, they end up overcooked and a little burnt, which isn't good coffee either.

:::{figure}
:label: ch07-30
:no-subfigures:

```{image} diagrams/30-light.svg
:alt: coffee-roasting
:class: dark:hidden
```

```{image} diagrams/30-dark.svg
:alt: coffee-roasting
:class: hidden dark:block
```
:::

***Only the points inside the small triangle correspond to good coffee.***

This example is a simplified version of real coffee roasting.

#### Task
Given a feature vector `x` containing both temperature and duration (say, 200 °C for 17 minutes), how can we run inference in a neural network so that it tells us whether this setting will result in good coffee?

:::{figure}
:label: ch07-31
:no-subfigures:

```{image} diagrams/31-light.svg
:alt: nn-coffee-roasting
:class: dark:hidden
```

```{image} diagrams/31-dark.svg
:alt: nn-coffee-roasting
:class: hidden dark:block
```
:::

First, set `x` to an array of two numbers: the input features, 200 °C and 17 minutes. Then create Layer 1, the first hidden layer of the network, as `Dense(units=3, activation='sigmoid')`. This means the layer has three hidden units and uses the sigmoid activation function. *Dense* is another name for the type of layer we've learned about so far. As you learn more about neural networks, you'll learn about other types of layers too, but for now we'll use dense layers for all of our examples.

Next, compute `a1` by applying Layer 1, which is actually a function, to `x`. Because Layer 1 has three units, `a1` is a list of three numbers; for illustration, it might be 0.2, 0.7, and 0.3.

Next, create Layer 2, the output layer. It's also a dense layer, this time with one unit and, again, a sigmoid activation function. Compute `a2` by applying Layer 2 to the activations from Layer 1, `a1`. For illustration, `a2` might be 0.8.

Finally, if you want to threshold the output at 0.5, check whether `a2` is greater than or equal to 0.5 and set `yhat` to 1 (positive class) or 0 (negative class) accordingly.

That's how you run inference in a neural network using TensorFlow. There are some additional details I didn't cover here, such as how to import the TensorFlow library and how to load the network's parameters $w$ and $b$. But these are the key steps of forward propagation: computing `a1` and `a2`, and optionally thresholding `a2`.

``` python
# Input Feature
x = np.array([200.0, 17.0])

# Create Layer 1
layer_1 = Dense(units=3, activation='sigmoid')

# Calculate a1
a1 = layer_1(x)

# Create Layer 2
layer_2 = Dense(units=1, activation='sigmoid')

# Calculate a2
a2 = layer_2(a1)

# Threshold a2
if a2 >= 0.5:
  yhat = 1
else:
  yhat = 0
```

### Example - Handwritten Digit Classification
In this example, the input `x` is a list of pixel intensity values, stored as a NumPy array.

:::{figure}
:label: ch07-32
:no-subfigures:

```{image} diagrams/27-light.svg
:alt: Step 4 - Threshold
:class: dark:hidden
```

```{image} diagrams/27-dark.svg
:alt: Step 4 - Threshold
:class: hidden dark:block
```
:::

To set up and run the first step of forward propagation, create Layer 1 as a dense layer with 25 units and a sigmoid activation function, then compute `a1` by applying Layer 1 to `x`. The second layer works the same way: create Layer 2 as a dense layer with 15 units, then compute `a2` by applying Layer 2 to `a1`. Layer 3 is the third and final dense layer, with a single unit; applying it to `a2` gives `a3`. Finally, you can optionally threshold `a3` to get a binary prediction, `yhat`.

That's the syntax for carrying out inference in TensorFlow.
``` python
# Input feature
x = np.array([0, ..., 245, ..., 240, ... 0])

# Create Layer 1
layer_1 = Dense(units=25, activation='sigmoid')

# Calculate a1
a1 = layer_1(x)

# Create Layer 2
layer_2 = Dense(units=15, activation='sigmoid')

# Calculate a2
a2 = layer_2(a1)

# Create Layer 3
layer_3 = Dense(units=1, activation='sigmoid')

# Calculate a3
a3 = layer_3(a2)

# Threshold a3
if a3 >= 0.5:
  yhat = 1
else:
  yhat = 0
```


:::{note}
One thing I only touched on briefly is the structure of the NumPy arrays. TensorFlow treats data in a particular way, and it's important to get it right.
:::

Next, let's look at how TensorFlow handles data.

## Data in TensorFlow