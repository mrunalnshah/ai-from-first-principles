---
title: "Chapter 1: What is Machine Learning?"
description: What is Machine Learning?
date: 2026-10-04
numbering:
  enumerator: 1.%s          # Figure 1.1, Equation (1.1), Table 1.1, …
---

# Machine "IRL" Learning

Before we define machine learning, let's look at where you already run into it. Chances are you've used it several times today without noticing.

- **Web search.** Want to know how to make [Pav Bhaji](https://en.wikipedia.org/wiki/Pav_bhaji)? Type the question into Google, Bing, or Baidu, and helpful recipes appear right at the top. That happens because machine learning has taught the search engine which pages deserve the top spots.
- **Photo tagging.** You share a group photo on Instagram or Snapchat and want your friends tagged so they see it too. The app can pick out each friend's face and suggest the right name, so you don't have to tag everyone by hand.
- **Recommendations.** The credits roll on a Star Wars movie (I'm a huge Star Wars fan, by the way!), and you want something similar to watch next. The suggestions your streaming service shows you, like a "Because you watched…" row, are usually chosen by machine learning.
- **Speech recognition.** Whether you dictate a text message ("Hey Grogu, how's it going?"), ask "Hey Siri, play a song by Lisa Gerrard," or say "OK Google, show me Indian restaurants near me," machine learning is what turns your speech into text and actions.
- **Spam filtering.** An email arrives with the subject line "Congratulations! You've won a million dollars!" Sadly, it's almost certainly a scam, and your email service usually knows that before you do. Spotting spam and moving it out of your inbox is another job for machine learning.

Machine learning isn't limited to the apps on your phone. It is also starting to help with some of the world's biggest challenges:

- **Energy and climate.** Climate change is something I care about a lot, so it's encouraging to see machine learning already helping to optimize how we generate power.
- **Healthcare.** Hospitals are starting to use AI as well, for example to help doctors diagnose diseases more accurately.

All of these are machine learning at work. In one sentence:

:::{prf:definition} Machine Learning
:label: def-machine-learning

**Machine learning** is the science of getting computers to learn without being explicitly programmed.[^samuel]
:::

# Applications of Machine Learning

## Why Machines Need to Learn

Machine learning started out as a branch of **artificial intelligence (AI)**, the field whose goal is to build intelligent machines.

A few basic tasks can be programmed directly. Finding the shortest route from A to B, which is exactly what your GPS does, is a good example: we know a precise procedure for solving it, so we can write that procedure as code.

Most of the really exciting problems are a different story. Nobody knew how to write explicit, step-by-step instructions for tasks like:

- searching the web
- recognizing human speech
- diagnosing diseases from X-rays
- building a self-driving car

:::{important} Key idea
For problems like these, there was only one option left: stop trying to write the instructions ourselves and let the machine learn the task *by itself*.
:::

## The Long-Term Dream: Super Intelligence

Further in the future lies one of AI's biggest dreams: machines that can think as well as people do. This goal is called **Super Intelligence (SI)**, and like many people, I find it exciting.

Even so, I think SI is getting far more hype than it deserves. We are nowhere close to building it yet. Will we get there in 50 years? 500? Even longer? I honestly don't know.

Most AI researchers believe, though, that **learning algorithms** are the most promising path toward it. And when it comes to designing those algorithms, perhaps the best place to look for inspiration is the brain itself, especially the human brain!

[^samuel]: This classic definition is usually credited to Arthur Samuel, the IBM researcher who coined the term "machine learning" in 1959. The wording here follows the version popularized by Andrew Ng.
