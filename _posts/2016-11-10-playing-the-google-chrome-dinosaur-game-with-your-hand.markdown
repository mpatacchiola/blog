---
layout: post
title:  "Playing the Google Chrome's dinosaur game using hand-tracking"
date:   2016-11-10 19:00:00 +0000
comments: true
---

Today I was looking for some robust/fast way to **track hands** in OpenCV. At the beginning I turned my attention to classifiers such as: Haar cascade, Convolutional Neural Networks, SVM, etc. Unfortunately there are two main problems with classifiers: slinding windows and datasets.

1. To classify parts of an image it is often used a standard approach called **sliding window**. A sliding window is a window of predefined size that is moved over an image and which content is given as input to the classifier. If you have a picture containing human faces and you want to identify in which part of the image the faces are then you have to use a sliding window. This approach is widely used in computer vision but it has a drawback: speed (in terms of computational efficiency). Since we do not know the size of the face we have to move windows of different size over the input image. This is time/resources consuming. A Region Of Interest (ROI) can be used to define a subframe of the input image to use as input feature. This solution introduce more problems, like defining the position of the ROI and update it when the object moves.

2. Another problem when you want to implement a classifier is to find **datasets** which can be used in the training phase. OpenCV has an [Haar cascade classifier](http://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html) for hand detection but it has been trained on a limited dataset and it recognises hands in predefined poses. I tried to use Convolutional Neural Networks (CNNs) for recognising open hands but I had the same problem, very few datasets available.

Given this two problems, what to do? 

Histograms in Python
--------------------

In python we can easilly play with histograms. In `numpy` we have the function:

```python
numpy.histogram(a, bins=10, range=None, normed=False, weights=None, density=None)
``` 

Histogram Backprojection for Color Detection
--------------------------------------------

One possibility is to use the **Histogram Backprojection** algorithm. This algorithm was proposed by Swain and Ballard in their article [Indexing via color histograms](http://link.springer.com/chapter/10.1007%2F978-3-642-77225-2_13). Since we are dealing with histograms it is necessary to briefly summarise what an histogram is. An **histogram** is a **graphical representation** of the value distribution of a digital image. The **value distribution** is a way to represent the colour appearance and in the HVS model it represents the saturation of a colour.



How does the algorithm work? We start with an input image and an object to track (let's say our hand). Next we take a subframe from the image, this subframe is often called **template** and it corresponds to the object we want to track (or a significant part of it).

Given the input image and the template the algorithm creates a output image of the same size of the input but with a single channel. Each pixel in the output image corresponds to the probability of that pixel of belonging to the reference object.

$$a^2 + b^2 = c^2$$

Python implementation of the Histogram Backprojection algorithm
---------------------------------------------------------------
Implementing the algorithm from scratch is not necessary since OpenCV has an useful function called `calcBackProject`

Playing the Google Chrome's Dinosaur Game using your hand
---------------------------------------------------------

Ok now that we know the theory and how to implement the algorithm in python it's time for a real application. While I was writing this post I lost my internet connection probably because [Valerio Biscione](http://valeriobiscione.com/) was doing something weird with utorrent. When you lost the connection and you are using Chrome a tiny dinosaur will appear. This **8-bit T-Rex** can jump when you press UP and drop when you press DOWN. The goal of the game is to go on until death (in my case it happens quite soon). To play the game you don't have to turn off your router, [here](http://apps.thecodepost.org/trex/trex.html) you can find an online version. My idea was to implement an hand controller to play this game. Hand Up > Dino Up, Hand Down > Dino Down. 

<div style="text-align: center;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/eoUOkV5vVpU" frameborder="0" allowfullscreen></iframe>
</div>

Conclusions
------------

The histogram back projection is a useful technique for detecting and tracking objects. It has some advantages (easy to implement, fast) and some disadvantages (noisy and unstable). Some other techniques can be used to reduce noise and stabilise the tracker, like particle filtering, but this is material for another post...




