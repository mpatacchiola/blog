---
layout: post
title:  "Playing the Google Chrome's dinosaur game using hand-tracking"
date:   2016-11-15 19:00:00 +0000
comments: true
---

![image-title-here]({{site.baseurl}}/images/dinosaur_backprojection_hand_tracking.png){:class="img-responsive"}

Today I was looking for some robust/fast way to **track hands** in OpenCV. At the beginning I turned my attention to classifiers such as: Haar cascade, Convolutional Neural Networks, SVM, etc. Unfortunately there are two main problems with classifiers: slinding windows and datasets.

1. To classify parts of an image it is often used a standard approach called **sliding window**. A sliding window is a window of predefined size that is moved over an image and which content is given as input to the classifier. This approach is widely used in computer vision but it has a drawback: speed (in terms of computational efficiency). Since we do not know the size of the face we have to move windows of different size over the input matrix. This is time/resources consuming. A Region Of Interest (ROI) can be used to define a subframe of the input image to use as input feature. This solution introduce more problems, like defining and updating the position of the ROI.

2. Another problem when you want to implement a classifier is to find **datasets** which can be used in the training phase. OpenCV has an [Haar cascade classifier](http://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html) for hand detection but it has been trained on a limited dataset and it recognises hands in predefined poses. I tried to use Convolutional Neural Networks (CNNs) for recognising open hands but I had the same problem, very few datasets available.

Given this two problems, what to do? One possibility is to use the **histogram backprojection** algorithm.


The histogram backprojection algorithm
---------------------------------------

The histogram backprojection algorithm was proposed by Swain and Ballard in their article ["Color Indexing"](http://link.springer.com/article/10.1007/BF00130487). If you did not read my [previous post]({{site.baseurl}}{% post_url 2016-11-12-the-simplest-classifier-histogram-intersection %}) about **histogram intersection** you should do it, since the backprojection can be considered the complementary of the intersection. Using the terminology of Swain and Ballard we can say that the **intersection** answers to the question "**What** is the object we are looking to?", whereas the **backprojection** answers to the question "**Where** in the image are the colors that belong to the object being looked for (the target)?". How does the backprojection work? Given the image histogram $$ I $$ and the model histogram $$ M $$ we define a ratio histogram $$ R $$ as follow:

$$ R_{i} = min \bigg( \frac{M_{i}}{I_{i}}, 1 \bigg) $$

We define also a function $$ h(c) $$ that maps the colour $$ c $$ of the pixel at $$ (x, y) $$ to the value of the histogram $$ R $$ it indexes. Using the slang of Swain and Ballard we can say that the function $$ h(c) $$ **backprojects** the histogram $$ R $$ onto the input image. The backprojected image is then convolved with a disk $$ D^{r} $$ of radius $$ r $$. The authors recommend to choose as area of $$ D $$ the expected area subtended by the object.

Implementation in python
---------------------------------------------------------------
To implement the algorithm in numpy is easy. First of all we need a function to get the ratio histogram $$ R $$ given the the **model histogram** $$ M $$ and the **image histogram** $$ I $$:

```python
def return_ratio_histogram(M, I):
    ones_array = np.ones(I.shape)
    R = np.minimum(np.true_divide(M, I), ones_array)
    return R
```

The next step is to implement the equivalent of the function $$ h(c) $$ which maps the pixel of the **input image** to the value of the **ratio histogram** $$ R $$:

```python
def return_backprojected_image(image, R):
    

```

Since the backprojection process can be hard to grasp I realised an image which explain what's going on. To simplify the explanation I take into account a **single channel greyscale image**. Imagine to take the first (3x4) pixels on the top right corner of the image. Each one of this pixel has a value in the range [0,255]. If we consider the pixel at the location (1,2) we get a value of 179. We have to access the ratio histogram and find the bin which corresponds to 179. Using a 10 bins histogram the position representing the value 179 is the number eight (range [175, 200]). The value stored in this position (1.0) is taken and assigned to the location (1,2) of the output matrix.

![image-title-here]({{site.baseurl}}/images/backprojection_figure.png){:class="img-responsive"}

Implementing the algorithm from scratch is not necessary since OpenCV has an useful function called `calcBackProject`.

Google Chrome's dinosaur game
----------------------------------

Ok now that we know the theory and how to implement the algorithm in python it's time for a real application. While I was writing this post I lost my internet connection probably because [Valerio Biscione](http://valeriobiscione.com/) was doing something weird with utorrent. When you lost the connection and you are using Chrome a tiny dinosaur will appear. This **8-bit T-Rex** can jump when you press UP and drop when you press DOWN. The goal of the game is to go on until death (in my case it happens quite soon). To play the game you don't have to turn off your router, [here](http://apps.thecodepost.org/trex/trex.html) you can find an online version. My idea was to implement an hand controller to play this game. Hand Up > Dino Up, Hand Down > Dino Down. 

<div style="text-align: center;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/eoUOkV5vVpU" frameborder="0" allowfullscreen></iframe>
</div>

Conclusions
------------

The histogram back projection is a useful technique for detecting and tracking objects. It has some advantages (easy to implement, fast) and some disadvantages (noisy and unstable). Some other techniques can be used to reduce noise and stabilise the tracker, like particle filtering, but this is material for another post...

References
------------

Swain, M. J., & Ballard, D. H. (1991). Color indexing. International journal of computer vision, 7(1), 11-32.

Swain, M. J., & Ballard, D. H. (1992). Indexing via color histograms. In Active Perception and Robot Vision (pp. 261-273). Springer Berlin Heidelberg.


