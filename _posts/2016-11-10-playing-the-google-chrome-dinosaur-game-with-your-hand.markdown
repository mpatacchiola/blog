---
layout: default
title:  "Playing the Google Chrome's dinosaur game using hand-tracking"
date:   2016-11-10 19:00:00 +0000
---
Today I was looking for some robust and fast way to track hands in OpenCV. Using a classifier (Haar cascade, Convolutional Neural Networks, SVM, etc) was not feasible due to the lack of datasets and to the hardware resources necessary to run everything smootly. What to do? One possibility was the **Histogram Backprojection** algorithm. This algorithm was proposed by Swain and Ballard in their article `Indexing via color histograms`. How does the algorithm work? We start with an input image and an object to track (let's say our hand). Next we take a subframe from the image, this subframe is often called `template` and it corresponds to the object we want to track (or a significant part of it).

Given the input image and the template the algorithm creates a output image of the same size of the input but with a single channel. Each pixel in the output image corresponds to the probability of that pixel of belonging to the reference object.
