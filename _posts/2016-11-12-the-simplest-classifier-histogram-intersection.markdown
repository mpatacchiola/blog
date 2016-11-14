---
layout: post
title:  "The Simplest Classifier: Histogram Intersection"
date:   2016-11-12 10:00:00 +0000
comments: true
---

Geometrical cues are the most reliable way to estimate the object identity. After all, we can recognise a chair also in a black and white image, we do not need colours or texture. However this is not always the case. In nature there are many examples where a colour identifies species, fruits and minerals. In our society colours are often used in trademark and logos. Think about Google and Facebook, the first is identified by a combination of blue, red, yellow and green, whereas the second is mainly blue. Another example are the **superheroes from the Marvel and DC universe**. Each superhero has a distinctive multicolour fingertip.


![image-title-here]({{site.baseurl}}/images/superheroes.png){:class="img-responsive"}

Identifying objects only through colours can be considered crazy at first glance. However, there are psychophysical and neuroscientific evidences which justify this intuition. 

Also one of the most effective classifier the **Convolutional Neural Network** takes advantage of colours. In some cases giving as input a RGB image instead of a greyscale one, can make a huge difference. 

The histogram intersection algorithm
-------------------------------------

The histogram intersection algorithm was proposed by Swain and Ballard in their article [Indexing via color histograms](http://link.springer.com/chapter/10.1007%2F978-3-642-77225-2_13). This algorithm is particular reliable when the colour is a strong predictor of the object identity. The histogram intersection does not require the accurate separation of the object from its background and it is robust to occluding objects in the foreground. An **histogram** is a **graphical representation** of the value distribution of a digital image. The **value distribution** is a way to represent the colour appearance and in the HVS model it represents the saturation of a colour.Histograms are invariant to translation and they change slowly under different view angles, scales and in presence of occlusions.

Now let's try to give a mathematical foundation to the algorithm. We start with an *input image* which represent the frame obtained from the camera, and an internal *model* which is the object frame which has been saved previously. Given the histogram $$I$$ of the input image and the histogram $$M$$ of the model, each one containing $$n$$ bins, the **intersection** is defined as:

$$ \sum_{j=1}^{n} min(I_{j}, M_{j}) $$

The $$min$$ function take as arguments two values and return the smallest one. **The result of the intersection is the number of pixels from the model that have corresponding pixels of the same colours in the input image**. To normalise the result between 0 and 1 we have to divide it by the number of pixels in the model histogram:

$$ \frac{ \sum_{j=1}^{n} min(I_{j}, M_{j}) }{ \sum_{j=1}^{n} M_{j} } $$

That's all. What we need in our classifier is an histogram for each object we want to identify. When an unknown object is given as input we compute the histogram intersection for all the stored models, the highest value is returned for the best image-model match.

Implementation in Python
----------------------------------------------------------

In python we can easily play with histograms, for instance **numpy** has the function `numpy.histogram()` and **OpenCV** the function `cv2.calcHist()`. To plot an histogram we can use the **matplotlib** function `matplotlib.pyplot.hist()`. The main parameters to give as input to these functions are the array (or image), the number of bins and the lower and upper range of the bins. In OpenCV two histograms can be compared using the function `cv2.compareHist()` which take as input the histograms and the comparison method. Among the possible methods there is also the `CV_COMP_INTERSECT` which is an implementation of the histogram intersection method. Numpy does not have a built-in function for comparing histograms, but we can implement it in a few lines. After importing the numpy library we have to load the two images as numpy arrays, then we can use the following function to compare them:

```python
def histogram_intersection(image, model):
    I = numpy.histogram(image)
    M = numpy.histogram(model)
    minima = numpy.minimum(I, M)
    intersection = numpy.sum(minima) / numpy.sum(M)
    return intersection
```
I have implemented a full version of the algorithm in [deepgaze](https://github.com/mpatacchiola/deepgaze). I suggest you to clone the deepgaze repository since it includes some full working examples to play with.



A real world example: delivery system
-----------------------------------

Let's suppose we have to build a vision system for a manipulator which is used in the **robotic delivery system** of the company Cup&Go. Cup&Go produces five different cups. Our system has to identify the cup and place it in a specific package. We have a database containing the cup ID, the cup description and some images. In this example **we will consider only five objects**, but the same reasoning can be applied for many more.

The objects move on a treadmill and a camera is positioned at the beginning of the tape, allowing the control unit to receive a clear image of one object at a time. At the end of the tape there is a robotic arm which has to grasp the object as fast as possible and move it to a specific container. In this example it is necessary to have an efficient classifier, which **can be trained on very few images** and which is fast to execute. Training a Convolutional Neural Network for each object is crazy because it will take ages. After some queries on the database you notice that the objects have some specific colour combinations and that this information could be used for a histogram-based classifier. Let's do it!

Acknowledgments
---------------

The superheroes images are courtesy of [Christopher Chong](https://www.flickr.com/photos/126293860@N05/).


References
------------

Swain, M. J., & Ballard, D. H. (1991). Color indexing. International journal of computer vision, 7(1), 11-32.







