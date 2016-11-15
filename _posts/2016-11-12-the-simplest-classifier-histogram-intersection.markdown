---
layout: post
title:  "The Simplest Classifier: Histogram Intersection"
date:   2016-11-12 10:00:00 +0000
comments: true
---

![image-title-here]({{site.baseurl}}/images/ironman_histogram_intersection.png){:class="img-responsive"}

**Geometrical cues** are the most reliable way to estimate the object identity. After all, we can recognise a chair also in a black and white image, we do not need colours or texture. However this is not always the case. In nature there are many examples where a colour identifies species, fruits and minerals. In our society **distinctive colours are often used in trademark and logos**. Think about Google and Facebook, the first is identified by a combination of blue, red, yellow and green, whereas the second is mainly blue. 

Identifying objects only through colours can be considered crazy at first glance. However, there are psychophysical and neuroscientific evidences which justify this intuition. 
One of the most effective classifier the **Convolutional Neural Network** takes advantage of colours. In some cases giving as input a RGB image instead of a greyscale one, can make a huge difference. 

The histogram intersection algorithm
-------------------------------------

The histogram intersection algorithm was proposed by Swain and Ballard in their article [Indexing via color histograms](http://link.springer.com/chapter/10.1007%2F978-3-642-77225-2_13). This algorithm is particular reliable when the colour is a strong predictor of the object identity. The histogram intersection does not require the accurate separation of the object from its background and it is robust to occluding objects in the foreground. An **histogram** is a **graphical representation** of the value distribution of a digital image. The **value distribution** is a way to represent the colour appearance and in the HVS model it represents the saturation of a colour. Histograms are invariant to translation and they change slowly under different view angles, scales and in presence of occlusions.

Now let's try to give a mathematical foundation to the algorithm. We start with an *input image* which represents the frame obtained from the camera, and an internal *model* which is the object frame which has been saved previously. Given the histogram $$I$$ of the input image and the histogram $$M$$ of the model, each one containing $$n$$ bins, the **intersection** is defined as:

$$ \sum_{j=1}^{n} min(I_{j}, M_{j}) $$

The $$min$$ function take as arguments two values and return the smallest one. **The result of the intersection is the number of pixels from the model that have corresponding pixels of the same colours in the input image**. To normalise the result between 0 and 1 we have to divide it by the number of pixels in the model histogram:

$$ \frac{ \sum_{j=1}^{n} min(I_{j}, M_{j}) }{ \sum_{j=1}^{n} M_{j} } $$

That's all. What we need in our classifier is an histogram for each object we want to identify. When an unknown object is given as input we compute the histogram intersection for all the stored models, the highest value is the best match.

Implementation in Python
----------------------------------------------------------

In python we can easily play with histograms, for instance **numpy** has the function `numpy.histogram()` and **OpenCV** the function `cv2.calcHist()`. To plot an histogram we can use the **matplotlib** function `matplotlib.pyplot.hist()`. The main parameters to give as input to these functions are the array (or image), the number of bins and the lower and upper range of the bins. In OpenCV two histograms can be compared using the function `cv2.compareHist()` which take as input the histogram parameters and the comparison method. Among the possible methods there is also the `CV_COMP_INTERSECT` which is an implementation of the histogram intersection method. Numpy does not have a built-in function for comparing histograms, but we can implement it in a few lines. After importing the numpy library we have to load the two images as numpy arrays, then we can use the following function to compare them:

```python
def histogram_intersection(image, model):
    I = numpy.histogram(image)
    M = numpy.histogram(model)
    minima = numpy.minimum(I, M)
    intersection = numpy.sum(minima) / numpy.sum(M)
    return intersection
```
I have implemented a **full version of the algorithm in** [deepgaze](https://github.com/mpatacchiola/deepgaze). I suggest you to clone the repository since it includes the full working example and some pre-processed images to play with. 

In deepgaze it is extremely easy to initialise a histogram classifier. First of all we have to import the histogram classifier module, then we can **initialise the classifier object** calling `HistogramColorClassifier()`. As parameter we can give the number of channel (in a RGB image there are three channels) then the number of bins for each channel and the range of the values inside each bin. The default values generally work well, then you can initialise the classifier with empty brackets. To **load a model** we can use the method `addModelHistogram()` which take as input a frame. Finally we can give as input to the classifier an **image to compare**, the method to call is `returnHistogramComparisonArray()`. The comparison function returns an array which contains the intersection values between the image and each model stored in the classifier. The highest value inside this array identifies the object with the best match. Let's see a practical example...


Superheroes classification
---------------------------

Maybe you already notices that each **superheroes from the Marvel and DC universe** has a distinctive multicolour fingertip. In the image below you can visually get the difference:

![image-title-here]({{site.baseurl}}/images/superheroes.png){:class="img-responsive"}

In this example I will use the deepgaze colour classifier to recognise **eight superheroes**. First of all we have to import some libraries and the deepgaze module:

```python
import cv2
import numpy as np
from deepgaze.color_classification import HistogramColorClassifier

my_classifier = HistogramColorClassifier(channels=[0, 1, 2], hist_size=[128, 128, 128], hist_range=[0, 256, 0, 256, 0, 256], hist_type='BGR')
```

Now we have to load the models inside the classifier. You can read the image with the OpenCV function `cv2.imread()` and then load it calling the deepgaze `addModelHistogram()` function. You have to repeat this operation for all the models you want to store in the classifier.

```python
model_1 = cv2.imread('model_1a.png') #Flash
my_classifier.addModelHistogram(model_1)

model_2 = cv2.imread('model_2a.png') #Batman
my_classifier.addModelHistogram(model_2)
```

In the article [Indexing via color histograms](http://link.springer.com/chapter/10.1007%2F978-3-642-77225-2_13) the authors recommend to crop the model from the background. This operation can be annoying but you can obtain good result with a gross cropping. In the [deepgaze repository](https://github.com/mpatacchiola/deepgaze) you can find 10 different images cropped and ready to use. Each model was taken from a frontal image of the puppet and manually cropped to isolate the foreground from the background:

![image-title-here]({{site.baseurl}}/images/superheroes_cropped.png){:class="img-responsive"}

Now is time to test the classifier. For the test I used an image of Batman where the puppet had a different pose. After loading the image with OpenCV I compared it with the models using a single call to `returnHistogramComparisonArray()` as follow:

```python
image = cv2.imread('image_2.jpg') #Load the image
comparison_array = my_classifier.returnHistogramComparisonArray(image, method="intersection")
```

The method `returnHistogramComparisonArray()` returns a comparison array which contains the result of the intersection between the image and the models. In this function it is possible to specify the comparison method, `intersection` refers to the method we discussed in this article. Other available methods are `correlation`, `chisqr` (Chi-Square) and `bhattacharyya` which is an implementation of the [Bhattacharyya distance measure](https://en.wikipedia.org/wiki/Bhattacharyya_distance). To check the result of the comparison we can print the values inside the `comparison_array`:

```python
[ 0.00818883  0.55411926  0.12405966  0.07735263  0.34388389  0.12672027 0.09870308  0.2225694 ]
```
As you can see the second value (*0.55*) is the highest one, meaning that the second model has the best match with the input image. This value is the raw distance value, to obtain a probability distribution we have to normalise the array dividing it by the sum of the values:

```python
probability_array = comparison_array / np.sum(comparison_array)
```

This simple operation gives a new vector which values sum up to 1:

```python
[ 0.00526411  0.35621003  0.07975051  0.04972536  0.22106232  0.08146086 0.06345029  0.14307651]
```

Now we can say that the second model has a probability of *35.6%* of being the correct match with the input image in our case Batman:

![image-title-here]({{site.baseurl}}/images/batman_histogram_intersection.png){:class="img-responsive"}

With Batman was easy to get an high confidence value (*35.6%*). This was possible because the Batman's multicolor fingerprint is pretty distinctive. Let's test the classifier on something more complicated. I gave as model a frontal image of Ironman with the helmet on. For the test I used an image where Ironman was turned on the right and did not have the helmet. I called the `returnHistogramComparisonArray()` and normalised the output, then with matplotlib I generated a bar chart:

![image-title-here]({{site.baseurl}}/images/ironman_b_histogram_intersection.png){:class="img-responsive"}

We got good results with Ironman (*26.6%*). What happens if the image is in a different position and there are some additional colours respect to the model? We can try with Wonder Woman:

![image-title-here]({{site.baseurl}}/images/wonder_woman_histogram_intersection.png){:class="img-responsive"}

We got a confidence of *25.4%* which was enough to identify the puppet. As we said the histogram intersection is reliable when there are small change in the object perspective and when there is noise in the background. If you want you can test the other images downloading the example from the [deepgaze repository](https://github.com/mpatacchiola/deepgaze). **For all the images the highest value identified the correct superhero**. In their [article](http://link.springer.com/chapter/10.1007%2F978-3-642-77225-2_13) Swain and Ballard tested the histogram intersection on a dataset containing 66 models. For the 66-object dataset the correct model was the **best matches 29 of 32 times** and in the other 3 cases the correct model was the second highest match. If you have the occasion to test the model with larger datasets please **share your results** in the comments below.

<!---
A real world example: delivery system
-----------------------------------

It was funny to play with superheroes, but how can we use the histogram intersection algorithm in the real world? Let's suppose we have to build a vision system for a manipulator which is used in the **robotic delivery system** of the company Cup&Go. Cup&Go produces 50 different cups. Our system has to identify the cup and place it in a specific package. We have a database containing the cup ID, the cup description and some images. The cups move on a treadmill and a camera is positioned at the beginning of the tape, allowing the control unit to receive a clear image of one cup at a time. At the end of the tape there is a robotic arm which has to grasp the object as fast as possible and move it to a specific container. In this example it is necessary to have an efficient classifier, which **can be trained on very few images** and which is fast to execute. Training a Convolutional Neural Network to recognise the cups is not possible because we do not have enough images. However, after some queries on the database you notice that the objects have some specific colour combinations and that this information could be used for a histogram-based classifier. Well now you know what to do...
-->

Acknowledgments
---------------

The superheroes images are courtesy of [Christopher Chong](https://www.flickr.com/photos/126293860@N05/).


References
------------

Swain, M. J., & Ballard, D. H. (1991). Color indexing. International journal of computer vision, 7(1), 11-32.







