---
layout: post
title:  "Head Pose Estimation using Convolutional Neural Networks"
date:   2017-08-22 09:00:00 +0000
description: Head pose estimation in the wild using Convolutional Neural Networks
author: Massimiliano Patacchiola
type: computer vision
comments: false
published: false
---


I suggest to give a look to this useful [Wiki article](https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions) about rotation formalism and conversion.

1. The CNN returns three angles in Euler convention. From the Euler angles we get the rotation matrix. The rotation matrix depends from the convention used for the Euler angles. To achieve this conversion we have to find the rotation matrices for the three axis given by the Euler angles Rx, Ry, Rz and then multiply the matrices to get the final rotation matrix. The order matters, and the standard convetion used here is: Z-Y-X (yaw-pitch-roll) convention meaning that R=Rz Ry Rx

2. We can create three vectors of pre-defined lenght and project them on 
To visualize the rotation using three vectors pose convention, we had to project the end-points of the three vectors on a two-dimensional plane. this is done through the method calle [projectPoints](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#projectpoints) which takes as input: `cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)`. What we need is the `rvec` term which can be obtained through `cv2.Rodrigues()` method. However `cv2.Rodrigues()` takes as input a rotation matrix and returns the `rvec` we are looking for. Where do we find the rotaion matrix? We should use the Euler angle returned by our CNN to obtain a rotation Matrix.

3. Use the OpenCV method [Rodrigues](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#rodrigues) (converts a rotation matrix to a rotation vector or vice versa.)
We need to get a rotation vector `rvect` because is used in the global 3D geometry optimization procedures like calibrateCamera(), stereoCalibrate(), solvePnP(), or projectPoints(). A rotation vector is a convenient and most compact representation of a rotation matrix (since any rotation matrix has just 3 degrees of freedom).  
Rodrigues parameters are also called axis-angle rotation. They are formed by 4 numbers [theta, x, y, z], which means that you have to rotate an angle "theta" around the axis described by unit vector v=[x, y, z]. Looking at cv2.Rodrigues function reference, it seems that OpenCV uses a "compact" representation of Rodrigues notation as vector with 3 elements rod2=[a, b, c], where:
Angle to rotate `theta` is the module of input vector `theta = sqrt(a^2 + b^2 + c^2)`
Rotation axis `v` is the normalized input vector: `v = rod2/theta = [a/theta, b/theta, c/theta]`
So, Rodrigues vector from solvePnP is not even slightly related with Euler angles notation, which represent three consecutive rotations around a combination of X, Y and Z axes.




References
------------

Swain, M. J., & Ballard, D. H. (1991). Color indexing. International journal of computer vision, 7(1), 11-32.

Swain, M. J., & Ballard, D. H. (1992). Indexing via color histograms. In Active Perception and Robot Vision (pp. 261-273). Springer Berlin Heidelberg.

Wichmann, F. A., Sharpe, L. T., & Gegenfurtner, K. R. (2002). The contributions of color to recognition memory for natural scenes. Journal of Experimental Psychology: Learning, Memory, and Cognition, 28(3), 509.




