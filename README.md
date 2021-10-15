# Description
A **numpy-based** implementation of RANSAC for fundamental matrix and homography estimation. 

The **degeneracy updating** and **local optimization** components are included and optional.

Since the only dependency is numpy, it should be very easy to install, debug and understand the code.

# Usage
Since the core functions are dependent solely on numpy, one only requires to add src files into system path to invoke RANSAC.

The file demo.py is given to show a simple example of fundamental matrix and homography estimation.

Note that scipy and opencv-python packages are additionally required to run the demo.

# Performance
The draw back of this repo is that it can be a bit slow, taking a few seconds when the data is challenging. But the accuracy is somewhat satisfying.

Below is the comparison result of npRANSAC (this repo) and several state-of-the-art robust estimators from the renowned image matching benchmark (https://github.com/ubc-vision/image-matching-benchmark).

The test sequences in the figure are reichstag, sacre coeur and st peters square from Phototourism dataset. The competitors follow the recommended parameter settings as in https://www.cs.ubc.ca/research/image-matching-challenge/2021/submit/. 

Basically in the experiment npRANSAC has the same parameters as DEGENSAC, except that the maximum iteration number is set to 10k for npRANSAC (the other methods have much higher numbers).

![ransac-rt-example](https://user-images.githubusercontent.com/59504874/137452189-1bdba4d4-efc7-44f9-b9bd-6bee5baff610.png)



