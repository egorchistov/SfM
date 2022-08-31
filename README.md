# Scale trajectory predicted by SfM Learner

**E. Chistov, M. Tregubenko, and S. Linok**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/egorchistov/SfM/blob/master/demo.ipynb)

## Overview

Visual odometry is the process of determining the position and orientation of a camera by analyzing images.
One of the visual odometry problems is the discrepancy between the scale of the predicted and ground truth trajectory.

![Scale discrepancy example](images/scale-discrepancy-example.jpg)

This repository contains algorithm to scale trajectory predicted by [SfM Learner](https://github.com/ClementPinard/SfmLearner-Pytorch).
For scaling we use metadata such as camera heigth and camera intrinsics.

![Algorithm overview](images/algorithm-overview.jpg)
