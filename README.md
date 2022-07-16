# Visual odometry using a mobile phone monocular camera

**E. Chistov, M. Tregubenko, and S. Linok**

The repository is under development and will be updated

1. [x] Create repository description in README
1. [ ] Add Google Colaboratory demo
1. [ ] Add proposed method description and results
1. [ ] Add documentation and tests
1. [ ] Think about trajectory length, frame rate and frame size

## Overview

Visual odometry is the process of determining the position and orientation of a phone by analyzing the camera images. One of the visual odometry problems is the discrepancy between the scale of the predicted and ground truth trajectory.

![Scale discrepancy example](images/scale-discrepancy-example.jpg)

This repository contains algorithm to scale trajectory predicted by [SfM Learner](https://github.com/ClementPinard/SfmLearner-Pytorch). For scaling we use metadata such as camera heigth, video frame rate and camera intrinsics.

![Algorithm overview](images/algorithm-overview.jpg)

## Usage

Try [demo](demo.ipynb) in Google Colaboratory!

