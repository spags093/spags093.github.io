---
layout: post
title:      "Deepfake Image Detection"
date:       2020-11-28 05:52:04 +0000
permalink:  deepfake_image_detection
---


In recent years, deepfake technology has made huge leaps in terms of accuracy, quality, and most of all: believability.  At one time or another, we've all been fooled by a photoshopped viral image or even someone's Snapchat filter.  The abilitly to determine whether or not an image is, in fact, real is quickly becoming a necessity in a world that's becoming more and more susceptible to "questionable" information.  Recently, doctored images have made headlines and divided people in their opinions of whether these images were legitimate.

In this project, I set out to build an image classification system that can determine whether an image is real or fake.  In order to achieve this, I built several iterations of convolutional neural networks that trained on hundreds of thousands of images of human faces, roughly half real images and half doctored images.  After creating a CNN that I personally tuned, creating a CNN with a pretrained model, and creating an ensemble of both, I was able to achieve a 97% accuracy in the detection of real vs. fake images.  With this level of accuracy on the training and test sets, I felt that it was ready to be deployed to web app where users can upload a new, untrained face image and the model can determine the validity of their image as well as explain how it came to that conclusion.  
