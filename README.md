# Two Stage Object Detector

A two-stage object detector, based on Faster R-CNN, which consists of two modules - Region Proposal Networks (RPN) and Fast R-CNN. Trained to detect a set of object classes and evaluate the detection accuracy using the classic metric mean Average Precision (mAP).

<p align="center">
  <img src="./gif/results.gif" alt="Image Description" width="400" height="300">
</p>


# Contents

[***Objective***](https://github.com/leob03/2Stage_Object_Detector#objective)

[***Concepts***](https://github.com/leob03/2Stage_Object_Detector#concepts)

[***Overview***](https://github.com/leob03/2Stage_Object_Detector#overview)

[***Dependencies***](https://github.com/leob03/2Stage_Object_Detector#dependencies)

[***Getting started***](https://github.com/leob03/2Stage_Object_Detector#getting-started)

[***Deeper dive into the code***](https://github.com/leob03/2Stage_Object_Detector#deeper-dive-into-the-code)

# Objective

** To estimate the 3D translation of an object by localizing its center in the image and predicting its distance from the camera.**

In this project, we implemented an **end-to-end** object pose estimator, based on [PoseCNN](https://arxiv.org/abs/1711.00199), which consists of two stages - feature extraction with a backbone network and pose estimation represented by instance segmentation, 3D translation estimation, and 3D rotation estimation.
We will train it to estimate the pose of a set of object classes and evaluate the estimation accuracy.

<p align="center">
  <img src="./img/pose_image.png" alt="Image Description" width="600" height="400">
</p>


# Concepts

* **Semantic Labeling**. In order to detect objects in images, we resort to semantic labeling, where the network classifies each image pixel into an
object class. Compared to recent 6D pose estimation methods that resort to object detection with bounding boxes, semantic labeling provides richer information about the objects and handles occlusions better.

* **3D Translation Estimation**. 3D translation estimation refers to the task of determining the spatial translation of an object in a three-dimensional coordinate system. It involves predicting the displacement or movement of an object from a reference position to its current position in 3D space. This dispacement can be represented by a translation vector that typically consists of three values representing the displacements along the x, y, and z axes.

* **3D Rotation Regression**. 3D rotation regression refers to the task of estimating the rotational orientation or pose of an object in three-dimensional space. It involves predicting the rotation parameters that describe the object's orientation relative to a reference position. The rotation parameters are typically represented as quaternions, Euler angles, or rotation matrices, which capture the object's orientation along the x, y, and z axes.
  
# Overview

This architecture is designed to take an RGB color image as input and produce a [6 degrees-of-freedom pose](https://en.wikipedia.org/wiki/Six_degrees_of_freedom) estimate for each instance of an object within the scene from which the image was taken. To do this, PoseCNN uses 5 operations within the architecture descried in the next pipeline:

- The **input** is a dataset of images and 5 sentence descriptions that were collected with Amazon Mechanical Turk. We will use the 2014 release of the [COCO Captions dataset](http://cocodataset.org/) which has become the standard testbed for image captioning. The dataset consists of 80,000 training images and 40,000 validation images, each annotated with 5 captions.
- In the **training stage**, the images are fed as input to RNN (or LSTM/LSTM with attention depending on the model) and the RNN is asked to predict the words of the sentence, conditioned on the current word and previous context as mediated by the hidden layers of the neural network. In this stage, the parameters of the networks are trained with backpropagation.
- In the **prediction stage**, a witheld set of images is passed to RNN and the RNN generates the sentence one word at a time. The code also includes utilities for visualizing the results.

- First, a backbone convolutional **feature extraction** network is used to produce a tensor representing learned features from the input image.
- Second, the extracted features are processed by an **embedding branch** to reduce the spatial resolution and memory overhead for downstream layers.
- Third, an **instance segmentation branch** uses the embedded features to identify regions in the image corresponding to each object instance (regions of interest).
- Fourth, the translations for each object instance are estimated using a **translation branch** along with the embedded features.
- Finally, a **rotation branch** uses the embedded features to estimate a rotation, in the form of a [quaternion](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation), for each region of interest.

The architecture is shown in more detail from Figure 2 of the [PoseCNN paper](https://arxiv.org/abs/1711.00199):

![architecture](https://deeprob.org/assets/images/posecnn_arch.png)

Now, we will implement a variant of this architecture that performs each of the 5 operations using PyTorch and data from our `PROPSPoseDataset`.

# Dependencies
**Python 3.10**, modern version of **PyTorch**, **numpy** and **scipy** module. Most of these are okay to install with **pip**. To install all dependencies at once, run the command `pip install -r requirements.txt`

I only tested this code with Ubuntu 20.04, but I tried to make it as generic as possible (e.g. use of **os** module for file system interactions etc. So it might work on Windows and Mac relatively easily.)


# Getting started

1. **Get the code.** `$ git clone` the repo and install the Python dependencies
2. **Train the models.** Run the training `$ train.py` and wait. You'll see that the learning code writes checkpoints into `cv/` and periodically print its status. 
3. **Evaluate the models checkpoints and Visualize the predictions.** To evaluate a checkpoint run the scripts `$ python test.py` and pass it the path to a checkpoint (by modifying the checkpoint in the code, default: posecnn_model.pth).

# Deeper dive into the code

### PROPS Pose Dataset
