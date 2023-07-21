import matplotlib.pyplot as plt
import torch
import torchvision

from p3_helper import *
from utils import reset_seed
from utils.grad import rel_error

# for plotting
plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["font.size"] = 16
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

# To download the dataset
!pip install wget

# for mAP evaluation
!rm -rf mAP
!git clone https://github.com/Cartucho/mAP.git
!rm -rf mAP/input/*
from two_stage_detector import DetectorBackboneWithFPN

backbone = DetectorBackboneWithFPN(out_channels=64)
