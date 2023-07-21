import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import torchvision

from func import *
from utils.utils import reset_seed
from utils.grad import rel_error

from two_stage_detector import RPNPredictionNetwork

rpn_pred_net = RPNPredictionNetwork(
    in_channels=64, stem_channels=[64], num_anchors=3
)

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

import multiprocessing
from func import PROPSDetectionDataset

# Set a few constants related to data loading.
NUM_CLASSES = 10
BATCH_SIZE = 16
IMAGE_SHAPE = (224, 224)
NUM_WORKERS = multiprocessing.cpu_count()

from torchvision import transforms
from rob599.utils import detection_visualizer

inverse_norm = transforms.Compose(
    [
        transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ]
)

from two_stage_detector import get_fpn_location_coords, generate_fpn_anchors, iou, rcnn_match_anchors_to_gt, nms

reset_seed(0)


def get_data():
  train_dataset = PROPSDetectionDataset(
      GOOGLE_DRIVE_PATH, "train", image_size=IMAGE_SHAPE[0],
      download=False  # True (for the first time)
  )
  val_dataset = PROPSDetectionDataset(GOOGLE_DRIVE_PATH, "val", image_size=IMAGE_SHAPE[0])
  # `pin_memory` speeds up CPU-GPU batch transfer, `num_workers=NUM_WORKERS` loads data
  # on the main CPU process, suitable for Colab.
  train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS
  )
  
  # Use batch_size = 1 during inference - during inference we do not center crop
  # the image to detect all objects, hence they may be of different size. It is
  # easier and less redundant to use batch_size=1 rather than zero-padding images.
  val_loader = torch.utils.data.DataLoader(
      val_dataset, batch_size=1, pin_memory=True, num_workers=NUM_WORKERS
  )
  
  train_loader_iter = iter(train_loader)
  image_paths, images, gt_boxes = next(train_loader_iter)
  return train_dataset, val_dataset

