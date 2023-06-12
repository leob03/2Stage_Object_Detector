import math
from typing import Dict, List, Optional, Tuple

import torch
import torchvision
from p3_helper import *
from torch import dtype, nn, zeros
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


# Short hand type notation:
TensorDict = Dict[str, torch.Tensor]


def hello_two_stage_detector():
    print("Hello from two_stage_detector.py!")
    
    
    
class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        # Initialization of additional Conv layers for FPN

        # THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)
        # such that they all end up with the same `out_channels`.
        # THREE "output" 3x3 conv layers to transform the merged
        # FPN features to output (p3, p4, p5) features.
        # THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        C3_in, C4_in, C5_in = dummy_out_shapes[0][1][1], dummy_out_shapes[1][1][1], dummy_out_shapes[2][1][1]
        self.fpn_params = nn.ModuleDict({
                'convl3': nn.Conv2d(C3_in,self.out_channels,1,stride=1,padding=0),
                'convl4': nn.Conv2d(C4_in,self.out_channels,1,stride=1,padding=0),
                'convl5': nn.Conv2d(C5_in,self.out_channels,1,stride=1,padding=0),
                'conv3': nn.Conv2d(self.out_channels,self.out_channels,3,stride=1,padding=1),
                'conv4': nn.Conv2d(self.out_channels,self.out_channels,3,stride=1,padding=1),
                'conv5': nn.Conv2d(self.out_channels,self.out_channels,3,stride=1,padding=1),                
        })


    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}

        p5_inter = self.fpn_params['convl5'](backbone_feats["c5"])
        fpn_feats["p5"]= self.fpn_params['conv5'](p5_inter)
        
        p4_inter=self.fpn_params['convl4'](backbone_feats["c4"])
        p5_inter_up2 = F.interpolate(p5_inter, scale_factor=2, mode="nearest")
        p4_inter+=p5_inter_up2
        fpn_feats["p4"]= self.fpn_params['conv4'](p4_inter)

        p3_inter=self.fpn_params['convl3'](backbone_feats["c3"])
        # p5_inter_up4 = F.interpolate(p5_inter_up2, scale_factor=2, mode="nearest")
        p4_inter_up2 = F.interpolate(p4_inter, scale_factor=2, mode="nearest")
        p3_inter+= p4_inter_up2
        fpn_feats["p3"]= self.fpn_params['conv3'](p3_inter)

        return fpn_feats

def weights_init_normal(m):
  classname = m.__class__.__name__
  if classname.find('Conv2d') != -1:
      m.weight.data.normal_(0.0,0.01)
      m.bias.data.fill_(0)
        # if type(m) == nn.Conv2d:
        #   nn.init.normal_(m.weight, mean=0, std=0.01)(m.weight)

class RPNPredictionNetwork(nn.Module):
    """
    RPN prediction network that accepts FPN feature maps from different levels
    and makes two predictions for every anchor: objectness and box deltas.

    Faster R-CNN typically uses (p2, p3, p4, p5) feature maps. We will exclude
    p2 for have a small enough model for Colab.

    Conceptually this module is quite similar to `FCOSPredictionNetwork`.
    """

    def __init__(
        self, in_channels: int, stem_channels: List[int], num_anchors: int = 3
    ):
        """
        Args:
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
            num_anchors: Number of anchor boxes assumed per location (say, `A`).
                Faster R-CNN without an FPN uses `A = 9`, anchors with three
                different sizes and aspect ratios. With FPN, it is more common
                to have a fixed size dependent on the stride of FPN level, hence
                `A = 3` is default - with three aspect ratios.
        """
        super().__init__()

        self.num_anchors = num_anchors
        # a stem of alternating 3x3 convolution layers and RELU
        # activation modules. RPN shares this stem for objectness and box
        # regression (unlike FCOS, that uses separate stems).

        stem_rpn = []
        self.num_layers = len(stem_channels)
        in_chan = in_channels
        for i in range(1, self.num_layers+1):
          out_chan = stem_channels[i-1]
          stem_rpn.append(nn.Conv2d(in_chan, out_chan, 3, stride=1, padding=1))
          stem_rpn.append(nn.ReLU())
          in_chan= out_chan
        # Wrap the layers defined by student into a `nn.Sequential` module:
        self.stem_rpn = nn.Sequential(*stem_rpn)
        self.stem_rpn.apply(weights_init_normal)
        # TWO 1x1 conv layers for individually to predict
        # objectness and box deltas for every anchor, at every location.

        # Objectness is obtained by applying sigmoid to its logits.

        self.pred_obj = nn.Conv2d(stem_channels[-1],num_anchors,1)  # Objectness conv
        self.pred_box = nn.Conv2d(stem_channels[-1],4*num_anchors,1)  # Box regression conv


    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict desired quantities for every anchor
        at every location. Format the output tensors such that feature height,
        width, and number of anchors are collapsed into a single dimension (see
        description below in "Returns" section) this is convenient for computing
        loss and perforning inference.

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}.
                Each tensor will have shape `(batch_size, fpn_channels, H, W)`.

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Objectness logits:     `(batch_size, H * W * num_anchors)`
            2. Box regression deltas: `(batch_size, H * W * num_anchors, 4)`
        """

        # Iteration over every FPN feature map and predictions using
        # the layers defined above.
        object_logits = {}
        boxreg_deltas = {}

        batch_size = feats_per_fpn_level["p3"].shape[0]

        p3_inter = self.stem_rpn(feats_per_fpn_level["p3"])
        p4_inter = self.stem_rpn(feats_per_fpn_level["p4"])
        p5_inter = self.stem_rpn(feats_per_fpn_level["p5"])

        p3_obj = self.pred_obj(p3_inter)
        object_logits["p3"]=p3_obj.reshape(batch_size,-1)
        p3_box = self.pred_box(p3_inter)
        boxreg_deltas["p3"]=p3_box.reshape(batch_size,-1,4)

        p4_obj = self.pred_obj(p4_inter)
        object_logits["p4"]=p4_obj.reshape(batch_size,-1)
        p4_box = self.pred_box(p4_inter)
        boxreg_deltas["p4"]=p4_box.reshape(batch_size,-1,4)

        p5_obj = self.pred_obj(p5_inter)
        object_logits["p5"]=p5_obj.reshape(batch_size,-1)
        p5_box = self.pred_box(p5_inter)
        boxreg_deltas["p5"]=p5_box.reshape(batch_size,-1,4)

        return [object_logits, boxreg_deltas]


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]
        #logic to get location co-ordinates below.
        loc=torch.zeros(feat_shape[2]*feat_shape[3],2,device=device,dtype=dtype)
        n=0
        for i in range(feat_shape[2]):
          for j in range(feat_shape[3]):
            loc[n]=torch.tensor((level_stride*(i+0.5),level_stride*(j+0.5)))
            n+=1
        location_coords[level_name]=loc

    return location_coords


@torch.no_grad()
def generate_fpn_anchors(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    stride_scale: int,
    aspect_ratios: List[float] = [0.5, 1.0, 2.0],
):
    """
    Generate multiple anchor boxes at every location of FPN level. Anchor boxes
    should be in XYXY format and they should be centered at the given locations.

    Args:
        locations_per_fpn_level: Centers at different levels of FPN (p3, p4, p5),
            that are already projected to absolute co-ordinates in input image
            dimension. Dictionary of three keys: (p3, p4, p5) giving tensors of
            shape `(H * W, 2)` where H, W is the size of FPN feature map.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `common.py` for more details.
        stride_scale: Size of square anchor at every FPN levels will be
            `(this value) * (FPN level stride)`. Default is 4, which will make
            anchor boxes of size (32x32), (64x64), (128x128) for FPN levels
            p3, p4, and p5 respectively.
        aspect_ratios: Anchor aspect ratios to consider at every location. We
            consider anchor area to be `(stride_scale * FPN level stride) ** 2`
            and set new width and height of anchors at every location:
                new_width = sqrt(area / aspect ratio)
                new_height = area / new_width

    Returns:
        TensorDict
            Dictionary with same keys as `locations_per_fpn_level` and values as
            tensors of shape `(HWA, 4)` giving anchors for all locations
            per FPN level, each location having `A = len(aspect_ratios)` anchors.
            All anchors are in XYXY format and their centers align with locations.
    """

    # `(N, A, 4)` Tensors giving anchor boxes in XYXY format.
    anchors_per_fpn_level = {
        level_name: None for level_name, _ in locations_per_fpn_level.items()
    }

    for level_name, locations in locations_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        # List of `A = len(aspect_ratios)` anchor boxes.
        anchor_boxes = []
        for aspect_ratio in aspect_ratios:
            # logic for anchor boxes below.
            # Vectorized implementation to generate anchors for a single aspect ratio.
            #
            # Calculate resulting width and height of the anchor box as per
            # `stride_scale` and `aspect_ratios` definitions. Then shift the
            # locations to get top-left and bottom-right co-ordinates.
            area = (stride_scale * level_stride) ** 2
            new_width = (area / aspect_ratio)**(1/2)
            new_height = area / new_width

            add1 = torch.tensor((-new_width/2,-new_height/2))
            anchor_box1=locations+add1
            add2 = torch.tensor((new_width/2,new_height/2))
            anchor_box2=locations+add2
            anchor_boxes.append(torch.hstack((anchor_box1,anchor_box2)))

        # shape: (A, H * W, 4)
        anchor_boxes = torch.stack(anchor_boxes)
        # Bring `H * W` first and collapse those dimensions.
        anchor_boxes = anchor_boxes.permute(1, 0, 2).contiguous().view(-1, 4)
        anchors_per_fpn_level[level_name] = anchor_boxes

    return anchors_per_fpn_level


@torch.no_grad()
def iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute intersection-over-union (IoU) between pairs of box tensors. Input
    box tensors must in XYXY format.

    Args:
        boxes1: Tensor of shape `(M, 4)` giving a set of box co-ordinates.
        boxes2: Tensor of shape `(N, 4)` giving another set of box co-ordinates.

    Returns:
        torch.Tensor
            Tensor of shape (M, N) with `iou[i, j]` giving IoU between i-th box
            in `boxes1` and j-th box in `boxes2`.
    """
    # iou = torch.zeros(boxes1.shape[0],boxes2.shape[0])
    dev = boxes1.device
    dtyp = boxes1.dtype
    M = boxes1.shape[0]
    N = boxes2.shape[0]

    x11 = boxes1[:,0].expand(N,-1).T
    x11 = x11.to(dev)
    y11 = boxes1[:,1].expand(N,-1).T
    y11 = y11.to(dev)
    x12 = boxes1[:,2].expand(N,-1).T
    x12 = x12.to(dev)
    y12 = boxes1[:,3].expand(N,-1).T
    y12 = y12.to(dev)

    x21 = boxes2[:,0].expand(M,-1)
    x21 = x21.to(dev)
    y21 = boxes2[:,1].expand(M,-1)
    y21 = y21.to(dev)
    x22 = boxes2[:,2].expand(M,-1)
    x22 = x22.to(dev)
    y22 = boxes2[:,3].expand(M,-1)
    y22 = y22.to(dev)

    xA = torch.maximum(x11,x21)
    yA = torch.maximum(y11,y21)
    xB = torch.minimum(x12,x22)
    yB = torch.minimum(y12,y22)

    interArea = torch.maximum((xB-xA),torch.tensor([0], device=dev, dtype=dtyp)) * torch.maximum((yB-yA),torch.tensor([0], device=dev,dtype=dtyp))
    box1Area = (x12 - x11) * (y12 - y11)
    box2Area = (x22 - x21) * (y22 - y21)
    iou = interArea / (box1Area + box2Area - interArea)
    return iou


@torch.no_grad()
def rcnn_match_anchors_to_gt(
    anchor_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresholds: Tuple[float, float],
) -> TensorDict:
    """
    Match anchor boxes (or RPN proposals) with a set of GT boxes. Anchors having
    high IoU with any GT box are assigned "foreground" and matched with that box
    or vice-versa.

    NOTE: This function is NOT BATCHED. Call separately for GT boxes per image.

    Args:
        anchor_boxes: Anchor boxes (or RPN proposals). Dictionary of three keys
            a combined tensor of some shape `(N, 4)` where `N` are total anchors
            from all FPN levels, or a set of RPN proposals.
        gt_boxes: GT boxes of a single image, a batch of `(M, 5)` boxes with
            absolute co-ordinates and class ID `(x1, y1, x2, y2, C)`. In this
            codebase, this tensor is directly served by the dataloader.
        iou_thresholds: Tuple of (low, high) IoU thresholds, both in [0, 1]
            giving thresholds to assign foreground/background anchors.
    """

    # Filter empty GT boxes:
    gt_boxes = gt_boxes[gt_boxes[:, 4] != -1]

    # If no GT boxes are available, match all anchors to background and return.
    if len(gt_boxes) == 0:
        fake_boxes = torch.zeros_like(anchor_boxes) - 1
        fake_class = torch.zeros_like(anchor_boxes[:, [0]]) - 1
        return torch.cat([fake_boxes, fake_class], dim=1)

    # Match matrix => pairwise IoU of anchors (rows) and GT boxes (columns).
    # STUDENTS: This matching depends on your IoU implementation.
    match_matrix = iou(anchor_boxes, gt_boxes[:, :4])

    # Find matched ground-truth instance per anchor:
    match_quality, matched_idxs = match_matrix.max(dim=1)
    matched_gt_boxes = gt_boxes[matched_idxs]

    # Set boxes with low IoU threshold to background (-1).
    matched_gt_boxes[match_quality <= iou_thresholds[0]] = -1

    # Set remaining boxes to neutral (-1e8).
    neutral_idxs = (match_quality > iou_thresholds[0]) & (
        match_quality < iou_thresholds[1]
    )
    matched_gt_boxes[neutral_idxs, :] = -1e8
    return matched_gt_boxes


def rcnn_get_deltas_from_anchors(
    anchors: torch.Tensor, gt_boxes: torch.Tensor
) -> torch.Tensor:
    """
    Get box regression deltas that transform `anchors` to `gt_boxes`. These
    deltas will become GT targets for box regression. Unlike FCOS, the deltas
    are in `(dx, dy, dw, dh)` format that represent offsets to anchor centers
    and scaling factors for anchor size. Box regression is only supervised by
    foreground anchors. If GT boxes are "background/neutral", then deltas
    must be `(-1e8, -1e8, -1e8, -1e8)` (just some LARGE negative number).

    Follow lecture 12:
        https://deeprob.org/calendar/#lec-12

    Args:
        anchors: Tensor of shape `(N, 4)` giving anchors boxes in XYXY format.
        gt_boxes: Tensor of shape `(N, 4)` giving matching GT boxes.

    Returns:
        torch.Tensor
            Tensor of shape `(N, 4)` giving anchor deltas.
    """
    # The logic to get deltas.                               #
    # Remember to set the deltas of "background/neutral" GT boxes to -1e8    #
    deltas = None
    deltas = torch.zeros_like(anchors)
    iou_matrix = iou(anchors, gt_boxes)
    for i in range(anchors.shape[0]):
      anchor_box = anchors[i]
      gt_box = gt_boxes[i]
      if torch.max(iou_matrix[:,i]) < 0.6 and gt_box[4]==-1:
        delta = torch.tensor((-1e8, -1e8, -1e8, -1e8))
      else:
        (xa1,ya1,xa2,ya2) = anchor_box
        (xg1,yg1,xg2,yg2) = gt_box[:4]
        pw= xa2-xa1
        ph= ya2-ya1
        bw= xg2-xg1
        bh= yg2-yg1
        px= xa1 + pw/2
        py= ya1 + ph/2
        bx= xg1 + bw/2
        by= yg1 + bh/2
        delta = torch.tensor(((bx-px)/pw,(by-py)/ph,torch.log(bw/pw),torch.log(bh/ph)))
      deltas[i]= delta

    return deltas


def rcnn_apply_deltas_to_anchors(
    deltas: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    """
    Implement the inverse of `rcnn_get_deltas_from_anchors` here.

    Args:
        deltas: Tensor of shape `(N, 4)` giving box regression deltas.
        anchors: Tensor of shape `(N, 4)` giving anchors to apply deltas on.

    Returns:
        torch.Tensor
            Same shape as deltas and locations, giving the resulting boxes in
            XYXY format.
    """

    # Clamp dw and dh such that they would transform a 8px box no larger than
    # 224px. This is necessary for numerical stability as we apply exponential.
    scale_clamp = math.log(224 / 8)
    deltas[:, 2] = torch.clamp(deltas[:, 2], max=scale_clamp)
    deltas[:, 3] = torch.clamp(deltas[:, 3], max=scale_clamp)
    # Transformation logic to get output boxes.
    output_boxes = None
    anchors = anchors.to(deltas.device)
    p_x = (anchors[:,0]+anchors[:,2])/2
    p_y = (anchors[:,1]+anchors[:,3])/2
    p_h = anchors[:,2]-anchors[:,0]
    p_w = anchors[:,3]-anchors[:,1]
    b_x = p_x + p_h*deltas[:,0]
    b_y = p_y + p_w*deltas[:,1]
    b_h = p_h*torch.exp(deltas[:,2])
    b_w = p_w*torch.exp(deltas[:,3])
    X_TL = b_x-b_h/2
    Y_TL = b_y-b_w/2
    X_BR = b_x+b_h/2
    Y_BR = b_y+b_w/2
    output_boxes = torch.stack((X_TL,Y_TL,X_BR,Y_BR),dim=1)
    output_boxes[deltas[:,0]==1e-8] = 1e-8
    return output_boxes


@torch.no_grad()
def sample_rpn_training(
    gt_boxes: torch.Tensor, num_samples: int, fg_fraction: float
):
    """
    Return `num_samples` (or fewer, if not enough found) random pairs of anchors
    and GT boxes without exceeding `fg_fraction * num_samples` positives, and
    then try to fill the remaining slots with background anchors. We will ignore
    "neutral" anchors in this sampling as they are not used for training.

    Args:
        gt_boxes: Tensor of shape `(N, 5)` giving GT box co-ordinates that are
            already matched with some anchor boxes (with GT class label at last
            dimension). Label -1 means background and -1e8 means meutral.
        num_samples: Total anchor-GT pairs with label >= -1 to return.
        fg_fraction: The number of subsampled labels with values >= 0 is
            `min(num_foreground, int(fg_fraction * num_samples))`. In other
            words, if there are not enough fg, the sample is filled with
            (duplicate) bg.

    Returns:
        fg_idx, bg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or
            fewer. Use these to index anchors, GT boxes, and model predictions.
    """
    foreground = (gt_boxes[:, 4] >= 0).nonzero().squeeze(1)
    background = (gt_boxes[:, 4] == -1).nonzero().squeeze(1)

    # Protect against not enough foreground examples.
    num_fg = min(int(num_samples * fg_fraction), foreground.numel())
    num_bg = num_samples - num_fg

    # Randomly select positive and negative examples.
    perm1 = torch.randperm(foreground.numel(), device=foreground.device)[:num_fg]
    perm2 = torch.randperm(background.numel(), device=background.device)[:num_bg]

    fg_idx = foreground[perm1]
    bg_idx = background[perm2]
    return fg_idx, bg_idx


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    # non-maximum suppression which iterates the following:     
    #       1. Select the highest-scoring box among the remaining ones,         
    #          which has not been chosen in this step before                    
    #       2. Eliminate boxes with IoU > threshold                             
    #       3. If any boxes remain, GOTO 1                                      
    #       Your implementation should not depend on a specific device type;    
    #       you can use the device of the input if necessary.                   
    # refer to the torchvision library code:
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    
    keep = []
    order = torch.argsort(scores)
    # print(order.shape)
    while len(order) > 0:
      index = order[-1]
      keep.append(index)
      IoU = iou(boxes[index].reshape(1,-1), boxes[order[:-1]])
      left = IoU.flatten() <= iou_threshold
      left = left.nonzero().flatten()
      order = order[left]
    keep = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    return keep


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


@torch.no_grad()
def reassign_proposals_to_fpn_levels(
    proposals_per_image: List[torch.Tensor],
    gt_boxes: Optional[torch.Tensor] = None,
    fpn_level_ids: List[int] = [3, 4, 5],
) -> Dict[str, List[torch.Tensor]]:
    """
    The first-stage in Faster R-CNN (RPN) gives a few proposals that are likely
    to contain any object. These proposals would have come from any FPN level -
    for example, they all maybe from level p5, and none from levels p3/p4 (= the
    image mostly has large objects and no small objects). In second stage, these
    proposals are used to extract image features (via RoI-align) and predict the
    class labels. But we do not know which level to use, due to two reasons:

        1. We did not keep track of which level each proposal came from.
        2. ... even if we did keep track, it may be possible that RPN deltas
           transformed a large anchor box from p5 to a tiny proposal (which could
           be more suitable for a lower FPN level).

    Hence, we re-assign proposals to different FPN levels according to sizes.
    Large proposals get assigned to higher FPN levels, and vice-versa.

    At start of training, RPN proposals may be low quality. It's possible that
    very few of these have high IoU with GT boxes. This may stall or de-stabilize
    training of second stage. This function also mixes GT boxes with RPN proposals
    to improve training. GT boxes are also assigned by their size.

    See Equation (1) in FPN paper (https://arxiv.org/abs/1612.03144).

    Args:
        proposals_per_image: List of proposals per image in batch. Same as the
            outputs from `RPN.forward()` method.
        gt_boxes: Tensor of shape `(B, M, 4 or 5)` giving GT boxes per image in
            batch (with or without GT class label, doesn't matter). These are
            not present during inference.
        fpn_levels: List of FPN level IDs. For this codebase this will always
            be [3, 4, 5] for levels (p3, p4, p5) -- we include this in input
            arguments to avoid any hard-coding in function body.

    Returns:
        Dict[str, List[torch.Tensor]]
            Dictionary with keys `{"p3", "p4", "p5"}` each containing a list
            of `B` (`batch_size`) tensors. The `i-th` element in this list will
            give proposals of `i-th` image, assigned to that FPN level. An image
            may not have any proposals for a particular FPN level, for which the
            tensor will be a tensor of shape `(0, 4)` -- PyTorch supports this!
    """

    # Make empty lists per FPN level to add assigned proposals for every image.
    proposals_per_fpn_level = {f"p{_id}": [] for _id in fpn_level_ids}

    # Usually 3 and 5.
    lowest_level_id, highest_level_id = min(fpn_level_ids), max(fpn_level_ids)

    for idx, _props in enumerate(proposals_per_image):

        # Mix ground-truth boxes for every example, per FPN level.
        if gt_boxes is not None:
            # Filter empty GT boxes and remove class label.
            _gtb = gt_boxes[idx]
            _props = torch.cat([_props, _gtb[_gtb[:, 4] != -1][:, :4]], dim=0)

        # Compute FPN level assignments for each GT box. This follows Equation (1)
        # of FPN paper (k0 = 4). `level_assn` has `(M, )` integers, one of {3,4,5}
        _areas = (_props[:, 2] - _props[:, 0]) * (_props[:, 3] - _props[:, 1])

        # Assigned FPN level ID for each proposal (an integer between lowest_level
        # and highest_level).
        level_assignments = torch.floor(4 + torch.log2(torch.sqrt(_areas) / 224))
        level_assignments = torch.clamp(
            level_assignments, min=lowest_level_id, max=highest_level_id
        )
        level_assignments = level_assignments.to(torch.int64)

        # Iterate over FPN level IDs and get proposals for each image, that are
        # assigned to that level.
        for _id in range(lowest_level_id, highest_level_id + 1):
            proposals_per_fpn_level[f"p{_id}"].append(
                # This tensor may have zero proposals, and that's okay.
                _props[level_assignments == _id]
            )

    return proposals_per_fpn_level


class RPN(nn.Module):
    """
    Region Proposal Network: First stage of Faster R-CNN detector.

    This class puts together everything you implemented so far. It accepts FPN
    features as input and uses `RPNPredictionNetwork` to predict objectness and
    box reg deltas. Computes proposal boxes for second stage (during both
    training and inference) and losses during training.
    """

    def __init__(
        self,
        fpn_channels: int,
        stem_channels: List[int],
        batch_size_per_image: int,
        anchor_stride_scale: int = 8,
        anchor_aspect_ratios: List[int] = [0.5, 1.0, 2.0],
        anchor_iou_thresholds: Tuple[int, int] = (0.3, 0.6),
        nms_thresh: float = 0.7,
        pre_nms_topk: int = 400,
        post_nms_topk: int = 100,
    ):
        """
        Args:
            batch_size_per_image: Anchors per image to sample for training.
            nms_thresh: IoU threshold for NMS - unlike FCOS, this is used
                during both, training and inference.
            pre_nms_topk: Number of top-K proposals to select before applying
                NMS, per FPN level. This helps in speeding up NMS.
            post_nms_topk: Number of top-K proposals to select after applying
                NMS, per FPN level. NMS is obviously going to be class-agnostic.

        Refer explanations of remaining args in the classes/functions above.
        """
        super().__init__()
        self.pred_net = RPNPredictionNetwork(
            fpn_channels, stem_channels, num_anchors=len(anchor_aspect_ratios)
        )
        # Record all input arguments:
        self.batch_size_per_image = batch_size_per_image
        self.anchor_stride_scale = anchor_stride_scale
        self.anchor_aspect_ratios = anchor_aspect_ratios
        self.anchor_iou_thresholds = anchor_iou_thresholds
        self.nms_thresh = nms_thresh
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk

    def forward(
        self,
        feats_per_fpn_level: TensorDict,
        strides_per_fpn_level: TensorDict,
        gt_boxes: Optional[torch.Tensor] = None,
    ):
        # Get batch size from FPN feats:
        num_images = feats_per_fpn_level["p3"].shape[0]

        # Steps for the training forward pass:
        #   1. Pass the FPN features per level to the RPN prediction network.
        #   2. Generate anchor boxes for all FPN levels.
        shape_per_fpn_level = {
            level_name: None for level_name, _ in feats_per_fpn_level.items()
        }
        for level_name, feat in feats_per_fpn_level.items():
            shape_per_fpn_level[level_name] = feat.shape
        pred_obj_logits, pred_boxreg_deltas = self.pred_net(feats_per_fpn_level)
        anchors_per_fpn_level = generate_fpn_anchors(get_fpn_location_coords(shape_per_fpn_level,strides_per_fpn_level),strides_per_fpn_level, self.anchor_stride_scale, self.anchor_aspect_ratios)

        # We will fill three values in this output dict - "proposals",
        # "loss_rpn_box" (training only), "loss_rpn_obj" (training only)
        output_dict = {}

        # Get image height and width according to feature sizes and strides.
        # We need these to clamp proposals (These should be (224, 224) but we
        # avoid hard-coding them).
        img_h = feats_per_fpn_level["p3"].shape[2] * strides_per_fpn_level["p3"]
        img_w = feats_per_fpn_level["p3"].shape[3] * strides_per_fpn_level["p3"]

        output_dict["proposals"] = self.predict_proposals(
            anchors_per_fpn_level,
            pred_obj_logits,
            pred_boxreg_deltas,
            (img_w, img_h),
        )
        if not self.training:
            return output_dict

        # Match the generated anchors with provided GT boxes.
        # Combine anchor boxes from all FPN levels - we do not need any
        # distinction of boxes across different levels (for training).
        anchor_boxes = self._cat_across_fpn_levels(anchors_per_fpn_level, dim=0)

        matched_gt_boxes = []

        for _batch_idx in range(gt_boxes.shape[0]):
          matched_gt_boxes.append(rcnn_match_anchors_to_gt(anchor_boxes,gt_boxes[_batch_idx],self.anchor_iou_thresholds))

        # for _batch_idx in range(num_images):
        #   matched_gt_boxes.append(rcnn_match_anchors_to_gt(anchor_boxes, gt_boxes[_batch_idx], self.anchor_iou_thresholds))

        # Combine matched boxes from all images to a `(B, HWA, 5)` tensor.
        matched_gt_boxes = torch.stack(matched_gt_boxes, dim=0)

        # Combine predictions across all FPN levels.
        pred_obj_logits = self._cat_across_fpn_levels(pred_obj_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)

        if self.training:
            # Repeat anchor boxes `batch_size` times so there is a 1:1
            # correspondence with GT boxes.
            anchor_boxes = anchor_boxes.unsqueeze(0).repeat(num_images, 1, 1)
            anchor_boxes = anchor_boxes.contiguous().view(-1, 4)

            # Collapse `batch_size`, and `HWA` to a single dimension so we have
            # simple `(-1, 4 or 5)` tensors. This simplifies loss computation.
            matched_gt_boxes = matched_gt_boxes.view(-1, 5)
            pred_obj_logits = pred_obj_logits.view(-1)
            pred_boxreg_deltas = pred_boxreg_deltas.view(-1, 4)

            # Steps for computing training losses.
            #   1. Sample a few anchor boxes for training. Pass the variable
            #      `matched_gt_boxes` to `sample_rpn_training` function and
            #      use those indices to get subset of predictions and targets.
            #      RPN samples 50-50% foreground/background anchors, unless
            #      there aren't enough foreground anchors.
            #
            #   2. Compute GT targets for box regression (you have implemented
            #      the transformation function already).
            #
            #   3. Calculate objectness and box reg losses per sampled anchor.
            #      Remember to set box loss for "background" anchors to 0.

            fg_idx, bg_idx = sample_rpn_training(matched_gt_boxes,num_images*self.batch_size_per_image,0.5)
            matched_gt_boxes = matched_gt_boxes.to(fg_idx.device)
            anchor_boxes = anchor_boxes.to(fg_idx.device)

            gt_targets = rcnn_get_deltas_from_anchors(anchor_boxes[fg_idx],matched_gt_boxes[fg_idx,:4])
            # gt_targets[bg_idx] = -1e8

            loss_box = F.l1_loss(pred_boxreg_deltas[fg_idx], gt_targets, reduction="none")
            loss_box[gt_targets == -1e8] *= 0.0

            pred_obj_logits_sampled = torch.cat((pred_obj_logits[fg_idx],pred_obj_logits[bg_idx]))
            gt_obj_logits_sampled = torch.zeros_like(pred_obj_logits_sampled)
            gt_obj_logits_sampled[:len(fg_idx)] = 1

            loss_obj = F.binary_cross_entropy_with_logits(pred_obj_logits_sampled.view(-1),gt_obj_logits_sampled.view(-1),reduction='none')

            # Sum losses and average by num(foreground + background) anchors.
            # In training code, we simply add these two and call `.backward()`
            total_batch_size = self.batch_size_per_image * num_images
            output_dict["loss_rpn_obj"] = loss_obj.sum() / total_batch_size
            output_dict["loss_rpn_box"] = loss_box.sum() / total_batch_size

        return output_dict

    
    @torch.no_grad()
    def predict_proposals(
        self,
        anchors_per_fpn_level: Dict[str, torch.Tensor],
        pred_obj_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        image_size: Tuple[int, int],  # (width, height)
    ) -> List[torch.Tensor]:
        """
        Predict proposals for a batch of images for the second stage. Other
        input arguments are same as those computed in `forward` method. This
        method should not be called from anywhere except from inside `forward`.

        Returns:
            List[torch.Tensor]
                proposals_per_image: List of B (`batch_size`) tensors giving RPN
                proposals per image. These are boxes in XYXY format, that are
                most likely to contain *any* object. Each tensor in the list has
                shape `(N, 4)` where N could be variable for each image (maximum
                value `post_nms_topk`). These will be anchors for second stage.
        """

        # Gather RPN proposals *from all FPN levels* per image. This will be a
        # list of B (batch_size) tensors giving `(N, 4)` proposal boxes in XYXY
        # format (maximum value of N should be `post_nms_topk`).
        proposals_per_image = []

        # Get batch size to iterate over:
        batch_size = pred_obj_logits["p3"].shape[0]

        for _batch_idx in range(batch_size):

            # For each image in batch, iterate over FPN levels. Fill proposals
            # AND scores per image, per FPN level, in these:
            proposals_per_fpn_level_per_image = {
                level_name: None for level_name in anchors_per_fpn_level.keys()
            }
            scores_per_fpn_level_per_image = {
                level_name: None for level_name in anchors_per_fpn_level.keys()
            }

            for level_name in anchors_per_fpn_level.keys():

                # Get anchor boxes and predictions from a single level.
                level_anchors = anchors_per_fpn_level[level_name]

                # Predictions for a single image - shape: (HWA, ), (HWA, 4)
                level_obj_logits = pred_obj_logits[level_name][_batch_idx]
                level_boxreg_deltas = pred_boxreg_deltas[level_name][_batch_idx]

                #   1. Transform the anchors to proposal boxes using predicted
                #      box deltas, clamp to image height and width.
                #   2. Sort all proposals by their predicted objectness, and
                #      retain `self.pre_nms_topk` proposals. This speeds up
                #      our NMS computation. HINT: `torch.topk`
                #   3. Apply NMS and add the filtered proposals and scores
                #      (logits, with or without sigmoid, doesn't matter) to
                #      the dicts above (`level_proposals_per_image` and
                #      `level_scores_per_image`).

                (width, height) = image_size
                level_boxreg_deltas[:, 2] = torch.clamp(level_boxreg_deltas[:, 2], max=height)
                level_boxreg_deltas[:, 3] = torch.clamp(level_boxreg_deltas[:, 3], max=width)
                proposal_boxes = rcnn_apply_deltas_to_anchors(level_boxreg_deltas, level_anchors)
                proposal_boxes = proposal_boxes.to(pred_obj_logits["p3"].device)
                # (proposal_boxes_topk, indices) = torch.topk(proposal_boxes, self.pre_nms_topk)
                # level_obj_logits_topk = level_obj_logits[indices]

                descending_idx = torch.argsort(-level_obj_logits)
                # descending_idx = descending_idx.to(pred_obj_logits["p3"].device)
                descending_proposal_boxes = proposal_boxes[descending_idx]
                level_obj_logits_topk = level_obj_logits[descending_idx][:self.pre_nms_topk]
                proposal_boxes_topk = descending_proposal_boxes[:self.pre_nms_topk]

                keep = torchvision.ops.nms(proposal_boxes_topk, level_obj_logits_topk, self.nms_thresh)
                proposals_per_fpn_level_per_image[level_name] = proposal_boxes_topk[keep]
                scores_per_fpn_level_per_image[level_name] = level_obj_logits_topk[keep]

            proposals_all_levels_per_image = self._cat_across_fpn_levels(
                proposals_per_fpn_level_per_image, dim=0
            )
            scores_all_levels_per_image = self._cat_across_fpn_levels(
                scores_per_fpn_level_per_image, dim=0
            )
            # Sort scores from highest to smallest and filter proposals.
            _inds = scores_all_levels_per_image.argsort(descending=True)
            _inds = _inds[: self.post_nms_topk]
            keep_proposals = proposals_all_levels_per_image[_inds]

            proposals_per_image.append(keep_proposals)

        return proposals_per_image


    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)


class FasterRCNN(nn.Module):
    """
    Faster R-CNN detector: this module combines backbone, RPN, ROI predictors.

    Unlike Faster R-CNN, we will use class-agnostic box regression and Focal
    Loss for classification. We opted for this design choice for you to re-use
    a lot of concepts that you already implemented in FCOS - choosing one loss
    over other matters less overall.
    """

    def __init__(
        self,
        backbone: nn.Module,
        rpn: nn.Module,
        stem_channels: List[int],
        num_classes: int,
        batch_size_per_image: int,
        roi_size: Tuple[int, int] = (7, 7),
    ):
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.batch_size_per_image = batch_size_per_image

        # A stem of alternating 3x3 convolution layers and RELU
        # activation modules using `stem_channels` arguments.

        # This stem will be applied on RoI-aligned FPN features.

        cls_pred = []
        self.num_layers = len(stem_channels)
        # in_chan = roi_size[1]
        in_chan = self.backbone.out_channels
        for i in range(1, self.num_layers+1):
          out_chan = stem_channels[i-1]
          cls_pred.append(nn.Conv2d(in_chan, out_chan, 3, stride=1, padding=1))
          cls_pred.append(nn.ReLU())
          in_chan= out_chan

        cls_pred.append(nn.Flatten())
        Linear_layer = nn.Linear(stem_channels[-1]*roi_size[0]*roi_size[1],self.num_classes+1)
        Linear_layer.weight.data.normal_(0,0.01)
        Linear_layer.bias.data.fill_(0)
        cls_pred.append(Linear_layer)

        # Faster R-CNN also predicts box offsets to "refine" RPN proposals, we
        # exclude it for simplicity and keep RPN proposal boxes as final boxes.
        self.cls_pred = nn.Sequential(*cls_pred)
        self.cls_pred.apply(weights_init_normal)

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        See documentation of `FCOS.forward` for more details.
        """

        feats_per_fpn_level = self.backbone(images)
        output_dict = self.rpn(
            feats_per_fpn_level, self.backbone.fpn_strides, gt_boxes
        )
        
        # List of B (`batch_size`) tensors giving RPN proposals per image.
        proposals_per_image = output_dict["proposals"]

        # Assign the proposals to different FPN levels for extracting features
        # using RoI-align. During training we also mix GT boxes with proposals.
        proposals_per_fpn_level = reassign_proposals_to_fpn_levels(
            proposals_per_image,
            gt_boxes
            # gt_boxes will be None during inference
        )

        # Get batch size from FPN feats:
        num_images = feats_per_fpn_level["p3"].shape[0]

        # Perform RoI-align using FPN features and proposal boxes.
        roi_feats_per_fpn_level = {
            level_name: None for level_name in feats_per_fpn_level.keys()
        }
        # Get RPN proposals from all levels.
        for level_name in feats_per_fpn_level.keys():
            level_feats = feats_per_fpn_level[level_name]
            level_props = proposals_per_fpn_level[level_name]
            level_stride = self.backbone.fpn_strides[level_name]

            roi_feats = torchvision.ops.roi_align(level_feats,level_props,self.roi_size,1/level_stride,aligned=True)

            roi_feats_per_fpn_level[level_name] = roi_feats

        # Combine ROI feats across FPN levels, do the same with proposals.
        # shape: (batch_size * total_proposals, fpn_channels, roi_h, roi_w)
        roi_feats = self._cat_across_fpn_levels(roi_feats_per_fpn_level, dim=0)

        # Obtain classification logits for all ROI features.
        # shape: (batch_size * total_proposals, num_classes)
        pred_cls_logits = self.cls_pred(roi_feats)

        if not self.training:
            # fmt: off
            return self.inference(
                images,
                proposals_per_fpn_level,
                pred_cls_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )
            # fmt: on

        # Match the RPN proposals with provided GT boxes and append to
        # `matched_gt_boxes`. Use `rcnn_match_anchors_to_gt` with IoU threshold
        # such that IoU > 0.5 is foreground, otherwise background.
        # There are no neutral proposals in second-stage.
        matched_gt_boxes = []
        for _idx in range(len(gt_boxes)):
            # Get proposals per image from this dictionary of list of tensors.
            proposals_per_fpn_level_per_image = {
                level_name: prop[_idx]
                for level_name, prop in proposals_per_fpn_level.items()
            }

            proposals_per_image = self._cat_across_fpn_levels(
                proposals_per_fpn_level_per_image, dim=0
            )
            gt_boxes_per_image = gt_boxes[_idx]
            matched_gt_boxes.append(rcnn_match_anchors_to_gt(proposals_per_image,gt_boxes_per_image,(0.5,0.5)))

        # Combine predictions and GT from across all FPN levels.
        matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)

        # Steps to train the classifier head:
        #   1. Sample a few RPN proposals, like you sampled 50-50% anchor boxes
        #      to train RPN objectness classifier. However this time, sample
        #      such that ~25% RPN proposals are foreground, and the rest are
        #      background. Faster R-CNN performed such weighted sampling to
        #      deal with class imbalance, before Focal Loss was published.
        #
        #   2. Use these indices to get GT class labels from `matched_gt_boxes`
        #      and obtain the corresponding logits predicted by classifier.
        #
        #   3. Compute cross entropy loss - use `F.cross_entropy`, see its API
        #      documentation on PyTorch website. Since background ID = -1, you
        #      may shift class labels by +1 such that background ID = 0 and
        #      other VC classes have IDs (1-20). Reverse shift
        #      this during inference, so that model predicts VOC IDs (0-19).

        fg_idx, bg_idx = sample_rpn_training(matched_gt_boxes,num_images*self.batch_size_per_image,0.25)

        gt_labels = torch.cat((matched_gt_boxes[fg_idx][:,4],matched_gt_boxes[bg_idx][:,4]),dim=0)
        gt_labels = gt_labels+1

        pred_classes,indices = torch.max(pred_cls_logits,dim=1)
        pred_logits = torch.cat((pred_cls_logits[fg_idx],pred_cls_logits[bg_idx]),dim=0)

        loss_cls = F.cross_entropy(pred_logits, gt_labels.long())

        return {
            "loss_rpn_obj": output_dict["loss_rpn_obj"],
            "loss_rpn_box": output_dict["loss_rpn_box"],
            "loss_cls": loss_cls,
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        proposals: torch.Tensor,
        pred_cls_logits: torch.Tensor,
        test_score_thresh: float,
        test_nms_thresh: float,
    ):
        """
        Run inference on a single input image (batch size = 1). Other input
        arguments are same as those computed in `forward` method. This method
        should not be called from anywhere except from inside `forward`.

        Returns:
            Three tensors:
                - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
                  co-ordinates of predicted boxes.

                - pred_classes: Tensor of shape `(N, )` giving predicted class
                  labels for these boxes (one of `num_classes` labels). Make
                  sure there are no background predictions (-1).

                - pred_scores: Tensor of shape `(N, )` giving confidence scores
                  for predictions.
        """

        # The second stage inference in Faster R-CNN is quite straightforward:
        # combine proposals from all FPN levels and perform a *class-specific
        # NMS*. There would have been more steps here if we further refined
        # RPN proposals by predicting box regression deltas.

        # Use `[0]` to remove the batch dimension.
        proposals = {level_name: prop[0] for level_name, prop in proposals.items()}
        pred_boxes = self._cat_across_fpn_levels(proposals, dim=0)

        ######################################################################
        # Faster R-CNN inference, perform the following steps in order:
        #   1. Get the most confident predicted class and score for every box.
        #      Note that the "score" of any class (including background) is its
        #      probability after applying C+1 softmax.
        #
        #   2. Only retain prediction that have a confidence score higher than
        #      provided threshold in arguments.
        #
        # `pred_classes` may contain background as ID = 0 (based on how
        # the classifier was supervised in `forward`)

        pred_scores, pred_classes = None, None

        pred_cls_logits_softmax = F.softmax(pred_cls_logits)
        pred_scores,pred_classes = torch.max(pred_cls_logits_softmax,dim=1)

        pred_classes = pred_classes-1

        keep = class_spec_nms(pred_boxes, pred_scores, pred_classes, iou_threshold=test_nms_thresh)
        pred_boxes = pred_boxes[keep]
        pred_classes = pred_classes[keep]
        pred_scores = pred_scores[keep]
        return pred_boxes, pred_classes, pred_scores

