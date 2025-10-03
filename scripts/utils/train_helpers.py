from sklearn.utils import class_weight
import numpy as np
import torch
import random
import torch.nn.functional as F
from models.AutoSAM.loss_functions.dice_loss import soft_dice_per_batch_2
import cv2

def compute_class_weights(train_masks):

    if isinstance(train_masks, str):
        train_masks = np.load(train_masks)

    masks_resh = train_masks.reshape(-1, 1)
    masks_resh_list = masks_resh.flatten().tolist()
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(masks_resh), y=masks_resh_list
    )
    return class_weights

def compute_pixel_distance_to_edge(train_masks, teta=3.0):
    # compute the pixel-wise distance to the edge of the object
    if isinstance(train_masks, str):
        train_masks = np.load(train_masks)

    all_distances = []
    for i in range(train_masks.shape[0]):
        mask = train_masks[i]
        edges_class_1 = cv2.Canny((mask == 1).astype(np.uint8) * 255, 100, 200)
        edges_class_2 = cv2.Canny((mask == 2).astype(np.uint8) * 255, 100, 200)
        edges_class_3 = cv2.Canny((mask == 3).astype(np.uint8) * 255, 100, 200)
        # combine all edges into one image
        edges = edges_class_1 | edges_class_2 | edges_class_3
        edges[edges > 0] = 1

        dist_transform = cv2.distanceTransform((1 - edges).astype(np.uint8), cv2.DIST_L2, 3)
        # exponentially decay the distance transform
        dist_transform = np.exp(-dist_transform / teta)
        # invert the distance transform
        dist_transform = 1 - dist_transform

        # normalize to 0-1
        dist_transform = dist_transform / np.max(dist_transform)
        all_distances.append(dist_transform)

    all_distances = np.array(all_distances)
    return all_distances


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print("Setting seed for GPU")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=[0.25, 0.25, 0.25], gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if self.alpha is not None and isinstance(self.alpha, (list, torch.Tensor)):
            if isinstance(self.alpha, list):
                self.alpha = torch.Tensor(self.alpha)

    def focal_loss_multiclass(self, inputs, targets, num_classes, distance_map=None):
        """ Focal loss for multi-class classification. """

        # this is only true for our case
        assert num_classes == 3

        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
        
        # convert distance map to tensor if not None
        if distance_map is not None and not isinstance(distance_map, torch.Tensor):
            distance_map = torch.tensor(distance_map, dtype=torch.float32).to(inputs.device)

        # Convert logits to probabilities with softmax
        assert inputs.shape[-1] == num_classes
        probs = F.softmax(inputs, dim=-1)
        log_probs = F.log_softmax(inputs, dim=-1)   # numerically stable log-softmax

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Compute cross-entropy for each class
        #ce_loss_old = -targets_one_hot * torch.log(probs)
        ce_loss = -targets_one_hot * log_probs

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=-1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            # we need to handle that alpha has shape (C,) and ce_loss has shape (B, H, W, C)
            alpha_t = self.alpha[targets]
            if distance_map is not None:
                print("Using label uncertainty with focal loss.")
                # Apply distance map to alpha
                alpha_t = alpha_t * distance_map

            ce_loss = alpha_t.unsqueeze(-1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(-1) * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def forward(self, inputs, targets, pixel_distances=None):

        num_classes = inputs.shape[1]

        # we need to make sure to have inputs of shape (B, H, W, C)
        if inputs.shape[-1] != num_classes:
            inputs = inputs.permute(0, 2, 3, 1)

        return self.focal_loss_multiclass(inputs, targets, num_classes=num_classes, distance_map=pixel_distances)