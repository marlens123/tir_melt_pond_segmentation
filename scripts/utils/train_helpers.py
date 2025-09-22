from sklearn.utils import class_weight
import numpy as np
import torch
import random
import torch.nn.functional as F
from models.AutoSAM.loss_functions.dice_loss import soft_dice_per_batch_2

def compute_class_weights(train_masks):

    if isinstance(train_masks, str):
        train_masks = np.load(train_masks)

    masks_resh = train_masks.reshape(-1, 1)
    masks_resh_list = masks_resh.flatten().tolist()
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(masks_resh), y=masks_resh_list
    )
    return class_weights

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

    def focal_loss_multiclass(self, inputs, targets, num_classes):
        """ Focal loss for multi-class classification. """

        # this is only true for our case
        assert num_classes == 3

        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        assert inputs.shape[-1] == num_classes
        probs = F.softmax(inputs, dim=-1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=-1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            # we need to handle that alpha has shape (C,) and ce_loss has shape (B, H, W, C)
            alpha_t = self.alpha[targets]

            ce_loss = alpha_t.unsqueeze(-1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(-1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def forward(self, inputs, targets):

        num_classes = inputs.shape[1]

        # we need to make sure to have inputs of shape (B, H, W, C)
        if inputs.shape[-1] != num_classes:
            inputs = inputs.permute(0, 2, 3, 1)

        return self.focal_loss_multiclass(inputs, targets, num_classes=num_classes)