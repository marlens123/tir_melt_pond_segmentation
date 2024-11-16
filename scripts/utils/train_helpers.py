from sklearn.utils import class_weight
import numpy as np
import torch
import random

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