# Inspired by 2019 Pavel Iakubovskii https://github.com/qubvel/segmentation_models/

import numpy as np
import keras
from .preprocess_helpers import expand_greyscale_channels, get_training_augmentation, get_preprocessing, patch_extraction
from torch.utils.data import Dataset as BaseDataset
from models.smp.encoders import get_preprocessing_fn


class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        mode (str): Image mode ('train' or 'test')
    """

    CLASSES = ["melt_pond", "sea_ice", "ocean"]
    classes = ['melt_pond', 'sea_ice']

    def __init__(
        self,
        cfg_model,
        cfg_training,
        mode,
        args=None,
        preprocessing=get_preprocessing(),
        preprocessing_fn=get_preprocessing_fn(encoder_name="resnet34", pretrained="imagenet"),
        images = None,
        masks = None,
    ):
        self.mode = mode
        self.im_size = cfg_model["im_size"]

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in self.classes]

        if images is not None and masks is not None:
            images, masks = patch_extraction(images, masks, size=self.im_size)
            self.images_fps = images.tolist()
            self.masks_fps = masks.tolist()
        else:
            if self.mode == "train":
                X_train, y_train = patch_extraction(np.load(args.path_to_X_train), np.load(args.path_to_y_train), size=self.im_size)
                self.images_fps = X_train.tolist()
                self.masks_fps = y_train.tolist()
            elif self.mode == "test":
                X_test, y_test = patch_extraction(np.load(args.path_to_X_test), np.load(args.path_to_y_test), size=self.im_size)
                self.images_fps = X_test.tolist()
                self.masks_fps = y_test.tolist()
            else:
                print("Specified mode must be either 'train' or 'test'")

        self.normalize = cfg_training["z_score_normalize"]

        self.augmentation = cfg_training["augmentation"]
        self.augment_mode = cfg_training["augmentation_mode"]

        self.preprocessing = preprocessing
        self.preprocessing_fn = preprocessing_fn
        if "pretrain" in cfg_model:
            self.encoder_weights = cfg_model["pretrain"]
        else:
            self.encoder_weights = None

    def __getitem__(self, i):
        image = self.images_fps[i]
        # reshape to 3 dims in last channel
        image = expand_greyscale_channels(image)

        mask = self.masks_fps[i]
        mask = np.array(mask)

        mask = np.expand_dims(mask, axis=-1)
        
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        # apply normalization
        if self.normalize:
            # z-score normalization
            image = (image - image.mean()) / image.std()

        if self.mode == "train" and self.augmentation:
            augmentation = get_training_augmentation(
                im_size=self.im_size, augment_mode=self.augment_mode
            )
            sample = augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            if self.encoder_weights in {"imagenet", "rsd46-whu", "aid"}:
                print("Using imagenet preprocessing")
                image = self.preprocessing_fn(image)
            sample = self.preprocessing(image=image, mask=mask, pretraining=self.encoder_weights)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.images_fps)