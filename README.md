Best weights for inference:
- UnetPlusPlus with AID Pretrain: 'models/weights/UnetPlusPlus/all_plusplus_aid_imnetprepro_epoch_145.pth'

(remove later)
2) UnetPlusPlus with ImageNet Pretrain: 'models/weights/UnetPlusPlus/all_plusplus_imnet_imnetprepro_epoch_145.pth'

Best weights for validation:
- UnetPlusPlus with AID Pretrain: 'models/weights/UnetPlusPlus/aidimnet2809_UnetPlusPlus_epoch_139.pth'

(remove later)
2) UnetPlusPlus with ImageNet Pretrain: 'models/weights/UnetPlusPlus/plusplus_imnet_imnetprepro_epoch_145.pth'

GitHub TODO:
- test code and requirements (test aid)
- add Apache-2.0 license
- fill in README in models/
- make repo public
- additional README in pre-training checkpoints

## Overview
- Purpose of the repository
- What is the research trying to solve/prove/explore
- What data/dataset is used
- Article abstract
- add example image

This repository accompanies the paper submission "Deep Learning for the Segmentation of Melt Ponds from Infrared Images". The flight csv files used to generate the fraction reports for classified helicopter flights are stored in runs/ and further documented in a separate README. In "reproduce_results.sh" we report how our results were generated.

Acknowledge original authors of AutoSAM and Segmentation Models here.

## Getting Started
This code requires ```python>=3.10```, as well as ```pytorch>=1.7``` and ```torchvision>=0.8```. Install additional packages using ```pip install -r requirements.txt```. The data and weights are git lfs tracked. Please install git lfs.

To use AutoSAM: Segment Anything model checkpoints can be downloaded from SAM and should be placed in ```pretraining_checkpoints/```.

AID and RSD46-WHU pre-training weights can be downloaded from here (https://github.com/lsh1994/remote_sensing_pretrained_models) and store them in the respective folders in ```pretraining_checkpoints/```.

To reproduce the helicopter flight classification, the flight data can be loaded from ... into ```data/prediction/temperatures/```.

## How to Use
Requirements (e.g., CUDA version, num of GPU and memory)

- how to run finetuning
- how to run cross-validation
- how to run inference: Put datasets in specific folders
- how to recreate the final model weights

## Acknowledgments
This project includes code from the following sources:

- [AutoSAM](https://github.com/xhu248/AutoSAM), which is licensed under the Apache License 2.0
- [Segmentation Models](https://github.com/qubvel-org/segmentation_models.pytorch), which is licensed under the MIT license

AutoSAM itself relies on code from [SAM](https://github.com/facebookresearch/segment-anything/). We gratefully acknowledge all original authors for making their code available.


## Related Publications Data
[1] Kanzow, Thorsten (2023). The Expedition PS131 of the Research Vessel POLARSTERN to the Fram Strait in 2022. Ed. by Horst Bornemann and Susan Amir Sawadkuhi. Bremerhaven. DOI: 10.57738/BzPM_0770_2023.

[2] Reil, Marlena; Huntemann, Marcus; Spreen, Gunnar (2024): Helicopter-borne melt pond, sea ice, and ocean surface temperatures with surface type classifications during the Polarstern PS131 ATWAICE expedition in the Arctic during summer 2022 [dataset]. PANGAEA, https://doi.org/10.1594/PANGAEA.971908.


Contact: marlena1@gmx.de