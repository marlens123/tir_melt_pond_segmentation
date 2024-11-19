## Overview

This directory stores `.csv` files of fraction values from classified helicopter flights.

Individual `.csv` files correspond to individual helicopter flights on the dates specified in the file name and in the first column of the files. Each row in a `.csv` file corresponds to an image of the corresponding flight. The coordinates of each image, the fraction values and the mean probability per image are reported.

The fractions are calculated as follows:

- `melt pond fraction (MPF) = MP / ( SI + MP )`
- `sea ice fraction (SIF) = OC / ( SI + MP + OC )`
- `ocean fraction (OCF) = ( MP + SI ) / ( SI + MP + OC )`

with MP = number of melt pond pixels within an image, SI = number of sea ice pixels within an image, and OC = number of ocean pixels within an image.

The mean probability per image corresponds to the mean model confidence for an image. The model confidence value for an individual pixel is obtained by applying a softmax operation on the logit output of the segmentation model. The overall mean probability for the image is computed by utilising the probability score of the assigned class for each pixel.

The files were generated with the `scripts/run_inference.py` script.
