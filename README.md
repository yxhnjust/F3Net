# F3-Net
# Abstract
The code of the paper [F3-Net: Feature Fusion and Filtration Network for Object Detection in Optical Remote Sensing Images](https://www.mdpi.com/2072-4292/12/24/4027).
The code will be updated on March. The method is based on Faster R-CNN and is completed by Xinhai Ye.
# Environment
1. python 3.5
2. cuda 10.0
3. opencv
4. tensorflow-gpu 1.13
5. tfplot 0.2.0
# Hardware
At least two NVIDIA GPUs with more than 10GB memory.
# Installation
cd $PATH_ROOT/libs/box_utils/cython_utils

python setup.py build_ext --inplace (or make)

cd $PATH_ROOT/libs/box_utils/

python setup.py build_ext --inplace
