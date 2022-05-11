## Introduction
Status: Archive (code is provided as-is, no updates expected)
### Inference code
Code for reproducing results in the paper __Detecting insulator strings as linked chain framework in smart grid inspection__ by Ning Wei, Xiangyang Li, Jiaqi Jin, Peng Chen, Shuifa Sun.

## Network Architecture
![pipeline](https://github.com/XCLXY0/Insulators/blob/master/pipeline.png)

## Results
<p align="center">
<img src="https://github.com/XCLXY0/Insulators/blob/master/result/000020.jpg", width="720">
</p>
<p align="center">
<img src="https://github.com/XCLXY0/Insulators/blob/master/result/000252.jpg", width="720">
</p>
<p align="center">
<img src="https://github.com/XCLXY0/Insulators/blob/master/result/001246.jpg", width="720">
</p>

###### For more test results, see the link below.

## Require
Please `pip install` the following packages:
-Cython
-torch>=1.5
-torchvision>=0.6.1
-progress
-matplotlib
-scipy
-numpy
-opencv

## Development Environment

Running on Ubuntu 18.04 system with pytorch 3.6, 8G VRAM.

## Inference
### step 1: Install Python packages in [requirement.txt](https://github.com/XCLXY0/Insulators/blob/master/requirement.txt) .

### step 2: Download the weight `model/Ours/paf_800X800_6000_80_14_8_SGD_0.1.pth` to the root directory.

  -Model weights and test results download link：[af9p](https://pan.baidu.com/s/1coFL9CIx0wu7twu5fD9gog).

### step 3: Run the following code to test the image.
  `python inference.py --image [image_path]`
- for example:
  `python inference.py --image ./picture/000033.JPG`
- Test results：

![test](https://github.com/XCLXY0/Insulators/blob/master/test_result.png)

![000033](https://github.com/XCLXY0/Insulators/blob/master/result/000033.jpg)

__Note: The pixels of the test image of this model are approximately `5400 px X 3600 px`__.

## Results
| model | AP | AP<sup>50</sup> | AP<sup>75</sup> |
| :---------: | :---------: |:---------: |:---------: |
|[RetinaNet](https://arxiv.org/abs/1708.02002)   | 0.605 |0.766 |0.625 |
|[CenterNet](https://arxiv.org/abs/1904.07850)   | 0.644 |0.875 |0.676 |
|[Centripetalnet](https://arxiv.org/abs/2003.09119)   | 0.651 |0.883 |0.684 |
|[R<sup>3</sup>Det](https://arxiv.org/abs/1908.05612)   | 0.723 |0.897 |0.731 |
|Ours   | 0.757 |0.895 |0.789 |

- Note：`RetinaNet, CenterNet, CentripetalNet, R<sup>3</sup>Det` is successfully debugged on [mmdetection](https://github.com/open-mmlab/mmdetection), to evaluate and test these models need to be on [mmdetection](https://github.com/open-mmlab/mmdetection).
- Moreover，`RetinaNet, CenterNet, CentripetalNet, R<sup>3</sup>Det` model weights and results see：[af9p](https://pan.baidu.com/s/1coFL9CIx0wu7twu5fD9gog)
