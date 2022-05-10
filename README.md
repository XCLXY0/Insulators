## Introduction

### 基于关键点关联域的绝缘子定向检测方法——推理代码

## Network Architecture

![pipeline](https://github.com/XCLXY0/Insulator/blob/master/pipeline.png)

## Results
<p align="center">
<img src="https://github.com/XCLXY0/Insulator/blob/master/result/000020.jpg", width="720">
</p>
<p align="center">
<img src="https://github.com/XCLXY0/Insulator/blob/master/result/000252.jpg", width="720">
</p>
<p align="center">
<img src="https://github.com/XCLXY0/Insulator/blob/master/result/001246.jpg", width="720">
</p>

###### 更多测试结果见下方百度网盘链接

## Require
- 第三方Python包详见[requirement.txt](https://github.com/XCLXY0/Insulator/blob/master/requirement.txt)

## Development Environment
- 使用pytorch 3.6 在Ubuntu 18.04系统上运行，显存8G

## Inference
### step 1: 安装 [requirement.txt](https://github.com/XCLXY0/Insulator/blob/master/requirement.txt) 中的Python包

### step 2: 下载权重 `model/Ours/paf_800X800_6000_80_14_8_SGD_0.1.pth` 到根目录

  - 模型权重及测试结果下载链接：[af9p](https://pan.baidu.com/s/1coFL9CIx0wu7twu5fD9gog)

### step 3: 运行以下代码测试图片
  `python inference.py --image [image_path]`
- 如：
  `python inference.py --image ./picture/000033.JPG`
- 测试结果：

![test](https://github.com/XCLXY0/Insulator/blob/master/test_result.png)

![000033](https://github.com/XCLXY0/Insulator/blob/master/result/000033.jpg)

__注意： 本模型测试图片像素约为`5400px X 3600px`__

## Results
| model | AP | AP<sup>50</sup> | AP<sup>75</sup> |
| :---------: | :---------: |:---------: |:---------: |
|[RetinaNet](https://arxiv.org/abs/1708.02002)   | 0.605 |0.766 |0.625 |
|[CenterNet](https://arxiv.org/abs/1904.07850)   | 0.644 |0.875 |0.676 |
|[Centripetalnet](https://arxiv.org/abs/2003.09119)   | 0.651 |0.883 |0.684 |
|[R<sup>3</sup>Det](https://arxiv.org/abs/1908.05612)   | 0.723 |0.897 |0.731 |
|Ours   | 0.757 |0.895 |0.789 |

- 注意：RetinaNet, CenterNet, CentripetalNet, R3Det均在[mmdetection](https://github.com/open-mmlab/mmdetection)上复现成功, 若要评估、测试这些模型需在[mmdetection](https://github.com/open-mmlab/mmdetection)上运行
- 此外，RetinaNet, CenterNet, CentripetalNet, R3Det等模型权重及结果见：[af9p](https://pan.baidu.com/s/1coFL9CIx0wu7twu5fD9gog)
