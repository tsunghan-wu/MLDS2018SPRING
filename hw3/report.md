## MLDS Homework 3 Report
<p align="right">b05902127 劉俊緯 b05902013 吳宗翰</p>

### 3-3 Style Transfer

#### 1. My result

|   Input Domain — Horse    |   Output Domain — Zabra    |
| :-----------------------: | :------------------------: |
| ![](./imgs/3_3_input.jpg) | ![](./imgs/3_3_output.jpg) |

#### 2. Analysis

##### Model

- 使用Cycle GAN，同時train了zebra2horse以及horse2zebra，兩個都是LSGAN (Least Square GAN)
- 

##### Observation

##### Reference

在Homework 3-3 Style Transfer中，我們是拿Github上面的原始碼，自己只是做Inference，因此在此我們也附上model的來源：

https://github.com/vanhuyz/CycleGAN-TensorFlow