### Model Description

#### 2. text-to-image generation

1. Model
   ![model](./model.png)
   * add batch norm$(0.9,0.999)$ and leaky relu$(0.2)$ between every transform without output layer and input layer.
     * generator output's acivation is $tanh$.
   * all convolution layer and deconvolution layer are setting to $strides=(2,2) , filter\,\,size=(5,5)$ except the last conv in the generator.
     * the last conv in generator is conv2d([1,1],[1,1]). 
       * I think that it's similar to ensemble.
   * We try to train in large scale image, and hope it will be clear than transform to $64 \times 64$
2. Training Method and details
   * Preprocessing: image normalize to $[-1,1]$.
   * Batch size: 32
   * Loss Function:
     * DRAGAN's constraint $\lambda = 8$
     * Spherical interpolation's $\alpha = (0,1)$

     ​

### Experiment settings and observation

2. text-to-image generation
|Itrs|pic|
|-|-|
|1600|![report_1](report.png)|
|3200|![report_1](report_1.png)|
|20000|![report_2](report_2.png)|

* 從一開始的1600 Iterations，我們得知GAN已經學習好眼睛和頭髮的位置。
* 而train到後面，20000 iterations時，與之前的圖片相比，頭髮分的較為清楚，並且有光澤。
* Training Details:
  - 0-10000 iterations: 
    - learning rate = $10^{-4}$
    - discriminator : generator =  4 : 1
  - 10000-20000 iterations: 
    - learning rate = $10^{-5}$ 
    - discriminator : generator =  4 : 1
  - Ensemble 4 discriminator.
* In inference, in order to decay the probability of bad picture, we generate 512 picture per one target picture, and pick the one that is the highest score judging by our discriminator.