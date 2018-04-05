## MLDS Homework 1 Report
<p align="right">b05902127 劉俊緯 b05902013 吳宗翰</p>
<!--Homework 1-1-->
### 1. Deep and Shallow

#### 1-1 Simulate a function

##### 實驗設計
1. Task: 試著讓deep和shallow的network去fit一個函數
	- task1：$f(x) = sin(x) + cos(x^2)$ 
	- task2：碎形函數(Weierstrass function)
2. Optimizer: Adam, learning rate: $0.01$, loss function: square loss
3. batch: $10^4$, epoch: $2 \times 10^5$
4. 網路架構：以下三個model分別為10351, 10367, 10351個參數

- Model 1 (DNN 1 Layer)
	- $Input(1) \rightarrow 3450 \rightarrow Output(1)$

- Model 2 (DNN 3 Layer)
	- $Input(1) \rightarrow 67 \rightarrow 70 \rightarrow 76 \rightarrow Output(1)$

- Model 3 (DNN 5 Layer)
	- $Input(1) \rightarrow 50 \rightarrow 50 \rightarrow 50 \rightarrow 50 \rightarrow 50 \rightarrow Output(1)$

##### 實驗結果 

###### A. **$f(x) = sin(x) + cos(x^2)$**

1. Training Loss
<img src="./1-1/function/ein2.png" align="center" style="zoom:40%" />
2. Predicted x Ground Truth Function Curve

巨觀             |  局部放大
:-------------------------:|:-------------------------:
![](./1-1/function/result2.png)  |  ![](./1-1/function/result_zoom2.png)

###### B. **$f(x) = \sum_{n=0}^{10} (\frac{1}{2})^n cos(2^n \pi x)$**

1. Training Loss
<img src="./1-1/function/ein.png" align="center" style="zoom:40%" />
2. Predicted x Ground Truth Function Curve

巨觀            |  局部放大
:-------------------------:|:-------------------------:
![](./1-1/function/result.png)  |  ![](./1-1/function/result_zoom.png)


##### 綜合討論

1. 實驗結果的GIF演進圖在 https://imgur.com/gallery/MdENL ，可以看到shallow是沿著x軸一塊一塊的去fit在收斂，deep則在前幾個epoch就在整個x軸上掌握了大部分的特徵，接下來的epoch只是在細部的地方微調而已，估計是模組化的現象。
2. 在function curve大圖上可以看到deep真的好在他產生比較多的pieces，符合上課提到的理論。
3. 從training loss的圖中可以看到越深的網路架構在收斂的過程中震盪比較大，這可能是因為在反向傳播的時候deep前面的參數改了一下會造成比較大的影響
4. 有點可惜的是照理說shallow一開始loss會掉得比較快，但是在這次的圖中不是很明顯

#### 1-2 An actual task

##### 實驗設計
1. Task: MNIST and CIFAR-10
2. Optimizer: Adam, learning rate: $10^{-4}$, loss function: Cross Entropy
3. batch: $100$, epoch: $10^5$
4. 網路架構：

###### A. MNIST：三個Model分別為：50535, 50551, 50598個參數

- Model 1 (CNN * 1 + Dense * 1)
	- $Input(28 \times 28) \rightarrow Conv (kernel=14 \times 14, filter=5) \rightarrow pool(2 \times 2)$
		$\rightarrow Flatten(14 \times 14 \times 5) \rightarrow 50 \rightarrow Output(10)$

- Model 2 (CNN * 1 + Dense * 2)
	- $Input(28 \times 28) \rightarrow Conv (kernel=14 \times 14, filter=5) \rightarrow pool(2 \times 2)$
		$\rightarrow Flatten(14 \times 14 \times 5) \rightarrow 48 \rightarrow 42 \rightarrow Output(10)$

- Model 3 (CNN * 2 + Dense * 2)
	- $Input(28 \times 28) \rightarrow Conv (kernel=5 \times 5, filter=32) \rightarrow pool(2 \times 2)$
		$\rightarrow Conv (kernel=3 \times 3, filter=20) \rightarrow pool(2 \times 2)$
		$\rightarrow Flatten(7 \times 7 \times 20) \rightarrow 48 \rightarrow 42 \rightarrow Output(10)$

###### B. CIFAR-10：32938, 32946, 32946個參數

- Model 1 (CNN * 2 + Dense * 1)
	- $Input(32 \times 32 \times 3) \rightarrow Conv (kernel=3 \times 3, filter=120) \rightarrow pool(2 \times 2)$
		$\rightarrow Conv (kernel=3 \times 3, filter=8) \rightarrow pool(2 \times 2)$
		$\rightarrow Flatten(8 \times 8 \times 8) \rightarrow 40 \rightarrow Output(10)$

- Model 2 (CNN * 2 + Dense * 3)
	- $Input(32 \times 32 \times 3) \rightarrow Conv (kernel=3 \times 3, filter=120) \rightarrow pool(2 \times 2)$
		$\rightarrow Conv (kernel=3 \times 3, filter=8) \rightarrow pool(2 \times 2)$
		$\rightarrow Flatten(8 \times 8 \times 8) \rightarrow 32 \rightarrow 64 \rightarrow 32 \rightarrow Output(10)$

- Model 3 (CNN * 3 + Dense * 2)
	- $Input(32 \times 32 \times 3) \rightarrow Conv (kernel=3 \times 3, filter=32) \rightarrow pool(2 \times 2)$
		$\rightarrow Conv (kernel=3 \times 3, filter=28) \rightarrow pool(2 \times 2)$
		$\rightarrow Conv (kernel=3 \times 3, filter=28) \rightarrow pool(2 \times 2)$
		$\rightarrow Flatten(4 \times 4 \times 28) \rightarrow 30 \rightarrow 30 \rightarrow Output(10)$

##### 實驗結果

###### A. MNIST

Training Accuracy            |  Training Loss
:-------------------------:|:-------------------------:
<img src="./1-1/actual_task/mnist_acc.png" align="center" style="zoom:30%" />|  <img src="./1-1/actual_task/mnist_err.png" align="center" style="zoom:30%" />

###### B. CIFAR-10

Training Accuracy            |  Training Loss
:-------------------------:|:-------------------------:
<img src="./1-1/actual_task/CIFAR_acc.png" align="center" style="zoom:18%" />|  <img src="./1-1/actual_task/CIFAR_err.png" align="center" style="zoom:18%" />

##### 綜合討論
1. 由於loss decrease的震盪太大，因此我們選擇數個epoch做一次平均，如此一來能比較清楚的看到整體的走向
2. 在接近參數的狀況下(綠色和橘色的比較中)，多一層CNN, Max_pooling比起多一層DNN的還來得好
3. 在接近參數的狀況下(藍色和橘色的比較中)，deep的確實比shallow的還來得好，符合我們上課所提到的理論

<!--Homework 1-2-->
### 2. Optimization

#### 2-1 Visualize the optimization process

##### 實驗設計
1. Task: MNIST, 每3個epoch就記錄一次所有的參數
2. 網路架構：DNN
	- $Input(28 \times 28) \rightarrow 20 \rightarrow 20 \rightarrow 20 \rightarrow Output(10)$
3. Dimension Reduction method
	- 使用sklearn套件的PCA直接把高維度降成二維，並沒有加上其他的參數

##### 實驗結果

1. Whole Model

巨觀           |  局部放大
:-------------------------:|:-------------------------:
<img src="./1-2/1-2-1whole.png" align="center" style="zoom:18%" />|  <img src="./1-2/1-2-1whole_zoom.png" align="center" style="zoom:18%" />

2. Only Layer-1

巨觀           |  局部放大
:-------------------------:|:-------------------------:
<img src="./1-2/1-2-1layer1.png" align="center" style="zoom:18%" />|  <img src="./1-2/1-2-1layer1_zoom.png" align="center" style="zoom:18%" />

##### 綜合討論

1. 由圖中train 8次的圖都很像可以推測我們可能是收到同一個local minimum
2. whole model和layer 1表現類似可能是因為這個neural network中layer 1佔總parameter量的80%以上，因此就dominate了

#### 2-2 Observe gradient norm during training

##### 實驗設計
1. Task: MNIST, 每個epoch都紀錄loss和gradient norm (使用DNN)

##### 實驗結果

Gradient norm           | Loss
:-------------------------:|:-------------------------:
<img src="./1-2/1-2-2norm.png" align="center" style="zoom:35%" />|  <img src="./1-2/1-2-2loss.png" align="center" style="zoom:35%" />


##### 綜合討論
1. 圖中看到最開始gradient norm很低只是因為初始化剛好在那裡而已
2. 雖然不是很明顯，不過整體而言gradient norm還是隨著loss下降而跟著在下降

#### 2-3 What happens when gradient is almost zero?

##### 實驗設計
1. Task: fit designed function $f(x) = sin(x) + cos(\frac{x^2}{10})$ (small DNN network)
2. Find gradient norm almost zero by changing objective function
	- 前$5000$個epoch: Minimize loss
	- 後$10000$個epoch: Monimize norm
3. Minimum ratio
	- 定義就如同課堂上所述的"proportion of eigenvalues greater than zero"
	- 使用tensorflow內建的`tf.hessians()`計算，並視他是diagonal的矩陣

##### 實驗結果

<img src="./1-2/1-2-3.png" align="center" style="zoom:20%" />

##### 綜合討論
1. 由上圖我們可以發現在$loss < 0.1$的狀況下，大部分的minimum ratio也都比0.5還來得高，因此我們覺得minimum ratio的這個理論還算是成功

#### 2-4 Visualize Error Surface (Bonus!)

##### 實驗設計
1. Task: fit designed function $f(x) = sin(x) + cos(\frac{x^2}{5})$ (使用DNN)

2. 觀察Error Surface的方法	(還要想怎麼寫比較好)

在5000個epoch後每100個epoch就在周圍隨機打50個點來計算loss，接著降維後用`matplotlib`的`plot_surface`畫出error surface

###### C. Dimension Reduction

先使用PCA降一點點維度再使用TSNE把它降到2 or 3維

##### 實驗結果

<img src="./1-2/1-2-4.png" align="center" style="zoom:40%" />

##### 綜合討論
1. 在這個例子中我們可以看到error surface相當的崎嶇，而我們做gradient descent的時候看起來也繞了不少路

<!--Homework 1-3-->
### 3. Generalization

#### 3-1 Can network fit random labels?

##### 實驗設計
1. Task: MNIST
1. Optimizer: Adam, learning rate: $10^{-4}$, loss function: Cross Entropy
2. batch: $100$, epoch: $3 \times 10^4$
3. 網路架構：DNN
	- $Input(28 \times 28) \rightarrow 700 \rightarrow 700 \rightarrow 700 \rightarrow Output(10)$

##### 實驗結果

Accuracy           |  Loss
:-------------------------:|:-------------------------:
<img src="./1-3/1-3-1acc.png" align="center" style="zoom:30%" />|  <img src="./1-3/1-3-1loss.png" align="center" style="zoom:30%" />


##### 綜合討論
1. MNIST的總維度是$28 \times 28 \times 255$，在實驗中我們所用的總參數比他還大(也就是VC dimension過大)，因此做出training accuracy=1.0是有可能的
2. 比起正常的label，發現random label需要更多的parameter才能去fit

#### 3-2 Number of parameters v.s. Generalization

##### 實驗設計
1. Task: train MNIST for 100 次
2. Optimizer: Adam, loss function: Cross Entropy
3. batch: $10^3$, epoch: $5 \times 10^3$
4. 每次training都使用DNN，不過不同的點如下表所示：	(待補)

參數 | uniform distrubution區間
----- | -----
learning rate 	| blah
	layer層數  	| $[10^{-5}, 10^{-3}]$
	每一層的neuron數量	 |	blah


##### 實驗結果

Accuracy           |  Loss
:-------------------------:|:-------------------------:
<img src="./1-3/1-3-2acc.png" align="center" style="zoom:30%" />|  <img src="./1-3/1-3-2loss.png" align="center" style="zoom:30%" />


##### 綜合討論
1. 從圖中可以看到隨著parameter的增加，不但training loss不斷地下降，連testing loss也跟著下降
2. 上面的結果一反我們傳統認為可能overfitting的狀況，雖然不知道為什麼會如此，不過總之就覺得deep learning實在是太神奇了

#### 3-3 Flatness v.s. Generalization

#### A. Two model interpolation

##### 實驗設計
1. Task: MNIST
2. Optimizer: Adam, learning rate: $10^{-4}$, loss function: Cross Entropy
3. ​batch: 分別為1024以及64, epoch: 40000
4. 網路結構：DNN
	- $Input(28 \times 28) \rightarrow 15 \rightarrow 15 \rightarrow Output(10)$

##### 實驗結果

<img src="./1-3/1-3-3-1.png" align="center" style="zoom:18%" />

##### 綜合討論
1. 有關於Cross Entropy的解釋：
	- 在$\alpha = 0.0, 1.0$兩個點的cross entropy是圖中的最小值，可以知道network是有學到東西的
	- 在$\alpha = [0, 1]$的區間中cross entropy都相對的低可以猜測兩個model都掉到比較flatness的地方
	- 在$\alpha < 0, \alpha > 1$的地方cross entropy都節節上升，可以猜測當初gradient descent時兩個model可能是從兩側滾下來的
2. 有關於Accuracy的解釋：
	- 圖中大部分的地方都呈現了Accuracy和Cross Entropy的趨勢相反，這相當的符合直覺

#### B. Sensitivity

##### 實驗設計
1. Task: MNIST
2. Optimizer: Adam, learning rate: $10^{-4}$, loss function: Cross Entropy
3. ​batch: 變數，我們測試了100到5000每???個做一個嘗試, epoch: 200000
4. 網路結構：DNN
	- $Input(28 \times 28) \rightarrow 10 \rightarrow 10 \rightarrow 10 \rightarrow Output(10)$

##### 實驗結果

<img src="./1-3/1-3-3-2-acc-sensitivity.png" align="center" style="zoom:35%" />

備註：藍色實線是train acc；藍色虛線是test acc；橫軸是batch size

##### 綜合討論
1. 如果根據Sensitivity的理論，結果應該是在Accuracy上升的過程中，Sensitivity先跟著增加(增加辨識能力)再逐步下降(不會動一下就造成predict改變)；然而我們做出的結果卻是在Accuracy增加的過程中Sensitivity一路掉下去
2. 根據理論大的batch size可以使Sensitivity上升，不過在這裡我們做出來的結果與理論相反

#### C. Sharpness (Bonus!)

##### 實驗設計
1. Task: MNIST (使用DNN)
2. Sharpness：做隨機打點並且loss平均視為sharpness

##### 實驗結果

Accuracy           |  Loss
:-------------------------:|:-------------------------:
<img src="./1-3/acc_sharpness.png" align="center" style="zoom:30%" />|  <img src="./1-3/loss_sharpness.png" align="center" style="zoom:30%" />

##### 綜合討論
1. 和上課提到的結論一樣，在batch越大的地方sharpness平均來說比較高
2. sharpness高的地方accuracy普遍低, loss普遍高也跟課堂結論一樣

##### Appendix: 分工表

- B05902013 吳宗翰
	- $\frac{1}{4}$ HW1-1 + $\frac{1}{2}$ HW1-2
	- All report
- B05902127 劉俊緯
	- $\frac{3}{4}$ HW1-1 + $\frac{1}{2}$ HW1-2 + All HW1-3
	- All bonus
