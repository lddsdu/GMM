### EM算法求解高斯混合模型

![a](data/origin.jpg)

![b](data/s-00.jpg)

![c](data/s-01.jpg)

![d](data/s-02.jpg)

![e](data/s-03.jpg)

求解过程非常迅速



TODO：使用梯度下降法来求解混合高斯模型。

1. 完全的梯度下降。
2. 采用梯度下降来最优化EM算法中的M-step的目标函数。



现在做了第二种的，直接采用EM算法中的M-step的目标函数作为损失函数进行优化，其中的优化过程中的如下面动图所示。

![sgd](data/gd.gif)

相比于EM算法，这里不是求取解析解，而是求取数值解。

