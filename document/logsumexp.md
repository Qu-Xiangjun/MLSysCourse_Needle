# 稳定的 LogSumExp 与 Sigmoid

### 什么是LSE

$$
\operatorname{LSE}(x_1,\cdots,x_n)=\log\sum_{i=1}^n\exp(x_i)=\log\left(\exp(x_1)+\cdots+\exp(x_n)\right)
$$

输入为n维向量，输出一个标量，是所有参数指数之和的对数。

### Stable LSE
> 编程语言中数值有表示范围，数值过大上溢出，数值过小下溢出

由于exp(x)取值范围是(0,+∞)，那么对应的指数函数会上下溢出.

那么如何解决这个问题呢？一个常见做法是指数归一化技巧，首先得到x中之中最大的值$b=\max_{i=1}^{n}x_{i}$。如此LSE函数会变成：
$$
\begin{equation}
\text{LogSumExp}(x) = \log (\sum_{i} \exp (x_i - \max{x})) + \max{x}
\end{equation}
$$

然后将$x_i-b$替换$x_i$，如此$x_i-b$的取值范围就是(-∞, 0], $exp(x_i-b)$的范围就成为了(0, 1],那么最终的LSE函数取值范围就是(b, b+1), 没有了上下溢出。

因此，这个就是稳定版了LSE

### Stable LSE in softmax
LSE 一般用在哪呢，比如计算概率输出或者loss 函数需要的Softmax函数：
$$
\text{Softmax}(x_i)=\frac{\exp(x_i)}{\sum_{j=1}^n\exp(x_j)}
$$
可以看到softmax存在上下溢出问题，如果xi过大，则分子分母都上溢出得到`nan`结果；如果xi无穷小，则下溢出为0，但分母不能为0。因此使用stable LSE解决
$$
Softmax(x_i - b)=\frac{\exp(x_i-b)}{\sum_{j=1}^n\exp(x_j-b)}
$$

### Stable LSE Gradient
对LogSumExp求导就得到了exp-normalize(Softmax)的形式，
$$
\frac{\partial\left(b+\log\sum_{j=1}^n\exp(x_j-b)\right)}{\partial x_j}=\frac{\exp(x_i-b)}{\sum_{j=1}^n\exp(x_j-b)}
$$


### Stable Sigmoid
首先是Sigmoid 的公式：
$$
\sigma(x)=\frac1{1+\exp(-x)}
$$

而Sigmoid可以变换为：
$$
\frac1{1+\exp(-x)}=\frac{\exp(x)}{1+\exp(x)}
$$
- 当x>=0时用左式子，避免分母无穷大Sigmoid下溢出；
- 当x<0的时用右式子，避免分子分母无穷大上溢出。