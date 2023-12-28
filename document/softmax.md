# Softmax 使用技巧

对多分类问题，最常见的是softmax损失函数，或者称之为交叉熵损失。

模型的原始输出是一种对“信心”的含糊表示，softmax损失函数将其作了一个变换，转换成一个更直观的概率分布形式：通过指数函数将模型的原始输出转变成正值，然后通过归一化让所有可能的结果之和为1，这样就满足了概率分布的要求。

引入一个向量 zi，其每个分量都看做是模型预测标签为i的概率，形式化地
$$
z_i=p(\mathrm{label}=i)=\frac{\exp(h_i(\boldsymbol{x}))}{\sum_{j=1}^k\exp(h_j(\boldsymbol{x}))}\Leftrightarrow\boldsymbol{z}\equiv\mathrm{normalize}(\exp(h(\boldsymbol{x})))
$$

这种情况下模型足够好的体现就在与正确标签y对应的i分类标签概率值足够大。但在loss函数的背景下“缩小损失”是更常用且自然的理解方式，但最小化概率值容易产生下溢出，则对概率值去对数，得到最终损失函数——负对数概率函数：
$$
\ell_\text{ce}(h(\boldsymbol{x}),y)=-\log p(\text{label}=y)=-h_y(\boldsymbol{x})+\log\sum_{j=1}^k\exp(h_j(\boldsymbol{x}))
$$

但是需要注意的是上式子的后一项是LogSumExp式子，可以使用stable LSE 替换防止LSE的上下溢出。