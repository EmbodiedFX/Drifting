# Generative Modeling via Drifting (玩具示例)

一个基于 NumPy 的 **Drifting 生成建模** 极简玩具示例，所有代码都在一个 Jupyter Notebook：`toy.ipynb` 中，用来帮助理解 Drifting 的训练过程和生成行为，而不是追求任何 SOTA 指标或完整工程实现。

English version: [README.md](./README-EN.md)

---

## 实验设定

- **真实数据分布**：  
  - 在 4 维空间构造一个双峰高斯混合；  
  - 两个 mode 的均值大致在 (1, 0, 0, 0) 和 (0, 1, 0, 0) 附近，协方差为各向同性的 $\sigma^2 I$；  
  - 在 notebook 运行开始时生成一批固定的数据点，在训练过程中不会改变。

- **噪声与生成器**：  
  - 噪声来自标准正态分布 $\varepsilon \sim \mathcal{N}(0, I)$，维度与数据维度相近；  
  - 生成器是一个两层隐藏层的 MLP（使用 `tanh` 激活），用小高斯初始化权重、偏置为零；  
  - 生成时只需一次前向：`x = f_theta(eps)`。

- **Drifting 场（漂移场）**：  
  - 正样本：固定的真实数据点；  
  - 负样本：当前 batch 中生成出来的样本；  
  - 使用 RBF 核 $k(x, y) = \exp(-\|x - y\|^2 / \tau)$ 做 mean-shift，定义  
    $V(x) = m_{\text{pos}}(x) - m_{\text{neg}}(x)$；  
  - 每一轮训练中，先将生成样本沿着 $V(x)$ 漂移到 $x_{\text{drift}} = x + V(x)$，再用均方误差让生成器拟合这些漂移后的目标。

---

## 预期结果

在默认超参数下，从头训练时，你大致可以看到：

- 早期迭代中，生成样本散落在数据分布附近的某个随机区域，与两团真实数据相差较大；  
- 随着迭代进行，Drifting 场会把生成样本逐步推向真实数据的两个 mode 附近，生成分布开始呈现“两团”的结构；  
- 正样本 mean-shift $m_{\text{pos}}(x)$ 负责把点云往真实数据的高密度区域拉；  
- 负样本 mean-shift $m_{\text{neg}}(x)$ 在生成分布内部提供“挤开”的信号，有助于缓解所有样本坍塌到单个 mode 上的模式坍塌问题。

---

## 运行步骤

1. **克隆仓库**

   ```bash
   git clone https://github.com/EmbodiedFX/Drifting.git
   cd Drifting
   ```

2. **安装依赖（如尚未安装）**

    建议使用 Python 虚拟环境，并安装至少以下包：

    ```bash
    pip install numpy matplotlib jupyter
    ```

3. 启动 Jupyter 并打开 notebook

    ```bash
    jupyter notebook
    ```
    在浏览器中打开 `toy.ipynb`，从上到下依次运行所有 cell。
    
    
运行完成后，你可以在 notebook 中看到训练过程中的数值输出和可视化图像，从而对 Drifting 这一生成建模范式有一个直观的理解。

---

## 更多可能性

`toy.ipynb` 里刻意把几个关键组件写得比较独立，方便你替换和做对比实验：

- **数据分布**：  
  生成真实数据的那一段代码可以直接改成：
  - 不同的 mode 位置（例如环形、更多个 mode）；  
  - 不同的协方差（各向异性、高维更稀疏等）；  
  - 不同的数据维度。  

- **生成器结构与超参数**：  
  你可以修改：
  - 噪声维度 / 数据维度；  
  - 隐藏层维度、隐藏层层数、激活函数（例如换成 ReLU）；  
  - 学习率、训练轮数等优化超参数。  

- **Drifting 相关参数**：  
  - kernel 的温度 $\tau$（更 local 或更 global）；  
  - `drift_scale`（每轮走大步还是小步）；  
  - `batch_size`，影响负样本 mean-shift 的统计稳定性。  

- **日志与可视化**：  
  - 是否打印详细的中间结果（单步样本 / mean-shift / loss 等）；  
  - 是否将日志重定向到文件；  
  - 如何画散点图与指标曲线。  

这些改动可以帮助你系统性地观察：不同设置下，Drifting 是如何缓解或加剧模式坍塌的，以及生成分布如何在几何上发生变化。
