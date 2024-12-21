## 1. Batch Normalization (BN)

Batch Normalization 的目标是使每一层的输入分布稳定。在序列任务中，BN 会对每个时间步的位置进行标准化，考虑批次中的所有样本。

### 计算过程：
假设输入数据为 $ X \in \mathbb{R}^{m \times L \times d} $，其中：

- $ m $ 是批次大小，
- $ L $ 是序列长度，
- $ d $ 是每个序列单元的特征维度。

1. **计算均值和方差：**

   对每个时间步 $ i $ 和特征维度 $ j $ 计算批次中的均值和方差：
   $$
   \mu_{i,j} = \frac{1}{m} \sum_{k=1}^{m} X_{k,i,j}
   $$
   $$
   \sigma_{i,j}^2 = \frac{1}{m} \sum_{k=1}^{m} (X_{k,i,j} - \mu_{i,j})^2
   $$

2. **标准化：**

   对每个样本的每个特征进行标准化：
   $$
   \hat{X}_{k,i,j} = \frac{X_{k,i,j} - \mu_{i,j}}{\sqrt{\sigma_{i,j}^2 + \epsilon}}
   $$

3. **线性变换：**

   最后进行线性变换：
   $$
   Y_{k,i,j} = \gamma_j \hat{X}_{k,i,j} + \beta_j
   $$

---

## 2. Layer Normalization (LN)

Layer Normalization 是对每个样本的所有特征进行标准化，而不是依赖于批次大小。它适用于序列数据中的每个时间步。

### 计算过程：
假设输入数据为 $ X \in \mathbb{R}^{m \times L \times d} $，其中：

- $ m $ 是批次大小，
- $ L $ 是序列长度，
- $ d $ 是每个序列单元的特征维度。

1. **计算均值和方差：**

   对每个样本 $ x_k $ 计算均值和方差：
   $$
   \mu_k = \frac{1}{d} \sum_{j=1}^{d} x_{k,j}
   $$
   $$
   \sigma_k^2 = \frac{1}{d} \sum_{j=1}^{d} (x_{k,j} - \mu_k)^2
   $$

2. **标准化：**

   对样本 $ x_k $ 中的每个特征进行标准化：
   $$
   \hat{x}_{k,j} = \frac{x_{k,j} - \mu_k}{\sqrt{\sigma_k^2 + \epsilon}}
   $$

3. **线性变换：**

   最后进行线性变换：
   $$
   y_{k,j} = \gamma_j \hat{x}_{k,j} + \beta_j
   $$

---

## 3. RMSNorm (Root Mean Square Normalization)

RMSNorm 是 Layer Normalization 的变体，使用均方根（RMS）代替均值和方差进行标准化，简化了计算过程。

### 计算过程：
假设输入数据为 $ X \in \mathbb{R}^{m \times L \times d} $，其中：

- $ m $ 是批次大小，
- $ L $ 是序列长度，
- $ d/j $ 是每个序列单元的特征维度。

1. **计算均方根（RMS）：**

   对每个样本 $ x_k $ 计算均方根（RMS）：
   $$
   \text{RMS}(x_k) = \sqrt{\frac{1}{d} \sum_{j=1}^{d} x_{k,j}^2}
   $$

2. **标准化：**

   对样本 $ x_k $ 中的每个特征进行标准化：
   $$
   \hat{x}_{k,j} = \frac{x_{k,j}}{\text{RMS}(x_k) + \epsilon}
   $$

3. **线性变换：**

   最后进行线性变换：
   $$
   y_{k,j} = \gamma_j \hat{x}_{k,j} + \beta_j
   $$

---

## 比较总结

- **Batch Normalization (BN)**：对批次中的每个时间步位置进行标准化，依赖批次的均值和方差。
- **Layer Normalization (LN)**：对每个样本的所有特征进行标准化，不依赖批次大小，适合序列数据。
- **RMSNorm**：与 LN 相似，但使用均方根（RMS）代替均值和方差进行标准化，计算更简洁。
