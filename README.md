# QVAE: Boltzmann Machine Prior Variational Autoencoder for Protein Sequence Generation

## 简介

本项目实现了一种使用**玻尔兹曼机 (Boltzmann Machine)** 作为隐空间先验的变分自编码器，用于**蛋白质跨膜信号肽 (transmembrane signal peptide)** 的生成式设计。

传统 VAE 使用标准正态分布 `N(0,I)` 作为先验，本项目使用玻尔兹曼机建模隐空间分布，使其能够学习更复杂的多模态分布，从而生成更具多样性和功能性的蛋白质序列。

通过 **Kaiwu SDK** 将玻尔兹曼机转化为 Ising 模型，使用模拟退火优化器进行采样，并预留了对量子退火硬件的接口支持。

### 核心特点

- **DVAE++ 离散隐变量**: 使用重叠分布 (overlap distribution) + 隐式梯度技巧实现可微采样
- **双先验架构**: 支持受限玻尔兹曼机 (RBM) 和全连接玻尔兹曼机 (Full BM) 两种先验
- **双目标联合优化**: VAE 的 ELBO 损失 + BM 的 Contrastive Divergence 损失
- **完整分析 Pipeline**: 物化性质、二级结构、亚细胞定位、物种分类
- **量子计算接口**: Ising 模型格式输出，可对接量子退火硬件

---

## 架构设计

```
+------------------------------------------------------------------+
|                      QVAE Architecture                           |
|                                                                  |
|  Protein Sequence (One-hot, 70 x 22 = 1540)                     |
|           |                                                      |
|           v                                                      |
|  +------------------+                                            |
|  |    Encoder       |  Linear(1540->512) -> ReLU ->             |
|  |   (FC/MLP)       |  Linear(512->latent_dim)  = q_logits      |
|  +------------------+                                            |
|           |                                                      |
|           v                                                      |
|  +------------------+                                            |
|  |  MixtureGeneric  |  DVAE++ overlap distribution               |
|  |  (Posterior)     |  Exponential smoothing + implicit grad     |
|  +------------------+                                            |
|           |                                                      |
|           v                                                      |
|     zeta (sample)         +------------------+                   |
|           |               |   Boltzmann      |                   |
|           |    +--------->|   Machine Prior  |                   |
|           |    |          |  (RBM / Full BM) |                   |
|           |    |          +------------------+                   |
|           v    |                    |                            |
|  +------------------+              v                            |
|  |    Decoder       |     Ising Matrix                          |
|  |   (FC/MLP)       |     +------------------+                  |
|  |                  |     |   Sampler        |                  |
|  |  Linear->ReLU->  |     |  (Sim Annealing) |                  |
|  |  Linear          |     +------------------+                  |
|  +------------------+                                           |
|           |                                                      |
|           v                                                      |
|  Reconstructed Logits (1540) -> One-hot Sequence                 |
|                                                                  |
|  Loss = Reconstruction + beta * KL(q || p_BM)                    |
|        + CD Loss (BM training via positive/negative phase)       |
+------------------------------------------------------------------+
```

### 两种模型变体

| 特性 | `QVAE` (RBM 先验) | `QVAE_BM` (Full BM 先验) |
|------|-------------------|-------------------------|
| **先验模型** | RestrictedBoltzmannMachine | BoltzmannMachine |
| **连接方式** | 可见层 <-> 隐藏层 (二分图) | 全连接 (含自连接) |
| **权重矩阵** | `(V, H)` 稀疏矩阵 | `(N, N)` 全矩阵 |
| **默认 Beta** | 1.0 | 0.1 |
| **默认 Latent Dim** | 64 | 32/64/128 |
| **采样复杂度** | O(V*H) | O(N^2) |
| **训练脚本** | `scripts/qvae/train.py` | `scripts/qvae_bm/train.py` |

---

## 目录结构

```
QVAE/
├── kaiwu_torch_plugin/               # 核心模型库
│   ├── __init__.py                   # 导出: RBM, BM, QVAE, QVAE_BM, DBN
│   ├── qvae.py                       # QVAE 模型 (RBM 先验)
│   ├── qvae_bm.py                    # QVAE_BM 模型 (Full BM 先验)
│   ├── qvae_dist_util.py             # 分布工具 (MixtureGeneric, FactorialBernoulli)
│   ├── abstract_boltzmann_machine.py # BM 基类 (Ising 转换, 采样接口)
│   ├── restricted_boltzmann_machine.py # 受限玻尔兹曼机
│   ├── full_boltzmann_machine.py     # 全连接玻尔兹曼机
│   └── dbn.py                        # 深度信念网络
├── scripts/                          # 训练和推理脚本
│   ├── qvae/                         # QVAE (RBM 先验)
│   ├── qvae_bm/                      # QVAE_BM (Full BM 先验)
│   └── vae/                          # Standard VAE (基线)
│   └── <model>/
│       ├── train.py                  # 模型训练
│       ├── generate.py               # 序列生成 (从 BM 先验采样)
│       ├── sample.py                 # 序列分析 (UMAP 聚类, 物种分类)
│       └── unirep.py                 # UniRep 嵌入计算
├── analysis/                         # 生物信息学分析工具
│   ├── characteristics.py            # 序列物化性质分析与可视化
│   ├── deeploc2.py                   # DeepLoc 2.0 结果对比分析
│   ├── gfp.py                        # GFP 报告蛋白融合 (用于 DeepLoc 提交)
│   ├── filter_sequences.py           # TargetP 结果过滤 (mTP 阈值筛选)
│   ├── organism.py                   # 跨物种比较分析
│   ├── seq_identity.py               # 序列同一性分析
│   └── distribution.py               # 分布可视化
├── data/                             # 数据和模型输出
│   ├── tv_sim_split_train.pkl        # 训练集
│   ├── tv_sim_split_valid.pkl        # 验证集
│   ├── mts_train.fasta               # 跨膜信号肽训练数据 (原始 FASTA)
│   ├── model_organism_sequences_mts.*# 模式生物序列 (FASTA + UniRep 嵌入)
│   ├── diversity_cluster_ss.fas      # s4pred 二级结构预测结果 (聚合)
│   ├── qvae/                         # QVAE (RBM) 模型输出
│   ├── qvae_bm/                      # QVAE_BM (Full BM) 模型输出
│   ├── vae/                          # Standard VAE 输出
│   └── analysis/                     # 分析图表输出
├── s4pred/                           # 二级结构预测工具 (第三方)
│   ├── run_model.py                  # S4PRED 推理脚本
│   ├── network.py                    # 模型网络定义
│   ├── utilities.py                  # 工具函数
│   └── weights.tar.gz                # 预训练权重
├── kaiwu-1.2.0-*.whl                 # Kaiwu SDK 安装包
├── requirements_qvae.txt             # Python 依赖
├── requirements_unirep.txt           # UniRep 依赖
└── .qvae/                            # Python 虚拟环境
```

---

## 安装

### 环境要求

- Python 3.10+
- CUDA 11.8+ (推荐，用于 GPU 加速训练)
- 8GB+ RAM
- 10GB+ 可用磁盘空间

### 安装步骤

```bash
# 1. 创建虚拟环境
python -m venv .qvae
source .qvae/bin/activate

# 2. 安装依赖
pip install -r requirements_qvae.txt

# 3. 安装 Kaiwu SDK (量子计算支持)
pip install kaiwu-1.2.0-cp310-none-manylinux1_x86_64.whl

# 4. 验证安装
python -c "import kaiwu_torch_plugin; print('OK')"
```

---

## 快速开始

### 1. 数据格式

数据使用 Pickle 格式存储，需包含 `sequence` 列的 DataFrame：

```python
# 数据格式示例
# DataFrame columns: ['sequence', ...]
# sequence: 蛋白质序列字符串，仅含标准 20 种氨基酸
```

数据编码规则：
- 氨基酸字符集: `FIWLVMYCATHGSQRKNEPD` (20 种)
- 特殊字符: `$` (序列终止符), `0` (填充符)
- 最大长度: 70 个氨基酸
- One-hot 维度: 70 x 22 = 1540

### 2. 训练模型

修改 `scripts/qvae_bm/train.py` 中的超参数后直接运行：

```bash
# 训练 QVAE_BM (Full BM 先验) - 推荐
python scripts/qvae_bm/train.py

# 训练 QVAE (RBM 先验)
python scripts/qvae/train.py

# 训练 Standard VAE (基线)
python scripts/vae/train.py
```

关键超参数 (在脚本顶部修改):

```python
BETA = 0.1              # KL 散度权重 (ELBO 中的 beta)
LATENT_DIM = 32         # 隐空间维度
BATCH_SIZE = 2048       # 批大小
LEARNING_RATE_VAE = 1e-4  # VAE 学习率
LEARNING_RATE_BM = 1e-4   # BM 学习率
EPOCHS = 50             # 训练轮数
```

训练输出将保存在 `data/qvae_bm/`:
- `model/*.chkpt` — 最佳模型权重
- `model/mean_x_*.pkl` — 数据偏置 (生成时需要)
- `log/train_log_*.txt` — 训练日志
- `log/*_loss_curves.png` — 损失曲线图
- `log/*_rbm_weights_epoch_final.png` — BM 权重热力图

### 3. 生成序列

```bash
python scripts/qvae_bm/generate.py \
    --model_path data/qvae_bm/model/qvae_best_b2048_ld32_beta0.1_bm0.0001.chkpt \
    --mean_x_path data/qvae_bm/model/mean_x_b2048_ld32_beta0.1_bm0.0001.pkl \
    --n_samples 5000 \
    --latent_dim 32 \
    --batch_size 2048 \
    --beta 0.1 \
    --temperature 1.0 \
    --decode_batch_size 512
```

生成参数说明:
| 参数 | 说明 |
|------|------|
| `--n_samples` | 生成序列总数 |
| `--temperature` | 采样温度 (>1.0 更随机, <1.0 更确定, 0.0 = argmax) |
| `--latent_dim` | 必须与训练时一致 |
| `--batch_size`, `--beta` | 用于构建输出路径名 |

### 4. 计算 UniRep 嵌入

```bash
python scripts/qvae_bm/unirep.py
```

### 5. 物种分类和 UMAP 可视化

```bash
python scripts/qvae_bm/sample.py
```

基于 4 种模式生物 (酵母、隐球酵母、人、烟草) 的 MTS 序列，使用 k-NN + 聚类中心距离为生成序列分配物种标签，并生成 UMAP 可视化图。

---

## 分析 Pipeline

### 完整分析流程

```
生成序列 (FASTA)
      |
      +---> characteristics.py  ---> 物化性质分析
      |         - 净电荷 (pH 7.0)
      |         - GRAVY 疏水指数
      |         - Eisenberg 疏水性
      |         - 氨基酸组成
      |         - 序列长度分布
      |         - 二级结构比例 (s4pred)
      |
      +---> gfp.py              ---> GFP 融合蛋白构建
      |         输出: *_GFP.fasta (用于 DeepLoc 提交)
      |
      +---> deeploc2.py         ---> DeepLoc 2.0 结果分析
      |         - 线粒体定位概率分布
      |         - 亚细胞定位比例对比
      |
      +---> filter_sequences.py  ---> TargetP 结果过滤
      |         筛选 mTP 概率 > 80% 的候选序列
      |
      +---> organism.py         ---> 跨物种比较分析
      |         人/小鼠/酵母 vs 生成序列
      |
      +---> seq_identity.py     ---> 序列同一性分析
            与训练集的相似性分布
```

### 运行分析

```bash
# 综合物化性质分析 (含多模型对比)
python analysis/characteristics.py

# DeepLoc 结果分析
python analysis/deeploc2.py

# GFP 融合 (提交 DeepLoc 前)
python analysis/gfp.py -i data/qvae_bm/.../generated_seqs.fasta -o output_GFP.fasta

# TargetP 过滤 (筛选高置信度信号肽)
python analysis/filter_sequences.py
```

分析图表统一输出至 `data/analysis/` 目录。

---

## 核心源码说明

### `kaiwu_torch_plugin/` — 模型库

#### `QVAE` / `QVAE_BM`

两个主模型类的核心方法:

```python
# 前向传播
recon_x, posterior, q, zeta = model(x)

# 计算负 ELBO 损失
output, recon_x, neg_elbo, wd_loss, total_kl, cost, q, zeta = model.neg_elbo(x, kl_beta=1.0)
```

损失函数构成:
- **Reconstruction**: Bernoulli 交叉熵 (per-variable log probability)
- **KL Divergence**: 通过后验熵 + 交叉熵计算，交叉熵项使用 BM 的 Ising 采样
- **Weight Decay**: `0.01 * ||J||^2 + 0.005 * ||h||^2` (BM 参数正则化)
- **CD Loss**: BM 对比散度 (positive phase from q, negative phase from sampler)

#### `MixtureGeneric` — 后验分布

实现 DVAE++ 中的重叠分布技巧:

```
z ~ Bernoulli(q)          # 离散采样
zeta ~ r(zeta|z)          # 平滑采样
zeta = z ? (1 - zeta) : zeta  # 翻转
```

使用隐式梯度 (DVAE# sec 3.4) 使采样过程可微。

#### `AbstractBoltzmannMachine` — 玻尔兹曼机基类

核心接口:

```python
# 转换为 Ising 模型矩阵 (供采样器使用)
ising_mat = bm.get_ising_matrix()

# 使用采样器采样
samples = bm.sample(sampler)  # sampler: Kaiwu SimulatedAnnealingOptimizer

# 计算目标函数 (负对数似然的梯度等价形式)
loss = bm.objective(s_positive, s_negative)

# 计算哈密顿量 (能量函数)
energy = bm(s_all)  # = s^T h + s^T J s
```

#### `UnsupervisedDBN` / `DBNTrainer` — 深度信念网络

逐层预训练 RBM 堆栈:

```python
dbn = UnsupervisedDBN(hidden_layers_structure=[128, 64, 32])
trainer = DBNTrainer(learning_rate_rbm=0.1, n_epochs_rbm=10, batch_size=100)
dbn = trainer.train(dbn, data)

# 特征提取
features = dbn.transform(data)

# 单层重建
recon, errors = dbn.reconstruct(data, layer_index=0)
```

### 网络结构

```
Encoder:  Linear(1540 -> 512) -> ReLU -> Linear(512 -> latent_dim)
Decoder:  Linear(latent_dim -> 512) -> ReLU -> Linear(512 -> 1540)

RBM Prior:  quadratic_coef (V x H), linear_bias (V + H)
Full BM:    quadratic_coef (N x N), linear_bias (N)
```

---

## 实验配置

### 已训练的模型

| 模型 | Batch Size | Latent Dim | Beta | 先验 | Checkpoint |
|------|-----------|-----------|------|------|------------|
| QVAE | 2048 | 32 | 1.0 | RBM | `data/qvae/model/qvae_best_b2048_ld32_beta1_bm0.0001.chkpt` |
| QVAE | 128 | 32 | 1.0 | RBM | `data/qvae/model/qvae_best_b128_ld32_beta1_bm0.0001.chkpt` |
| QVAE | 2048 | 32 | 0.1 | RBM | `data/qvae/model/qvae_best_b2048_ld32_beta0.1_bm0.0001.chkpt` |
| QVAE_BM | 2048 | 32 | 0.1 | Full BM | `data/qvae_bm/model/qvae_best_b2048_ld32_beta0.1_bm0.0001.chkpt` |
| QVAE_BM | 2048 | 64 | 0.1 | Full BM | `data/qvae_bm/model/qvae_best_b2048_ld64_beta0.1_bm0.0001.chkpt` |
| QVAE_BM | 2048 | 128 | 0.1 | Full BM | `data/qvae_bm/model/qvae_best_b2048_ld128_beta0.1_bm0.0001.chkpt` |

---

## 依赖

### 主要依赖

- `torch >= 2.0` — 深度学习框架
- `kaiwu == 1.2.0` — 量子计算 SDK (含模拟退火优化器)
- `biopython >= 1.85` — 生物信息学工具
- `pandas`, `numpy` — 数据处理
- `matplotlib`, `seaborn` — 可视化
- `scikit-learn`, `scipy` — 分析工具
- `umap-learn` — 降维可视化

### UniRep 嵌入 (可选)

```bash
pip install -r requirements_unirep.txt
```

用于 `scripts/*/unirep.py` 和 `scripts/*/sample.py`。

---

## 参考文献

- **DVAE++**: Thorpe, M. et al. (2022). "Discrete Variational Autoencoders with Relaxed Boltzmann Priors." arXiv:1905.07458
- **DVAE**: Shazeer, N. et al. (2017). "Discrete Variational Autoencoders." arXiv:1609.02200
- **Quantum Boltzmann Machines**: LeCun, Y. et al. (2016). "A Tutorial on Energy-Based Learning." arXiv:1608.00627
- **S4PRED**: Moffat, L. et al. (2021). "S4PRED: A Deep Learning Model for Protein Secondary Structure Prediction." Bioinformatics
- **UniRep**: Rao, R. et al. (2019). "Evaluating the transferability of protein language models." ICML

---

## 许可

本项目仅供学术研究使用。
