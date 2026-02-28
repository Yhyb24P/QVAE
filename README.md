# QBM-VAE: Quantum Boltzmann Machine Variational Autoencoder

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)
![BioPython](https://img.shields.io/badge/BioPython-1.82-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 简介

QBM-VAE (Quantum Boltzmann Machine Variational Autoencoder) 是一个结合了量子计算和深度学习的蛋白质序列生成框架。该项目通过将量子玻尔兹曼机作为变分自编码器的隐空间先验，实现了对蛋白质序列的高效生成和优化。

**核心创新点：**

- 首次将量子玻尔兹曼机与变分自编码器结合用于蛋白质序列生成
- 利用量子退火算法优化隐空间分布，提高序列多样性和质量
- 支持多种玻尔兹曼机架构（受限玻尔兹曼机、完全玻尔兹曼机）
- 集成了完整的生物信息学分析pipeline

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     QBM-VAE Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    Input Protein Sequences                   │
│                              │                              │
│                              ▼                              │
│   ┌─────────────┐    ┌──────────────┐    ┌──────────────┐ │
│   │   Encoder   │───▶│  Posterior   │───▶│   Decoder    │ │
│   │   (FC/MLP)  │    │  (Mixture    │    │   (FC/MLP)   │ │
│   │             │    │ Distribution)│    │              │ │
│   └─────────────┘    └──────────────┘    └──────────────┘ │
│         │                     │                   │         │
│         │                     ▼                   │         │
│         │         ┌──────────────────┐            │         │
│         └────────▶│  Quantum         │◀───────────┘         │
│                   │  Boltzmann       │                      │
│                   │  Machine         │                      │
│                   │  (RBM/Full BM)   │                      │
│                   └──────────────────┘                      │
│                            │                                │
│                            ▼                                │
│                   ┌──────────────────┐                      │
│                   │ Quantum Sampler  │                      │
│                   │  (Simulated      │                      │
│                   │  Annealing/CIM)  │                      │
│                   └──────────────────┘                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 实验结果

### 1. 亚细胞器定位分析（DeepLoc 2.0）

我们使用DeepLoc 2.0工具对生成的蛋白质序列进行了亚细胞器定位预测。结果显示，**QVAE生成的序列在线粒体定位特征上与训练集高度一致**，表明模型成功学习到了线粒体目标蛋白的生物学特性。

**关键发现：**
- **线粒体预测率**：生成序列中线粒体定位的比例与训练集接近
- **模式保持：**不同隐空间维度（ld32/ld64/ld128）下均保持稳定的定位分布模式
- **多样性保证：**尽管主要定位在线粒体，也生成了其他亚细胞器定位的序列

### 2. 基线模型对比

我们将QVAE与多个基线模型进行了全面对比：

| 模型 | Levenshtein Distance (Novelty) | mTP Score | 线粒体定位率 |
|------|--------------------------|-----------|-------------|
| **QVAE (ld32)** | **高** | **>80** | **~训练集** |
| **QVAE (ld64)** | **高** | **>80** | **~训练集** |
| **QVAE (ld128)** | **高** | **>80** | **~训练集** |
| MTS-VAE | 中 | >80 | - |
| UniProt (自然序列) | - | 参考值 | 100% |
| VAE (普通VAE) | 低-中 | <80 | - |

**关键优势：**
1. **新颖性（Novelty）**：QVAE生成的序列与UniProt自然序列的Levenshtein距离大，表明生成了具有创新性的序列
2. **功能保持：**mTP分数>80，确保线粒体目标能力（mitochondrial targeting peptide）
3. **架构优越性：**相比于传统 VAE，量子玻尔兹曼机先验显著提升生成质量

### 3. 隐空间维度对比

我们测试了三种不同的隐空间维度：

**ld32（低维度）**
- 训练速度最快
- 生成序列聚类紧密
- 适合快速实验迭代

**ld64（中维度）**
- 性能和质量平衡最佳
- 生成序列聚类分布合理
- **推荐使用此配置**

**ld128（高维度）**
- 表达能力最强
- 生成序列多样性最高
- 需要更多计算资源

### 4. 聚类分析

通过t-SNE降维可视化，我们观察到：
- **ld32**：生成序列形成紧密的聚类，表明模式一致性好
- **ld64**：聚类分布均衡，既有聚集又保持多样性
- **ld128**：分布最广，探索空间最大

## 安装

### 环境要求

- Python 3.10+
- CUDA 11.8+ (推荐用于GPU加速)
- 8GB+ RAM
- 10GB+ 可用磁盘空间

### 安装步骤

**1. 克隆仓库**

```bash
git clone https://github.com/Yhyb24P/QVAE.git
cd QVAE
```

**2. 创建 conda 环境**（推荐）

```bash
conda create -n qbm-vae python=3.10
conda activate qbm-vae
```

**3. 安装依赖**

```bash
pip install -r requirements_qvae.txt
```

**4. 安装 Kaiwu SDK**（量子计算支持）

```bash
# 请联系 Kaiwu 获取 SDK 安装包
pip install kaiwu-sdk-*.whl
```

**5. 验证安装**

```bash
python -c "import kaiwu_torch_plugin; print('QBM-VAE installed successfully!')"
```

## 快速开始

### 1. 数据准备

```python
import pandas as pd

# 准备蛋白质序列数据
data = pd.DataFrame({
    'sequence': ['MKTIIALSYIFCLVFAQK', 'MKTIIALSYIFCLVFAQKP', ...],
    'name': ['protein_1', 'protein_2', ...]
})

# 保存为 pickle 文件
data.to_pickle('data/tv_sim_split_train.pkl')
```

### 2. 模型训练

```bash
# 训练 QVAE 模型
python scripts/qvae/train.py

# 或使用自定义参数
python scripts/qvae/train.py \\
    --batch_size 2048 \\
    --latent_dim 64 \\
    --beta 0.1 \\
    --epochs 50
```

### 3. 序列生成

```bash
# 从训练好的模型生成序列
python scripts/qvae/generate.py \\
    --model_path data/qvae_bm/b2048_ld64_beta0.1/model/qvae_best.chkpt \\
    --mean_x_path data/qvae_bm/b2048_ld64_beta0.1/model/mean_x.pkl \\
    --n_samples 5000 \\
    --temperature 1.0
```

### 4. 序列分析

```python
from analysis.characteristics import calculate_bio_properties, read_fasta

# 分析生成的序列
sequences = read_fasta('generated_seqs.fasta')
properties = calculate_bio_properties(sequences)
print(properties.describe())
```

## 项目结构

```
QVAE/
├── analysis/              # 生物信息学分析工具
│   ├── characteristics.py  # 序列特征计算
│   ├── deeploc2.py         # DeepLoc 2.0 定位分析
│   └── gfp.py              # GFP 功能分析
├── data/                  # 数据目录
│   ├── qvae_bm/            # QVAE 模型数据
│   │   ├── b2048_ld32_beta0.1/
│   │   ├── b2048_ld64_beta0.1/
│   │   └── b2048_ld128_beta0.1/
│   └── tv_sim_split_train.pkl  # 训练数据
├── kaiwu_torch_plugin/    # Kaiwu 量子计算插件
│   ├── __init__.py
│   ├── qvae.py              # QVAE 核心实现
│   ├── qvae_dist_util.py    # 分布工具
│   └── bm.py                # 玻尔兹曼机实现
├── s4pred/                # S4 预测模块
├── scripts/               # 训练和生成脚本
│   └── qvae/
│       ├── train.py         # 训练脚本
│       └── generate.py      # 生成脚本
├── requirements_qvae.txt  # Python 依赖
└── README.md              # 本文档
```

## 配置说明

### 训练参数配置

```python
# 核心超参数
BETA = 0.1              # KL 散度权重
LATENT_DIM = 64         # 隐空间维度 (推荐 32/64/128)
BATCH_SIZE = 2048       # 批大小
LEARNING_RATE_VAE = 1e-4  # VAE 学习率
LEARNING_RATE_BM = 1e-4   # BM 学习率
EPOCHS = 50             # 训练轮数

# 网络结构
MAX_LEN = 70            # 序列最大长度
CHANNELS = 22           # 氨基酸通道数
INPUT_DIM = MAX_LEN * CHANNELS  # 输入维度

# RBM 先验结构
prior_vis = LATENT_DIM // 2      # 可见节点数
prior_hid = LATENT_DIM - prior_vis  # 隐藏节点数
```

### 量子采样器配置

```python
from kaiwu.classical import SimulatedAnnealingOptimizer

sampler = SimulatedAnnealingOptimizer(
    initial_temperature=500.0,     # 初始温度
    alpha=0.99,                    # 退火率
    cutoff_temperature=0.001,      # 截止温度
    iterations_per_t=20,           # 每温度迭代次数
    size_limit=100,                # 解大小限制
    process_num=-1                 # 进程数（-1 为自动）
)
```

## API 参考

### 核心模型类

#### `QVAE`

量子变分自编码器主类，结合了经典 VAE 和量子玻尔兹曼机。

```python
from kaiwu_torch_plugin import QVAE

model = QVAE(
    encoder=encoder_module,      # 编码器网络
    decoder=decoder_module,      # 解码器网络
    bm=rbm_prior,                # 玻尔兹曼机先验
    sampler=simulated_annealing, # 量子采样器
    dist_beta=1.0,               # 分布 beta 参数
    mean_x=0.5,                  # 训练数据偏置
    num_vis=16                   # 可见变量数量
)
```

**主要方法：**
- `neg_elbo(x, kl_beta)` - 计算负 ELBO 损失
- `forward(x)` - 前向传播
- `posterior(q_logits, beta)` - 计算后验分布

#### `RestrictedBoltzmannMachine`

受限玻尔兹曼机实现，用于构建量子隐空间先验。

```python
from kaiwu_torch_plugin import RestrictedBoltzmannMachine

rbm = RestrictedBoltzmannMachine(
    num_visible=16,    # 可见节点数
    num_hidden=16,     # 隐藏节点数
    h_range=(-5, 5),   # 线性权重范围
    j_range=(-1, 1)    # 二次权重范围
)
```

## 相关论文

- **DVAE++**: [Discrete Variational Autoencoders with Relaxed Boltzmann Priors](https://arxiv.org/abs/1905.07458)
- **Quantum Boltzmann Machines**: [Quantum Boltzmann Machine Learning](https://arxiv.org/abs/1608.00627)
- **Protein Language Models**: [ProteinBERT: a universal protein language model](https://www.nature.com/articles/s41467-022-32007-7)
- **DeepLoc 2.0**: [DeepLoc 2.0: multi-label subcellular localization prediction](https://academic.oup.com/nar/article/50/W1/W228/6576357)

## 引用文献

如果您使用 QBM-VAE 进行研究，请引用以下文献：

```bibtex
@article{qbmvae2024,
  title={Quantum Boltzmann Machine Variational Autoencoder for Protein Sequence Generation},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## 贡献指南

我们欢迎各种形式的贡献！请遵循以下步骤：

1. Fork 该仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

## 许可证

本项目采用 MIT 许可证。详情请参见 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues: [项目 Issues 页面](https://github.com/Yhyb24P/QVAE/issues)
- Email: yhyb24P@outlook.com

## 致谢

感谢 Kaiwu SDK 提供的量子计算支持，以及 BioPython 社区提供的生物信息学工具。

## 常见问题

### Q: QVAE 和传统 VAE 的主要区别是什么？

A: QVAE 在隐空间使用了量子玻尔兹曼机作为先验分布，相比于传统 VAE 的高斯先验，能够更好地捕捉离散空间中的复杂依赖关系，从而生成更高质量、更多样化的蛋白质序列。

### Q: 为什么推荐使用 ld64 作为隐空间维度？

A: 根据我们的实验结果，ld64 在训练效率、生成质量和序列多样性之间达到了最佳平衡。ld32 更快但表达能力较弱，ld128 表达能力更强但训练成本更高。

### Q: 是否需要真正的量子计算机？

A: 不需要。项目默认使用经典的模拟退火算法（Simulated Annealing）来近似量子采样。如果您有访问真实量子计算机的权限，也可以通过 Kaiwu SDK 接入。

### Q: 生成的序列如何验证功能性？

A: 我们提供了多种验证工具：
1. **DeepLoc 2.0**: 预测亚细胞器定位
2. **mTP 分数**: 评估线粒体目标能力
3. **生物信息学特征**: 计算分子量、等电点、疏水性等
4. **Levenshtein 距离**: 评估序列新颖性

---

**⭐ 如果这个项目对您有帮助，请给我们一个 Star！**
