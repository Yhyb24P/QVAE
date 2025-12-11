# QBM-VAE: Quantum Boltzmann Machine Variational Autoencoder

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![BioPython](https://img.shields.io/badge/BioPython-1.85+-green.svg)](https://biopython.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

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
│                    QBM-VAE Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Protein Sequences                                    │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Encoder    │───▶│   Posterior  │───▶│   Decoder    │   │
│  │  (FC/MLP)   │    │   (Mixture   │    │  (FC/MLP)    │   │
│  │             │    │ Distribution)│    │              │   │
│  └─────────────┘    └──────────────┘    └──────────────┘   │
│         │                   │              │              │
│         │                   ▼              │              │
│         │         ┌──────────────────┐     │              │
│         │         │  Quantum Boltzmann │     │              │
│         └────────▶│     Machine       │◀────┘              │
│                   │  (RBM/Full BM)     │                    │
│                   └──────────────────┘                    │
│                            │                              │
│                            ▼                              │
│                   ┌──────────────────┐                    │
│                   │  Quantum Sampler  │                    │
│                   │  (Simulated       │                    │
│                   │   Annealing/CIM)  │                    │
│                   └──────────────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 安装

### 环境要求
- Python 3.10+
- CUDA 11.8+ (推荐用于GPU加速)
- 8GB+ RAM
- 10GB+ 可用磁盘空间

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/your-repo/qbm-vae.git
cd qbm-vae
```

2. **创建conda环境**（推荐）
```bash
conda create -n qbm-vae python=3.10
conda activate qbm-vae
```

3. **安装依赖**
```bash
pip install -r requirements_qvae.txt
```

4. **安装Kaiwu SDK**（量子计算支持）
```bash
# 请联系Kaiwu获取SDK安装包
pip install kaiwu-sdk-*.whl
```

5. **验证安装**
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

# 保存为pickle文件
data.to_pickle('data/tv_sim_split_train.pkl')
```

### 2. 模型训练
```bash
# 训练QVAE模型
python scripts/qvae/train.py

# 或使用自定义参数
python scripts/qvae/train.py \
    --batch_size 2048 \
    --latent_dim 32 \
    --beta 1.0 \
    --epochs 50
```

### 3. 序列生成
```bash
# 从训练好的模型生成序列
python scripts/qvae/generate.py \
    --model_path data/qvae/model/qvae_best_b2048_ld32_beta1.0_bm0.0001.chkpt \
    --mean_x_path data/qvae/model/mean_x_b2048_ld32_beta1.0_bm0.0001.pkl \
    --n_samples 5000 \
    --temperature 1.0
```

### 4. 序列分析
```python
from analysis.characteristics import calculate_bio_properties

# 分析生成的序列
sequences = pd.read_fasta('generated_seqs.fasta')
properties = calculate_bio_properties(sequences)
print(properties.describe())
```

## API参考

### 核心模型类

#### `QVAE`
量子变分自编码器主类，结合了经典VAE和量子玻尔兹曼机。

```python
from kaiwu_torch_plugin import QVAE

model = QVAE(
    encoder=encoder_module,      # 编码器网络
    decoder=decoder_module,      # 解码器网络
    bm=rbm_prior,               # 玻尔兹曼机先验
    sampler=simulated_annealing, # 量子采样器
    dist_beta=1.0,              # 分布beta参数
    mean_x=0.5,                 # 训练数据偏置
    num_vis=16                  # 可见变量数量
)
```

**主要方法：**
- `neg_elbo(x, kl_beta)` - 计算负ELBO损失
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

#### `BoltzmannMachine`
完全连接的玻尔兹曼机，支持更复杂的量子关联。

```python
from kaiwu_torch_plugin import BoltzmannMachine

bm = BoltzmannMachine(
    num_nodes=32,      # 总节点数
    h_range=(-5, 5),   # 线性权重范围
    j_range=(-1, 1)    # 二次权重范围
)
```

### 分布工具类

#### `MixtureGeneric`
混合分布类，实现DVAE++中的重叠分布技巧。

```python
from kaiwu_torch_plugin.qvae_dist_util import MixtureGeneric

posterior = MixtureGeneric(
    param=logits,              # 编码器输出logits
    smoothing_dist_beta=1.0    # 平滑分布beta参数
)

# 重参数化采样
zeta = posterior.reparameterize(is_training=True)
```

#### `FactorialBernoulliUtil`
伯努利分布工具类，处理二值随机变量的概率分布。

```python
from kaiwu_torch_plugin.qvae_dist_util import FactorialBernoulliUtil

dist = FactorialBernoulliUtil(logits)
entropy = dist.entropy()
log_prob = dist.log_prob_per_var(samples)
```

### 分析工具

#### `calculate_bio_properties`
计算蛋白质序列的生物学属性。

```python
from analysis.characteristics import calculate_bio_properties

properties = calculate_bio_properties(sequence)
# 返回：分子量、等电点、净电荷、疏水性等
```

#### `read_fasta`
读取FASTA格式的序列文件。

```python
from analysis.characteristics import read_fasta

sequences = read_fasta('generated_seqs')
# 返回：[[name1, seq1], [name2, seq2], ...]
```

## 配置说明

### 训练参数配置

```python
# 核心超参数
BETA = 1.0              # KL散度权重
LATENT_DIM = 32         # 隐空间维度
BATCH_SIZE = 2048       # 批大小
LEARNING_RATE_VAE = 1e-4 # VAE学习率
LEARNING_RATE_BM = 1e-4  # BM学习率
EPOCHS = 50             # 训练轮数

# 网络结构
MAX_LEN = 70            # 序列最大长度
CHANNELS = 22           # 氨基酸通道数（20种氨基酸 + 特殊字符）
INPUT_DIM = MAX_LEN * CHANNELS  # 输入维度

# RBM先验结构
prior_vis = LATENT_DIM // 2    # 可见节点数
prior_hid = LATENT_DIM - prior_vis  # 隐藏节点数
```

### 量子采样器配置

```python
from kaiwu.classical import SimulatedAnnealingOptimizer

sampler = SimulatedAnnealingOptimizer(
    initial_temperature=500.0,    # 初始温度
    alpha=0.99,                   # 退火率
    cutoff_temperature=0.001,     # 截止温度
    iterations_per_t=20,          # 每温度迭代次数
    size_limit=100,               # 解大小限制
    process_num=-1                # 进程数（-1为自动）
)
```

### 生成参数配置

```python
# 序列生成参数
n_samples = 5000        # 生成序列数量
temperature = 1.0       # 采样温度（>1.0更随机，<1.0更确定）
decode_batch_size = 512 # 解码批大小
```

## 示例流程

### 完整训练流程

```python
import torch
import pandas as pd
from kaiwu_torch_plugin import QVAE, RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer

# 1. 数据加载
data = pd.read_pickle('data/tv_sim_split_train.pkl')
mean_x = calculate_mean_x(data)  # 计算数据偏置

# 2. 模型组件定义
encoder = EncoderFC(INPUT_DIM, LATENT_DIM)
decoder = DecoderFC(LATENT_DIM, INPUT_DIM)
rbm_prior = RestrictedBoltzmannMachine(num_visible=16, num_hidden=16)
sampler = SimulatedAnnealingOptimizer()

# 3. QVAE模型初始化
model = QVAE(
    encoder=encoder,
    decoder=decoder,
    bm=rbm_prior,
    sampler=sampler,
    dist_beta=1.0,
    mean_x=mean_x,
    num_vis=16
)

# 4. 训练循环
optimizer_vae = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(EPOCHS):
    for batch in dataloader:
        loss = model.neg_elbo(batch, kl_beta=BETA)
        optimizer_vae.zero_grad()
        loss.backward()
        optimizer_vae.step()
```

### 序列分析流程

```python
from analysis import characteristics, deeploc2, gfp

# 1. 基础特征分析
sequences = characteristics.read_fasta('generated_seqs')
properties = characteristics.calculate_bio_properties(sequences)

# 2. 定位预测
localization = deeploc2.predict_localization(sequences)

# 3. GFP功能分析
gfp_scores = gfp.calculate_gfp_activity(sequences)

# 4. 可视化
characteristics.plot_property_distribution(properties)
deeploc2.plot_localization_comparison(localization)
```

## 引用文献

如果您使用QBM-VAE进行研究，请引用以下文献：

```bibtex
@article{qbmvae2024,
  title={Quantum Boltzmann Machine Variational Autoencoder for Protein Sequence Generation},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## 相关论文

- **DVAE++**: [Discrete Variational Autoencoders with Relaxed Boltzmann Priors](https://arxiv.org/abs/1905.07458)
- **Quantum Boltzmann Machines**: [Quantum Boltzmann Machine Learning](https://arxiv.org/abs/1608.00627)
- **Protein Language Models**: [ProteinBERT: a universal protein language model](https://www.nature.com/articles/s41467-022-32007-7)

## 许可证

本项目采用MIT许可证。详情请参见[LICENSE](LICENSE)文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: [项目Issues页面](https://github.com/your-repo/qbm-vae/issues)
- 邮箱: yhybpjy@gmail.com

## 致谢

感谢Kaiwu SDK提供的量子计算支持，以及BioPython社区提供的生物信息学工具。
