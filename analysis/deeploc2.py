import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import SeqIO
import random
from collections import Counter
import logging

# --- 1. 加载 DeepLoc 2.0 预测结果 ---

# 加载 Standard VAE (高斯先验) 基线的结果
vae_tp = pd.read_csv('data/VAE.csv')
logging.info(f"成功加载 Standard VAE 结果 (共 {len(vae_tp)} 条)")

# 加载 β-QBM-VAE
QVAE_CSV_PATH = 'data/rbm.csv' 
qvae_tp = pd.read_csv(QVAE_CSV_PATH)
logging.info(f"成功加载 β-QBM-VAE 结果 (共 {len(qvae_tp)} 条)")
QVAE_BM_CSV_PATH = 'data/rbm.csv' 
qvae_bm_tp = pd.read_csv(QVAE_BM_CSV_PATH)
logging.info(f"成功加载 β-QBM-BM-VAE 结果 (共 {len(qvae_bm_tp)} 条)")

# --- 2. 计算线粒体功能性成功率 ---
THRESHOLD = 0.6373
vae_prob = list(vae_tp['Mitochondrion'])
qvae_prob = list(qvae_tp['Mitochondrion'])
qvae_bm_prob = list(qvae_bm_tp['Mitochondrion']) 

# 计算并打印功能性（线粒体）的成功率
vae_functional_count = len([v for v in vae_prob if v > THRESHOLD])
vae_success_rate = (vae_functional_count * 100) / len(vae_prob)
print(f'Standard VAE: {vae_functional_count} / {len(vae_prob)} = {vae_success_rate:.2f}% 的序列 > {THRESHOLD}')

qvae_functional_count = len([v for v in qvae_prob if v > THRESHOLD])
qvae_success_rate = (qvae_functional_count * 100) / len(qvae_prob)
print(f'β-QBM-VAE (Ours): {qvae_functional_count} / {len(qvae_prob)} = {qvae_success_rate:.2f}% 的序列 > {THRESHOLD}')


qvae_bm_functional_count = len([v for v in qvae_bm_prob if v > THRESHOLD])
qvae_bm_success_rate = (qvae_bm_functional_count * 100) / len(qvae_bm_prob)
print(f'β-QBM-BM-VAE (Ours-BM): {qvae_bm_functional_count} / {len(qvae_bm_prob)} = {qvae_bm_success_rate:.2f}% 的序列 > {THRESHOLD}')


# --- 3. 绘图：线粒体概率分布对比 (直方图) ---

# 为每个 DataFrame 添加 'Model' 列
vae_tp['Model'] = 'Standard VAE (Baseline)'
qvae_tp['Model'] = 'β-QBM-VAE (Ours)'
qvae_bm_tp['Model'] = 'β-QBM-BM-VAE (Ours-BM)' 

# 合并数据以便于绘图
combined_prob_data = pd.concat([
    vae_tp[['Mitochondrion', 'Model']], 
    qvae_tp[['Mitochondrion', 'Model']],
    qvae_bm_tp[['Mitochondrion', 'Model']] 
])

sns.set(style="white")
plt.figure(figsize=(12, 7))

# 定义调色板
palette = {
    'Standard VAE (Baseline)': '#4C72B0', 
    'β-QBM-VAE (Ours)': '#DD8452',
    'β-QBM-BM-VAE (Ours-BM)': '#55A868' 
}

# 使用 histplot (或 kdeplot) 并设置 hue='Model' 来创建对比图
sns.histplot(
    data=combined_prob_data, 
    x='Mitochondrion', 
    hue='Model', 
    multiple='dodge',  # 并排显示
    kde=True,          # 显示核密度估计曲线
    bins=50,
    palette=palette # 使用调色板
)

plt.axvline(THRESHOLD, color='red', linestyle='--', label=f'Functional Threshold ({THRESHOLD})')
plt.title('Mitochondrion Probability Distribution (3-Model Comparison)', fontsize=16) 
plt.xlabel('Predicted Mitochondrion Probability (DeepLoc 2.0)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend()
<<<<<<< HEAD
plt.savefig('data/analysis/comparison_DeepLoc_prob_distribution_VAE_vs_QVAE.png', dpi=400, bbox_inches="tight")
=======
plt.savefig('data/analysis/comparison_DeepLoc_prob_distribution_3_models.png', dpi=400, bbox_inches="tight") 
>>>>>>> 19398fa (保存)
print("已保存概率分布对比图。")


# --- 4. 绘图：细胞定位分布对比 (计数图) ---

# 准备定位数据
vae_locs = vae_tp[['Localizations', 'Model']]
qvae_locs = qvae_tp[['Localizations', 'Model']]
qvae_bm_locs = qvae_bm_tp[['Localizations', 'Model']] 
combined_loc_data = pd.concat([vae_locs, qvae_locs, qvae_bm_locs]) # 更新

# 计算排序（基于总数）
label_counts = Counter(combined_loc_data['Localizations'])
sorted_labels = sorted(label_counts, key=lambda x: label_counts[x], reverse=True)

plt.figure(figsize=(12, 7))
ax = sns.countplot(
    data=combined_loc_data, 
    x='Localizations', 
    hue='Model', 
    order=sorted_labels,
    palette=palette # 复用上面的调色板
)

sns.despine(top=True, right=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha="right", fontsize=12)
plt.title('Predicted Localization Distribution (3-Model Comparison)', fontsize=16)
plt.xlabel('Localization', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Model')
<<<<<<< HEAD
plt.savefig('data/analysis/comparison_DeepLoc_localization_distribution_VAE_vs_QVAE.png', dpi=400, bbox_inches="tight")
print("已保存定位分布对比图。")
=======
plt.savefig('data/analysis/comparison_DeepLoc_localization_distribution_3_models.png', dpi=400, bbox_inches="tight") 
>>>>>>> 19398fa (保存)
