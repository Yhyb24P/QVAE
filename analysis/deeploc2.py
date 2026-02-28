import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import logging
import os
import sys

# --- 设置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- 1. 加载 DeepLoc 2.0 预测结果 ---

# 定义数据集配置
# 格式: (文件路径, 显示名称, 颜色代码)
datasets_config = [
    # Baseline
    ('data/VAE.csv', 'Standard VAE', '#58a365'),
    
    # QVAE 实验组
    ('data/qvae_bm/b2048_ld32_beta0.1/output/deeploc2.0.csv', 'QVAE (ld32)', '#f0c929'),
    ('data/qvae_bm/b2048_ld64_beta0.1/output/deeploc2.0.csv', 'QVAE (ld64)', '#e67e22'),
    ('data/qvae_bm/b2048_ld128_beta0.1/output/deeploc2.0.csv', 'QVAE (ld128)', '#8e44ad')
]

data_frames = []
palette = {}

print("--- 开始加载 DeepLoc 结果 ---")

for path, label, color in datasets_config:
    if not os.path.exists(path):
        logging.warning(f"文件不存在: {path}。跳过该数据集。")
        continue
        
    try:
        df = pd.read_csv(path)
        if 'Mitochondrion' not in df.columns or 'Localizations' not in df.columns:
            logging.warning(f"文件 {path} 缺少必要的列。跳过。")
            continue
            
        df['Model'] = label 
        data_frames.append(df)
        palette[label] = color
        logging.info(f"成功加载 {label}: {len(df)} 条记录")
        
    except Exception as e:
        logging.error(f"读取 {path} 时出错: {e}")

if not data_frames:
    logging.error("没有加载到任何有效数据。程序退出。")
    sys.exit()

# --- 2. 统计线粒体功能性成功率 ---
THRESHOLD = 0.6373 

print("\n--- 线粒体定位成功率统计 (Threshold > {:.4f}) ---".format(THRESHOLD))

for df in data_frames:
    model_name = df['Model'].iloc[0]
    total = len(df)
    functional_count = len(df[df['Mitochondrion'] > THRESHOLD])
    success_rate = (functional_count * 100) / total if total > 0 else 0
    print(f"{model_name:<15}: {functional_count}/{total} ({success_rate:.2f}%)")

# --- 3. 数据预处理（用于比例分析） ---
combined_data = pd.concat(data_frames, ignore_index=True)

# 确保输出目录存在
os.makedirs('data/analysis', exist_ok=True)

# --- 4. 绘图：线粒体概率分布对比（归一化概率密度） ---
plt.figure(figsize=(12, 7))
sns.set(style="white")

# 使用 stat="probability" 进行归一化
# common_norm=False 非常关键：确保每个组的占比总和为 1，而不是全局总和为 1
sns.histplot(
    data=combined_data, 
    x='Mitochondrion', 
    hue='Model', 
    stat="probability", 
    common_norm=False,  
    multiple='layer',   
    kde=True,          
    bins=30,           
    palette=palette,
    element="step",     # 使用 step 模式在多组对比时更清晰
    alpha=0.3
)

plt.axvline(THRESHOLD, color='red', linestyle='--', label=f'Functional Threshold ({THRESHOLD})')
plt.title('Mitochondrion Probability Distribution (Normalized Comparison)', fontsize=16) 
plt.xlabel('Predicted Mitochondrion Probability', fontsize=12)
plt.ylabel('Probability (Normalized per Model)', fontsize=12)
plt.legend()

save_path_prob = 'data/analysis/comparison_DeepLoc_prob_normalized.png'
plt.savefig(save_path_prob, dpi=400, bbox_inches="tight")
logging.info(f"已保存归一化概率分布图: {save_path_prob}")
plt.clf() 


# --- 5. 绘图：细胞定位分布对比（百分比模式） ---

# 步骤 A: 计算每个模型中各个定位的百分比
# 先按 Model 和 Localizations 分组计数
dist_df = combined_data.groupby(['Model', 'Localizations']).size().reset_index(name='count')

# 计算每个模型的总序列数
model_totals = combined_data.groupby('Model').size().reset_index(name='total')

# 合并并计算百分比
dist_df = pd.merge(dist_df, model_totals, on='Model')
dist_df['percentage'] = (dist_df['count'] / dist_df['total']) * 100

# 步骤 B: 确定 X 轴排序（以线粒体占比最高的模型为准）
sorted_locs = dist_df.groupby('Localizations')['percentage'].sum().sort_values(ascending=False).index

plt.figure(figsize=(12, 7))
ax = sns.barplot(
    data=dist_df, 
    x='Localizations', 
    y='percentage', 
    hue='Model', 
    order=sorted_labels if 'sorted_labels' in locals() else sorted_locs,
    palette=palette
)

sns.despine(top=True, right=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
plt.title('Predicted Localization Proportion (Normalized %)', fontsize=16)
plt.xlabel('Subcellular Localization', fontsize=12)
plt.ylabel('Percentage of Sequences (%)', fontsize=12)
plt.legend(title='Model')

# 在柱状图上方标注具体数值（可选，如果太挤可以注释掉）
for p in ax.patches:
    if p.get_height() > 1: # 仅标注占比大于1%的
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=8, alpha=0.7)

save_path_loc = 'data/analysis/comparison_DeepLoc_loc_percentage.png'
plt.savefig(save_path_loc, dpi=400, bbox_inches="tight")
logging.info(f"已保存百分比定位分布图: {save_path_loc}")

print("\n--- 比例对比分析完成 ---")