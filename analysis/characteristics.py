import numpy as np
import pandas as pd
import re
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from matplotlib.patches import Patch
import sys # 用于退出
import os # 用于检查文件
import pickle 

# --- 1. Functions ---

def calculate_amino_acid_fraction(peptide):
    """计算单个肽的氨基酸组成百分比"""
    try:
        prot_param = ProteinAnalysis(str(peptide))
        #  修复 BiopythonDeprecationWarning
        return prot_param.amino_acids_percent
    except (ValueError, TypeError):
        # 处理无效序列（例如，包含非标准氨基酸或为空）
        return pd.Series(dtype=float)


def read_fasta(name):
    """从 FASTA 文件读取数据"""
    # 自动处理是否有 .fasta 后缀
    if not name.endswith('.fasta'):
        filepath = name + '.fasta'
    else:
        filepath = name

    if not os.path.exists(filepath):
        print(f"错误：FASTA 文件未找到: {filepath}")
        return [] # 返回空列表
    
    print(f"Reading FASTA file: {filepath}")
    data = []
    try:
        for fasta in SeqIO.parse(open(filepath),'fasta'):
            data.append([fasta.id, str(fasta.seq).strip()])
    except Exception as e:
        print(f"读取 FASTA 文件时出错: {e}")
        return []
        
    return data

def calculate_bio_properties(sequence):
    """
     辅助函数：用于 .apply()，一次性计算所有 Biopython 属性
    """
    try:
        prot_seq = ProteinAnalysis(str(sequence))
        net_charge = prot_seq.charge_at_pH(7.0)
        gravy = prot_seq.gravy()
        eisenberg_hydrophobicity = prot_seq.gravy(scale='Eisenberg')
        return net_charge, gravy, eisenberg_hydrophobicity
    except (ValueError, TypeError):
        return np.nan, np.nan, np.nan

# Function to calculate secondary structure element percentages
def calculate_secondary_structure_percentages(structure):
    """计算二级结构百分比 (C, H, E)"""
    if not isinstance(structure, str):
        return np.nan, np.nan, np.nan
        
    length = len(structure)
    if length == 0: # 增加一个对空字符串的检查
        return 0.0, 0.0, 0.0
        
    c_count = structure.count("C")
    h_count = structure.count("H")
    e_count = structure.count("E")
    
    c_percentage = (c_count / length) * 100
    h_percentage = (h_count / length) * 100
    e_percentage = (e_count / length) * 100
    
    return c_percentage, h_percentage, e_percentage

def lowercase_sample(name):
    """辅助函数：标准化 'SAMPLE' 名称"""
    if isinstance(name, str) and 'SAMPLE' in name:
        return name.lower()
    return name

def read_s4pred_fasta(filepath):
    """辅助函数：解析 s4pred 的3行 .fas 文件"""
    data = {"Name": [], "Sequence": [], "Structure": []}
    if not os.path.exists(filepath):
        print(f"  [跳过] 找不到 s4pred 文件: {filepath}")
        return pd.DataFrame(data)

    try:
        print(f"  正在读取 s4pred 文件: {filepath}")
        with open(filepath, "r") as file:
            lines = file.readlines()

        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                # print(f"  警告: s4pred 文件 {filepath} 在第 {i} 行处记录不完整。")
                continue
                
            protein_name = lines[i].strip()[1:]
            protein_sequence = lines[i + 1].strip()
            protein_structure = lines[i + 2].strip()

            data["Name"].append(protein_name)
            data["Sequence"].append(protein_sequence)
            data["Structure"].append(protein_structure)
        
        print(f"  -> 成功加载 {len(data['Name'])} 条结构数据。")
    
    except Exception as e:
        print(f"  读取 s4pred 文件 {filepath} 时出错: {e}")
        return pd.DataFrame(data) # 返回空/部分数据
        
    return pd.DataFrame(data)


# --- 2. Data Loading ---

# 定义数据集配置
# 格式: (文件路径 (不含后缀), 标签, 颜色代码)
datasets_config = [
    # 1. 训练集 (基准 - 蓝紫色系)
    ('data/mts_train', 'Train Data', '#5865a3'),
    
    # 2. Standard VAE (基线模型 - 绿色系)
    ('data/vae/output/amts', 'AMTS (VAE)', '#58a365'),

    # 3. QVAE 实验组 (不同 Latent Dim - 调整后的配色)
    # ld32: 金色/明黄
    ('data/qvae_bm/b2048_ld32_beta0.1/output/generated_seqs_n5000_T1.0', 'QVAE (ld32)', '#f0c929'), 
    # ld64: 深橙色
    ('data/qvae_bm/b2048_ld64_beta0.1/output/generated_seqs_n5000_T1.0', 'QVAE (ld64)', '#e67e22'), 
    # ld128: 紫红色 (避免与橙色混淆)
    ('data/qvae_bm/b2048_ld128_beta0.1/output/generated_seqs_n5000_T1.0', 'QVAE (ld128)', '#8e44ad') 
]

dfs = []
label_colors = {} # 用于箱线图填充
colors_dict = {}  # 用于直方图线条

print("--- 开始加载序列数据 ---")

for path, label, color in datasets_config:
    # 尝试加载
    print(f"Loading {label}...")
    temp_df = pd.DataFrame(read_fasta(path), columns=['Name', 'Sequence'])
    
    if not temp_df.empty:
        temp_df['Label'] = label
        dfs.append(temp_df)
        label_colors[label] = color # 记录颜色
        colors_dict[label] = color
    else:
        print(f"警告: 数据集 {label} ({path}) 未加载到任何数据或文件不存在。")

# 合并所有数据集
if not dfs:
    print("错误: 没有加载到任何有效数据。程序退出。")
    sys.exit()

df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)


# --- 3. Biopython Property Calculation (Optimized) ---

print("\nCalculating Biopython properties (Charge, GRAVY, Eisenberg)...")
#  使用 .apply() 代替慢速 for 循环
bio_props_df = df['Sequence'].apply(calculate_bio_properties).apply(pd.Series)
bio_props_df.columns = ['Net Charge', 'GRAVY', 'Eisenberg hydrophobicity']
df = pd.concat([df, bio_props_df], axis=1)

print("Calculating Amino Acid fractions...")
# 计算氨基酸组成
aa_fraction_df = df['Sequence'].apply(calculate_amino_acid_fraction).apply(pd.Series)
# 确保所有20种标准氨基酸都存在
all_aas = list('ACDEFGHIKLMNPQRSTVWY')
for aa in all_aas:
    if aa not in aa_fraction_df.columns:
        aa_fraction_df[aa] = 0.0
# 按字母顺序排序
amino_acid_columns = sorted([col for col in aa_fraction_df.columns if col in all_aas])
df = pd.concat([df, aa_fraction_df[amino_acid_columns]], axis=1)


# 计算长度
df['Length'] = df['Sequence'].apply(len)

# --- 4. Plotting Setup ---
print("\nGenerating plots...")
# 确保标签顺序与 config 一致，保持图例整洁
labels = [cfg[1] for cfg in datasets_config if cfg[1] in df['Label'].unique()]

# 确保输出目录存在
os.makedirs('data/analysis', exist_ok=True)

# --- 5. Plots (Using Optimized sns.histplot) ---

# Net charge
plt.figure()
for label in labels:
    data = df[df['Label'] == label]['Net Charge'].dropna()
    if data.empty: continue
    sns.histplot(data, label=label, bins=15, kde=True, stat="density", 
                 color=colors_dict[label], element="step", kde_kws={'bw_adjust': 0.6}) 

plt.xlabel('Net Charge')
plt.ylabel('Density')
plt.legend()
plt.savefig('data/analysis/diversity_net_charge_comparison.png', dpi = 400, bbox_inches = "tight")
plt.clf()

# Amino acid composition (动态支持任意数量数据集)
plt.figure(figsize=(15, 6)) # 数据集变多，画布加宽
positions = list(range(1, len(amino_acid_columns) + 1))
label_list = list(labels) 

# 动态计算箱子宽度和间隙
N = len(label_list)
total_width_per_group = 0.85 # 稍微紧凑一点
widths = total_width_per_group / N 

legend_elements = []
for i, label in enumerate(label_list):
    # 计算每个箱子的位置偏移
    shift = (i - (N - 1) / 2.0) * widths
    
    data_to_plot = df[df['Label'] == label][amino_acid_columns].dropna()
    if data_to_plot.empty: continue
    
    plt.boxplot(data_to_plot, 
                positions=[pos + shift for pos in positions], 
                widths=widths * 0.95, # 留一点小缝隙
                patch_artist=True, 
                boxprops=dict(facecolor= label_colors[label], alpha=0.8), 
                flierprops={'markersize': 1})
    
    legend_elements.append(Patch(facecolor= label_colors[label], label=label, alpha=0.8))
    
plt.xticks(range(1, len(amino_acid_columns) + 1), amino_acid_columns)
plt.xlabel('Amino Acids')
plt.ylabel('Fraction')
plt.grid(visible=False, axis='both')
plt.legend(handles=legend_elements)
plt.savefig('data/analysis/diversity_aa_fraction_comparison.png', dpi = 400, bbox_inches = "tight")
plt.clf()

# GRAVY
plt.figure()
for label in labels:
    data = df[df['Label'] == label]['GRAVY'].dropna()
    if data.empty: continue
    sns.histplot(data, label=label, bins=15, kde=True, stat="density",
                 color=colors_dict[label], element="step")

plt.xlabel('GRAVY')
plt.ylabel('Density')
plt.legend()
plt.savefig('data/analysis/diversity_gravy_comparison.png', dpi = 400, bbox_inches = "tight")
plt.clf()

# Eisenberg hydrophobicity
plt.figure()
for label in labels:
    data = df[df['Label'] == label]['Eisenberg hydrophobicity'].dropna()
    if data.empty: continue
    sns.histplot(data, label=label, bins=15, kde=True, stat="density",
                 color=colors_dict[label], element="step")

plt.xlabel('Eisenberg hydrophobicity')
plt.ylabel('Density')
plt.legend()
plt.savefig('data/analysis/diversity_Eisenberg_hydrophobicity_comparison.png', dpi = 400, bbox_inches = "tight")
plt.clf()

# Length
plt.figure()
for l in labels:
    class_data = df[df['Label'] == l]
    if class_data.empty: continue
    sns.histplot(class_data['Length'], bins=15, kde=True, stat="density",
                 label=l, color=colors_dict[l], element="step")

plt.xlabel('Length')
plt.ylabel('Density')
plt.legend()
plt.savefig('data/analysis/diversity_Length_comparison.png', dpi = 400, bbox_inches = "tight")
plt.clf()

# --- 6. s4pred secondary structure ---
print("\nCalculating Secondary Structure properties (s4pred)...")

# --- 改进策略: 加载所有可能的 s4pred 文件并合并 ---
# 您可以在这里添加更多可能的文件路径
possible_s4pred_files = [
    # 1. 通用文件 (可能包含所有数据)
    'data/diversity_cluster_ss.fas',
    
    # 2. 特定的 QVAE 结果文件 (如果有单独生成)
    'data/qvae_bm/b2048_ld32_beta0.1/output/generated_seqs_n5000_T1.0_ss.fas',
    'data/qvae_bm/b2048_ld64_beta0.1/output/generated_seqs_n5000_T1.0_ss.fas',
    'data/qvae_bm/b2048_ld128_beta0.1/output/generated_seqs_n5000_T1.0_ss.fas'
]

all_s4pred_dfs = []

# 加载所有存在的结构文件
for filepath in possible_s4pred_files:
    ss_df = read_s4pred_fasta(filepath)
    if not ss_df.empty:
        all_s4pred_dfs.append(ss_df)

if not all_s4pred_dfs:
    print("警告: 未能加载任何 s4pred SS 数据。跳过二级结构绘图。")
else:
    # 合并为一个大的查找表 (drop_duplicates 防止同一个序列在多个文件中导致重复)
    s4pred_df_total = pd.concat(all_s4pred_dfs, ignore_index=True)
    # 优先保留最新的记录 (如果有重复名)
    s4pred_df_total = s4pred_df_total.drop_duplicates(subset='Name', keep='last')
    
    print(f"  总共加载了 {len(s4pred_df_total)} 条唯一的二级结构记录。")

    # 应用名称标准化
    s4pred_df_total['Name'] = s4pred_df_total['Name'].apply(lowercase_sample)
    df['Name'] = df['Name'].apply(lowercase_sample)

    # 合并: 将结构信息映射到主 DataFrame
    combined_df = df.merge(s4pred_df_total[['Name', 'Structure']], on='Name', how='left')

    # 检查哪些标签完全缺失结构信息
    missing_structure_labels = []
    for label in labels:
        if combined_df[combined_df['Label'] == label]['Structure'].isnull().all():
            missing_structure_labels.append(label)
    
    if missing_structure_labels:
        print(f"  注意: 以下数据集在 s4pred 文件中未找到匹配的序列: {missing_structure_labels}")
        print("  请检查是否已为这些序列运行 s4pred 并将结果保存到 data/diversity_cluster_ss.fas 或对应的 _ss.fas 文件中。")

    # 计算 SS 属性
    # 仅计算非空的
    valid_mask = combined_df['Structure'].notna()
    if valid_mask.any():
        ss_props = combined_df.loc[valid_mask, 'Structure'].apply(calculate_secondary_structure_percentages).apply(pd.Series)
        ss_props.columns = ['Coil', 'Helix', 'Strand']
        
        # 将计算结果合并回去
        combined_df = pd.concat([combined_df, ss_props], axis=1)
    else:
        combined_df['Coil'] = np.nan
        combined_df['Helix'] = np.nan
        combined_df['Strand'] = np.nan

    # --- 绘制二级结构图 ---
    plt.figure()
    positions = [1, 2, 3]
    ss_columns = ['Coil', 'Helix', 'Strand']
    
    # 仅绘制有数据的标签
    ss_plot_labels = [l for l in labels if not combined_df[combined_df['Label'] == l][ss_columns].dropna().empty]
    
    if len(ss_plot_labels) > 0:
        N = len(ss_plot_labels)
        # 动态调整宽度
        total_width = 0.85
        widths = total_width / N

        legend_elements = []

        for i, label in enumerate(ss_plot_labels):
            data_to_plot = combined_df[combined_df['Label'] == label].dropna(subset=ss_columns)
            if data_to_plot.empty: continue

            shift = (i - (N - 1) / 2.0) * widths
            
            plt.boxplot(data_to_plot[ss_columns], 
                        positions=[pos + shift for pos in positions], 
                        widths=widths * 0.95, 
                        patch_artist=True, 
                        boxprops=dict(facecolor=label_colors[label], alpha=0.8), 
                        flierprops={'markersize': 2})
            
            legend_elements.append(Patch(facecolor= label_colors[label], label=label, alpha=0.8))

        plt.xticks(positions, ss_columns)
        plt.xlabel("Secondary Structure")
        plt.ylabel('Percentage')
        plt.grid(visible=False, axis='both')
        plt.legend(handles=legend_elements)

        plt.savefig('data/analysis/diversity_ss_fraction_comparison.png', dpi = 400, bbox_inches = "tight")
        print("二级结构图已保存。")
    else:
        print("合并后没有有效的二级结构数据可供绘图。")

print("\n--- 脚本执行完毕 ---")
print("所有分析图表已保存至 data/analysis/ 目录。")