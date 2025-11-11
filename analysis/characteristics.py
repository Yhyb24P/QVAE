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
import pickle # <--- 导入 PICKLE

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
    filepath = f'{name}.fasta'
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
    try:
        with open(filepath, "r") as file:
            lines = file.readlines()

        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                print(f"  警告: s4pred 文件 {filepath} 在第 {i} 行处记录不完整。")
                continue
                
            protein_name = lines[i].strip()[1:]
            protein_sequence = lines[i + 1].strip()
            protein_structure = lines[i + 2].strip()

            data["Name"].append(protein_name)
            data["Sequence"].append(protein_sequence)
            data["Structure"].append(protein_structure)
    
    except Exception as e:
        print(f"  读取 s4pred 文件 {filepath} 时出错: {e}")
        return pd.DataFrame(data) # 返回空/部分数据
        
    return pd.DataFrame(data)


# --- 2. Data Loading ---

# --- 数据集 1 (AMTS(QVAE)) ---
# QVAE 序列文件
qvae_seq_path = 'data/qvae/b2048_ld32_beta0.1/output/generated_seqs_n5000_T1.0'
amts_df = pd.DataFrame(read_fasta(qvae_seq_path), columns = ['Name','Sequence'])
amts_df['Label'] = 'AMTS(QVAE)' 

# --- 数据集 2 (AMTS(VAE)) ---
# VAE 序列文件
vae_seq_path = 'data/vae/output/amts'
print(f"Loading VAE data from FASTA: {vae_seq_path}.fasta") 
mts_df = pd.DataFrame(read_fasta(vae_seq_path), columns = ['Name','Sequence'])
mts_df['Label'] = 'AMTS(VAE)'

# --- 数据集 3 (Train Data) ---
# 训练集 序列文件
new_data_name = 'data/mts_train'    
new_data_label = 'train_data'        
original_df = pd.DataFrame(read_fasta(new_data_name), columns = ['Name','Sequence'])
print(f"Loading NEW data from FASTA: {new_data_name}.fasta")
original_df['Label'] = new_data_label

# --- 合并所有数据集 ---
df_list = [amts_df, mts_df, original_df]
df = pd.concat([d for d in df_list if not d.empty], ignore_index=True).reset_index(drop = True)

# --- 3. Biopython Property Calculation (Optimized) ---

print("Calculating Biopython properties (Charge, GRAVY, Eisenberg)...")
#  使用 .apply() 代替慢速 for 循环
bio_props_df = df['Sequence'].apply(calculate_bio_properties).apply(pd.Series)
bio_props_df.columns = ['Net Charge', 'GRAVY', 'Eisenberg hydrophobicity']
df = pd.concat([df, bio_props_df], axis=1)

print("Calculating Amino Acid fractions...")
# 计算氨基酸组成
aa_fraction_df = df['Sequence'].apply(calculate_amino_acid_fraction).apply(pd.Series)
# 确保所有20种标准氨基酸都存在，以防某些序列导致列缺失
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
print("Generating plots...")
labels = np.unique(df['Label']) # 自动获取所有加载的标签


label_colors = {
    'AMTS(VAE)': '#A2DEA5',     
    'AMTS(QVAE)': 'mistyrose',  
    'train_data': '#A2A9DE'     
}
colors_dict = {
    'AMTS(VAE)': '#58a365',    
    'AMTS(QVAE)': '#d4664f',  
    'train_data': '#5865a3'     
}


# --- 5. Plots (Using Optimized sns.histplot) ---

# Net charge
plt.figure()
for label in labels:
    if label not in colors_dict: continue # 跳过未定义颜色的标签
    data = df[df['Label'] == label]['Net Charge'].dropna()
    if data.empty: continue
    sns.histplot(data, label=label, bins=10, kde=True, stat="density", 
                 color=colors_dict[label], element="step", kde_kws={'bw_adjust': 0.5}) 

plt.xlabel('Net Charge')
plt.ylabel('Density') # Y 轴现在是 'Density' (密度)
plt.legend()
plt.savefig('data/analysis/diversity_net_charge_biopython_3way.png', dpi = 400, bbox_inches = "tight")
plt.clf()

# Amino acid composition (为支持3个数据集)
plt.figure(figsize=(12, 4.8))
positions = list(range(1, len(amino_acid_columns) + 1))
gap = 0.4 # 加大总间隙
widths = 0.12 # 减小每个箱子的宽度
label_list = list(labels) # 获取标签的固定顺序

# 动态绘制所有标签的箱线图
legend_elements = []
for i, label in enumerate(label_list):
    if label not in label_colors: continue
    
    # 计算每个箱子的位置

    N = len(label_list)
    shift = (i - (N - 1) / 2.0) * (widths + 0.01) # 0.01 是箱子间的小间隙
    
    data_to_plot = df[df['Label'] == label][amino_acid_columns].dropna()
    if data_to_plot.empty: continue
    
    plt.boxplot(data_to_plot, 
                positions=[pos + shift for pos in positions], 
                widths=widths, 
                patch_artist=True, 
                boxprops=dict(facecolor= label_colors[label]), 
                flierprops={'markersize': 2})
    
    legend_elements.append(Patch(facecolor= label_colors[label], label=label))
    
plt.xticks(range(1, len(amino_acid_columns) + 1), amino_acid_columns)
plt.xlabel('Amino Acids')
plt.ylabel('Fraction')
plt.grid(visible=False, axis='both')
plt.legend(handles=legend_elements)
plt.savefig('data/analysis/diversity_aa_fraction_biopython_3way.png', dpi = 400, bbox_inches = "tight")
plt.clf()

# GRAVY
plt.figure(figsize=(6.4, 4.8))
for label in labels:
    if label not in colors_dict: continue
    data = df[df['Label'] == label]['GRAVY'].dropna()
    if data.empty: continue
    sns.histplot(data, label=label, bins=10, kde=True, stat="density",
                 color=colors_dict[label], element="step")

plt.xlabel('GRAVY')
plt.ylabel('Density')
plt.legend()
plt.savefig('data/analysis/diversity_gravy_biopython_3way.png', dpi = 400, bbox_inches = "tight")
plt.clf()

# Eisenberg hydrophobicity
plt.figure()
for label in labels:
    if label not in colors_dict: continue
    data = df[df['Label'] == label]['Eisenberg hydrophobicity'].dropna()
    if data.empty: continue
    sns.histplot(data, label=label, bins=10, kde=True, stat="density",
                 color=colors_dict[label], element="step")

plt.xlabel('Eisenberg hydrophobicity')
plt.ylabel('Density')
plt.legend()
plt.savefig('data/analysis/diversity_Eisenberg_hydrophobicity_biopython_3way.png', dpi = 400, bbox_inches = "tight")
plt.clf()

# Length
plt.figure()
for l in labels:
    if l not in colors_dict: continue
    class_data = df[df['Label'] == l]
    if class_data.empty: continue
    sns.histplot(class_data['Length'], bins=10, kde=True, stat="density",
                 label=l, color=colors_dict[l], element="step")

plt.xlabel('Length')
plt.ylabel('Density') # Y 轴是 'Density'
plt.legend()
plt.savefig('data/analysis/diversity_Length_3way.png', dpi = 400, bbox_inches = "tight")
plt.clf()

# --- 6. s4pred secondary structure ---
print("Calculating Secondary Structure properties (s4pred)...")

# 路径基于第 2 节中加载序列文件
s4pred_files_to_load = {
    'AMTS(QVAE)': 'data/diversity_cluster_ss.fas',
    'AMTS(VAE)': 'data/diversity_cluster_ss.fas', 
    'train_data': 'data/diversity_cluster_ss.fas'
}
all_s4pred_dfs = []
# 仅加载已加载数据集的 SS 文件
for label in labels: 
    if label not in s4pred_files_to_load:
        print(f"警告: 标签 '{label}' 没有定义的 s4pred SS 文件路径。")
        continue
    
    filepath = s4pred_files_to_load[label]
    
    if not os.path.exists(filepath):
        print(f"警告: s4pred 文件未找到: {filepath}。将跳过 '{label}' 的二级结构分析。")
        continue
    
    print(f"Loading SS data for '{label}' from: {filepath}")
    ss_df = read_s4pred_fasta(filepath)
    if not ss_df.empty:
        all_s4pred_dfs.append(ss_df)
    else:
        print(f"  警告: {filepath} 为空或读取失败。")

if not all_s4pred_dfs:
    print("警告: 未能加载任何 s4pred SS 数据。跳过二级结构分析。")
    combined_df = df.copy() # 复制df以保持流程
    combined_df['Coil'] = np.nan
    combined_df['Helix'] = np.nan
    combined_df['Strand'] = np.nan
else:
    # 成功加载了至少一个 SS 文件
    s4pred_df = pd.concat(all_s4pred_dfs, ignore_index=True)
    
    # 应用名称标准化 (与主 df 相同)
    s4pred_df['Name'] = s4pred_df['Name'].apply(lowercase_sample)
    df['Name'] = df['Name'].apply(lowercase_sample)

    # 合并
    combined_df = df.merge(s4pred_df[['Name', 'Structure']], on='Name', how='left')

    # 检查合并是否成功 (即，是否有任何非 NaN 的 Structure)
    if combined_df['Structure'].isnull().all():
        print("警告: s4pred 数据已加载，但没有一个 'Name' 成功匹配主数据集。")
        print("  请检查 .fas 文件中的序列名称是否与 .fasta 文件中的序列名称一致。")
        combined_df['Coil'] = np.nan
        combined_df['Helix'] = np.nan
        combined_df['Strand'] = np.nan
    else:
        # 计算 SS 属性
        ss_props_df = combined_df['Structure'].apply(calculate_secondary_structure_percentages).apply(pd.Series)
        ss_props_df.columns = ['Coil', 'Helix', 'Strand']

        combined_df = combined_df.drop(columns=['Coil', 'Helix', 'Strand'], errors='ignore')
        combined_df = pd.concat([combined_df, ss_props_df], axis=1)


# --- 绘制二级结构图 (为支持3个数据集) ---
plt.figure()
positions = [1, 2, 3]
ss_columns = ['Coil', 'Helix', 'Strand']
label_list = list(labels) # 获取标签的固定顺序

# 动态计算位置
N = len(label_list)
gap = 0.4 # 总间隙
widths = max(0.05, (gap * 0.8) / N) # 动态计算宽度，并设置一个最小值

legend_elements = []

for i, label in enumerate(label_list):
    if label not in label_colors: continue
    
    data_to_plot = combined_df[combined_df['Label'] == label].dropna(subset=ss_columns)
    if data_to_plot.empty:
        # 即使文件加载了，也可能因为名称不匹配而没有数据
        print(f"警告: 标签 '{label}' 没有有效的二级结构数据用于绘图。")
        continue

    # 计算位移
    shift = (i - (N - 1) / 2.0) * (widths + 0.01) # 0.01 是箱子间的小间隙
    
    plt.boxplot(data_to_plot[ss_columns], 
                positions=[pos + shift for pos in positions], 
                widths=widths, 
                patch_artist=True, 
                boxprops=dict(facecolor=label_colors[label]), 
                flierprops={'markersize': 2})
    
    legend_elements.append(Patch(facecolor= label_colors[label], label=label))

    
plt.xticks(positions, ss_columns)
plt.xlabel("Secondary Structure")
plt.ylabel('Percentage')
plt.grid(visible=False, axis='both')

if legend_elements: # 仅当有图例时才显示
    plt.legend(handles=legend_elements)

plt.savefig('data/analysis/diversity_ss_fraction_s4pred_3way.png', dpi = 400, bbox_inches = "tight")
plt.clf()

print("--- 脚本执行完毕 ---")