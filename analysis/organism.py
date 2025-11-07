## Analyzing peptide composition, net charge, secondary structure properties 
# Modules 
import numpy as np
import pandas as pd 
import re
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from matplotlib.patches import Patch
# 移除了未使用的 modlamp.descriptors

# --- 1. Functions ---

def read_fasta(name):
    """从 FASTA 文件读取数据"""
    filepath = f'{name}.fasta'
    print(f"Reading FASTA file: {filepath}")
    data = []
    # 移除了 try-except
    for fasta in SeqIO.parse(open(filepath),'fasta'):
        data.append([fasta.id, str(fasta.seq).strip()])
            
    return data

def calculate_amino_acid_fraction(peptide):
    """[优化] 计算单个肽的氨基酸组成百分比"""
    # 移除了 try-except
    prot_param = ProteinAnalysis(str(peptide))
    # [修复] 修复 BiopythonDeprecationWarning
    return prot_param.amino_acids_percent

def get_color_by_charge(amino_acid):
    # (此函数未在脚本中使用，但保留)
    if amino_acid in ['R', 'K', 'H']:
        return 'red'  # Positive charge
    elif amino_acid in ['D', 'E']:
        return 'blue'  # Negative charge
    else:
        return 'gray'  # Neutral charge

def validate(seq, pattern=re.compile(r'^[FIWLVMYCATHGSQRKNEPD]+$')):
    # (此函数已被优化的 clean() 取代)
    if (pattern.match(seq)):
        return True
    return False

def clean(sequence_df, pattern=re.compile(r'^[FIWLVMYCATHGSQRKNEPD]+$')):
    """[优化] 使用矢量化操作清理 DataFrame"""
    initial_count = len(sequence_df)
    
    # 使用 str.match 进行矢量化正则表达式匹配
    valid_mask = sequence_df['Sequence'].str.match(pattern).fillna(False)
    
    dropped_count = initial_count - valid_mask.sum()
    print(f'Total number of sequences dropped: {dropped_count}')
    
    cleaned_df = sequence_df[valid_mask].reset_index(drop=True)
    print(f'Total number of sequences remaining: {len(cleaned_df)}')
    
    return cleaned_df

def calculate_bio_properties(sequence):
    """
    [优化] 辅助函数：用于 .apply()，一次性计算所有 Biopython 属性
    """
    # 移除了 try-except
    prot_seq = ProteinAnalysis(str(sequence))
    net_charge = prot_seq.charge_at_pH(7.0)
    gravy = prot_seq.gravy()
    eisenberg_hydrophobicity = prot_seq.gravy(scale='Eisenberg')
    return net_charge, gravy, eisenberg_hydrophobicity

def calculate_secondary_structure_percentages(structure):
    """[优化] 计算二级结构百分比 (C, H, E)"""
    # 移除了鲁棒性检查 (if not isinstance)
        
    length = len(structure)
    if length == 0: 
        return 0.0, 0.0, 0.0
        
    c_count = structure.count("C")
    h_count = structure.count("H")
    e_count = structure.count("E")
    
    c_percentage = (c_count / length) * 100
    h_percentage = (h_count / length) * 100
    e_percentage = (e_count / length) * 100
    
    return c_percentage, h_percentage, e_percentage

def lowercase_sample(name):
    """辅助函数：标准化 'SAMPLE' 名称 (用于 s4pred 合并)"""
    if 'SAMPLE' in name:
        return name.lower()
    return name

# --- 2. Data Loading & Cleaning ---

human_df = pd.DataFrame(read_fasta('data/human_tp_cd_hit_cluster'), columns = ['Name','Sequence'])
mouse_df = pd.DataFrame(read_fasta('data/mouse_tp_cd_hit_cluster'), columns = ['Name','Sequence'])
yeast_df = pd.DataFrame(read_fasta('data/yeast_tp_cd_hit_cluster'), columns = ['Name','Sequence'])
amts_df = pd.DataFrame(read_fasta('scripts/qvae-v/data/qvae-fc/b2048_ld32_beta0.1/output/generated_seqs_fc_n5000_T1.0'), columns = ['Name','Sequence'])

print("\nCleaning Human data...")
human_df = clean(human_df)
print("\nCleaning Mouse data...")
mouse_df = clean(mouse_df)
print("\nCleaning Yeast data...")
yeast_df = clean(yeast_df)
print("\nCleaning AMTS (generated) data...")
# [修复] 对 AMTS 应用同样的清理
amts_df = clean(amts_df)

human_df['Label'] = 1 #'Human'
mouse_df['Label'] = 2 #'Mouse'
yeast_df['Label'] = 3 #'Yeast'
amts_df['Label'] = 4 #'AMTS'

df = pd.concat([human_df, mouse_df, yeast_df, amts_df], ignore_index=True).reset_index(drop = True)

# --- 3. Property Calculation (Optimized) ---

print("\nCalculating Biopython properties (Charge, GRAVY, Eisenberg)...")
# [优化] 使用 .apply() 代替慢速 for 循环 (Biopython)
bio_props_df = df['Sequence'].apply(calculate_bio_properties).apply(pd.Series)
bio_props_df.columns = ['Net Charge', 'GRAVY', 'Eisenberg hydrophobicity']
df = pd.concat([df, bio_props_df], axis=1)

print("Calculating Amino Acid fractions...")
# [优化] 使用 .apply() 代替慢速 for 循环 (AA Fraction)
aa_fraction_df = df['Sequence'].apply(calculate_amino_acid_fraction).apply(pd.Series)
amino_acid_columns = aa_fraction_df.columns # 获取实际的氨基酸列名
df = pd.concat([df, aa_fraction_df], axis=1)

# 计算长度 (这已经是最高效的方式)
df['Length'] = df['Sequence'].apply(len)


# --- 4. s4pred secondary structure (Optimized) ---

s4pred_filepath = "data/organism_cluster_ss.fas"
data = {"Name": [], "Sequence": [], "Structure": []}

# 移除了 try-except
print(f"\nReading Secondary Structure file: {s4pred_filepath}")
with open(s4pred_filepath, "r") as file:
    lines = file.readlines()

for i in range(0, len(lines), 3):
    # 移除了不完整记录检查
    protein_name = lines[i].strip()[1:]
    protein_sequence = lines[i + 1].strip()
    protein_structure = lines[i + 2].strip()
    data["Name"].append(protein_name)
    data["Sequence"].append(protein_sequence)
    data["Structure"].append(protein_structure)

s4pred_df = pd.DataFrame(data)

# 应用名称标准化
s4pred_df['Name'] = s4pred_df['Name'].apply(lowercase_sample)

# 合并
df = df.merge(s4pred_df[['Name', 'Structure']], on='Name', how='left')

print("Calculating Secondary Structure properties (s4pred)...")
# [优化] 使用 .apply() 代替慢速 for 循环 (s4pred)
ss_props_df = df['Structure'].apply(calculate_secondary_structure_percentages).apply(pd.Series)
ss_props_df.columns = ['Coil', 'Helix', 'Strand']

# 在合并回 df 之前，删除可能因 .apply() 产生的旧列（如果存在）
df = df.drop(columns=['Coil', 'Helix', 'Strand'], errors='ignore')
df = pd.concat([df, ss_props_df], axis=1)


# --- 5. Plotting ---

print("Generating plots...")

# 定义标签和颜色 (更稳健的方式)
label_map = {1: 'Human', 2: 'Mouse', 3: 'Yeast', 4: 'AMTS'}
color_map = {1: 'lightblue', 2: 'bisque', 3: '#A2DEA5', 4: 'mistyrose'}
labels = [label_map[l] for l in sorted(df['Label'].unique())] # 确保顺序

# Amino acid composition
plt.figure(figsize=(12, 4.8))
positions = list(range(1, len(amino_acid_columns) + 1))
gap = 0.4
widths = 0.15

# 移除了 .dropna()
data_to_plot_1 = df[df['Label'] == 1]
data_to_plot_2 = df[df['Label'] == 2]
data_to_plot_3 = df[df['Label'] == 3]
data_to_plot_4 = df[df['Label'] == 4]

plt.boxplot(data_to_plot_1[amino_acid_columns], positions=[pos-1.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_map[1]), flierprops={'markersize': 2})
plt.boxplot(data_to_plot_2[amino_acid_columns], positions=[pos-0.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_map[2]), flierprops={'markersize': 2})
plt.boxplot(data_to_plot_3[amino_acid_columns], positions=[pos+0.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_map[3]), flierprops={'markersize': 2})
plt.boxplot(data_to_plot_4[amino_acid_columns], positions=[pos+1.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_map[4]), flierprops={'markersize': 2})

plt.xticks(range(1, len(amino_acid_columns) + 1), amino_acid_columns)
plt.xlabel('Amino Acids')
plt.ylabel('Fraction')
plt.grid(visible=False, axis='both')
legend_elements = [Patch(facecolor=color_map[k], label=label_map[k]) for k in sorted(color_map.keys())]
plt.legend(handles=legend_elements)
plt.savefig('data/organism_aa_fraction_biopython.png', dpi = 400, bbox_inches = "tight")
plt.clf()

# Secondary structure
# 移除了 if 'Coil' in df.columns 检查
plt.figure()
positions = [1, 2, 3]
gap = 0.2
widths = 0.15

# 移除了 .dropna()
data_to_plot_1 = df[df['Label'] == 1]
data_to_plot_2 = df[df['Label'] == 2]
data_to_plot_3 = df[df['Label'] == 3]
data_to_plot_4 = df[df['Label'] == 4]

plt.boxplot(data_to_plot_1[['Coil','Helix','Strand']], positions=[pos-1.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_map[1]), flierprops={'markersize': 2})
plt.boxplot(data_to_plot_2[['Coil','Helix','Strand']], positions=[pos-0.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_map[2]), flierprops={'markersize': 2})
plt.boxplot(data_to_plot_3[['Coil','Helix','Strand']], positions=[pos+0.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_map[3]), flierprops={'markersize': 2})
plt.boxplot(data_to_plot_4[['Coil','Helix','Strand']], positions=[pos+1.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_map[4]), flierprops={'markersize': 2})
    
plt.xticks(positions, ['Coil', 'Helix', 'Strand'])
plt.xlabel("Secondary Structure")
plt.ylabel('Percentage')
plt.grid(visible=False, axis='both')
plt.legend(handles=legend_elements)
plt.savefig('data/organism_ss_fraction_s4pred.png', dpi = 400, bbox_inches = "tight")
plt.clf()

# 绘制其他属性 (Length, Net Charge, GRAVY, Eisenberg)
plot_properties = ['Length', 'Net Charge', 'GRAVY', 'Eisenberg hydrophobicity']
plot_colors_hist = {1: '#89cff0', 2: '#faddb3', 3: '#58a365', 4: '#d4664f'} # 用于 histplot 的更深的颜色

for prop in plot_properties:
    plt.figure()
    for label_id, label_name in label_map.items():
        # 移除了 .dropna() 和 if not data.empty
        data = df[df['Label'] == label_id][prop]
        # [修复] 使用 sns.histplot 代替 sns.distplot
        sns.histplot(data, bins=10, kde=True, stat="density",
                     label=label_name, color=plot_colors_hist[label_id], element="step")
    
    plt.xlabel(prop)
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'data/organism_{prop}_biopython.png', dpi = 400, bbox_inches = "tight")
    plt.clf()

print("\n--- 脚本执行完毕 ---")

