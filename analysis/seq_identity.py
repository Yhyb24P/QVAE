import pandas as pd
import numpy as np
import re
from Bio import SeqIO
import seaborn as sns
import matplotlib.pyplot as plt
from Levenshtein import distance as lv
from tqdm import tqdm
import pickle
import multiprocessing
from functools import partial
import time
import os

# --- 1. Functions (Optimized) ---

def validate(seq, pattern=re.compile(r'^[FIWLVMYCATHGSQRKNEPD]+$')):
    """验证序列是否仅包含标准氨基酸"""
    if (pattern.match(seq)):
        return True
    return False

def clean(sequence_df):
    """清理非标准序列"""
    print(f"Cleaning sequences... Initial count: {len(sequence_df)}")
    valid_mask = sequence_df['sequence'].str.match(r'^[FIWLVMYCATHGSQRKNEPD]+$').fillna(False)
    cleaned_df = sequence_df[valid_mask].reset_index(drop=True)
    print(f'Total number of sequences remaining: {len(cleaned_df)}')
    return cleaned_df

def read_fasta(name):
    """读取 FASTA 文件"""
    # 自动处理是否有 .fasta 后缀
    if not name.endswith('.fasta'):
        filepath = name + '.fasta'
    else:
        filepath = name
        
    print(f"Reading FASTA file: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return []

    data = []
    # eGFP 序列 (用于移除)
    egfp = 'VSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK'
    
    try:
        fasta_seqs = SeqIO.parse(open(filepath),'fasta')
        for fasta in fasta_seqs:
            seq_cleaned = str(fasta.seq).strip().replace(egfp,'')
            data.append([fasta.id, seq_cleaned])
    except Exception as e:
        print(f"Error reading FASTA: {e}")
        return []
    
    print(f"Read {len(data)} sequences from FASTA.")
    return data

def parse_uniprot_row(row):
    """
    解析 UniProt Excel 行。
    假设列结构: [0]Entry, [1]Entry Name, [2]Sequence, [3]Features
    """
    try:
        # 注意：这里假设列索引为 0, 2, 3。如果 Excel 结构不同，需要修改这里
        col_name = row[0]
        col_seq = row[2]
        col_features = str(row[3]) # 确保转为字符串
        
        if 'mitochondrion' not in col_features.lower():
            return None
        
        # 提取 Transit peptide 位置
        # --- 修复更新: 兼容 'TRANSIT 1..97' 和 'Transit peptide 1..97' ---
        # 正则解释:
        # transit          匹配 "transit" (忽略大小写)
        # (?:\s+peptide)?  可选匹配 " peptide" (忽略大小写)
        # \s+              匹配空格
        # \d+\.\.          匹配 "数字.." (起始位置)
        # (\d+)            捕获组 1: 结束位置
        tp_match = re.search(r'transit(?:\s+peptide)?\s+\d+\.\.(\d+)', col_features, re.IGNORECASE)
        
        # 如果第一种正则失败，尝试更宽松的匹配（针对某些包含非标准字符的情况）
        if not tp_match:
             # 备用方案
             tp_match = re.search(r'transit.*?\.\.(\d+)', col_features, re.IGNORECASE)
        
        if tp_match:
            tp_end = int(tp_match.group(1))
        else:
            return None

        if tp_end <= 5: 
            return None
            
        tp_seq = col_seq[:tp_end]
        
        if not validate(tp_seq):
            return None
            
        return [col_name, tp_seq]
        
    except Exception as e:
        return None

def find_min_distance(query_seq, target_sequences):
    """计算最小编辑距离"""
    if not query_seq: return np.inf
    min_dist = np.inf
    for target_seq in target_sequences:
        dist = lv(query_seq, target_seq)
        if dist < min_dist:
            min_dist = dist
    return min_dist

# --- 2. Main Execution ---

if __name__ == '__main__':
    print("--- 启动序列相似性分析 ---")
    start_time = time.time()

    # --- UniProt 数据处理 ---
    print("Loading and parsing UniProt data...")
    uniprot_file = 'data/uniprot_transit_peptide.xlsx'
    
    if os.path.exists(uniprot_file):
        # 读取 Excel
        uniprot_raw = pd.read_excel(uniprot_file, header=None)
        
        # --- DEBUG: 打印 Excel 结构以供检查 ---
        print(f"  [DEBUG] Excel Shape: {uniprot_raw.shape}")
        # print(f"  [DEBUG] Row 0 (Header?): {uniprot_raw.iloc[0].tolist()}")
        # print(f"  [DEBUG] Row 1 (Data?):   {uniprot_raw.iloc[1].tolist()}")
        # ------------------------------------

        # 解析数据
        parsed_data = uniprot_raw.iloc[1:].apply(parse_uniprot_row, axis=1).dropna().tolist()
        uniprot_tp = pd.DataFrame(parsed_data, columns = ['name', 'sequence'])
        
        print(f'Total valid sequences parsed: {len(uniprot_tp)}')
        
        if len(uniprot_tp) > 0:
            uniprot_tp = uniprot_tp.drop_duplicates(subset='sequence').reset_index(drop=True)
            print(f'Total sequences remaining after duplicate removal: {len(uniprot_tp)}')
        else:
            print("⚠️ 警告: 未能解析出任何 UniProt 序列。请检查上方的 [DEBUG] 信息和 Excel 列结构。")
    else:
        print(f"Error: UniProt file not found: {uniprot_file}")
        uniprot_tp = pd.DataFrame(columns=['name', 'sequence'])

    # --- VAE 和 训练数据 加载 ---
    print("Loading VAE generated sequences...")
    # 更新为您日志中的路径 (ld32)
    vae_seq_path = 'data/vae/output/amts'
    vae_tp = pd.DataFrame(read_fasta(vae_seq_path), columns = ['name','sequence'])
    
    if vae_tp.empty:
        print("Error: No VAE sequences loaded. Exiting.")
        exit()

    print("Loading training data (X_train)...")
    try:
        with open('data/tv_sim_split_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        X_train = pd.DataFrame(columns=['sequence'])

    # --- 3. 并行计算 Levenshtein 距离 ---
    query_vae_seqs = vae_tp['sequence'].tolist()
    target_uniprot_seqs = uniprot_tp['sequence'].tolist()
    target_train_seqs = X_train['sequence'].tolist()

    num_queries = len(query_vae_seqs)
    print(f"Starting parallel distance calculation for {num_queries} query sequences...")

    n_cores = multiprocessing.cpu_count()
    print(f"Using {n_cores} CPU cores for parallel processing.")

    min_lev_h = []
    min_lev = []

    with multiprocessing.Pool(processes=n_cores) as pool:
        
        # --- 任务 1: VAE vs UniProt ---
        if len(target_uniprot_seqs) > 0:
            print(f"Calculating distances to {len(target_uniprot_seqs)} UniProt sequences...")
            task_h = partial(find_min_distance, target_sequences=target_uniprot_seqs)
            min_lev_h = list(tqdm(pool.imap(task_h, query_vae_seqs), total=num_queries, desc="VAE vs UniProt"))
        else:
            print("Skipping UniProt distance calculation (No data).")

        # --- 任务 2: VAE vs Training Data ---
        if len(target_train_seqs) > 0:
            print(f"Calculating distances to {len(target_train_seqs)} Training sequences...")
            task_train = partial(find_min_distance, target_sequences=target_train_seqs)
            min_lev = list(tqdm(pool.imap(task_train, query_vae_seqs), total=num_queries, desc="VAE vs Training"))

    print("Distance calculations complete.")

    # --- 4. 绘图 (Safe Plotting) ---
    print("Generating plot...")
    plt.figure(figsize=(9, 6))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # 绘制 Length
    vae_tp_len = list(vae_tp['sequence'].str.len())
    sns.histplot(vae_tp_len, kde=True, label='Length', stat="density", element="step")

    # 绘制 Distance to Training
    if len(min_lev) > 0:
        sns.histplot(min_lev, kde=True, label='Distance to training data', stat="density", element="step")
    
    # 绘制 Distance to UniProt (仅当有数据时)
    if len(min_lev_h) > 0:
        sns.histplot(min_lev_h, kde=True, label='Distance to MTSs in UniProt', stat="density", element="step")
    else:
        print("⚠️ 跳过绘制 'Distance to MTSs in UniProt'，因为数据为空。")

    plt.legend(fontsize=12)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(vae_seq_path)
    save_path = os.path.join(output_dir, 'Edit_Distance_Optimized.png')
    
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"Plot saved to {save_path}")

    end_time = time.time()
    print(f"--- 脚本总运行时间: {end_time - start_time:.2f} 秒 ---")