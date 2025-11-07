import pandas as pd
import numpy as np
import re
from Bio import SeqIO
# from Bio import pairwise2 as pw2 # 未使用，已注释
import seaborn as sns
import matplotlib.pyplot as plt
from Levenshtein import distance as lv
from tqdm import tqdm
import pickle # <--- 在这里添加缺失的导入

# --- 导入并行处理库 ---
import multiprocessing
from functools import partial
import time # 用于计时

# --- 1. Functions (Optimized) ---

def validate(seq, pattern=re.compile(r'^[FIWLVMYCATHGSQRKNEPD]+$')):
    """使用预编译的 regex 验证序列"""
    if (pattern.match(seq)):
        return True
    return False

def clean(sequence_df):
    """
    使用 Pandas 矢量化操作 (str.match) 优化 clean 函数
    """
    print(f"Cleaning sequences... Initial count: {len(sequence_df)}")
    
    # 矢量化操作：检查 'sequence' 列中的每个字符串是否匹配
    valid_mask = sequence_df['sequence'].str.match(r'^[FIWLVMYCATHGSQRKNEPD]+$').fillna(False)
    
    invalid_count = len(sequence_df) - valid_mask.sum()
    print(f'Total number of sequences dropped: {invalid_count}')
    
    # 按掩码过滤并重置索引
    cleaned_df = sequence_df[valid_mask].reset_index(drop=True)
    print(f'Total number of sequences remaining: {len(cleaned_df)}')
    
    return cleaned_df

def read_fasta(name):
    """读取 FASTA 文件并移除 eGFP 序列"""
    print(f"Reading FASTA file: {name}.fasta")
    fasta_seqs = SeqIO.parse(open(name + '.fasta'),'fasta')
    data = []
    # eGFP 序列保持不变
    egfp = 'VSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK'
    
    count = 0
    for fasta in fasta_seqs:
        count += 1
        # 清理序列中的空格和 eGFP
        seq_cleaned = str(fasta.seq).strip().replace(egfp,'')
        data.append([fasta.id, seq_cleaned])
    
    print(f"Read {count} sequences from FASTA.")
    return data

def parse_uniprot_row(row):
    """
    用于 .apply() 的辅助函数，解析 UniProt excel 的单行。
    使用 try-except 增加鲁棒性。
    """
    try:
        col_name, col_seq, col_features = row[0], row[2], row[3]
        
        # 过滤：寻找 'Mitochondrion' (不区分大小写)
        if 'mitochondrion' not in col_features.lower():
            return None
        
        # 提取 'note=' (不区分大小写)
        note_match = re.search(r'note="([^"]+)"', col_features, re.IGNORECASE)
        if not note_match:
            return None
            
        organelle = note_match.group(1)
        if 'mitochondrion' not in organelle.lower():
             return None

        # 提取 'Transit peptide' 范围 (不区分大小写)
        tp_match = re.search(r'transit peptide\s+\d+\.\.(\d+)', col_features, re.IGNORECASE)
        if not tp_match:
            return None

        tp_end_str = tp_match.group(1)
        
        # 确保 tp_end 是有效数字
        if '?' in tp_end_str:
            return None
            
        tp_end = int(tp_end_str)
        if tp_end <= 5: # 过滤长度
            return None
            
        # 提取序列
        tp_seq = col_seq[:tp_end]
        
        # 验证序列字符
        if not validate(tp_seq):
            return None
            
        return [col_name, tp_seq]
        
    except Exception as e:
        # 捕获解析错误
        # print(f"Skipping row due to error: {e}")
        return None

def find_min_distance(query_seq, target_sequences):
    """
    计算单个 query_seq 与 target_sequences 列表中所有序列的最小 Levenshtein 距离。
    这是用于并行化的工作单元。
    """
    if not query_seq: # 处理空字符串
        return np.inf
        
    min_dist = np.inf
    for target_seq in target_sequences:
        if not target_seq: # 跳过空的目标
            continue
        dist = lv(query_seq, target_seq)
        if dist < min_dist:
            min_dist = dist
    return min_dist


# --- 2. Main Execution ---

print("--- 启动序列相似性分析 ---")
start_time = time.time()

# --- UniProt 数据处理 (Optimized) ---
print("Loading and parsing UniProt data...")
uniprot_raw = pd.read_excel('data/uniprot_transit_peptide.xlsx', header = None) 

# 使用 .apply() 代替慢速循环
# 我们跳过标题行 (index 0)
parsed_data = uniprot_raw.iloc[1:].apply(parse_uniprot_row, axis=1).dropna().tolist()

uniprot_tp = pd.DataFrame(parsed_data, columns = ['name', 'sequence'])
print(f'Total valid sequences parsed: {len(uniprot_tp)}')

# 删除重复项
uniprot_tp = uniprot_tp.drop_duplicates(subset='sequence').reset_index(drop=True)
print(f'Total sequences remaining after duplicate removal: {len(uniprot_tp)}')

# --- VAE 和 训练数据 加载 ---
print("Loading VAE generated sequences...")
# --- 修改下面这一行 ---
vae_tp = pd.DataFrame(read_fasta('scripts/qvae-v/data/qvae-fc/b2048_ld32_beta0.1/output/generated_seqs_fc_n5000_T1.0'), columns = ['name','sequence'])
# --- 修改上面这一行 ---

# 可选：清理 VAE 序列
vae_tp = clean(vae_tp) 

print("Loading training data (X_train)...")
with open('data/tv_sim_split_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
# 确保 X_train 也是干净的（如果需要）
# X_train = clean(X_train) 

# --- 3. 并行计算 Levenshtein 距离 ---
query_vae_seqs = vae_tp['sequence'].tolist()
target_uniprot_seqs = uniprot_tp['sequence'].tolist()
target_train_seqs = X_train['sequence'].tolist()

num_queries = len(query_vae_seqs)
print(f"Starting parallel distance calculation for {num_queries} query sequences...")

# 获取 CPU 核心数
n_cores = multiprocessing.cpu_count()
print(f"Using {n_cores} CPU cores for parallel processing.")

min_lev_h = []
min_lev = []

# 使用 multiprocessing.Pool
with multiprocessing.Pool(processes=n_cores) as pool:
    
    # --- 任务 1: VAE vs UniProt ---
    print(f"Calculating distances to {len(target_uniprot_seqs)} UniProt sequences...")
    # 使用 partial 锁定 'target_sequences' 参数
    task_h = partial(find_min_distance, target_sequences=target_uniprot_seqs)
    
    # pool.imap 允许 tqdm 显示进度条
    min_lev_h = list(tqdm(
        pool.imap(task_h, query_vae_seqs), 
        total=num_queries, 
        desc="VAE vs UniProt"
    ))

    # --- 任务 2: VAE vs Training Data ---
    print(f"Calculating distances to {len(target_train_seqs)} Training sequences...")
    task_train = partial(find_min_distance, target_sequences=target_train_seqs)
    
    min_lev = list(tqdm(
        pool.imap(task_train, query_vae_seqs), 
        total=num_queries, 
        desc="VAE vs Training"
    ))

print("Distance calculations complete.")

# --- 4. 绘图 (Optimized) ---
print("Generating plot...")
# --- 修复：将 'Sequence' 改为 'sequence' ---
vae_tp_len = list(vae_tp['sequence'].str.len())

plt.figure(figsize=(9, 6))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# --- 使用 sns.histplot ---
sns.histplot(vae_tp_len, kde=True, label='Length', stat="density", element="step")
sns.histplot(min_lev, kde=True, label='Distance to training data', stat="density", element="step")
sns.histplot(min_lev_h, kde=True, label='Distance to MTSs in UniProt', stat="density", element="step")

plt.legend(fontsize=12)
save_path = 'data/Edit_Distance_Optimized.png'
plt.savefig(save_path, dpi=400, bbox_inches="tight")
print(f"Plot saved to {save_path}")

end_time = time.time()
print(f"--- 脚本总运行时间: {end_time - start_time:.2f} 秒 ---")



