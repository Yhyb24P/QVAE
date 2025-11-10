import pandas as pd
import numpy as np
import pickle 
from tqdm import tqdm
import re
from Bio import SeqIO
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.nn import functional as F
import torch
import argparse
import sys
import os
import logging

# --- 设置 sys.path ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..')) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    # 日志将在 main() 中配置, 这里先
    print(f"[Info] 已将 {project_root} 添加到 sys.path 以查找 kaiwu_torch_plugin")

# --- Kaiwu 许可初始化 ---
import kaiwu as kw
kw.license.init(user_id="105879747841515522", sdk_code="4vCbDDWqIdUEXDdEHKK0L4MtOOXvMF")
from kaiwu_torch_plugin import QVAE_BM, BoltzmannMachine, RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- 辅助函数 ---
def write_fasta(name, sequence_df):
    """ 将包含序列的 DataFrame 写入 FASTA 文件。"""
    try:
        with open(name + '.fasta', "w") as out_file:
            for i in range(len(sequence_df)):
                out_file.write('>' + sequence_df.name[i] + '\n')
                out_file.write(sequence_df.sequence[i] + '\n')
        logging.info(f"FASTA 文件已成功写入: {name}.fasta")
    except Exception as e:
        logging.error(f"写入 FASTA 文件失败: {e}")

# --- 1. 定义 QVAE 组件 (FC/MLP 架构) ---

# --- 数据相关常量 ---
MAX_LEN = 70
CHANNELS = 22
INPUT_DIM = MAX_LEN * CHANNELS # 1540
# 索引到字符的反向映射 (用于解码)
REV_MAPPING = {j:i for i,j in dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(CHANNELS))).items()}

class EncoderFC(nn.Module):
    """
    基于全连接层 (MLP) 的编码器
    """
    def __init__(self, input_dim, latent_dim):
        super(EncoderFC, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.logits = nn.Linear(512, latent_dim) 
        self.relu = nn.ReLU()
        self.input_dim = input_dim

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        return self.logits(h1)

class DecoderFC(nn.Module):
    """
    基于全连接层 (MLP) 的解码器
    """
    def __init__(self, latent_dim, output_dim):
        super(DecoderFC, self).__init__()
        self.fc3 = nn.Linear(latent_dim, 512)
        self.fc4 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()
        self.output_dim = output_dim

    def forward(self, z):
        h3 = self.relu(self.fc3(z))
        return self.fc4(h3).view(-1, self.output_dim)

# --- 2. 主生成函数 ---
def main():
    
    # --- 参数解析 (命令行参数) ---
    parser = argparse.ArgumentParser(description="从训练好的 QVAE (FC 架构) 生成序列")
    parser.add_argument('--model_path', type=str, required=True, help="指向 .chkpt 模型文件的路径")
    parser.add_argument('--mean_x_path', type=str, required=True, help="指向训练期间保存的 mean_x.pkl 文件的路径")
    parser.add_argument('--n_samples', type=int, default=5000, help="要生成的样本总数")
    parser.add_argument('--batch_size', type=int, default=2048, help="训练时使用的 BATCH_SIZE (用于日志和输出路径)")
    parser.add_argument('--latent_dim', type=int, default=32, help="模型的 LATENT_DIM")
    parser.add_argument('--beta', type=float, default=0.1, help="训练时使用的 BETA (用于日志和输出路径)")
    parser.add_argument('--decode_batch_size', type=int, default=512, help="解码时使用的批次大小")
    # --- 新增: 用于解码的温度参数 ---
    parser.add_argument('--temperature', type=float, default=1.0, help="解码采样温度 (T > 1.0 更随机, T < 1.0 更贪婪, T=0.0 即 argmax)")
    
    args = parser.parse_args()

    # --- 设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    logging.info(f"--- QVAE 序列生成开始 (FC 架构) ---")
    logging.info(f"解码温度: {args.temperature}")

    # --- 加载 mean_x ---
    try:
        with open(args.mean_x_path, 'rb') as f:
            loaded_mean_x = pickle.load(f)
        logging.info(f"成功从 {args.mean_x_path} 加载 mean_x: {loaded_mean_x}")
    except FileNotFoundError:
        logging.error(f"错误: 找不到 mean_x 文件 {args.mean_x_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"加载 mean_x 失败: {e}")
        sys.exit(1)
    
    # # --- 初始化 RBM (隐空间先验) ---
    # logging.info("初始化 RBM...")
    # vis_units = args.latent_dim // 2
    # hid_units = args.latent_dim - vis_units
    # bm = RestrictedBoltzmannMachine(
    #     num_visible=vis_units,
    #     num_hidden=hid_units
    # ).to(device)
    logging.info("初始化 Full BoltzmannMachine...")
    vis_units = args.latent_dim // 2
    bm = BoltzmannMachine(
        num_nodes=args.latent_dim
    ).to(device)

    # --- 初始化 BM 采样器 (用于 QVAE) ---
    logging.info("设置 BM 采样器 (模拟退火)...")
    sampler = SimulatedAnnealingOptimizer(
        initial_temperature=5000.0,
        alpha=0.995,
        iterations_per_t=100,
        size_limit=100,
        process_num=-1
    )

    # --- 初始化 QVAE 模型 ---
    logging.info("初始化 QVAE 模型 (FC 架构)...")
    model = QVAE_BM(
        encoder=EncoderFC(INPUT_DIM, args.latent_dim),
        decoder=DecoderFC(args.latent_dim, INPUT_DIM),
        bm=bm,
        sampler=sampler,           
        dist_beta=1.0, # 必须匹配 train.py (硬编码为 1.0)
        mean_x=loaded_mean_x, # 必须使用加载的 mean_x
        num_vis=vis_units # 必须匹配 train.py
    ).to(device)

    # --- 加载预训练模型 ---
    try:
        logging.info(f"正在从 {args.model_path} 加载模型权重...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval() # 设置为评估模式
    except FileNotFoundError:
        logging.error(f"错误: 找不到模型文件 {args.model_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        logging.error("这通常是因为模型架构或 QVAE 初始化参数与训练时不匹配。")
        sys.exit(1)

    # --- 3. 从 BM 先验采样 ---
    logging.info(f"正在从 BM 先验分布采样 {args.n_samples} 个隐向量 z ...")
    
    all_samples = []
    pbar = tqdm(total=args.n_samples, desc="Sampling from BM")
    total_collected = 0
    with torch.no_grad():
        while total_collected < args.n_samples:
            samples_batch = model.bm.sample(sampler) 
            all_samples.append(samples_batch.cpu())
            total_collected += samples_batch.shape[0]
            pbar.update(samples_batch.shape[0])
    pbar.close()
    
    sampled_z_tensor = torch.cat(all_samples, dim=0)[:args.n_samples].to(device)
    logging.info(f"采样完成。Z 张量形状: {sampled_z_tensor.shape}")

    # --- 4. 解码 Z 向量 ---
    logging.info("分批解码 Z 向量...")
    all_decoded_logits = []
    
    with torch.no_grad():
        for i in tqdm(range(0, args.n_samples, args.decode_batch_size), desc="解码中"):
            batch_z = sampled_z_tensor[i : i + args.decode_batch_size]
            logits = model.decoder(batch_z.float()) # (B, 1540)
            logits_reshaped = logits.view(-1, MAX_LEN, CHANNELS) # (B, 70, 22)
            all_decoded_logits.append(logits_reshaped.cpu())

    final_logits = torch.cat(all_decoded_logits, dim=0)
    logging.info(f"解码完成。Logits 形状: {final_logits.shape}")

    # --- 5. 后处理与 FASTA 保存 ---
    if args.temperature <= 0.0:
        logging.info("使用 argmax (贪婪) 采样...")
        indices = torch.argmax(final_logits, dim=2) # (B, L)
    else:
        logging.info(f"使用 temperature={args.temperature} 进行多项式采样...")
        B, L, C = final_logits.shape
        
        # 1. 缩放 Logits
        scaled_logits = final_logits / args.temperature
        
        # 2. 计算概率 (B*L, C)
        probs = F.softmax(scaled_logits.view(-1, C), dim=-1)
        
        # 3. 采样索引
        probs = probs.clamp(min=1e-8) 
        indices_flat = torch.multinomial(probs, num_samples=1) # (B*L, 1)
        
        # 4. Reshape 回 (B, L)
        indices = indices_flat.view(B, L)

    logging.info("正在将索引转换回序列并进行过滤...")
    sampled_seqs = []
    for seq_indices in tqdm(indices, desc="后处理"):
        out_seq = [REV_MAPPING[idx.item()] for idx in seq_indices]
        # 移除末尾的 '0' 和 '$'
        final_seq = ''.join(out_seq).rstrip('0').rstrip('$')
        sampled_seqs.append(final_seq)

    # 过滤无效序列 (在序列*中间*包含 '0' 或 '$')
    seq_to_check = []
    count = 0
    for seq in sampled_seqs:
        if seq and '$' not in seq and '0' not in seq:
            count += 1
            seq_to_check.append([f'>sample{count}', seq])

    logging.info(f"总共生成的有效序列 (去重前): {len(seq_to_check)}")
    
    if not seq_to_check:
        logging.warning("没有生成有效的序列。退出。")
        sys.exit(0)
        
    filtered_seq_to_check = pd.DataFrame(seq_to_check, columns = ['name', 'sequence'])
        
    logging.info(f'序列总数 (去重前): {len(filtered_seq_to_check)}')
    filtered_seq_to_check = filtered_seq_to_check.drop_duplicates(subset='sequence').reset_index(drop=True)
    logging.info(f'序列总数 (去重后): {len(filtered_seq_to_check)}')

    # 定义输出路径
    output_dir = f"data/qvae_bm/b{args.batch_size}_ld{args.latent_dim}_beta{args.beta}/output"
    os.makedirs(output_dir, exist_ok=True)
    
    output_fasta_path = os.path.join(output_dir, f"generated_seqs_n{args.n_samples}_T{args.temperature}")
    
    # 写入 FASTA
    write_fasta(output_fasta_path, filtered_seq_to_check)

    logging.info(f"--- QVAE 序列生成完毕 ---")


if __name__ == "__main__":
    main()

