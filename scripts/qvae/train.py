import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import logging
import itertools
import sys
import os
import kaiwu as kw

# --- 设置 sys.path  ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..')) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logging.info(f"已将 {project_root} 添加到 sys.path 以查找 kaiwu_torch_plugin")

plugin_path = os.path.join(project_root, 'kaiwu_torch_plugin')
if not os.path.isdir(plugin_path):
    logging.warning(f"未在 {plugin_path} 找到 'kaiwu_torch_plugin' 目录。")
    logging.warning("请确保 'kaiwu_torch_plugin/' 目录与此脚本的父目录位于同一级别。")

# --- Kaiwu 许可初始化  ---
kw.license.init(user_id="105879747841515522", sdk_code="4vCbDDWqIdUEXDdEHKK0L4MtOOXvMF")

from kaiwu_torch_plugin import QVAE, BoltzmannMachine, RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer

# --- 1. 设置与常量  ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mpl.use('Agg') 
logging.info(f"使用设备: {device}")

# --- 关键超参数  ---
BETA = 0.01        
LATENT_DIM = 32
BATCH_SIZE = 2048
LEARNING_RATE_VAE = 1e-4
LEARNING_RATE_BM = 1e-4 
EPOCHS = 50
NUM_WORKERS = 4
MAX_LEN = 70
CHANNELS = 22
INPUT_DIM = MAX_LEN * CHANNELS
EXPERIMENT_TAG = f"b{BATCH_SIZE}_ld{LATENT_DIM}_beta{BETA}_bm{LEARNING_RATE_BM}"

# RBM 先验网络结构 
prior_vis = LATENT_DIM // 2
prior_hid = LATENT_DIM - prior_vis

# 路径配置 
QVAE_OUTPUT_ROOT = os.path.join(project_root, "data", "qvae") 
log_save_dir = os.path.join(QVAE_OUTPUT_ROOT, "log")
model_save_dir = os.path.join(QVAE_OUTPUT_ROOT, "model")
os.makedirs(log_save_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

# 加载数据 
data_dir = os.path.join(project_root, 'data') 
logging.info(f"正在从数据目录加载: {data_dir}")

log_file_path = os.path.join(log_save_dir, f"train_log_{EXPERIMENT_TAG}.txt")
# --- 2. 数据加载与预处理  ---
MAPPING = dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(CHANNELS)))

def one_hot_encode(seq):
    seq2 = [MAPPING[i] for i in seq]
    return np.eye(CHANNELS)[seq2]

class SequenceDataset(Dataset):
    def __init__(self, pkl_file_path, max_len):
        logging.info(f"正在加载数据: {pkl_file_path}...")
        try:
            with open(pkl_file_path, 'rb') as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            logging.error(f"错误: 找不到 .pkl 文件: {pkl_file_path}")
            logging.error("请确保 'tv_sim_split_train.pkl' 和 'tv_sim_split_valid.pkl' 位于 'data/' 文件夹中。")
            sys.exit(1)
            
        self.max_len = max_len
        self.pad_char = '0'
        self.end_char = '$'
        logging.info(f"成功加载 {len(self.data)} 条序列。")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_raw = self.data.sequence[idx]
        seq_terminated = (seq_raw + self.end_char)[:self.max_len]
        seq_padded = seq_terminated.ljust(self.max_len, self.pad_char)
        ohe_seq = one_hot_encode(seq_padded)
        ohe_tensor_flat = torch.FloatTensor(ohe_seq).view(-1)
        return ohe_tensor_flat

def get_mean_x(dataset, batch_size, num_workers):
    logging.info("计算训练数据均值 (mean_x)...")
    temp_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
    total_sum = 0.0
    total_count = 0
    for batch_data in tqdm(temp_loader, desc="计算 mean_x"):
        total_sum += batch_data.sum().item()
        total_count += batch_data.numel()
    
    if total_count == 0:
        logging.warning("数据加载器为空，无法计算 mean_x。返回默认值 0.5。")
        return 0.5
        
    mean_x = total_sum / total_count
    logging.info(f"计算得到的 mean_x: {mean_x}")
    if mean_x < 0.001 or mean_x > 0.999:
        logging.warning(f"mean_x ({mean_x}) 接近 0 或 1，可能导致 QVAE 初始化 'train_bias' 时出现数值不稳定。")
    return mean_x

# --- 3. 定义 FC (MLP) 模型组件  ---
class EncoderFC(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(EncoderFC, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.logits = nn.Linear(512, latent_dim)
        self.relu = nn.ReLU()
        self.input_dim = input_dim

    def forward(self, x):
        h = x.view(-1, self.input_dim)
        h1 = self.relu(self.fc1(h))
        return self.logits(h1)

class DecoderFC(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(DecoderFC, self).__init__()
        self.fc3 = nn.Linear(latent_dim, 512)
        self.fc4 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()
        self.output_dim = output_dim

    def forward(self, z):
        h3 = self.relu(self.fc3(z.float()))
        return self.fc4(h3).view(-1, self.output_dim)

# --- 4. 绘图函数  ---
def plot_losses(train_elbo, valid_elbo, train_cost, valid_cost, train_bm, save_dir, prefix):
    try:
        logging.info("正在生成损失曲线图...")
        epochs_range = range(1, len(train_elbo) + 1)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))
        fig.suptitle(f'QVAE (FC) Training Loss Curves (LD={LATENT_DIM}, β={BETA})', fontsize=16)
        
        ax1.plot(epochs_range, train_elbo, 'b-', label='Train ELBO')
        ax1.plot(epochs_range, valid_elbo, 'r-', label='Valid ELBO')
        ax1.set_title(f'ELBO Loss (Cost + {BETA}*KL)')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs_range, train_cost, 'b-', label='Train Reconstruction Loss (Cost)')
        ax2.plot(epochs_range, valid_cost, 'r-', label='Valid Reconstruction Loss (Cost)')
        ax2.set_title('Reconstruction Loss (Cost)')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        ax3.plot(epochs_range, train_bm, 'g-', label='Train BM CD Loss')
        ax3.set_title('BM Contrastive Divergence Loss')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(save_dir, f"{prefix}_loss_curves.png")
        plt.savefig(save_path)
        plt.close(fig)
        logging.info(f"损失曲线图已保存至: {save_path}")
    except Exception as e:
        logging.error(f"绘制损失曲线时发生错误: {e}")

def plot_rbm_weights(weights_matrix, save_dir, epoch_str, prefix):
    try:
        logging.info(f"正在生成 Epoch {epoch_str} 的 RBM 权重热力图...")
        plt.figure(figsize=(10, 8))
        sns.heatmap(weights_matrix, cmap='viridis', annot=False, cbar=True)
        plt.title(f'RBM Weight Matrix (V-H) - Epoch {epoch_str}\n{prefix}')
        plt.xlabel(f'Hidden Nodes ({weights_matrix.shape[1]})')
        plt.ylabel(f'Visible Nodes ({weights_matrix.shape[0]})')
        save_path = os.path.join(save_dir, f"{prefix}_rbm_weights_epoch_{epoch_str}.png")
        plt.savefig(save_path)
        plt.close()
        logging.info(f"RBM 权重热力图已保存至: {save_path}")
    except Exception as e:
        logging.error(f"绘制 RBM 权重热力图时发生错误: {e}")

# --- 5. 主训练函数 ---
def main():
    
    # --- 配置日志记录  ---
    file_handler = logging.FileHandler(log_file_path, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == file_handler.baseFilename for h in root_logger.handlers):
         root_logger.addHandler(file_handler)
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
         root_logger.addHandler(logging.StreamHandler(sys.stdout))

    logging.info("--- QVAE 训练和评估开始 (FC 架构) ---")
    logging.info(f"使用设备: {device}")
    logging.info(f"BETA = {BETA}, LATENT_DIM = {LATENT_DIM}, BATCH_SIZE = {BATCH_SIZE}")

    # --- 实例化数据集和加载器  ---
    train_pkl_path = os.path.join(data_dir, 'tv_sim_split_train.pkl')
    valid_pkl_path = os.path.join(data_dir, 'tv_sim_split_valid.pkl')
    
    train_dataset = SequenceDataset(pkl_file_path=train_pkl_path, max_len=MAX_LEN)
    valid_dataset = SequenceDataset(pkl_file_path=valid_pkl_path, max_len=MAX_LEN)

    if len(train_dataset) == 0:
        logging.error("训练数据集为空，请检查数据文件。")
        return

    mean_x = get_mean_x(train_dataset, BATCH_SIZE, NUM_WORKERS)  
    mean_x_save_path = os.path.join(model_save_dir, f"mean_x_{EXPERIMENT_TAG}.pkl")
    with open(mean_x_save_path, 'wb') as f:
        pickle.dump(mean_x, f)
    logging.info(f"已将 mean_x 保存到: {mean_x_save_path}")

    logging.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(valid_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        sampler=None
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # --- 实例化模型和优化器  ---
    logging.info(f"初始化 RBM (隐空间先验)... V={prior_vis}, H={prior_hid}")
    bm_prior = RestrictedBoltzmannMachine(
        num_visible=prior_vis,
        num_hidden=prior_hid
    ).to(device)
    
    logging.info("初始化 RBM 采样器 (模拟退火)...")
    train_sampler = SimulatedAnnealingOptimizer(
        initial_temperature=500.0,
        alpha=0.99,
        cutoff_temperature=0.001,
        iterations_per_t=20,
        size_limit=100,
        process_num=-1
    )
    
    logging.info("初始化 QVAE 模型 (FC 架构)...")
    model = QVAE(
        encoder=EncoderFC(INPUT_DIM, LATENT_DIM),
        decoder=DecoderFC(LATENT_DIM, INPUT_DIM),
        bm=bm_prior,
        sampler=train_sampler,
        dist_beta=1.0, 
        mean_x=mean_x,
        num_vis=bm_prior.num_visible
    ).to(device)

    vae_params = itertools.chain(model.encoder.parameters(), model.decoder.parameters())
    bm_params = model.bm.parameters()
    optimizer_vae = optim.Adam(vae_params, lr=LEARNING_RATE_VAE)
    optimizer_bm = optim.Adam(bm_params, lr=LEARNING_RATE_BM)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_vae, mode='min', factor=0.2, patience=3, min_lr=1e-6)

    # --- 6. 训练循环 ---
    logging.info(f"--- 开始训练: {EPOCHS} 个 Epochs, β={BETA} (ELBO KL 权重) ---")
    best_valid_elbo = float('inf')

    train_history_elbo = []
    valid_history_elbo = []
    valid_history_cost = [] 
    train_history_cost = []
    train_history_bm_loss = []
    
    file_prefix = f"{EXPERIMENT_TAG}"

    try: 
        for epoch in range(EPOCHS):
            model.train()
            train_loss_elbo = 0.0
            train_loss_cost = 0.0
            train_loss_kl = 0.0
            train_loss_bm = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [训练]", leave=False)
            for batch_X in pbar:
                batch_X = batch_X.to(device) 
                
                optimizer_vae.zero_grad()
                optimizer_bm.zero_grad()
                
                (output, recon_x, neg_elbo, wd_loss,
                 total_kl, cost, q, zeta) = model.neg_elbo(batch_X, kl_beta=BETA) 

                vae_loss = neg_elbo 
                
                with torch.no_grad():
                    q_probs = torch.sigmoid(q)
                    s_positive = torch.bernoulli(q_probs) 
                    
                s_negative = model.bm.sample(train_sampler)
                
                cd_loss = model.bm.objective(s_positive, s_negative)
                bm_loss = cd_loss 

                vae_loss.backward(retain_graph=True) 
                bm_loss.backward() 
                
                optimizer_vae.step()
                optimizer_bm.step()

                train_loss_elbo += neg_elbo.item()
                train_loss_cost += cost.item()
                train_loss_kl += total_kl.item()
                train_loss_bm += bm_loss.item()
                
                pbar.set_postfix(ELBO=f"{neg_elbo.item():.4f}", Cost=f"{cost.item():.4f}", BM_Loss=f"{bm_loss.item():.4f}")

            avg_train_elbo = train_loss_elbo / len(train_loader)
            avg_train_cost = train_loss_cost / len(train_loader)
            avg_train_kl = train_loss_kl / len(train_loader)
            avg_train_bm_loss = train_loss_bm / len(train_loader)
            train_history_elbo.append(avg_train_elbo)
            train_history_cost.append(avg_train_cost)
            train_history_bm_loss.append(avg_train_bm_loss)
            
            log_msg_train = (
                f"Epoch {epoch+1} [训练] | ELBO: {avg_train_elbo:.4f} | Cost: {avg_train_cost:.4f} | "
                f"KL: {avg_train_kl:.4f} | BM Loss: {avg_train_bm_loss:.4f}"
            )
            logging.info(log_msg_train)

            # --- 验证循环  ---
            model.eval()
            valid_loss_elbo = 0.0
            valid_loss_cost = 0.0
            valid_loss_kl = 0.0
            
            with torch.no_grad():
                for batchv_X in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [验证]", leave=False):
                    batchv_X = batchv_X.to(device)

                    (v_output, v_recon_x, v_neg_elbo, v_wd_loss,
                     v_total_kl, v_cost, v_q, v_zeta) = model.neg_elbo(batchv_X, kl_beta=BETA)
                    
                    valid_loss_elbo += v_neg_elbo.item()
                    valid_loss_cost += v_cost.item()
                    valid_loss_kl += v_total_kl.item()
            
            avg_valid_elbo = valid_loss_elbo / len(valid_loader)
            avg_valid_cost = valid_loss_cost / len(valid_loader)
            avg_valid_kl = valid_loss_kl / len(valid_loader)
            valid_history_elbo.append(avg_valid_elbo)
            valid_history_cost.append(avg_valid_cost)

            log_msg_valid = (
                f"Epoch {epoch+1} [验证] | ELBO: {avg_valid_elbo:.4f} | Cost: {avg_valid_cost:.4f} | KL: {avg_valid_kl:.4f}"
            )
            logging.info(log_msg_valid)

            if avg_valid_elbo < best_valid_elbo:
                best_valid_elbo = avg_valid_elbo
                best_model_path = os.path.join(model_save_dir, f"qvae_best_{EXPERIMENT_TAG}.chkpt")
                torch.save(model.state_dict(), best_model_path)
                logging.info(f" 新的最佳模型已保存至: {best_model_path} (Valid ELBO: {best_valid_elbo:.4f}) ")

            scheduler.step(avg_valid_elbo)

    except KeyboardInterrupt:
        logging.warning("--- 训练中断 (KeyboardInterrupt) ---")
    finally:
        logging.info("--- QVAE 训练和评估结束 ---")
        if train_history_elbo:
            plot_losses(
                train_history_elbo,
                valid_history_elbo,
                train_history_cost,
                valid_history_cost,
                train_history_bm_loss,
                log_save_dir,
                file_prefix
            )
        else:
            logging.info("没有足够的训练数据来绘制损失曲线。")
        try:
            logging.info("正在获取并绘制最终的 RBM 权重...")
            with torch.no_grad():
                final_weights = model.bm.quadratic_coef.detach().cpu().numpy()
            plot_rbm_weights(final_weights, log_save_dir, "final", file_prefix)
        except Exception as e:
            logging.error(f"无法绘制最终的 RBM 权重热力图: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout) 
        ]
    )
    main()


