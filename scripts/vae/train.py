import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torch
# 导入 DataLoader
from torch.utils.data import TensorDataset, DataLoader
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# --- 1. 设置设备 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# --- 2. 加载数据 ---
# 注意：假设 'data/' 目录存在。如果不存在，请创建它或修改路径。
# 为了使脚本可直接运行，我们使用模拟数据。
# 在您的环境中，请取消注释下一节以加载您的 .pkl 文件。

# --- 模拟数据（如果未找到 .pkl 文件） ---
def create_mock_data():
    logging.warning("Mock data creation: Simulating pkl files.")
    # 模拟 'data/tv_sim_split_train.pkl'
    train_sequences = ['FIWLVMYCATHGSQRKNEPD', 'PDENKRSQGHTACMYVLWIF', 'ATHGSQRKNEPDFIWLVMYC'] * 100
    mock_train_df = pd.DataFrame({'sequence': train_sequences})
    
    # 模拟 'data/tv_sim_split_valid.pkl'
    valid_sequences = ['FIWLVMYCATHGSQRKNEPD', 'PDENKRSQGHTACMYVLWIF'] * 20
    mock_valid_df = pd.DataFrame({'sequence': valid_sequences})
    
    return mock_train_df, mock_valid_df

try:
    with open('data/tv_sim_split_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('data/tv_sim_split_valid.pkl', 'rb') as f:
        X_valid = pickle.load(f)
    logging.info("Successfully loaded data from .pkl files.")
except FileNotFoundError:
    logging.warning("Could not find .pkl files in 'data/'. Using mock data instead.")
    logging.warning("Please ensure 'data/tv_sim_split_train.pkl' and 'data/tv_sim_split_valid.pkl' exist for real training.")
    X_train, X_valid = create_mock_data()
except Exception as e:
    logging.error(f"Error loading data: {e}. Exiting.")
    exit()


def one_hot_encode(seq):
    mapping = dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(22)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(22)[seq2]

logging.info("One-hot encoding training data...")
X_ohe_train_list = []
for i in tqdm(range(np.shape(X_train)[0])):
    seq = X_train.sequence[i]+'$'
    pad_seq = seq.ljust(70,'0')
    X_ohe_train_list.append(one_hot_encode(pad_seq))

logging.info("One-hot encoding validation data...")
X_ohe_valid_list = []
for i in tqdm(range(np.shape(X_valid)[0])):
    seq = X_valid.sequence[i]+'$'
    pad_seq = seq.ljust(70,'0')
    X_ohe_valid_list.append(one_hot_encode(pad_seq))
    
logging.info("Converting training list to single tensor...")
X_ohe_train_tensor = torch.FloatTensor(np.array(X_ohe_train_list)).view(-1, 1540)

logging.info("Converting validation list to single tensor...")
X_ohe_valid_tensor = torch.FloatTensor(np.array(X_ohe_valid_list)).view(-1, 1540)

logging.info(f"Train tensor shape: {X_ohe_train_tensor.shape}")
logging.info(f"Valid tensor shape: {X_ohe_valid_tensor.shape}")

# --- 4. 定义模型 (和以前一样) ---
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1540, 512)
        self.fc21 = nn.Linear(512, 32)
        self.fc22 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 512)
        self.fc4 = nn.Linear(512, 1540)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# --- 修改 loss_function ---
# 现在返回 (总损失, 重构损失, KL损失)
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # 返回 ELBO (BCE + KLD), BCE, KLD
    return BCE + KLD, BCE, KLD

# --- 5. 设置 DataLoader ---
batch_size = 128
epochs = 50 # 您可以根据需要调整

train_dataset = TensorDataset(X_ohe_train_tensor)
valid_dataset = TensorDataset(X_ohe_valid_tensor)

# 检查数据集大小，如果太小，减少 num_workers
num_workers = 4
if len(train_dataset) < batch_size * num_workers:
    num_workers = 0
    logging.warning(f"Training dataset is small. Setting num_workers to 0.")

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers,
    pin_memory=True
)

# --- 6. 实例化模型并移动到 GPU ---
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)  

# --- 7. 训练与验证循环 ---

# --- 为绘图初始化列表 ---
train_elbo_losses = []
train_recon_losses = []
valid_elbo_losses = []
valid_recon_losses = []

# 确保 'data/vae/model/' 和 'model/vae/' 目录存在
import os
os.makedirs('data/vae/model', exist_ok=True)
os.makedirs('model/vae', exist_ok=True)

f = open("data/vae/model/loss_w.txt", "a")
for epoch in range(epochs):
    # --- 训练 ---
    model.train() # 设置为训练模式
    total_train_elbo_loss = 0
    total_train_recon_loss = 0
    
    for (batch_X,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        
        batch_X = batch_X.to(device)
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch_X) 
        
        # 获取分解的损失
        elbo_loss, recon_loss, kld_loss = loss_function(recon_batch, batch_X, mu, logvar)
        
        elbo_loss.backward() # 仅反向传播总损失
        
        total_train_elbo_loss += elbo_loss.item()
        total_train_recon_loss += recon_loss.item() # 累加重构损失
        
        optimizer.step()
    
    # 计算平均损失
    avg_train_elbo = total_train_elbo_loss / len(train_loader.dataset)
    avg_train_recon = total_train_recon_loss / len(train_loader.dataset)
    
    # 添加到列表
    train_elbo_losses.append(avg_train_elbo)
    train_recon_losses.append(avg_train_recon)
    
    f.write(f"Epoch: {epoch}. Train ELBO: {avg_train_elbo}. Train Recon: {avg_train_recon}\n")
    logging.info(f"Epoch: {epoch}. Train ELBO: {avg_train_elbo}. Train Recon: {avg_train_recon}")
    
    # 保存模型
    torch.save(model.state_dict(), f"model/vae/vae_self_tv_sim_split_kl_weight_1_batch_size_{batch_size}_epochs{epoch}.chkpt")

    # --- 验证 ---
    model.eval() # 设置为评估模式
    with torch.no_grad(): 
        total_valid_elbo_loss = 0
        total_valid_recon_loss = 0
        
        for (batchv_X,) in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]"):
            
            batchv_X = batchv_X.to(device)
            
            recon_batch, mu, logvar = model(batchv_X)
            
            elbo_loss, recon_loss, kld_loss = loss_function(recon_batch, batchv_X, mu, logvar)
            
            total_valid_elbo_loss += elbo_loss.item()
            total_valid_recon_loss += recon_loss.item()
    
        # 计算平均损失
        avg_valid_elbo = total_valid_elbo_loss / len(valid_loader.dataset)
        avg_valid_recon = total_valid_recon_loss / len(valid_loader.dataset)
        
        # 添加到列表
        valid_elbo_losses.append(avg_valid_elbo)
        valid_recon_losses.append(avg_valid_recon)

        f.write(f"Epoch: {epoch}. Valid ELBO: {avg_valid_elbo}. Valid Recon: {avg_valid_recon}\n")
        logging.info(f"Epoch: {epoch}. Valid ELBO: {avg_valid_elbo}. Valid Recon: {avg_valid_recon}")

f.close()
logging.info("训练完毕")

# --- 8. 绘制并保存子图 ---
logging.info("Generating and saving loss subplots...")
plot_filename = 'vae_loss_subplots.png'

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

ax1.plot(range(epochs), train_elbo_losses, 'b-', label='Train ELBO')
ax1.plot(range(epochs), valid_elbo_losses, 'r-', label='Valid ELBO')
ax1.set_title('ELBO Loss (Reconstruction + KLD)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(range(epochs), train_recon_losses, 'b-', label='Train Reconstruction Loss (Cost)')
ax2.plot(range(epochs), valid_recon_losses, 'r-', label='Valid Reconstruction Loss (Cost)')
ax2.set_title('Reconstruction Loss (BCE)')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

 # 保存图像
save_dir = 'data/vae/model'
save_path = os.path.join(save_dir, f"loss_curves.png")
plt.savefig(save_path)
plt.close(fig) # 关闭图像
logging.info(f"损失曲线图已保存至: {save_path}")

