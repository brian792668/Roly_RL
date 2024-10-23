import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

xyz_array = np.load('Roly/Inverse_kinematics/label200point.npy')
joints_array = np.load('Roly/Inverse_kinematics/joint200point.npy')

class IKDataset(Dataset):
    def __init__(self, xyz_data, angles_data):
        self.xyz_data = xyz_data
        self.angles_data = angles_data
    
    def __len__(self):
        return len(self.xyz_data)
    
    def __getitem__(self, idx):
        # 獲取對應的 x, y, z 和 a1, a2, a3, a4
        xyz = self.xyz_data[idx]
        angles = self.angles_data[idx]
        # NumPy to PyTorch tensor form
        xyz_tensor = torch.tensor(xyz, dtype=torch.float32)
        angles_tensor = torch.tensor(angles, dtype=torch.float32)
        
        return xyz_tensor, angles_tensor

# create dataset from numpy array
ik_dataset = IKDataset(xyz_array, joints_array)
dataloader = DataLoader(ik_dataset, batch_size=10, shuffle=True)

# # print dataset in dataloader form
# for batch_idx, (xyz, angles) in enumerate(dataloader):
#     print(f"Batch {batch_idx+1}:")
#     print("xyz: ", xyz)
#     print("angles: ", angles)
#     print("\n")  # 每個 batch 之間加點空行讓輸出更清楚


# ==========================================================
# ----------------------- 建立 MLP --------------------------

class IKMLP(nn.Module):
    def __init__(self):
        super(IKMLP, self).__init__()
        # 定義三層線性層
        self.fc1 = nn.Linear(3, 64)  # 輸入3維，隱藏層64
        self.fc2 = nn.Linear(64, 128) # 隱藏層64輸出128
        self.fc3 = nn.Linear(128, 4)  # 輸出4維（4個關節角度）
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一層用 ReLU 激活
        x = F.relu(self.fc2(x))  # 第二層用 ReLU 激活
        x = self.fc3(x)          # 最後一層直接輸出
        return x
    

# ==========================================================
# ----------------------- Training -------------------------

model = IKMLP()
criterion = nn.MSELoss()  # 使用均方誤差作為損失函數
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 1000

# 開始訓練
for epoch in range(epochs):
    for i, (xyz, target_angles) in enumerate(dataloader):
        # 前向傳播
        predicted_angles = model(xyz)
        loss = criterion(predicted_angles, target_angles)
        
        # 反向傳播和優化
        optimizer.zero_grad()  # 清空上一次的梯度
        loss.backward()        # 計算梯度
        optimizer.step()       # 更新參數

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# save model
torch.save(model.state_dict(), 'Roly/Inverse_kinematics/Roly_IK_model.pth')