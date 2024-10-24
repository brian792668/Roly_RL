import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

class IKMLP(nn.Module):
    def __init__(self):
        super(IKMLP, self).__init__()
        # 定義三層線性層 3 - 64 - 128 - 4
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 4)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一層用 ReLU
        x = F.relu(self.fc2(x))  # 第二層用 ReLU
        x = self.fc3(x)          # 最後一層直接輸出
        return x

def train(numberofpoints, version):
    # ---------------- Create datasets from numpy array -----------------
    xyz_array = np.load(f'Roly/Inverse_kinematics/datasets/{numberofpoints}points/xyz.npy')
    joints_array = np.load(f'Roly/Inverse_kinematics/datasets/{numberofpoints}points/joints.npy')
    ik_dataset = IKDataset(xyz_array, joints_array)
    dataloader = DataLoader(ik_dataset, batch_size=10, shuffle=True)

    # ----------------------- Create MLP model --------------------------
    model = IKMLP()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --------------------------- Training ------------------------------
    epochs = 1000
    all_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (xyz, target_angles) in enumerate(dataloader):
            # 前向傳播
            predicted_angles = model(xyz)
            loss = criterion(predicted_angles, target_angles)

            # 反向傳播和優化
            optimizer.zero_grad()  # 清空上一次的梯度
            loss.backward()        # 計算梯度
            optimizer.step()       # 更新參數

            running_loss += loss.item() # 累加當前batch的loss

        epoch_loss = running_loss / len(dataloader) # epoch avg loss
        all_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.2f}')

    # save model
    torch.save(model.state_dict(), f'Roly/Inverse_kinematics/models/{numberofpoints}points_{version}.pth')

    # plot
    fig = plt.figure(figsize=(20, 12))
    plt.plot(range(1, epochs+1), all_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 200)
    plt.title('Training Loss vs. Epochs')
    plt.legend()
    plt.savefig(f"Roly/Inverse_kinematics/models/TrainingLoss_vs_epoch ({numberofpoints}points {version}).png")
    plt.show()
    plt.close()

if __name__ == '__main__':
    train(numberofpoints=10, version="v1")