import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class IKDataset(Dataset):
    def __init__(self, xyz_data, collision_data):
        self.xyz_data = xyz_data
        self.collision_data = collision_data
    
    def __len__(self):
        return len(self.xyz_data)
    
    def __getitem__(self, idx):
        xyz = self.xyz_data[idx]
        collision = self.collision_data[idx]
        xyz_tensor = torch.tensor(xyz, dtype=torch.float32)
        collision_tensor = torch.tensor(collision, dtype=torch.float32)
        return xyz_tensor, collision_tensor

class IKMLP(nn.Module):
    def __init__(self): # 3 -> 32 -> 16 -> 2
        super(IKMLP, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.tanh(x) * 1.6
        return x
    

def train(numberofpoints, version):
    file_path = os.path.dirname(os.path.abspath(__file__))
    # ---------------- Create datasets -----------------
    EE_xyz_array = np.load(os.path.join(file_path, f"datasets/{numberofpoints}points/EE_xyz_label.npy"))
    collision_array = np.load(os.path.join(file_path, f"datasets/{numberofpoints}points/collision_label.npy"))
    ik_dataset = IKDataset(EE_xyz_array, collision_array)
    dataloader = DataLoader(ik_dataset, batch_size=256, shuffle=True)
    # ---------------- Create testing datasets -----------------
    test_EE_xyz_array = np.load(os.path.join(file_path, "datasets/409points/EE_xyz_label.npy"))
    test_collision_array = np.load(os.path.join(file_path, "datasets/409points/collision_label.npy"))
    test_dataset = IKDataset(test_EE_xyz_array, test_collision_array)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # ----------------------- Model --------------------------
    model = IKMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    # --------------------------- Training ------------------------------
    epochs = 100
    train_losses = []
    test_losses = []
    min_train_loss = 1000

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for xyz_batch, collision_batch in dataloader:
            # 預測模型的輸出 c 和 d
            predicted = model(xyz_batch)

            # 取得 c 和 d 的預測值
            c = predicted[:, 0]
            d = predicted[:, 1]

            # 根據公式計算 boundary_value
            boundary_value = (collision_batch[:, 0] - c) * (collision_batch[:, 0] - d)
            normalized = 1 / (1 + torch.exp(-500 * torch.clamp(boundary_value, -0.02, 0.02)))
            # print(normalized, collision_batch[:, 1])

            # 計算 loss: 使用 MSE 來與 collision_batch 中的 b 進行比較
            # loss = torch.mean(torch.abs(normalized - collision_batch[:, 1]))
            loss = criterion(normalized, collision_batch[:, 1])

            # 清空梯度
            optimizer.zero_grad()

            # 反向傳播
            loss.backward()

            # 更新權重
            optimizer.step()

            # 累積當前 batch 的 loss
            running_loss += loss.item()


        # 計算訓練的平均 loss
        epoch_loss = running_loss / len(dataloader)
        train_losses.append(epoch_loss)

        # ---------------------- Testing ----------------------
        model.eval()
        test_running_loss = 0.0
        with torch.no_grad():
            for xyz_batch, collision_batch in test_dataloader:
                predicted = model(xyz_batch)
                c = predicted[:, 0]
                d = predicted[:, 1]
                boundary_value = (collision_batch[:, 0] - c) * (collision_batch[:, 0] - d)
                normalized = 1 / (1 + torch.exp(-500 * torch.clamp(boundary_value, -0.02, 0.02)))
                # loss = torch.mean(torch.abs(normalized - collision_batch[:, 1]))
                loss = criterion(normalized, collision_batch[:, 1])
                test_running_loss += loss.item()
        
        test_epoch_loss = test_running_loss / len(test_dataloader)
        test_losses.append(test_epoch_loss)

        # 儲存最小的 train loss
        if epoch_loss < min_train_loss:
            min_train_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(file_path, f'collision_bound_{numberofpoints}points_{version}.pth'))

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_epoch_loss:.4f}')

        # 每個 epoch 繪製並儲存圖
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, epoch+2), train_losses, label='Training Loss')
        plt.plot(range(1, epoch+2), test_losses, label='Testing Loss', linestyle='dashed')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Testing Loss vs. Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(file_path, f"collision_bound_{version}.png"))
        plt.close()

    # # 儲存模型
    # torch.save(model.state_dict(), os.path.join(file_path, f'collision_bound_{numberofpoints}points_{version}.pth'))


if __name__ == '__main__':
    train(numberofpoints=2048000, version="v1")
