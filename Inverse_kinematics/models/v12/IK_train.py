import os
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
        xyz = self.xyz_data[idx]
        angles = self.angles_data[idx]
        xyz_tensor = torch.tensor(xyz, dtype=torch.float32)
        angles_tensor = torch.tensor(angles, dtype=torch.float32)
        return xyz_tensor, angles_tensor

class IKMLP(nn.Module):
    def __init__(self):
        super(IKMLP, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = torch.tanh(x) * 95
        return x

def train(numberofpoints, version):
    # file path
    file_path = os.path.dirname(os.path.abspath(__file__))

    # ---------------- Create datasets -----------------
    xyz_array = np.load(os.path.join(file_path, f"../../datasets/new_origin/{numberofpoints}points/xyz.npy"))
    joints_array = np.load(os.path.join(file_path, f"../../datasets/new_origin/{numberofpoints}points/joints.npy"))
    ik_dataset = IKDataset(xyz_array, joints_array)
    dataloader = DataLoader(ik_dataset, batch_size=32, shuffle=True)

    # Load test data
    test_xyz_array = np.load(os.path.join(file_path, f"../../datasets/new_origin/100points/xyz.npy"))
    test_joints_array = np.load(os.path.join(file_path, f"../../datasets/new_origin/100points/joints.npy"))
    test_dataset = IKDataset(test_xyz_array, test_joints_array)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # ----------------------- Model --------------------------
    model = IKMLP()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --------------------------- Training ------------------------------
    epochs = 10000
    train_losses = []
    test_losses = []
    min_test_loss = 1000

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (xyz, target_angles) in enumerate(dataloader):
            target_n = target_angles[:, 2].unsqueeze(1)  # 只取 index 2 的元素，並保留 2D shape
            predicted_angles = model(xyz)
            loss = criterion(predicted_angles, target_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 計算訓練的平均 loss
        epoch_loss = running_loss / len(dataloader)
        train_losses.append(epoch_loss)
        
        # 在測試資料集上計算 loss
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xyz, target_angles in test_dataloader:
                target_n = target_angles[:, 2].unsqueeze(1)
                predicted_angles = model(xyz)
                loss = criterion(predicted_angles, target_n)
                test_loss += loss.item()

        test_loss /= len(test_dataloader)
        test_losses.append(test_loss)
        if test_loss <= min_test_loss:
            min_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(file_path, f"IKmodel_{version}.pth"))


        print(f'Epoch [{epoch+1}/{epochs}]   Train/Test Loss: {epoch_loss:.1f}, {test_loss:.1f}    min test loss: {min_test_loss:.3f}')

        # 每個 epoch 結束後繪製並儲存圖
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, epoch+2), test_losses, label='Test Loss', alpha=0.5)
        plt.plot(range(1, epoch+2), train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0, 200)
        plt.title('Training and Test Loss vs. Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(file_path, f"Loss_vs_epoch_{version}.png"))
        plt.close()


if __name__ == '__main__':
    train(numberofpoints=4997, version="v12")
