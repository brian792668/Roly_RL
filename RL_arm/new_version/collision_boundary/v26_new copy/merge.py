import os
import numpy as np

def merge_npy_files(input_dir, output_dir, file_prefix):
    # 確保輸出資料夾存在
    os.makedirs(output_dir, exist_ok=True)
    
    merged_array = np.array([])  # 預先分配記憶體
    
    for i in range(2):
        file_path = os.path.join(input_dir, f"{file_prefix}_epoch{i}.npy")
        if os.path.exists(file_path):
            data = np.load(file_path)  # 讀取 .npy 檔案
            
            if merged_array.size == 0:
                merged_array = data.copy()
            else:
                merged_array = np.concatenate((merged_array, data.copy()), axis=0)

        else:
            raise FileNotFoundError(f"File not found: {file_path}")
        
    
    # 儲存合併後的檔案
    output_file = os.path.join(output_dir, f"{file_prefix}.npy")
    np.save(output_file, merged_array)
    print(f"Saved: {output_file}")

# 設定資料夾路徑
file_path = os.path.dirname(os.path.abspath(__file__))
input_directory = os.path.join(file_path, f"datasets/new/500_epoch_points")
output_directory = os.path.join(file_path, f"datasets/408points")

# 合併 collision_label 檔案
merge_npy_files(input_directory, output_directory, "collision_label")
merge_npy_files(input_directory, output_directory, "EE_xyz_label")


position_data = np.load(os.path.join(file_path, f"datasets/408points/EE_xyz_label.npy"))
collision_data = np.load(os.path.join(file_path, f"datasets/408points/collision_label.npy"))
print("EE_xyz_label:", position_data.shape, "collision_label:", collision_data.shape)