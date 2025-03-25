import numpy as np
import matplotlib.pyplot as plt

# 參數設定
k1, k2 = 50, 50  # 控制平滑程度

# 定義函數
def F_final(x, y):
    return 1 - ((-np.tanh(k1 * (x - 0.07)) + 1) / 2) * ((np.tanh(k2 * (y + 0.16)) + 1) / 2)

# 生成 x, y 網格
x_vals = np.linspace(-1, 1, 500)
y_vals = np.linspace(-1, 1, 500)
X, Y = np.meshgrid(x_vals, y_vals)
Z_final = F_final(X, Y)

# 繪製散點圖
plt.figure(figsize=(12, 12))
plt.scatter(X, Y, c=Z_final, cmap='coolwarm', alpha=0.8)
plt.colorbar(label='F(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of Final Function')
plt.show()