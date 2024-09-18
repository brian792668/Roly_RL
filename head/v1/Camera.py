import cv2
import numpy as np

class Camera():
    def __init__(self, renderer, camID):
        self.renderer = renderer
        self.camID = camID
        self.rgbimg = np.zeros((240, 640, 3), dtype=np.uint8)
        self.depthimg = np.zeros((240, 640), dtype=np.uint8)
        self.target = [0.0, 0.0]
        self.target_depth = float('nan')

    def get_img(self, data, rgb=True, depth=True):
        if rgb == True:
            self.renderer.disable_depth_rendering()
            self.renderer.disable_segmentation_rendering()
            self.renderer.update_scene(data, camera=self.camID)
            self.rgbimg = self.renderer.render()
        if depth == True:
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(data, camera=self.camID)
            self.depthimg = self.renderer.render()
            # self.depthimg = (255*np.clip(depth, 0, 1)).astype(np.uint8)

    def show(self, rgb = False, depth = False):
        if rgb == True:
            cv2.imshow("RGB", cv2.cvtColor(self.rgbimg, cv2.COLOR_RGB2BGR))
        if depth == True:
            cv2.imshow("Depth", cv2.cvtColor(self.depthimg, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def get_target(self):
        # 定義紅色的RGB範圍
        lower_red = np.array([100, 0, 0], dtype=np.uint8)
        upper_red = np.array([255, 50, 50], dtype=np.uint8)
        # 創建紅色遮罩
        mask = cv2.inRange(self.rgbimg, lower_red, upper_red)

        # 確保在圖像中有紅色物體
        if np.any(mask):
            # 獲取紅色物體的像素坐標
            coords = np.column_stack(np.where(mask > 0))

            # 計算紅色物體的中心點
            center = np.mean(coords, axis=0)
            center_y, center_x = center

            # 在RGB圖像上畫十字標
            cross_size = 5  # 十字標的大小
            color = (255, 255, 255)  # 綠色
            thickness = 1  # 線條的粗細

            # 畫橫線
            cv2.line(self.rgbimg, (int(center_x) - cross_size, int(center_y)), 
                     (int(center_x) + cross_size, int(center_y)), color, thickness)
            # 畫縱線
            cv2.line(self.rgbimg, (int(center_x), int(center_y) - cross_size), 
                     (int(center_x), int(center_y) + cross_size), color, thickness)

            # 從深度圖像中獲取對應的深度值
            self.target_depth = 100*self.depthimg[int(center_y), int(center_x)]

            cv2.putText(self.rgbimg, f"{self.target_depth:.1f}", (int(center_x) + 10, int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # 將像素座標轉換至[-1, 1]區間
            norm_x = (center_x / self.rgbimg.shape[1]) * 2 - 1
            norm_y = (center_y / self.rgbimg.shape[0]) * 2 - 1
            self.target = [norm_x, norm_y]

        else:
            # 若無紅色物體，返回None並設置target為無效值
            self.target = [float('nan'), float('nan')]
            self.target = [0.0, 0.0]
            self.target_depth = float('nan')