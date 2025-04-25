import cv2
import numpy as np

class Camera():
    def __init__(self, renderer, camID):
        self.renderer = renderer
        self.camID = camID
        self.rgbimg = np.zeros((240, 320, 3), dtype=np.uint8)
        self.depthimg = np.zeros((240, 320), dtype=np.uint8)
        self.target = ['nan', 'nan']
        self.target_depth = float('nan')
        self.track_done = False

        self.center_height = 180
        self.center_width = 240
        self.feature_points = np.array([1.0]*225)

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
            # scaled_image = cv2.resize(self.rgbimg, (640, 480), interpolation=cv2.INTER_LINEAR)
            # cv2.imshow("RGB", cv2.cvtColor(scaled_image, cv2.COLOR_RGB2BGR))
            cv2.imshow("RGB", cv2.cvtColor(self.rgbimg, cv2.COLOR_RGB2BGR))
        if depth == True:
            cv2.imshow("Depth", cv2.cvtColor(self.depthimg, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def get_target(self, depth=False):
        # 定義紅色的RGB範圍
        lower_red = np.array([100, 0, 0], dtype=np.uint8)
        upper_red = np.array([255, 50, 50], dtype=np.uint8)
        mask = cv2.inRange(self.rgbimg, lower_red, upper_red)

        if np.any(mask):
            # 獲取紅色物體的像素坐標
            coords = np.column_stack(np.where(mask > 0))

            # 計算紅色物體的中心點
            center = np.mean(coords, axis=0)
            center_y, center_x = center

            size = 3
            thickness = 1

            # 畫橫縱線
            cv2.line(self.rgbimg, (int(center_x) - size, int(center_y)), 
                     (int(center_x) + size, int(center_y)), (255, 255, 255), thickness)
            cv2.line(self.rgbimg, (int(center_x), int(center_y) - size), 
                     (int(center_x), int(center_y) + size), (255, 255, 255), thickness)

            if depth == True:
                self.target_depth = 100*self.depthimg[int(center_y), int(center_x)]
                cv2.putText(self.rgbimg, f"{self.target_depth:.1f}", (int(center_x) + 10, int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # nomalize x,y to [-1, 1]
            norm_x = (center_x / self.rgbimg.shape[1]) * 2 - 1
            norm_y = (center_y / self.rgbimg.shape[0]) * 2 - 1
            self.target = [norm_x, norm_y]

        else:
            # 若無紅色物體，返回None並設置target為無效值
            self.track_done = False
            self.target = [float('nan'), float('nan')]
            if depth == True:
                self.target_depth = float('nan')
            
    def track(self, ctrlpos, speed=1.0):
        new_pos = ctrlpos.copy()
        if np.abs(self.target[0]) <= 0.01 and np.abs(self.target[1]) <= 0.01:
            self.track_done = True
        else:
            self.track_done = False
            if np.isnan(self.target[0]) == False:
                new_pos[0] += -0.1*self.target[0]*speed
                new_pos[1] += -0.1*self.target[1]*speed
        return new_pos
    
    def depth_feature(self):
        # 提取中央範圍
        start_h = (self.depthimg.shape[0] - self.center_height) // 2
        start_w = (self.depthimg.shape[1] - self.center_width) // 2
        center_region = self.depthimg[start_h:start_h + self.center_height, start_w:start_w + self.center_width]

        # 均勻選取 10x10 特徵點
        rows, cols = 15, 15
        y_indices = np.linspace(0, self.center_height - 1, rows, dtype=int)
        x_indices = np.linspace(0, self.center_width - 1, cols, dtype=int)

        self.feature_points = center_region[np.ix_(y_indices, x_indices)].flatten()
        self.feature_points = np.clip(self.feature_points, 0.15, 1.0)
        # print("特徵點值:", self.feature_points)
        # print("特徵點數量:", len(self.feature_points))

        # 繪製特徵點在 RGB 圖像上
        for y in y_indices:
            for x in x_indices:
                # 計算該特徵點在深度影像中的值
                depth_value = center_region[y, x]

                # 決定白色圓形的半徑（1 到 6 像素，越小越大）
                radius = int(6*np.exp(-2*depth_value**2))  # 深度值越小，半徑越大

                # 轉換到原圖上的座標
                original_y = y + start_h
                original_x = x + start_w

                # 繪製白色圓點
                # cv2.circle(self.rgbimg, (original_x, original_y), radius, (100, 255, 255), -1)  # 白色 (B, G, R)
                cv2.circle(self.depthimg, (original_x, original_y), radius, 0, -1)  # 黑色 (灰階為 0)