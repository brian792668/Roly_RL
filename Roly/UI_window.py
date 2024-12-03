from PyQt5 import QtWidgets
import UI
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imports.Camera import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = UI.my_UI(self)
        self.setup_button()
        self.head_camera = Camera()

    def setup_button(self):
        self.ui.Camera_start.clicked.connect(self.Camera_start)
        self.ui.Camera_close.clicked.connect(self.Camera_close)

    def Camera_start(self):
        self.head_camera.get_img(rgb=True, depth=True)
        self.head_camera.get_target(depth=True)
        self.head_camera.show(rgb=True, depth=True)
        print("1")

    def Camera_close(self):
        print("0")