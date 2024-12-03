from PyQt5 import QtCore, QtWidgets

class my_UI(object):
    def __init__(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowTitle("Roly")
        MainWindow.resize(1050, 550)
        self.mainwidget = QtWidgets.QWidget(MainWindow)
        self.mainwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.mainwidget)

        self.init_ui_groupbox1()

    def init_ui_groupbox1(self): # groupbox 1
        self.groupBox1 = QtWidgets.QGroupBox(self.mainwidget)
        self.groupBox1.setGeometry(QtCore.QRect(20, 20, 270, 140))
        self.groupBox1.setObjectName("groupBox1")
        self.groupBox1.setTitle("groupBox1")

        self.text_1_1 = QtWidgets.QLabel(self.groupBox1)
        self.text_1_1.setGeometry(QtCore.QRect(20, 30, 50, 30))
        self.text_1_1.setObjectName("Camera")
        self.text_1_1.setText("Camera")
        self.Camera_start = QtWidgets.QPushButton(self.groupBox1)
        self.Camera_start.setGeometry(QtCore.QRect(80, 30, 80, 30))
        self.Camera_start.setObjectName("Camera_start")
        self.Camera_start.setText("Start")
        self.Camera_close = QtWidgets.QPushButton(self.groupBox1)
        self.Camera_close.setGeometry(QtCore.QRect(170, 30, 80, 30))
        self.Camera_close.setObjectName("Camera_close")
        self.Camera_close.setText("Close")