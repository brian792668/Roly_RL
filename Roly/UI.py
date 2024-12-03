from PyQt5 import QtCore, QtWidgets

class Roly_UI(object):
    def __init__(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowTitle("Roly")
        MainWindow.resize(350, 330)
        self.mainwidget = QtWidgets.QWidget(MainWindow)
        self.mainwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.mainwidget)

        self.init_ui_groupbox1()

    def init_ui_groupbox1(self): # groupbox 1
        self.groupBox1 = QtWidgets.QGroupBox(self.mainwidget)
        self.groupBox1.setGeometry(QtCore.QRect(20, 20, 310, 230))
        self.groupBox1.setObjectName("groupBox1")

        self.camera_status_light = QtWidgets.QLabel(self.groupBox1)
        self.camera_status_light.setGeometry(QtCore.QRect(25, 40, 10, 10))
        self.camera_status_light.setStyleSheet(""" background-color: gray;  border-radius: 5px;  border: 2px solid white;""")
        self.text_Camera = QtWidgets.QLabel(self.groupBox1)
        self.text_Camera.setGeometry(QtCore.QRect(45, 30, 50, 30))
        self.text_Camera.setText("Camera")
        self.camera_start = QtWidgets.QPushButton(self.groupBox1)
        self.camera_start.setGeometry(QtCore.QRect(105, 30, 80, 30))
        self.camera_start.setText("Start")
        self.camera_stop = QtWidgets.QPushButton(self.groupBox1)
        self.camera_stop.setGeometry(QtCore.QRect(195, 30, 80, 30))
        self.camera_stop.setText("Stop")

        self.RL_status_light = QtWidgets.QLabel(self.groupBox1)
        self.RL_status_light.setGeometry(QtCore.QRect(25, 90, 10, 10))
        self.RL_status_light.setStyleSheet("""background-color: gray;  border-radius: 5px;  border: 2px solid white; """)
        self.text_RL = QtWidgets.QLabel(self.groupBox1)
        self.text_RL.setGeometry(QtCore.QRect(45, 80, 50, 30))
        self.text_RL.setText("RL")
        self.RL_start = QtWidgets.QPushButton(self.groupBox1)
        self.RL_start.setGeometry(QtCore.QRect(105, 80, 80, 30))
        self.RL_start.setText("Start")
        self.RL_stop = QtWidgets.QPushButton(self.groupBox1)
        self.RL_stop.setGeometry(QtCore.QRect(195, 80, 80, 30))
        self.RL_stop.setText("Stop")

        self.motor_status_light = QtWidgets.QLabel(self.groupBox1)
        self.motor_status_light.setGeometry(QtCore.QRect(25, 140, 10, 10))
        self.motor_status_light.setStyleSheet("""background-color: gray;  border-radius: 5px;  border: 2px solid white; """)
        self.text_motor = QtWidgets.QLabel(self.groupBox1)
        self.text_motor.setGeometry(QtCore.QRect(45, 130, 50, 30))
        self.text_motor.setText("Motor")
        self.motor_start = QtWidgets.QPushButton(self.groupBox1)
        self.motor_start.setGeometry(QtCore.QRect(105, 130, 80, 30))
        self.motor_start.setText("Start")
        self.motor_stop = QtWidgets.QPushButton(self.groupBox1)
        self.motor_stop.setGeometry(QtCore.QRect(195, 130, 80, 30))
        self.motor_stop.setText("Stop")


        self.stop_all = QtWidgets.QPushButton(self.mainwidget)
        self.stop_all.setGeometry(QtCore.QRect(250, 280, 80, 30))
        self.stop_all.setText("Stop All")