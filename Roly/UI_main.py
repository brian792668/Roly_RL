from PyQt5 import QtWidgets
import UI_window
import sys

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = UI_window.mywindow()
    window.show()
    sys.exit(app.exec_())