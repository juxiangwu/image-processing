#coding:utf-8
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Develop\DeepLearning\deeplearning-learning\temp\eric6-workspaces\ImageProcessing\MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Develop\DeepLearning\deeplearning-learning\temp\eric6-workspaces\ImageProcessing\MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

import sys
import os
sys.path.append(os.getcwd())
from PyQt5 import QtCore, QtGui, QtWidgets
from  PyQt5.QtWidgets import QAction,QFileDialog,QApplication
from PyQt5.QtGui import QIcon
from utils import imageloader

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 680)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.labelImage = QtWidgets.QLabel(self.centralWidget)
        self.labelImage.setGeometry(QtCore.QRect(0, 0, 800, 600))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelImage.sizePolicy().hasHeightForWidth())
        self.labelImage.setSizePolicy(sizePolicy)
        self.labelImage.setMinimumSize(QtCore.QSize(800, 600))
        self.labelImage.setMaximumSize(QtCore.QSize(800, 600))
        self.labelImage.setBaseSize(QtCore.QSize(800, 600))
        self.labelImage.setText("")
        self.labelImage.setObjectName("labelImage")

        openFileAction = QAction(QIcon('resources/images/icon.jpg'),'Open', MainWindow)
        openFileAction.setShortcut('Ctrl+O')
        openFileAction.setStatusTip('Open Image')
        openFileAction.triggered.connect(self.loadImage)

        menubar = MainWindow.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFileAction)

        MainWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

    def loadImage(self):
        fname = QFileDialog.getOpenFileName(MainWindow, 'Open Image', 'C:/')
        if fname[0] == None or len(fname[0]) == 0:
            return
        print('loadImage:fname=',fname[0])
        image = imageloader.load_image(fname[0])
        pixelmap = imageloader.array2pixmap(image)
        self.labelImage.setPixmap(pixelmap)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


