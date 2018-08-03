## #!/usr/local/bin/python3

from __future__ import print_function
import sys
import os

try:
	# PyQt4
	from PyQt4 import QtCore, QtGui
	from PyQt4.QtGui import *
	from PyQt4.QtCore import QSettings
	from PyQt4.uic import loadUiType
	print('Using PyQT4')
except ImportError:
	try:
		from PyQt5.QtCore import Qt,QSettings
		from PyQt5.QtWidgets import QApplication, QWidget,QMainWindow,QFileDialog
		from PyQt5.QtGui import QStandardItemModel,QStandardItem,QImage,QTransform,QPixmap
		from PyQt5.uic import loadUiType
		print('Using PyQT5')
	except ImportError:
		sys.exit('No PyQt found in our system!')

# VTK
import vtk
from scenemanager import SceneManager

# We'll need to access home directory
from os.path import expanduser

# numpy imports
import numpy as np
import scipy.io as sio

curdir,fname = os.path.split(__file__)
print('curdir:',curdir)
# load GUI
Ui_MainWindow = loadUiType(os.path.join(curdir,"mainwindow.ui"))[0]

class MainWindow(QMainWindow, Ui_MainWindow):

	def __init__(self, parent=None):
		super(MainWindow, self).__init__(parent)
		self.setupUi(self)
		self.initVTK()


	def initVTK(self):
		self.show()		# We need to call QVTKWidget's show function before initializing the interactor
		self.SceneManager = SceneManager(self.vtkContext)

	def FileOpen(self):
		title = "Open File"
		flags = QFileDialog.ShowDirsOnly
		dbpath = QFileDialog.getExistingDirectory(self,
						title,
						expanduser("~"),
						flags)

	def FileExit(self):
		app.quit()

	def ShowAboutDialog(self):
		title = QString("About Qt VTK Skeleton")
		text = QString("Qt Vtk Skeleton \n")
		text.append("Minimal code to implement a basic\n")
		text.append("Qt application with a VTK widget \n")
		text.append("\n")
		text.append("Developed by Toni Gurgui Valverde \n")
		aboutbox = QMessageBox(QMessageBox.Information,title,text,0,self,Qt.Sheet )
		aboutbox.show()
		#aboutbox.exec()

	def SetViewXY(self):
		# Ensure UI is sync (if view was selected from menu)
		self.radioButtonXY.setChecked(True)
		self.SceneManager.SetViewXY()

	def SetViewXZ(self):
		# Ensure UI is sync (if view was selected from menu)
		self.radioButtonXZ.setChecked(True)
		self.SceneManager.SetViewXZ()

	def SetViewYZ(self):
		# Ensure UI is sync (if view was selected from menu)
		self.radioButtonYZ.setChecked(True)
		self.SceneManager.SetViewYZ()

	def Snapshot(self):
		self.SceneManager.Snapshot()

	def ToggleVisualizeAxis(self, visible):
		# Ensure UI is sync
		self.actionVisualize_Axis.setChecked(visible)
		self.checkVisualizeAxis.setChecked(visible)
		self.SceneManager.ToggleVisualizeAxis(visible)

	def ToggleVisibility(self, visible):
		# Ensure UI is sync
		self.actionVisibility.setChecked(visible)
		self.checkVisibility.setChecked(visible)
		self.SceneManager.ToggleVisibility(visible)

if __name__ == '__main__':

	app = QApplication(sys.argv)

	mw = MainWindow()
	mw.show()

	sys.exit(app.exec_())
