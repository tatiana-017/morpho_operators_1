from ast import Import
from asyncio.windows_events import NULL
import os 
import sys
import numpy as np
import pydicom as dcm
import mypdi_module as pdi

 
## Importamos las librerias de diseño de apps
from PyQt5.QtCore import*
from PyQt5.QtGui import*
from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi

class BasicWindow(QMainWindow):
    __img = []

    def __init__(self):
        QMainWindow.__init__(self)
        self.initializeUi()

    def initializeUi(self):
        txtpath = os.path.dirname(os.path.abspath(__file__))
        loadUi(os.path.join(txtpath, "mypdi.ui"),self)
        self.pushButton1.clicked.connect(self.readImage)
        self.pushButton2.clicked.connect(self.equalizeImage)
        self.pushButton2.setEnabled(False)
        self.show()

    def equalizeImage(self):
        img = self.__img.copy()
        img = pdi.normImage(img, 2000)
        _, cdf = pdi.perform_hist_equalizer(img)
        imgequ = pdi.performHistTrans(img, cdf)
        qimg16 = self.convertImageQPixmap(imgequ)
        self.label2.setPixmap(qimg16)
        self.label2.setScaledContents(True)

    def readImage(self):
        global fname 
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'd:\\', "Image File (*.dcm)")
        if fname == NULL:
            return
        objdcm = dcm.dcmread(fname[0])
        self.__img = objdcm.pixel_array
        qimg16 = self.convertImageQPixmap(self.__img)
        self.label1.setPixmap(qimg16)
        self.label1.setScaledContents(True)
        self.pushButton2.setEnabled(True)

    def convertImageQPixmap(self, img):
        width, height = img.shape
        imgq16 = pdi.normImage(img, 65535)
        qimg16 = QImage(imgq16, width, height, QImage.Format_Grayscale16)
        pixmap = QPixmap(qimg16)
        return pixmap

app = QApplication(sys.argv)
win = BasicWindow()
sys.exit(app.exec_())
