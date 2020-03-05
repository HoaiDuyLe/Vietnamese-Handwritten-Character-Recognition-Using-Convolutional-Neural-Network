from PyQt5.QtCore import QDir, QPoint, QRect, QSize, Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

from nist_model import cnn_model
from utilities.predict import predict_char

import cv2
import numpy as np
import os
import sys

weight_path = os.path.join(os.getcwd(),'Model')

class MyPaint(QWidget):
    def __init__(self):
        super(MyPaint, self).__init__()
        self.myPenWidth = 5
        self.myPenColor = Qt.black
        self.maxWidth = 550
        self.maxHeight = 400
        self.imageSize = QSize(550,400)

        self.mEnable = False
        self.modified = False
        self.filebuf = None
        self.formatlist = ['png','PNG','jpg','JPG','jpeg','JPEG']

        self.image = QImage(self.imageSize, QImage.Format_RGB32)
        self.image.fill(Qt.white)

        self.lastPoint = QPoint()

        self.resultTab = QTabWidget(self)
        self.resultTab.setGeometry(600,0,200,400)
        self.setResultTab(self.resultTab)

    def setResultTab(self,resultTab):
        self.tab1 = QWidget()
        self.tab2 = QWidget()

        resultTab.addTab(self.tab1,'Tab 1')
        resultTab.addTab(self.tab2,'Tab 2')

        resultTab.setTabText(0,'Result')
        resultTab.setTabText(1,'Detail')

        self.settab1()
        self.settab2()

    def settab1(self):
        self.tab1Lab1_1 = QLabel()
        self.tab1Lab1_1.hide()
        self.tab1Lab1_1.setText('Predict: ')

        self.tab1Lab1_2 = QLabel()
        self.tab1Lab1_2.setStyleSheet("QLabel{color:red ; background-color:white}")
        self.tab1Lab1_2.setAlignment(Qt.AlignCenter)

        tab1Layout = QFormLayout()
        tab1Layout.addRow(self.tab1Lab1_1,self.tab1Lab1_2)
        self.tab1.setLayout(tab1Layout)

    def settab2(self):
        self.tab2Lab1_1 = QLabel()
        self.tab2Lab1_1.setStyleSheet("QLabel{color:red ; background-color:white}")

        self.tab2Lab1_2 = QLabel()
        self.tab2Lab1_2.setStyleSheet("QLabel{color:red ; background-color:white}")
        self.tab2Lab1_2.setAlignment(Qt.AlignCenter)

        self.tab2Lab2_1 = QLabel()
        self.tab2Lab2_1.setStyleSheet("QLabel{color:blue ; background-color:white}")

        self.tab2Lab2_2 = QLabel()
        self.tab2Lab2_2.setStyleSheet("QLabel{color:blue ; background-color:white}")
        self.tab2Lab2_2.setAlignment(Qt.AlignCenter)

        self.tab2Lab3_1 = QLabel()
        self.tab2Lab3_1.setStyleSheet("QLabel{color:green ; background-color:white}")

        self.tab2Lab3_2 = QLabel()
        self.tab2Lab3_2.setStyleSheet("QLabel{color:green ; background-color:white}")
        self.tab2Lab3_2.setAlignment(Qt.AlignCenter)

        self.tab2Lab4_1 = QLabel()
        self.tab2Lab4_1.setStyleSheet("QLabel{color:darkYellow ; background-color:white}")

        self.tab2Lab4_2 = QLabel()
        self.tab2Lab4_2.setStyleSheet("QLabel{color:darkYellow ; background-color:white}")
        self.tab2Lab4_2.setAlignment(Qt.AlignCenter)

        self.tab2Lab5_1 = QLabel()
        self.tab2Lab5_1.setStyleSheet("QLabel{color:darkCyan ; background-color:white}")

        self.tab2Lab5_2 = QLabel()
        self.tab2Lab5_2.setStyleSheet("QLabel{color:darkCyan ; background-color:white}")
        self.tab2Lab5_2.setAlignment(Qt.AlignCenter)

        tab2Layout = QFormLayout()
        tab2Layout.addRow(self.tab2Lab1_1,self.tab2Lab1_2)
        tab2Layout.addRow(self.tab2Lab2_1,self.tab2Lab2_2)
        tab2Layout.addRow(self.tab2Lab3_1,self.tab2Lab3_2)
        tab2Layout.addRow(self.tab2Lab4_1,self.tab2Lab4_2)
        tab2Layout.addRow(self.tab2Lab5_1,self.tab2Lab5_2)
        self.tab2.setLayout(tab2Layout)

    def clear(self):
        newimage = QImage(self.image.size(), QImage.Format_RGB32)
        newimage.fill(Qt.white)
        self.image = newimage
        self.modified = False
        self.update()

    def new(self):
        newwidth,_ =  QInputDialog.getInt(self, "New Image","Width (min: 100, max: %d)"%self.maxWidth, 200, 100, self.maxWidth, 10)
        newheight,_ =  QInputDialog.getInt(self, "New Image","Height (min:100, max: %d)"%self.maxHeight, 200, 100, self.maxHeight, 10)
        newsize = QSize(newwidth,newheight)
        newimage = QImage(newsize, QImage.Format_RGB32)

        newimage.fill(Qt.white)
        self.image = newimage
        self.modified = False
        self.update()

    def save(self):
        if self.modified == False:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            filename, extent = QFileDialog.getSaveFileName(self,'Save Image File','',\
                                                            'PNG Images (*.png, *.PNG);;\
                                                            JPEG Images (*.jpeg, *.jpg, *.JPEG, *.JPG)',\
                                                            options = options)
            if filename:
                ext = extent.split(' ')
                ext = ext[0]
                if self.isExtension(filename) == True:
                    filename = filename
                else:
                    filename = filename + '.%s'%(ext.lower())

                self.filebuf = (filename,ext)
                self.image.save(filename,ext)
                self.modified = True
        else:
            filename, ext = self.filebuf
            self.image.save(filename,ext)
            self.modified = True

    def open(self):
        loadimage = QImage()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, extent = QFileDialog.getOpenFileName(self,'Open Image File','',\
                                                        'PNG Images (*.png, *.PNG);;\
                                                        JPEG Images (*.jpeg, *.jpg, *.JPEG, *.JPG)',\
                                                        options = options)
        if filename:
            ext = extent.split(' ')
            ext = ext[0].lower()
            loadimage.load(filename,ext)
            self.filebuf = (filename,ext)
            if (loadimage.width() > self.maxWidth) or (loadimage.height() > self.maxHeight):
                loadimage = loadimage.scaled(self.image.size())

            self.image = loadimage
            self.modified = True
            self.update()

    def predict(self):
        image = self.getpixel(self.image)
        result,prob = predict_char(image,cnn_model,weight_path)

        self.tab1Lab1_1.show()
        self.tab1Lab1_2.setText(result)
        self.tab2Lab1_1.setText(prob[0][0])
        self.tab2Lab1_2.setText('%.4f'%prob[0][1])
        self.tab2Lab2_1.setText(prob[1][0])
        self.tab2Lab2_2.setText('%.4f'%prob[1][1])
        self.tab2Lab3_1.setText(prob[2][0])
        self.tab2Lab3_2.setText('%.4f'%prob[2][1])
        self.tab2Lab4_1.setText(prob[3][0])
        self.tab2Lab4_2.setText('%.4f'%prob[3][1])
        self.tab2Lab5_1.setText(prob[4][0])
        self.tab2Lab5_2.setText('%.4f'%prob[4][1])

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.mEnable = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.mEnable:
            self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.mEnable:
            self.drawLineTo(event.pos())
            self.mEnable = False

    def paintEvent(self, event):
        painter = QPainter(self)
        dirtyRect = event.rect()
        painter.drawImage(dirtyRect, self.image, dirtyRect)

    def drawLineTo(self, endPoint):
        painter = QPainter(self.image)
        painter.setPen(QPen(self.myPenColor, self.myPenWidth, Qt.SolidLine,\
                Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(self.lastPoint, endPoint)

        self.lastPoint = QPoint(endPoint)
        self.update()

    def isExtension(self,filename):
        for ext in self.formatlist:
            if ext in filename:
                return True
        return False

    def getpixel(self,image):
        output = []
        for i in range(image.height()):
            for j in range(image.width()):
                rgb = image.pixel(j,i)
                gray = QColor(rgb).red()  #red
                output.append(gray)
        output = np.asarray(output)
        output = np.reshape(output,(image.height(),image.width()))
        return output

class Recognizion(QMainWindow):
    def __init__(self):
        super(Recognizion,self).__init__()
        self.myPaint = MyPaint()
        self.setCentralWidget(self.myPaint)
        self.setUI()

    def setUI(self):
        self.setGeometry(0,0,850,600)
        self.setWindowTitle('Vietnamese Handwritten Character Recognizion')

        self.painttoolbar = self.addToolBar('File')
        new = QAction(QIcon('Image/new.png'),'New',self)
        self.painttoolbar.addAction(new)
        new.triggered.connect(self.myPaint.new)

        open = QAction(QIcon('Image/open.png'),'Open',self)
        self.painttoolbar.addAction(open)
        open.triggered.connect(self.myPaint.open)

        save = QAction(QIcon('Image/save.png'),'Save',self)
        self.painttoolbar.addAction(save)
        save.triggered.connect(self.myPaint.save)
        self.painttoolbar.addSeparator()

        self.brushbox = QComboBox()
        self.brushbox.addItems(['5','6','7','8','9','10','11','12','13','14','15','16','17','20','25','30'])
        self.brushbox.currentIndexChanged.connect(self.change_brush)
        self.boxlab = QLabel()
        self.boxlab.setText('Brush width:   ')

        self.painttoolbar.addWidget(self.boxlab)
        self.painttoolbar.addWidget(self.brushbox)

        self.infoBox = QGroupBox('Information',self)
        self.infoBox.setGeometry(0,450,500,125)
        self.infoBox.setStyleSheet('''QGroupBox {color:red} ''')
        self.infoBox.setAlignment(Qt.AlignCenter)

        self.infoLab = QLabel(self)
        self.infoLab.setText('Vietnamese Handwritten Character Recognizion Project - 2018' + \
                            '\nLe Hoai Duy' + \
                            '\nNguyen Thanh Nhan' + \
                            '\nBach khoa University')
        self.infoLab.setGeometry(20,475,450,75)
        self.infoLab.setStyleSheet('''QLabel {color:blue }''')

        butWidget = QWidget(self)
        butWidget.setGeometry(550,550,300,50)
        self.predPBut = QPushButton(self)
        self.predPBut.setText('Recognize')
        self.predPBut.setStyleSheet('QPushButton {color:white ; background-color:darkBlue}')
        self.predPBut.clicked.connect(self.myPaint.predict)

        self.clearPBut = QPushButton(self)
        self.clearPBut.setText('Clear')
        self.clearPBut.setStyleSheet('QPushButton {color:white ; background-color:darkCyan}')
        self.clearPBut.clicked.connect(self.myPaint.clear)

        self.exitPBut = QPushButton(self)
        self.exitPBut.setText('Exit')
        self.exitPBut.setStyleSheet('QPushButton {color:white ; background-color:darkRed}')
        self.exitPBut.clicked.connect(self.exit)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.predPBut)
        hlayout.addWidget(self.clearPBut)
        hlayout.addWidget(self.exitPBut)
        butWidget.setLayout(hlayout)

    def exit(self):
        self.close()

    def change_brush(self):
        self.myPaint.myPenWidth = int(self.brushbox.currentText())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Recognizion()
    window.show()
    sys.exit(app.exec_())
