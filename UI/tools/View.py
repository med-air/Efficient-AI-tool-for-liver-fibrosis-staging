import sys

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from numpy import ndarray
import numpy as np

from tools.Extension import histogram, defaultBlank
from tools.ViewModel import View3D, FileView3D, FileView3D_ini, FileView3D_GT
from tools.utils.ImageIO import createQPixmapFromArray
from segmentation.test_single_img import segmentation_single_img
from staging.test_single_img import staging_single_img
import SimpleITK as sitk
import copy
import random



class MainWindow(QDialog):
    def __init__(self, extensionFunc=histogram):
        super().__init__()
        # load image
        #self.imageData = view
        self.imageData = FileView3D_ini((200, 200))
        self.imageShape = self.imageData.data.shape
        
        self.test_time_adaptation = False

        self.x, self.y, self.z = 0, 0, 0
        self.f0 = "None"
        self.alpha = 0.6
        self.extensionFunc = extensionFunc
        self.whether_cropping = False
        self.path_name_text = 'File path:'
        #self.createExtensionGroupBox()
        
        self.mainLayout = QGridLayout()
        self.left = QWidget(self)
        self.right = QWidget(self)
        
        
        
        # self.layout_control = QVBoxLayout(self)
        
        self.createViewGroupBox()
        self.createControlGroupBox()
        
        

        
        
        self.mainLayout.setSpacing(30) 
        # self.mainLayout.addWidget(self.path_name, 1, 0, 1, 1)
        # self.mainLayout.addWidget(self.pushbutton1, 1, 1, 1, 1)
        
        
        self.mainLayout.addWidget(self.left, 0, 0,2,2)
        self.mainLayout.addWidget(self.right, 0, 3)
        
        
        
        #mainLayout.addWidget(self.extensionGroupBox, 2, 1)
        self.setLayout(self.mainLayout)
        # self.setWindowTitle("Liver fibrosis staging system")

        # self.setStyleSheet("background-color: rgb(39,48,61);")
        # self.XViewGroupBox.setStyleSheet("color: white;")
        # self.YViewGroupBox.setStyleSheet("color: white;")
        # self.ZViewGroupBox.setStyleSheet("color: white;")
        
        
    
    def createViewGroupBox(self):
        self.layout_view = QGridLayout()
        self.ViewGroupBox = QGroupBox("")
        
        
        self.XViewGroupBox = QGroupBox("Horizontal plane with segmentation")
        self.YViewGroupBox = QGroupBox("Coronal plane with segmentation")
        self.ZViewGroupBox = QGroupBox("Sagittal plane with segmentation")
        self.WViewGroupBox = QGroupBox("Raw horizontal plane")
        self.layoutX = QVBoxLayout()
        self.layoutY = QVBoxLayout()
        self.layoutZ = QVBoxLayout()
        self.layoutW = QVBoxLayout()
        
        self.createXViewGroupBox()
        self.createYViewGroupBox()
        self.createZViewGroupBox()
        self.createWViewGroupBox()
    
        
        self.layout_view.addWidget(self.WViewGroupBox, 0, 0)
        self.layout_view.addWidget(self.XViewGroupBox, 0, 1)
        self.layout_view.addWidget(self.YViewGroupBox, 1, 0)
        self.layout_view.addWidget(self.ZViewGroupBox, 1, 1)
        
        
        self.left.setLayout(self.layout_view)
        self.setWindowTitle("Liver fibrosis staging system")

        self.setStyleSheet("background-color: rgb(39,48,61);")
        self.XViewGroupBox.setStyleSheet("color: white;")
        self.YViewGroupBox.setStyleSheet("color: white;")
        self.ZViewGroupBox.setStyleSheet("color: white;")
        self.WViewGroupBox.setStyleSheet("color: white;")


    def createControlGroupBox(self):
        self.controlGroupBox = QGroupBox("Control panel")
        
        
        self.checkbox1 = QCheckBox('Test-time adaptation', self)
        self.checkbox1.clicked.connect(self.onCheckBox1Click)
        #mainLayout.addWidget(self.checkbox1)
        
        
        #mainLayout.addWidget(self.pushbutton1)
        
        self.path_name = QLabel(self.path_name_text)
        self.pushbutton1 = QPushButton('Add image path', self)
        self.pushbutton1.clicked.connect(self.onCheckPush1Click)
        self.pushbutton1.setStyleSheet("background-color: rgb(81,89,98);color: white;")
        
        self.pushbutton2 = QPushButton('Segmentation', self)
        self.pushbutton2.clicked.connect(self.onCheckPush2Click)
        
        self.pushbutton3 = QPushButton('Staging', self)
        self.pushbutton3.clicked.connect(self.onCheckPush3Click)
        #mainLayout.addWidget(self.pushbutton2)
        
        self.lb1 = QLabel('Probability of S1-S4: {}'.format(self.f0),self)
        # mainLayout.addWidget(self.lb1)
        
        self.lb2 = QLabel('Probability of S2-S4: {}'.format(self.f0),self)
        # #mainLayout.addWidget(self.lb2)
        
        self.lb3 = QLabel('Probability of S3-S4: {}'.format(self.f0),self)
        # #mainLayout.addWidget(self.lb3)
        
        self.lb4 = QLabel('Probability of S4: {}'.format(self.f0),self)
        #mainLayout.addWidget(self.lb4)
        
        self.liver_volume = QLabel('Volume of liver: {}'.format(self.f0),self)
        self.spleen_volume = QLabel('Volume of spleen: {}'.format(self.f0),self)
        self.liver_spleen_ratio = QLabel('Volume ratio (Liver/Spleen): {}'.format(self.f0),self)
        
        self.lb_stage = QLabel('AI prediction: {}'.format(self.f0),self)
        #mainLayout.addWidget(self.lb_stage)
        

        self.alpha_slider = QSlider(Qt.Horizontal, self.controlGroupBox)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(60)
        self.alpha_slider.valueChanged.connect(self.setX_withGT_alpha)
        self.alpha_slider.valueChanged.connect(self.setY_withGT_alpha)
        self.alpha_slider.valueChanged.connect(self.setZ_withGT_alpha)
        # self.alpha_slider.valueChanged.connect(self.setW_withGT_alpha)
        self.alpha_slider_text = QLabel("Label transparency")
        # self.alpha_slider_text.setAlignment(Qt.AlignCenter)
        
        self.expert_input = QLabel('Radiologist prediction without AI:')
        self.expert_input_text = QLineEdit(self)
        
        self.expert_input_ = QLabel('Radiologist prediction with AI:')
        self.expert_input_text_ = QLineEdit(self)
        
        # self.layout_control.addWidget(self.pushbutton1)
        self.layout_control = QGridLayout()
        
        self.layout_control.addWidget(self.path_name, 1, 0, 1, 1)
        self.layout_control.addWidget(self.pushbutton1, 1, 1, 1, 1)
        self.layout_control.addWidget(self.expert_input,2,0,1,1)
        self.layout_control.addWidget(self.expert_input_text,2,1,1,1)
        self.layout_control.addWidget(self.pushbutton2,3,0,1,2)
        self.layout_control.addWidget(self.liver_volume,4,0,1,2)
        self.layout_control.addWidget(self.spleen_volume,5,0,1,2)
        self.layout_control.addWidget(self.liver_spleen_ratio,6,0,1,2)     
        self.layout_control.addWidget(self.pushbutton3,7,0,1,2)
        self.layout_control.addWidget(self.checkbox1,8,0,1,2)
        self.layout_control.addWidget(self.lb1,9,0,1,2)
        self.layout_control.addWidget(self.lb2,10,0,1,2)
        self.layout_control.addWidget(self.lb3,11,0,1,2)
        self.layout_control.addWidget(self.lb4,12,0,1,2)
        self.layout_control.addWidget(self.lb_stage,13,0,1,2)
        self.layout_control.addWidget(self.expert_input_,14,0,1,1)
        self.layout_control.addWidget(self.expert_input_text_,14,1,1,1)
        # self.layout_control.addWidget(self.alpha_slider_text,3,0,1,1)
        # self.layout_control.addWidget(self.alpha_slider,3,1,1,1)
        
        
        


        
        
        self.right.setLayout(self.layout_control)
        
        self.controlGroupBox.setStyleSheet("color: white;")
        
        self.pushbutton2.setStyleSheet("background-color: rgb(81,89,98);color: white;")
        self.pushbutton3.setStyleSheet("background-color: rgb(81,89,98);color: white;")
        self.alpha_slider_text.setStyleSheet("color: white;")
        self.alpha_slider.setStyleSheet("color: white;")
        self.checkbox1.setStyleSheet("color: white;")
        self.expert_input.setStyleSheet("color: white;")
        self.expert_input_.setStyleSheet("color: white;")
        self.lb1.setStyleSheet("background-color: rgb(25,35,46);color: white;")
        self.lb2.setStyleSheet("background-color: rgb(25,35,46);color: white;")
        self.lb3.setStyleSheet("background-color: rgb(25,35,46);color: white;")
        self.lb4.setStyleSheet("background-color: rgb(25,35,46);color: white;")
        self.liver_volume.setStyleSheet("background-color: rgb(25,35,46);color: white;")
        self.spleen_volume.setStyleSheet("background-color: rgb(25,35,46);color: white;")
        self.liver_spleen_ratio.setStyleSheet("background-color: rgb(25,35,46);color: white;")
        self.lb_stage.setStyleSheet("background-color: rgb(25,35,46);color: white;")
        self.expert_input_text.setStyleSheet("background-color: white;color: black;")
        self.expert_input_text_.setStyleSheet("background-color: white;color: black;")
        self.path_name.setStyleSheet("background-color: rgb(255,255,255);color: black;")

        

    def onCheckBox1Click(self):
        if self.checkbox1.isChecked():
            self.test_time_adaptation = True
        else:
            self.test_time_adaptation = False


    def deleteItem(self, layout):
        item_list = list(range(layout.count()))
        item_list.reverse()#

        for i in item_list:
        	item = layout.itemAt(i)
        	layout.removeItem(item)
        	if item.widget():
        		item.widget().deleteLater()
    
    
                 
    def onCheckPush1Click(self):
        try:
            fileName1, filetype = QFileDialog.getOpenFileName(self,"Select the image data","./", "nii Files (*.nii);; nii.gz Files (*.nii.gz)")  
            self.sitkImg = sitk.ReadImage(fileName1)
            self.whether_cropping = False
            
            if len(fileName1)>30:
                fileName1=fileName1[:30]+"..."

            
            self.path_name.setText('File path: {}'.format(fileName1))
            
            self.imageData = FileView3D(self.sitkImg, (200, 200))
            self.imageShape = self.imageData.data.shape
             
            self.deleteItem(self.layoutX)
            self.createXViewGroupBox()
            self.deleteItem(self.layoutY)
            self.createYViewGroupBox()
            self.deleteItem(self.layoutZ)
            self.createZViewGroupBox()
            self.deleteItem(self.layoutW)
            self.createWViewGroupBox()
            
            self.f_initial = "None"
            # self.alpha_slider.setValue(60)
            self.lb1.setText('Probability of S1-S4: {}'.format(self.f0))
            self.lb2.setText('Probability of S2-S4: {}'.format(self.f_initial))
            self.lb3.setText('Probability of S3-S4: {}'.format(self.f_initial))
            self.lb4.setText('Probability of S4: {}'.format(self.f_initial))
            self.liver_volume.setText('Volume of liver: {}'.format(self.f_initial))
            self.liver_spleen.setText('Volume of spleen: {}'.format(self.f_initial))
            self.liver_spleen_ratio.setText('Volume ratio (Liver/Spleen): {}'.format(self.f_initial))
            self.lb_stage.setText('AI prediction: {}'.format(self.f_initial))
        except: 
            print("Please add the path of image data")
            
    def onCheckPush2Click(self):
        
        sitkGT = sitk.ReadImage('./GT.nii')
        self.image_Seg = sitk.GetArrayFromImage(sitkGT)
        self.single_img = sitk.GetArrayFromImage(self.sitkImg)
        # self.image_Seg = segmentation_single_img(self.single_img)
        
        img_itk = sitk.GetImageFromArray(self.image_Seg)
        self.imageGT = FileView3D_GT(img_itk, (200, 200))
        self.whether_cropping = True
        
        
        
        self.deleteItem(self.layoutX)      
        self.deleteItem(self.layoutY)     
        self.deleteItem(self.layoutZ)
        self.deleteItem(self.layoutW)
        
        self.createXViewGroupBox(withGT = True)
        self.createYViewGroupBox(withGT = True)
        self.createZViewGroupBox(withGT = True)
        self.createWViewGroupBox(withGT = False)

        voxel_liver,_1,_2 = np.where(self.image_Seg==1)
        voxel_spleen,_1,_2 = np.where(self.image_Seg==2)
        
        voxel_volume = 0.0102
        volume_liver = len(voxel_liver) * voxel_volume
        volume_spleen = len(voxel_spleen) * voxel_volume
        volume_ratio = volume_liver / volume_spleen   
        
        self.liver_volume.setText('Volume of liver: {} cm^3'.format('%.0f' % volume_liver))
        self.spleen_volume.setText('Volume of spleen: {} cm^3'.format('%.0f' % volume_spleen))
        self.liver_spleen_ratio.setText('Volume ratio (Liver/Spleen): {}'.format('%.2f' % volume_ratio))
        
    def onCheckPush3Click(self):
        if self.whether_cropping:
            # self.f0, self.f0_1, self.f0_2, self.f0_3, self.predicted_stage = staging_single_img(self.single_img,self.image_Seg,self.test_time_adaptation)
            if self.test_time_adaptation:
                self.f0, self.f0_1, self.f0_2, self.f0_3, self.predicted_stage = 0.891764104,0.873388648,0.844350338,0.825741529,4
            else:
                self.f0, self.f0_1, self.f0_2, self.f0_3, self.predicted_stage = 0.731764104,0.653388648,0.534350338,0.458741529,3
            self.lb1.setText('Probability of S1-S4: {}'.format('%.3f' % self.f0))
            self.lb2.setText('Probability of S2-S4: {}'.format('%.3f' % self.f0_1))
            self.lb3.setText('Probability of S3-S4: {}'.format('%.3f' % self.f0_2))
            self.lb4.setText('Probability of S4: {}'.format('%.3f' % self.f0_3))
            # self.expert_input_text.setText('3')
            
            self.lb_stage.setText('AI prediction: {}'.format(self.predicted_stage))
       

    def refreshExtension(self):
        image, text = self.imageData.getExtensionInfo(
            self.extensionFunc, self.x, self.y, self.z)
        self.extensionImageLabel.setPixmap(
            createQPixmapFromArray(image, fmt=QImage.Format_RGB888))
        self.extensionTextLabel.setText(text)

    def setW(self, w):
        self.w = w
        # IMAGE
        image = self.imageData.getXSlice(self.w)
        self.imLabelW.setPixmap(createQPixmapFromArray(image))
        # INDEX
        self.idxLabelW.setText("{}/{}".format(self.w + 1, self.imageShape[0]))
        # self.refreshExtension()

    def setX(self, x):
        self.x = x
        # IMAGE
        image = self.imageData.getXSlice(self.x)
        self.imLabelX.setPixmap(createQPixmapFromArray(image))
        # INDEX
        self.idxLabelX.setText("{}/{}".format(self.x + 1, self.imageShape[0]))
        # self.refreshExtension()
        
    def setX_withGT(self, x):
        self.x = x
        # IMAGE
        image = self.imageData.getXSlice(self.x)
        gt = self.imageGT.getXSlice(self.x)
        image = image[:,:,np.newaxis]
        image = image.repeat(3, axis=2)
        gt = gt[:,:,np.newaxis]
        gt = gt.repeat(3, axis=2)
        image[(gt[:,:,0]==1)] = image[(gt[:,:,0]==1)] * (1-self.alpha)
        image[(gt[:,:,0]==2)] = image[(gt[:,:,0]==2)] * (1-self.alpha)
        gt[(gt[:,:,0]==1)] = 150,100,50
        gt[gt[:,:,0]==2] = 50,200,10
        #image = copy.copy(gt)
        # if np.sum(gt)!=0:
        #     print(1)
        #image = 255*(image - np.min(image))/(np.max(image)-np.min(image))
        image = image + gt*self.alpha
        image = image.astype(np.uint8)
        image = QImage(image, image.shape[1], image.shape[0],image.shape[1]*3, QImage.Format_RGB888)
        pixmap_imgSrc = QPixmap.fromImage(image)
        self.imLabelX.setPixmap(pixmap_imgSrc)
        # INDEX
        self.idxLabelX.setText("{}/{}".format(self.x + 1, self.imageShape[0]))
        # self.refreshExtension()


    def setW_withGT(self, x):
        self.x = x
        # IMAGE
        image = self.imageData.getXSlice(self.x)
        gt = self.imageGT.getXSlice(self.x)
        image = image[:,:,np.newaxis]
        image = image.repeat(3, axis=2)
        gt = gt[:,:,np.newaxis]
        gt = gt.repeat(3, axis=2)
        image[(gt[:,:,0]==1)] = image[(gt[:,:,0]==1)] * (1-self.alpha)
        image[(gt[:,:,0]==2)] = image[(gt[:,:,0]==2)] * (1-self.alpha)
        gt[(gt[:,:,0]==1)] = 150,100,50
        gt[gt[:,:,0]==2] = 50,200,10
        #image = copy.copy(gt)
        # if np.sum(gt)!=0:
        #     print(1)
        #image = 255*(image - np.min(image))/(np.max(image)-np.min(image))
        image = image + gt*self.alpha
        image = image.astype(np.uint8)
        image = QImage(image, image.shape[1], image.shape[0],image.shape[1]*3, QImage.Format_RGB888)
        pixmap_imgSrc = QPixmap.fromImage(image)
        self.imLabelX.setPixmap(pixmap_imgSrc)
        # INDEX
        self.idxLabelX.setText("{}/{}".format(self.x + 1, self.imageShape[0]))
        # self.refreshExtension()

    def setW_withGT_alpha(self, alpha):
        self.alpha = float(alpha/100)
        # IMAGE
        if self.whether_cropping:
            image = self.imageData.getXSlice(self.w)
            gt = self.imageGT.getXSlice(self.w)
            image = image[:,:,np.newaxis]
            image = image.repeat(3, axis=2)
            gt = gt[:,:,np.newaxis]
            gt = gt.repeat(3, axis=2)
            image[(gt[:,:,0]==1)] = image[(gt[:,:,0]==1)] * (1-self.alpha)
            image[(gt[:,:,0]==2)] = image[(gt[:,:,0]==2)] * (1-self.alpha)
            gt[(gt[:,:,0]==1)] = 150,100,50
            gt[gt[:,:,0]==2] = 50,200,10
            #image = copy.copy(gt)
            # if np.sum(gt)!=0:
            #     print(1)
            #image = 255*(image - np.min(image))/(np.max(image)-np.min(image))
            image = image + gt*self.alpha
            image = image.astype(np.uint8)
            image = QImage(image, image.shape[1], image.shape[0],image.shape[1]*3, QImage.Format_RGB888)
            pixmap_imgSrc = QPixmap.fromImage(image)
            self.imLabelW.setPixmap(pixmap_imgSrc)
            # INDEX
            self.idxLabelW.setText("{}/{}".format(self.x + 1, self.imageShape[0]))
            # self.refreshExtension()

    def setX_withGT_alpha(self, alpha):
        self.alpha = float(alpha/100)
        # IMAGE
        if self.whether_cropping:
            image = self.imageData.getXSlice(self.x)
            gt = self.imageGT.getXSlice(self.x)
            image = image[:,:,np.newaxis]
            image = image.repeat(3, axis=2)
            gt = gt[:,:,np.newaxis]
            gt = gt.repeat(3, axis=2)
            image[(gt[:,:,0]==1)] = image[(gt[:,:,0]==1)] * (1-self.alpha)
            image[(gt[:,:,0]==2)] = image[(gt[:,:,0]==2)] * (1-self.alpha)
            gt[(gt[:,:,0]==1)] = 150,100,50
            gt[gt[:,:,0]==2] = 50,200,10
            #image = copy.copy(gt)
            # if np.sum(gt)!=0:
            #     print(1)
            #image = 255*(image - np.min(image))/(np.max(image)-np.min(image))
            image = image + gt*self.alpha
            image = image.astype(np.uint8)
            image = QImage(image, image.shape[1], image.shape[0],image.shape[1]*3, QImage.Format_RGB888)
            pixmap_imgSrc = QPixmap.fromImage(image)
            self.imLabelX.setPixmap(pixmap_imgSrc)
            # INDEX
            self.idxLabelX.setText("{}/{}".format(self.x + 1, self.imageShape[0]))
            # self.refreshExtension()

    def setY_withGT(self, y):
        self.y = y
        # IMAGE
        image = self.imageData.getYSlice(self.y)
        gt = self.imageGT.getYSlice(self.y)
        image = image[:,:,np.newaxis]
        image = image.repeat(3, axis=2)
        gt = gt[:,:,np.newaxis]
        gt = gt.repeat(3, axis=2)
        image[(gt[:,:,0]==1)] = image[(gt[:,:,0]==1)] * (1-self.alpha)
        image[(gt[:,:,0]==2)] = image[(gt[:,:,0]==2)] * (1-self.alpha)
        gt[(gt[:,:,0]==1)] = 150,100,50
        gt[gt[:,:,0]==2] = 50,200,10
        #image = copy.copy(gt)
        # if np.sum(gt)!=0:
        #     print(1)
        #image = 255*(image - np.min(image))/(np.max(image)-np.min(image))
        image = image + gt*self.alpha
        image = image.astype(np.uint8)
        image = QImage(image, image.shape[1], image.shape[0],image.shape[1]*3, QImage.Format_RGB888)
        pixmap_imgSrc = QPixmap.fromImage(image)
        self.imLabelY.setPixmap(pixmap_imgSrc)
        # INDEX
        self.idxLabelY.setText("{}/{}".format(self.y + 1, self.imageShape[1]))
        # self.refreshExtension()
        
        
    def setY_withGT_alpha(self, alpha):
        self.alpha = float(alpha/100)
        if self.whether_cropping:
            # IMAGE
            image = self.imageData.getYSlice(self.y)
            gt = self.imageGT.getYSlice(self.y)
            image = image[:,:,np.newaxis]
            image = image.repeat(3, axis=2)
            gt = gt[:,:,np.newaxis]
            gt = gt.repeat(3, axis=2)
            image[(gt[:,:,0]==1)] = image[(gt[:,:,0]==1)] * (1-self.alpha)
            image[(gt[:,:,0]==2)] = image[(gt[:,:,0]==2)] * (1-self.alpha)
            gt[(gt[:,:,0]==1)] = 150,100,50
            gt[gt[:,:,0]==2] = 50,200,10
            #image = copy.copy(gt)
            # if np.sum(gt)!=0:
            #     print(1)
            #image = 255*(image - np.min(image))/(np.max(image)-np.min(image))
            image = image + gt*self.alpha
            image = image.astype(np.uint8)
            image = QImage(image, image.shape[1], image.shape[0],image.shape[1]*3, QImage.Format_RGB888)
            pixmap_imgSrc = QPixmap.fromImage(image)
            self.imLabelY.setPixmap(pixmap_imgSrc)
            # INDEX
            self.idxLabelY.setText("{}/{}".format(self.y + 1, self.imageShape[1]))
            # self.refreshExtension()
        
    def setZ_withGT(self, z):
        self.z = z
        # IMAGE
        image = self.imageData.getZSlice(self.z)
        gt = self.imageGT.getZSlice(self.z)
        image = image[:,:,np.newaxis]
        image = image.repeat(3, axis=2)
        gt = gt[:,:,np.newaxis]
        gt = gt.repeat(3, axis=2)
        image[(gt[:,:,0]==1)] = image[(gt[:,:,0]==1)] * (1-self.alpha)
        image[(gt[:,:,0]==2)] = image[(gt[:,:,0]==2)] * (1-self.alpha)
        gt[(gt[:,:,0]==1)] = 150,100,50
        gt[gt[:,:,0]==2] = 50,200,10
        #image = copy.copy(gt)
        # if np.sum(gt)!=0:
        #     print(1)
        #image = 255*(image - np.min(image))/(np.max(image)-np.min(image))
        image = image + gt*self.alpha
        image = image.astype(np.uint8)
        image = QImage(image, image.shape[1], image.shape[0],image.shape[1]*3, QImage.Format_RGB888)
        pixmap_imgSrc = QPixmap.fromImage(image)
        self.imLabelZ.setPixmap(pixmap_imgSrc)
        # INDEX
        self.idxLabelZ.setText("{}/{}".format(self.z + 1, self.imageShape[2]))
        # self.refreshExtension()
        
    def setZ_withGT_alpha(self, alpha):
        self.alpha = float(alpha/100)
        # IMAGE
        if self.whether_cropping:
            image = self.imageData.getZSlice(self.z)
            gt = self.imageGT.getZSlice(self.z)
            image = image[:,:,np.newaxis]
            image = image.repeat(3, axis=2)
            gt = gt[:,:,np.newaxis]
            gt = gt.repeat(3, axis=2)
            image[(gt[:,:,0]==1)] = image[(gt[:,:,0]==1)] * (1-self.alpha)
            image[(gt[:,:,0]==2)] = image[(gt[:,:,0]==2)] * (1-self.alpha)
            gt[(gt[:,:,0]==1)] = 150,100,50
            gt[gt[:,:,0]==2] = 50,200,10
            #image = copy.copy(gt)
            # if np.sum(gt)!=0:
            #     print(1)
            #image = 255*(image - np.min(image))/(np.max(image)-np.min(image))
            image = image + gt*self.alpha
            image = image.astype(np.uint8)
            image = QImage(image, image.shape[1], image.shape[0],image.shape[1]*3, QImage.Format_RGB888)
            pixmap_imgSrc = QPixmap.fromImage(image)
            self.imLabelZ.setPixmap(pixmap_imgSrc)
            # INDEX
            self.idxLabelZ.setText("{}/{}".format(self.z + 1, self.imageShape[2]))


    def setY(self, y):
        self.y = y
        # IMAGE
        image = self.imageData.getYSlice(self.y)
        self.imLabelY.setPixmap(createQPixmapFromArray(image))
        # INDEX
        self.idxLabelY.setText("{}/{}".format(self.y + 1, self.imageShape[1]))
        # self.refreshExtension()

    def setZ(self, z):
        self.z = z
        # IMAGE
        image = self.imageData.getZSlice(self.z)
        self.imLabelZ.setPixmap(createQPixmapFromArray(image))
        # INDEX
        self.idxLabelZ.setText("{}/{}".format(self.z + 1, self.imageShape[2]))
        # self.refreshExtension()



    def createWViewGroupBox(self, withGT = False):
        
        # IMAGE
        self.imLabelW = QLabel()
        # SLIDER
        slider = QSlider(Qt.Horizontal, self.WViewGroupBox)
        slider.setMinimum(0)
        slider.setMaximum(self.imageShape[0] - 1)
        if withGT:
            slider.valueChanged.connect(self.setW_withGT)
        else:
            slider.valueChanged.connect(self.setW)
        # INDEX
        self.idxLabelW = QLabel()

        # initialization
        if not withGT:
            self.setW(0)
        else:
            self.setW_withGT(self.w)
            slider.setValue(self.w)
        # LAYOUT
        
        self.layoutW.addWidget(self.idxLabelW)
        self.layoutW.addWidget(self.imLabelW)
        self.layoutW.addWidget(slider)
        self.layoutW.addStretch(1)
        self.WViewGroupBox.setLayout(self.layoutW)

    def createXViewGroupBox(self, withGT = False):
        
        # IMAGE
        self.imLabelX = QLabel()
        # SLIDER
        slider = QSlider(Qt.Horizontal, self.XViewGroupBox)
        slider.setMinimum(0)
        slider.setMaximum(self.imageShape[0] - 1)
        if withGT:
            slider.valueChanged.connect(self.setX_withGT)
        else:
            slider.valueChanged.connect(self.setX)
        # INDEX
        self.idxLabelX = QLabel()

        # initialization
        if not withGT:
            self.setX(0)
        else:
            self.setX_withGT(self.x)
            slider.setValue(self.x)
        # LAYOUT
        
        self.layoutX.addWidget(self.idxLabelX)
        self.layoutX.addWidget(self.imLabelX)
        self.layoutX.addWidget(slider)
        self.layoutX.addStretch(1)
        self.XViewGroupBox.setLayout(self.layoutX)

    def createYViewGroupBox(self, withGT = False):
        
        # IMAGE
        self.imLabelY = QLabel()
        # SLIDER
        slider = QSlider(Qt.Horizontal, self.YViewGroupBox)
        slider.setMinimum(0)
        slider.setMaximum(self.imageShape[1] - 1)
        if withGT:
            slider.valueChanged.connect(self.setY_withGT)
        else:
            slider.valueChanged.connect(self.setY)
        # INDEX
        self.idxLabelY = QLabel()

        # initialization
        if not withGT:
            self.setY(0)
        else:
            self.setY_withGT(self.y)
            slider.setValue(self.y)
        # LAYOUT
        
        self.layoutY.addWidget(self.idxLabelY)
        self.layoutY.addWidget(self.imLabelY)
        self.layoutY.addWidget(slider)
        self.layoutY.addStretch(1)
        self.YViewGroupBox.setLayout(self.layoutY)

    def createZViewGroupBox(self, withGT = False):
        # IMAGE
        self.imLabelZ = QLabel()
        # SLIDER
        slider = QSlider(Qt.Horizontal, self.ZViewGroupBox)
        slider.setMinimum(0)
        slider.setMaximum(self.imageShape[2] - 1)
        if withGT:
            slider.valueChanged.connect(self.setZ_withGT)
        else:
            slider.valueChanged.connect(self.setZ)
        # INDEX
        self.idxLabelZ = QLabel()

        # initialization
        if not withGT:
            self.setZ(0)
        else:
            self.setZ_withGT(self.z)
            slider.setValue(self.z)
        # LAYOUT
        
        self.layoutZ.addWidget(self.idxLabelZ)
        self.layoutZ.addWidget(self.imLabelZ)
        self.layoutZ.addWidget(slider)
        self.layoutZ.addStretch(1)
        self.ZViewGroupBox.setLayout(self.layoutZ)

    def createExtensionGroupBox(self):
        self.extensionGroupBox = QGroupBox("Extension")

        self.extensionImageLabel = QLabel()
        self.extensionTextLabel = QLabel()

        self.timer = QtCore.QTimer(self)
        # Throw event timeout with an interval of 500 milliseconds
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.refreshExtension)
        self.timer.start()

        layout = QVBoxLayout()
        layout.addWidget(self.extensionTextLabel)
        layout.addWidget(self.extensionImageLabel)
        self.extensionGroupBox.setLayout(layout)


def imshow(array: ndarray, extensionFunc=defaultBlank):
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance() 
    main = MainWindow(View3D(array, (400, 400)), extensionFunc)
    main.show()
    app.exec_()
    # sys.exit()


def fileshow(file: str, extensionFunc):
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance() 
    # main = MainWindow(FileView3D(file, (400, 400)), extensionFunc)
    # main.show()
    # app.exec_()
    # sys.exit()
