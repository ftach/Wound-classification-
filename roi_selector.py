# -*- coding: utf-8 -*-
"""
Created on Sun May 30 01:27:18 2021

@author: flore
"""
from audioop import add
from ctypes import alignment
from pickletools import uint8
from PIL import Image 
from scipy import signal
import numpy as np
import sys, os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import random


class Example(QMainWindow):
    
    def __init__(self):
        super().__init__()
            # GRANDE FENETRE
        self.initUI()
    
    def initUI(self):

        # WINDOW TITLE AND LOGO
        self.setWindowTitle("Segmentation Tool")
        
        # MAIN WIDGET
        window_layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(window_layout)
        
        # HELP BUTTON
        help_button = QPushButton("Aide")
        help_button.setMaximumWidth(100)
        window_layout.addWidget(help_button, alignment=Qt.AlignRight)

        # MAIN UNDER WIDGETS
        parametre_box = QGroupBox("Paramètres")
        parametre_box.setMaximumHeight(200)
        window_layout.addWidget(parametre_box)
        
        images_box = QGroupBox("Images")
        self.max_img_w = images_box.frameGeometry().width()
        self.max_img_h = images_box.frameGeometry().height()
        window_layout.addWidget(images_box)
        
        interaction_box = QFrame()
        interaction_box = QGroupBox("Intéraction")
        interaction_box.setMaximumSize(400, 100)
        window_layout.addWidget(interaction_box, alignment=Qt.AlignRight)
    
    # FRAME PARAMETRES
    
        choose_us_image = QLabel("Choisir image")
        choose_us_image.setMaximumWidth(200)
        
        browse_us_image = QPushButton("Parcourir")
        browse_us_image.setMaximumWidth(200)
        browse_us_image.clicked.connect(self.getUsFile)
        
        contrast_image = QPushButton("Améliorer le contraste de l'image")
        contrast_image.setIcon(QIcon("stars2.png"))
        contrast_image.setMaximumWidth(200)
        contrast_image.clicked.connect(self.contrastImage)
        
        crop_image = QPushButton("Rogner l'image")
        crop_image.setMaximumWidth(200)
        crop_image.clicked.connect(self.cropImage)

        parametres_layout = QGridLayout(parametre_box)

        parametres_layout.addWidget(choose_us_image, 1, 0)
        parametres_layout.addWidget(browse_us_image, 1, 1)
        parametres_layout.addWidget(contrast_image, 1, 2)
        parametres_layout.addWidget(crop_image, 1, 3)
    
    
    #  FRAME AFFICHAGE D'IMAGE
        # US IMAGE FRAME
        us_img_frame = QFrame()
        self.configureImageFrame(us_img_frame)

        us_img_label = QLabel("Image", us_img_frame)
        us_img_label.setAlignment(Qt.AlignCenter)
        self.us_img_view = QLabel(us_img_frame)
        self.us_img_view.setAlignment(Qt.AlignCenter)
        
        self.configureImageLayout(us_img_frame, us_img_label, self.us_img_view)
        
        # CONTRASTED IMAGE FRAME
        self.contrasted_img_frame = QFrame()
        self.configureImageFrame(self.contrasted_img_frame)
        
        self.contrast_activated = False
        contrasted_us_img_label = QLabel("Image contrastée")
        contrasted_us_img_label.setAlignment(Qt.AlignCenter)
        self.contrasted_us_img_view = QLabel()
        self.contrasted_us_img_view.setAlignment(Qt.AlignCenter)
        
        self.configureImageLayout(self.contrasted_img_frame, contrasted_us_img_label, self.contrasted_us_img_view)
                
        # ALL IMAGES LAYOUT
        image_layout = QGridLayout(images_box)
        image_layout.addWidget(us_img_frame, 0, 0)
        image_layout.addWidget(self.contrasted_img_frame, 0, 1)

        # FRAME A CACHER
        self.contrasted_img_frame.hide()

    # FRAME INTERACTION        

        self.save_seg_button = QPushButton("Sauvegarder image")
        self.save_seg_button.setIcon(QIcon("save.ico"))
        self.save_seg_button.setMaximumWidth(300)
        self.save_seg_button.clicked.connect(self.saveSegmentation)
        
        interaction_layout = QGridLayout(interaction_box)
        
        interaction_layout.addWidget(self.save_seg_button, 0, 0)

        self.setCentralWidget(widget)
        
    def configureImageFrame(self, frame):
        frame.setFrameStyle(QFrame.Box)# frame us_img
        frame.setLineWidth(1)
        frame.setFrameShadow(QFrame.Sunken)
        
    def configureImageLayout(self, parent, text_label, image_view):
        layout = QGridLayout(parent)
        layout.addWidget(text_label, 0, 1, 1, 1, alignment=Qt.AlignTop)
        layout.addWidget(image_view, 1, 0, 3, 3)
        return layout

    def getUsFile(self):

        self.us_filename = QFileDialog.getOpenFileName(self, "Open image")[0]
        if self.us_filename != '':
            self.us_img_array = prepare_image(self.us_filename, self.max_img_h, self.max_img_h)
            self.dispUsImage(self.us_img_array)
        
    def dispUsImage(self, img_array_to_disp):
        h, w, ch = img_array_to_disp.shape
        bytesPerLine = ch * w
        self.q_us_img = QImage(img_array_to_disp, w, h, bytesPerLine, QImage.Format_RGB888)
        self.us_img_view.setPixmap(QPixmap.fromImage(self.q_us_img))
        
    def contrastImage(self):
        self.contrast_image_array = equalize_hist(self.us_img_array).astype(np.uint8)  # has to be changed if you want to use it
        self.contrast_image = QImage(self.contrast_image_array, self.contrast_image_array.shape[1], self.contrast_image_array.shape[0],  
                                     QImage.Format_RGB32)
        self.contrasted_us_img_view.setPixmap(QPixmap(self.contrast_image))
        self.contrast_activated = True
        self.contrasted_img_frame.show()
      
    def cropImage(self):
        cropWidget = cropTool(self.us_img_array, self.q_us_img)
        cropWidget.exec()
        if cropWidget.sendImage() != 0:
            self.croped_img_array = np.array(cropWidget.sendImage()) 
            self.dispUsImage(self.croped_img_array)
        
    def saveSegmentation(self):
        seg_filename = QFileDialog().getSaveFileName(self, "Save image")[0] + ".jpg"
        save_image(self.croped_img_array, seg_filename)


class cropTool(QDialog): # utiliser héritage dans un futur ou je saurai coder
    
    def __init__(self, us_img_array, q_us_img):
        super().__init__()
        self.cropWidget = QDialog()
        
        self.cropWidget.setModal(True)
        self.setWindowTitle("Crop Tool")
        self.setFixedSize(1280, 800)

        self.image_to_crop = Image.fromarray(us_img_array)
        self.q_image_to_crop = QPixmap.fromImage(q_us_img)

        self.window_width = self.size().width()
        self.window_height = self.size().height()
        
        self.img_width = self.q_image_to_crop.size().width()
        self.img_height = self.q_image_to_crop.size().height()

        self.img_shape = (self.img_height, self.img_width)

        self.rect_width = 105
        self.rect_height = 105

        self.rect = QRect(QPoint(*random.sample(range(200), 2)), QSize(self.rect_width, self.rect_height))

        self.img_croped = False # if the img has been croped or not
        
        crop_button = QPushButton("Rogner image")
        crop_button.setMaximumWidth(200)
        crop_button.clicked.connect(self.cropImage)

        cancel_button = QPushButton("Annuler")
        cancel_button.setMaximumWidth(200)
        cancel_button.clicked.connect(self.cancelCrop)

        window_layout = QVBoxLayout(self)
        self.setLayout(window_layout)

        button_box = QGroupBox()
        window_layout.addWidget(button_box, alignment=Qt.AlignBottom)

        button_layout = QGridLayout(button_box)
        button_layout.addWidget(crop_button, 1, 0)
        button_layout.addWidget(cancel_button, 1, 1)

        self.drag_position = QPoint()
        
    def dispImage(self, img_array_to_disp):
        # Créer point d'ancrage de l'image au centre de l'écran
        self.center_point = QPoint()
        self.center_point.setX(round((self.window_width/2)-(self.img_shape[1]/2)))
        self.center_point.setY(round((self.window_height/2)-(self.img_shape[0]/2)))

        # Créer image pixmap à partir du np.array
        h, w, ch = img_array_to_disp.shape
        bytesPerLine = ch * w
        self.q_image_to_crop = QPixmap.fromImage(QImage(img_array_to_disp, w, h, bytesPerLine, QImage.Format_RGB888))    

        self.painter.drawPixmap(self.center_point, self.q_image_to_crop) # Afficher image dans fenêtre

    def zoomImage(self, facteur):
        self.img_shape = (round(self.img_shape[0]*facteur), round(self.img_shape[1]*facteur)) # Calcul de la taille voulue en fonction du facteur de zoom
        self.image_to_crop = self.image_to_crop.resize((self.img_shape[1], self.img_shape[0])) # Zoom de l'image 
        self.dispImage(np.array(self.image_to_crop)) # afficher l'image avec dispImage 
        self.update() # Actualiser le changement de taille de l'image 

    def cropImage(self):
        crop_coord = (self.rect_point.x() - self.center_point.x(), self.rect_point.y() - self.center_point.y(), self.rect_point.x() - self.center_point.x() + self.rect_width, self.rect_point.y() - self.center_point.y() + self.rect_height)
        if crop_coord[0]>=0 and crop_coord[0]<=self.img_shape[1] and crop_coord[1]>=0 and crop_coord[1]<=self.img_shape[0] and crop_coord[2]>=0 and crop_coord[2]<=self.img_shape[1] and crop_coord[3]>=0 and crop_coord[3]<=self.img_shape[0]:
            self.croped_img = self.image_to_crop.crop(crop_coord) # left, top, right, bottom
            self.img_croped = True
        else:
            self.img_croped = False
            raise ValueError("Those crop coordinates entered are not possible. Crop canceled.")
        self.done(0)

    def cancelCrop(self):
        self.img_croped = False
        self.done(0)
       
    def sendImage(self):
        if self.img_croped==True:
            return self.croped_img
        else: 
            return 0
        
    def paintEvent(self, event):
        super().paintEvent(event)
        self.painter = QPainter(self)
        self.dispImage(np.array(self.image_to_crop))
        self.painter.setPen(QPen(QColor(141, 205, 204), 5, Qt.SolidLine)) # peppermint
        self.painter.setRenderHint(QPainter.Antialiasing)
        self.painter.drawRect(self.rect)
        self.painter.end()
            
    def mousePressEvent(self, event): 
        if self.rect.contains(event.pos()):
            self.drag_position = event.pos() - self.rect.topLeft()
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event): 
        if not self.drag_position.isNull():
            self.rect_point = QPoint(event.pos() - self.drag_position)
            self.rect.moveTopLeft(event.pos() - self.drag_position)
            self.update()
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        self.drag_position = QPoint()
        super().mouseReleaseEvent(event)  

    def wheelEvent(self, event):
        facteur = event.angleDelta().y() / 100 # Pour avoir 1.2 au lieu de 120
        if facteur < 0: 
            facteur = -1/facteur 
        self.zoomImage(facteur)

def save_image(seg_img_array, seg_filename):
    seg_img = Image.fromarray(seg_img_array)
    seg_img = seg_img.convert("RGB") 
    seg_img.save(seg_filename)

def prepare_image(img_path, max_w, max_h):
    '''Prepares the image to be displayed on a window. Reduces the size array to the nearest width or height coeficient if necessary.  
    Parameters
    ----------
    img_path: str   
        Path leading to the image to display
    max_w: int
        Maximum width wanted
    max_h: int 
        Maximum height wanted

    Returns     
    ----------
    img_array: np.array   
        3D Array of the image to display downsampled to the size of the window '''

    # Get the image and its size 
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    img_h = img_array.shape[0]
    img_w = img_array.shape[1]

    if img_w>max_w or img_h>max_h:
        # set the base width or height 
        if max_w/img_w < max_h/img_h: # si le coef est + petit alors on prend celui là comme base 
            final_width = max_w 
            final_height = round(img_h*(final_width/img_w))
        else: 
            final_height = max_h
            final_width = round(img_w*(final_height/img_h))
        # resize 
        img = img.resize((final_width, final_height))

    return np.array(img)

def main():
    app = QApplication(sys.argv)
    ex = Example()
    ex.showMaximized()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()