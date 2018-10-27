import cv2
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt

class PreprocessFile:
    def __init__(self,dataDir):
        self.dataDir="/home/berhe/Desktop/Geni_Project/Hand-written-digit-Recognition-Project/Data collected"
    def getImages_labels(self):
        self.imagesDirs=[]
        self.imagesLabels=[]
        for subdir, dirs, i in os.walk(self.dataDir):
            for j in i:
                file=subdir+'/'+j
                if (len(j.split('.')[0])==1):
                    self.imagesDirs.append(file)
                    self.imagesLabels.append('0'+j.split('.')[0])
                else:
                    self.imagesDirs.append(file)
                    self.imagesLabels.append(j.split('.')[0])
        return self.imagesDirs, self.imagesLabels

    def PreprocessImage(self, images,labels):
        #images,labels=self.getImages_labels()
        x=[]# representation of images as anarray
        y=[]# labels of the images: the arabic number representation of the geez handwritten images
        WIDTH = 128
        HEIGHT = 128
        print('processing images'),
        for img in images:
            print('..'),
            base=os.path.basename(img)

            full_size_image=cv2.imread(img)
            x.append(cv2.resize(full_size_image,(WIDTH, HEIGHT),interpolation=cv2.INTER_CUBIC))
        #y=[int(i) for i in labels]
        return x,y
