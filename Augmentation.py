
# coding: utf-8

# In[ ]:


import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import logging

class Augmentation():
    '''
    Input format:
        image : list of image
        xml : list of xml
        classes : list of class name, ex : [coating_Particle, Mask, ...]
        imPath : path for saving new images
        xmlPath : path for saving new xmls
        saveName : New images and xmls file name
    '''    
    def __init__(self, image, xml, classes, saveName, imPath, xmlPath):
        self.image = image
        self.xml = xml
        self.classes = classes
        self.saveName = saveName
        self.imPath = imPath
        self.xmlPath = xmlPath
        
    def _saveImage(self, image, fileName):
        Suffix = '.jpg'
        Path = os.path.join(self.imPath, fileName + Suffix)
        success = cv2.imwrite(Path, image)
        if not success:
            raise Exception('Image Saving Error! Please check your image or save dir.')
            
    def _saveXML(self, xml, fileName):
        Suffix = '.xml'
        Path = os.path.join(self.xmlPath, fileName + Suffix)
        xml.write(Path)
    

    def _XmlInfo(self, tree):
        root = tree.getroot()
        boxSet = []
        if root.find('size'):
            size = root.find('size')
            w = int(size.find('width').text)    
            h = int(size.find('height').text)   

            for obj in root.iter('object'):

                difficult = obj.find('difficult').text
                cls = obj.find('name').text

                if cls not in self.classes or int(difficult)==1:
                    continue
                cls_id = self.classes.index(cls)
                xmlbox = obj.find('bndbox')
       
                b = [float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)]
                boxSet.append(b)
        return boxSet
    
    def _checkBox(self, w, h, box, new):
        if new[0] < 0:
            new[0] = 0
        if new[1] >= w:
            new[1] = w - 1
        if new[2] < 0:
            new[2] = 0
        if new[3] >= h:
            new[3] = h - 1
        boxArea = (box[3] - box[2]) * (box[1] - box[0])
        newArea = (new[3] - new[2]) * (new[1] - new[0])
        if newArea / boxArea < 0.3 :
            new = [-1 for i in new]
        return new
        
    def _makeXml(self, tree, box):
        root = tree.getroot()
        if root.find('size'):
            for (obj, b) in zip(root.iter('object'), box):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text

                if cls not in self.classes or int(difficult)==1:
                    logging.warning("Object class not in classes.")
                    continue
                
                cls_id = self.classes.index(cls)
                xmlbox = obj.find('bndbox')
                xmlbox.find('xmin').text = str(b[0])
                xmlbox.find('xmax').text = str(b[1])
                xmlbox.find('ymin').text = str(b[2])
                xmlbox.find('ymax').text = str(b[3])
            for obj in root.findall('object'):
                box = obj.find('bndbox')
                flag = float(box.find('xmin').text)
                if flag < 0:
                    root.remove(obj)                  
        return tree
    
    def _Reflect(self, image, xml, dim, fileName):
        h, w, _ = image.shape
        image = cv2.flip(image, dim)
        self._saveImage(image, fileName)
        boxSet = self._XmlInfo(xml)
        if dim == 0:
            for box in boxSet:
                newYmin = h - box[3]
                newYmax = h - box[2]
                box[2] = newYmin
                box[3] = newYmax
        elif dim == 1:
            for box in boxSet:
                newXmin = w - box[1]
                newXmax = w - box[0]
                box[0] = newXmin
                box[1] = newXmax
        elif dim == -1:
            for box in boxSet:
                newXmin = w - box[1]
                newXmax = w - box[0]
                newYmin = h - box[3]
                newYmax = h - box[2]
                box[0] = newXmin
                box[1] = newXmax
                box[2] = newYmin
                box[3] = newYmax
        else:
            raise Exception("The parameter dim must be -1, 0 or 1!")
        xml = self._makeXml(xml, boxSet)
        self._saveXML(xml, fileName)
        return True
    
    def _Translate(self, x, y, image, xml, fileName):
        h, w, _ = image.shape
        if abs(x) > w or abs(y) > h:
            raise Exception("|x| and |y| must lower than w and y respectively.")
        M = np.float32([[1, 0, x],[0, 1, y]])
        image = cv2.warpAffine(image, M, (w, h))
        self._saveImage(image, fileName)
        boxSet = self._XmlInfo(xml)
        for box in boxSet:
            newXmin, newXmax = box[0] + x, box[1] + x
            newYmin, newYmax = box[2] + y, box[3] + y
            new = [newXmin, newXmax, newYmin, newYmax]
            new = self._checkBox(w, h, box, new)
            box[0] = new[0]
            box[1] = new[1]
            box[2] = new[2]
            box[3] = new[3]
        xml = self._makeXml(xml, boxSet)
        self._saveXML(xml, fileName)
        return True
    
    def _Zoom(self, x, y, image, method, xml, fileName):
        w = int(img.shape[1] * x)
        h = int(img.shape[0] * y)
        dim = (w, h)
        image = cv2.resize(image, dim, interpolation = method)
        top, bottom, left, right = 0, 0, 0, 0
        if x < 1:
            parallel_pad = img.shape[1] - w
            left = parallel_pad // 2          
            right = parallel_pad - left
        if y < 1:
            vertical_pad = img.shape[0] - h      
            top = vertical_pad // 2
            bottom = vertical_pad - top       
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
        self._saveImage(image, fileName)
        boxSet = self._XmlInfo(xml)
        for box in boxSet:
            box[0] = box[0] * x + left
            box[1] = box[1] * x + left
            box[2] = box[2] * y + top
            box[3] = box[3] * y + top
        xml = self._makeXml(xml, boxSet)
        self._saveXML(xml, fileName)
        return True
         
    def _check(self):
        if len(self.image) != len(self.xml) or len(self.image) != len(self.saveName) or len(self.xml) != len(self.saveName):
            raise Exception("Image, xml and saveName list must be the same size.")

    def reflect(self, dim):
        '''
        Usage:
            reflect along x axis for dim = 0
            reflect along y axis for dim = 1
            reflect along both axis for dim = -1
        '''
        self._check()
        for i in range(len(self.image)):
            check = self._Reflect(self.image[i], self.xml[i], dim, self.saveName[i])
            if not check:
                raise Exception("Reflect Function Error When Transferring Image {}".format(self.saveName[i]))
    
    def translate(self, x, y):
        '''
        Usage:
            shift (x, y) pixels and padding with zeros
        '''
        self._check()

        for i in range(len(self.image)):
            check = self._Translate(x, y, self.image[i], self.xml[i], self.saveName[i])
            if not check:
                raise Exception("Translate Function Error When Transferring Image {}".format(self.saveName[i]))
                
    def zoom(self, rx, ry, method = "bilinear"):
        '''
        Usage:
            rx : Original x multiply by rx (i.e. x = x * rx)
            ry : Original y multiply by ry (i.e. x = x * ry)
            Note that rx and ry are ratio and must greater than zero
            method : interpolation method
                     1. nearest neighbor : 'nearest' 
                     2. bilinear : 'bilinear' (default) 
                     3. bicubic : 'bicubic' 
        '''
        if rx or ry <= 0:
            raise Exception("Ratio Error!! rx and ry must greater then 0.")
        self._check()
        if method != "nearest" and method != "bilinear" and method != "bicubic":
            raise Exception("Method Error!! The available methods are nearest, bilinear and bicubic.")
        if method == 'nearest':
            method = cv2.INTER_NEAREST
        elif method == 'bicubic':
            method = cv2.INTER_CUBIC
        else:
            method = cv2.INTER_LINEAR
        
        for i in range(len(self.image)):
            check = self._Zoom(rx, ry, self.image[i], method, self.xml[i], self.saveName[i])
            if not check:
                raise Exception("Translate Function Error When Scaling Image {}".format(self.saveName[i]))

