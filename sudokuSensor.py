
from multiprocessing.connection import wait
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

import tensorflow as tf


sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))


model = load_model('model.h5') 
class SudokuClassifier:
    def __init__(self):
       
        
        self.numberList=[]
   
    def preprocessSudoku(self,sudoku):
        gray = cv2.cvtColor(sudoku,cv2.COLOR_BGR2GRAY)
        blur =  cv2.medianBlur(gray,3)
        
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (5,5))
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, (5,5))

        return closing

    def findBox(self,problem,procesedImage):
        edge = cv2.Canny(procesedImage, 175, 175)
        contours,hiearchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]

        x,y,w,h = cv2.boundingRect(cnt)

        cv2.rectangle(problem,(x,y),(x+w,y+h),(0,255,0),1)
        
        return (x,y,w,h)

        
    
    def findNumberBoxes(self,image,points):
        (x,y,w,h) = points

      
        
        
        """ x+=2

        temp_x=x
        temp_y=y
        box_w = int((x+w+25)/10)
        box_h = int((y+h-22)/10)
        box_w = int((w-x)/8.2)
        box_h = int((h-y)/7.5)  """

        x += 2

        temp_x = x
        temp_y = y

        box_w = int((x + w - 10) / 9)
        box_h = int((y + h - 35) / 9)

        for k in range(1,10):
            for j in range(1,10):
                cv2.rectangle(image, (x, y), (x + box_w, y + box_h), (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)), 3)
                
                self.tobeSudoku(image[y:y+box_h,x:x+box_w])



                x = x + box_w

                if j%3==0:
                    x = x - 3*box_w
                    y = y + box_h
            
            y = temp_y
            x = x + 3*box_w 

            if k%3==0:
                x = temp_x
                
                y = temp_y + 3*box_h
                temp_y=y

                
            
                

    def preprocessNumber(self,image):
        
        
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,8)
        #image = cv2.bilateralFilter(image,9,70,80)
        
        
        image  = self.preProcessImage(image)
        
        return image
    def preProcessImage(self,img):
        img = cv2.resize(img,(32,32))
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img[img>50]=255
        
        img = cv2.equalizeHist(img)
        
        img = img/255
        
        img = np.asarray(img)

        return img

    def tobeSudoku(self,image):
        
        image = self.preprocessNumber(image[5:-5,5:-5])
        #print(image)
        
        #plt.imshow(image, cmap='gray')
        #plt.show()

       
        if len(np.unique(image))==1:
            self.numberList.append(0)
        else:
            image = image.reshape(1, 32, 32, 1)
            predictions = model.predict(image)
            classIndex = np.argmax(predictions,axis=1)
            probabilityValue = np.amax(predictions)
        
            if probabilityValue > 0.78:
            
                self.numberList.append(classIndex[0])
            else:
                self.numberList.append(0)
        
        

        


if __name__ == '__main__':
    sc = SudokuClassifier()
     
    problem = cv2.imread('sudoku2.png')
    
    preprocessedImage = sc.preprocessSudoku(problem)
    points = sc.findBox(problem,preprocessedImage)
    #(x,y,w,h)=points
    sc.findNumberBoxes(problem,points)
    print(np.reshape(sc.numberList,(9,3,3)))

    cv2.imshow('preprocessedImage',preprocessedImage)
    #cv2.imshow('sudokuArea',problem.copy()[y:y+h,x:x+w])
    cv2.imshow('problem',problem.copy())
    cv2.waitKey(0)