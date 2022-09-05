
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt

class SudokuClassifier:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract'
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
        x,y,w,h = points
        """ x+=2
        
        temp_x=x
        temp_y=y

        box_w = int((x+w+25)/10)
        box_h = int((y+h-22)/10)
        box_w = int((w-x)/8.2)
        box_h = int((h-y)/7.5)  """
       
        x+=2
        
        temp_x=x
        temp_y=y

        box_w = int((x+w-10)/9)
        box_h = int((y+h-35)/9)
        
       

        for k in range(1,10):
            for j in range(1,10):
                
                print("Box {} : index {} " .format(k,j))

                cv2.rectangle(image, (x, y), (x + box_w, y + box_h), (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)), 3)
                #plt.imshow(image)
                #plt.show()
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
                
                
            
                

    def preprocessNumber(self,number):
        
        
        gray = cv2.cvtColor(number,cv2.COLOR_BGR2GRAY)
       
        bilFilter = cv2.bilateralFilter(gray,9,70,80)
        
        
        opening = cv2.morphologyEx(bilFilter, cv2.MORPH_OPEN, (5,5))
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, (2,2))
        
        return closing

    def tobeSudoku(self,image):
        
        image = self.preprocessNumber(image)
        
        
        
        #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #plt.show()
        
        number = pytesseract.image_to_string(image,config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789') 
        self.numberList.append(int(number) if number != "" else 0 )
        print(">>>> "+number)
        

        


if __name__ == '__main__':
    sc = SudokuClassifier()
     
    problem = cv2.imread('sudoku2.png')
    
    preprocessedImage = sc.preprocessSudoku(problem)
    points = sc.findBox(problem,preprocessedImage)
    sc.findNumberBoxes(problem,points)
    print(np.reshape(sc.numberList,(9,3,3)))

    #cv2.imshow('outputRectangle',sudokuArea)
    cv2.imshow('problem',problem)
    cv2.waitKey(0)