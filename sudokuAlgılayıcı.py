import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
class SudokuClassifier:

   
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
        
        

        return problem[y:y+h,x:x+w]
    
    def findNumberBoxes(self,procesedImage):
       
        
        edge = cv2.Canny(procesedImage, 175, 175)
        contours,hiearchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
     
        result = np.array(list(map(cv2.contourArea, contours)),dtype=np.int32)
        print("Median : {} , Sapma : {} ,".format(np.median(result) , np.std(result)))

        """ 
        plt.boxplot(result, notch=None, vert=None, patch_artist=None, widths=None)
        
        plt.show() """
        print(np.bincount(result).argmax())
        
        

        max  = np.bincount(result).argmax()+200
        min  = np.bincount(result).argmax()-200
        max = 2800
        min = 2300
        BOX = 0
        for cnt in contours:
            if cv2.contourArea(cnt)>min and cv2.contourArea(cnt)<max:
                BOX +=1
                x,y,w,h = cv2.boundingRect(cnt)
                print(cv2.contourArea(cnt))
                
                cv2.rectangle(procesedImage,(x,y),(x+w,y+h),(0,0,255),1)
                cv2.circle(procesedImage,(int(x),int(y)),4,(255,0,0),-1)
        print(BOX)
        return procesedImage

    def tobeSudoku(self,sudokuArea):
        
        pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract'
        print(" Rakamlar "+pytesseract.image_to_string(sudokuArea))


if __name__ == '__main__':
    sc = SudokuClassifier()
    
    problem = cv2.imread('sudokuUnFilled.jpg')
    
    preprocessedImage = sc.preprocessSudoku(problem.copy())
    sudokuArea = sc.findBox(problem.copy(),preprocessedImage)
    sudokuArea = sc.findNumberBoxes(sudokuArea)
    #sc.tobeSudoku(sudokuArea)

    cv2.imshow('outputRectangle',sudokuArea)
    cv2.imshow('problem',problem)
    cv2.waitKey(0)