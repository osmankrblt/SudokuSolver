import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import sudokuSolver as ss
from matplotlib import pyplot as plt

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

model = load_model('model.h5') 

class SudokuClassifier:
    def __init__(self):
       
        self.numberList=[]
        self.posArray=[]
   
    def preprocessSudoku(self,sudoku):
        sudoku = cv2.cvtColor(sudoku,cv2.COLOR_BGR2GRAY)
        sudoku =  cv2.medianBlur(sudoku,3)
        
        sudoku = cv2.adaptiveThreshold(sudoku,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,8)
        
        return sudoku

    def findBox(self,problem,procesedImage):
        edge = cv2.Canny(procesedImage, 175, 175)
        contours,hiearchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]

        x,y,w,h = cv2.boundingRect(cnt)

        cv2.rectangle(problem,(x,y),(x+w,y+h),(0,255,0),5)
        cv2.circle(problem, (x,y), 5, (0,0,255), -1)
        cv2.circle(problem, (x+w,y+h), 5, (0,0,255), -1)
       
        return (x,y,w,h)

        
    def drawSolvedNumbers(self,image,solvedNumbers):
        (x,y,w,h) = points

    
        

        box_w = ( w ) // 9
        box_h = ( h ) // 9

      
        for j in range(1,82):
                if self.posArray[j-1]==1:

                    cv2.putText(image, str(solvedNumbers[j-1]), (x+box_w//4,y+box_h//2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255), 1, cv2.LINE_AA, False)
                
                x = x + box_w

                if j%9==0:
                    x = x - 9*box_w
                    y = y + box_h
        
        return image

    def findNumberBoxes(self,image,points):
        (x,y,w,h) = points

        box_w = ( w ) // 9
        box_h = ( h ) // 9

        for j in range(1,82):
               
                cv2.rectangle(image, (x, y), (x + box_w, y + box_h), (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)), 2)
                
                self.findNumbers(image[y:y+box_h,x:x+box_w])
                
                x = x + box_w

                if j%9==0:
                    x = x - 9*box_w
                    y = y + box_h
            
           
                
            
                

    def preprocessNumber(self,image):
        

        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
        image = cv2.resize(image,(32,32))
        
        image = cv2.equalizeHist(image)
        
        image = image/255
        
        image = np.asarray(image)


        return image

    def solve(self,):
        numbers  = np.asarray(sc.numberList)
        self.posArray = np.where(numbers > 0, 0, 1)

        board = np.array_split(numbers,9)
    
        try:    
        
            ss.solve(board)
      
        except Exception as e: 
            print("Error solving problem: " + str(e))

        return board
    def drawNumbers(self,image):
        flatList = []
        for sublist in board:
            for item in sublist:
                flatList.append(item)
 
        solvedNumbers =flatList*self.posArray

        filled = sc.drawSolvedNumbers(image,solvedNumbers)
        return filled
       
    def findNumbers(self,image):
        
        
        cutX = 2*(image.shape[1]//10)
        cutY = 2*(image.shape[0]//10)

        image = self.preprocessNumber(image[cutY:-cutY,cutX:-cutX])
        
        #plt.imshow(image,cmap="gray")
        #plt.show()
        
        
        if len(np.unique(image))==1:
            self.numberList.append(0)
        else:
            image = image.reshape(1, 32, 32, 1)
            predictions = model.predict(image)
            classIndex = np.argmax(predictions,axis=1)
            probabilityValue = np.amax(predictions)
        
            if probabilityValue > 0.50:
            
                self.numberList.append(classIndex[0])
            else:
                self.numberList.append(0)
        
        

        


if __name__ == '__main__':
    
    sc = SudokuClassifier()
     
    problem = cv2.imread('sudoku2.png')
  
    preprocessedImage = sc.preprocessSudoku(problem)
    points = sc.findBox(problem,preprocessedImage)

    sc.findNumberBoxes(problem,points)
    

    board = sc.solve()
    
    ss.print_board(board)
    
    filled = sc.drawNumbers(problem)

    cv2.imshow('preprocessedImage',preprocessedImage)
    cv2.imshow('problem',problem.copy())
    cv2.imshow('filled',filled)
    cv2.waitKey(0)