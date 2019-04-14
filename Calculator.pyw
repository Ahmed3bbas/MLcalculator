import pygame.font, pygame.event, pygame.draw, pygame.image
import numpy as np
import cv2

from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import os
import math
from scipy import ndimage

#from data import load_data

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted 
  

#load Data
#features, labels, X_test, Y_test = load_data()

classfirerPath = "calculator_svm.pkl"
# Load the classifier
if not os.path.exists(classfirerPath):
    raise AssertionError()
    #import trainClassfier
  
clf, pp = joblib.load(classfirerPath)

def predict(im):
    # Convert to grayscale and apply Gaussian filtering
    #im = np.transpose(im)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    #cv2.imshow("im_gray",im_gray)

    # Threshold the image
    # ret is a original image
    # im_th is threshold Image
    ret, im_th = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY_INV  | cv2.THRESH_OTSU)
    #cv2.imshow("im_th",im_th)

    # Find contours in the image
    ctrs, hier,*_ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(im, ctrs, -1, (0,255,0), 3)
    #cv2.imshow("hier",im)
    
    #cv2.imshow("HIER",ctrs)
    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    rects.sort(key=lambda x:x[0])
    #print(rects)
    
    equation = ''
    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    for rect in rects:
        # Draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 3) 
        # Make the rectangular region around the digit
        '''leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        '''
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        padding = 10
        roi = im_th[y - padding : y+h + padding, x - padding : x+ w + padding] 
        # Resize the image
        roi = cv2.resize( roi, (20, 20), interpolation=cv2.INTER_AREA)
        roi = np.lib.pad(roi,((4,4),(4,4)),'constant')
        gray = roi

        shiftx,shifty = getBestShift(gray)
        shifted = shift(gray,shiftx,shifty)
        gray = shifted
        
        roi = gray
        #cv2.imshow('resize' + str(rect),roi)
        roi = cv2.dilate(roi, (3, 3))
        #print("dilate")
        #cv2.imshow("dilate" + str(rect),roi)
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        #print(roi_hog_fd)
        r = np.array([roi_hog_fd],dtype=np.float32)#np.float32(256)
        #print('r  ',r.shape)
        roi_hog_fd = pp.transform(r)
        #print('roi_hog_fd',roi_hog_fd)
        nbr = clf.predict(roi_hog_fd)
        print("nbr : ",nbr)
        equation += str(nbr[0])
        cv2.putText(im, str((nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 3)

    #cv2.imshow("OUTPUT IMAGE",im)
    print(equation)
    try:
      res = eval(equation)
      print(res)
    except:
      res = "The Equation is Not Correct"
      print(res)
    finally:
      return im,equation, res
#predict(im)

def clac_accurcy():
  list_h = []
  for f in X_test:
    roi_hog_fd = hog(f.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_h.append(roi_hog_fd)
  r = np.array(list_h,dtype=np.float32)#np.float32(256)
  roi_hog_fd = pp.transform(r)
  return clf.score(roi_hog_fd,Y_test)

def showStats(lineWidth, value, result):
    """ shows the current statistics """
    
    myFont = pygame.font.SysFont("Verdana", int(lineWidth))
    stats = "Estimate Equation: %s    result: %s" % (value, result)
    statSurf = myFont.render(stats, 1, ((255, 255, 255)))
    return statSurf

def drawStatistics(screen):  
    """Draw statistics about training set"""

   
    start = 30
    myFont = pygame.font.SysFont("Verdana", 24)
    myFont2 = pygame.font.SysFont("Verdana", 18)
    myFont3 = pygame.font.SysFont("Verdana", 16)
    pygame.draw.rect(screen,(255,255,255),(640,0,1000,360))
    acc = 85 #clac_accurcy()
    screen.blit(myFont.render("Accuracy: %s" % str(acc)+"%", 1, ((0, 0, 0))), (670, start))
    screen.blit(myFont2.render("Instuctions Number = %s" % (13), 1, ((0, 0, 0))), (670, start + 60))
    screen.blit(myFont3.render("INSTRUCTIONS:", 1, ((0, 0, 0))), (670, start + 40))
    instruction = ['+','-','*']
    for i in range(10):
        screen.blit(myFont2.render("Insturuction %s = %s" % (i, i), 1, ((0, 0, 0))), (670, start + 80+i*20))
    screen.blit(myFont2.render("Operator = %s" % (instruction), 1, ((0, 0, 0))), (670, start + 290))

def drawPixelated(A, screen):  
    """Draw 30x30 image of input""" 
    
    A = A.ravel()
    A = (255-A*255).transpose()
    size = 30
    for x in range(size):
        for y in range(size):
            z=x*30+y
            c = int(A[z])
            pygame.draw.rect(screen,(c,c,c),(x*11+640,y*11,30,30))


def checkKeys(myData):
    """test for various keyboard inputs"""
    
    (event, background, drawColor, lineWidth, keepGoing, screen, background2) = myData
    
    if event.key == pygame.K_q:
        keepGoing = False
    elif event.key == pygame.K_c:
        background.fill((255, 255, 255))
        background2.fill((255, 255, 255))
        drawPixelated(np.zeros((30,30)), screen)

    elif event.key == pygame.K_x:
        imgdata = cv2.transpose(pygame.surfarray.array3d(background))
        #cv2.imshow('image',imgdata)
        cv2.imwrite("111.jpg",imgdata)
        #Predicting Image Content
        image, equation, res = predict(imgdata)
        prdicted_image_name = "prdicted_image.jpg"
        cv2.imwrite(prdicted_image_name,cv2.resize(image,(350,360)))
        screen.fill((0, 0, 0))
        screen.blit(background2, (270, 0)) #(370,0)
        image = pygame.image.load(prdicted_image_name)
        screen.blit(image,(640,0))
        myLabel = showStats(25,equation, res)
        (x,y) = screen.get_size()
        screen.blit(myLabel, (17, y-90))
        #drawPixelated(image.resize((360,360)), screen)
        #calculateImage(background, screen, lineWidth, equation, res)
              #image = cv2.imread(imgdata)
              #gray = cv2.cvtColor(imgdata, cv2.COLOR_BGR2GRAY)

    
    myData = (event, background, drawColor, lineWidth, keepGoing)
    return myData
pygame.init()
screen = pygame.display.set_mode((1000, 450))
pygame.display.set_caption("Handwriting recognition")

background = pygame.Surface((630,360))
background.fill((255, 255, 255))
background2 = pygame.Surface((360,360))
background2.fill((255, 255, 255))
drawStatistics(screen)

clock = pygame.time.Clock()
keepGoing = True
lineStart = (0, 0)
drawColor = (255, 0, 0)
lineWidth = 4

pygame.display.update()

while keepGoing:
      clock.tick(30)
      for event in pygame.event.get():
          if event.type == pygame.QUIT:
              keepGoing = False
          elif event.type == pygame.MOUSEMOTION:
              lineEnd = pygame.mouse.get_pos()
              if pygame.mouse.get_pressed() == (1, 0, 0):
                  pygame.draw.line(background, drawColor, lineStart, lineEnd, lineWidth)
              lineStart = lineEnd
          elif event.type == pygame.MOUSEBUTTONUP:
            pass
              #screen.fill((0, 0, 0))
              #screen.blit(background2, (280, 0)) #(370,0)
              
              #background2.blit(pygame.Surface.get_buffer(background))
              #w = threading.Thread(name='worker', target=worker)
              #image = calculateImage(background, screen, Theta1, Theta2, lineWidth)

          elif event.type == pygame.KEYDOWN:
              myData = (event, background, drawColor, lineWidth, keepGoing, screen, background2)
              myData = checkKeys(myData)
              (event, background, drawColor, lineWidth, keepGoing) = myData
              


      screen.blit(background, (0, 0))
      pygame.display.flip()

