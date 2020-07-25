import argparse
import numpy as np
import cv2
import imutils
### DEFINED FUNCTIONS ################################

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,help = "Path to the image")    #گرفتن عکس از کاربر
args = vars(ap.parse_args())
def mousecall(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK: # چپ کلیک برای زوم کردن
        
        zoomin(x,y)
        
    elif event == cv2.EVENT_RBUTTONDBLCLK: # راست کلیک برای مات کردن
        fblur(x,y)

def mousenone(event,x,y,flags,param):
    return 0
def zoomin(x,y):

    cv2.setMouseCallback('frame', mousenone)
    while(True):
        image = cv2.imread(args["image"])
        if percent==200:
            x1 = x - 200
            if x1 < 0:     # نزدیک به دیواره ی سمت چپ
                x1 = 0
        
            x2 = x + 200
 
            y1 = y - 110
        
            if y1 < 0:        #نزدیک به دیواره ی بالا
                y1 = 0
        
            y2 = y + 110
         #### مرحله زوم کردن
            if y2>image.shape[0] and x2>image.shape[1]: #زوم کردن در گوشه عکس پایین سمت راست
                x2=image.shape[1]
                y2=image.shape[0]
                pts1 = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
                pts2 = np.float32([[0,0],[320,0],[0,192],[320,192]])
                M = cv2.getPerspectiveTransform(pts1,pts2)
                dst = cv2.warpPerspective(image,M,(320,192))
            elif y2 > image.shape[0]:                      #وقتی نقطه دلخواه کاربر به دیواره ی پایین نزدیک است
                y2=image.shape[0]
                
                pts1 = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
                pts2 = np.float32([[0,0],[320,0],[0,192],[320,192]])
                M = cv2.getPerspectiveTransform(pts1,pts2)

                dst = cv2.warpPerspective(image,M,(320,192))
            elif  x2 > image.shape[1]:                  # وقتی نقطه دلخواه کاربر به دیواره سمت راست نزدیک است
                x2=image.shape[1]
                pts1 = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
                pts2 = np.float32([[0,0],[320,0],[0,192],[320,192]])
                M = cv2.getPerspectiveTransform(pts1,pts2)
                dst = cv2.warpPerspective(image,M,(320,192))

            
            else:                                           #در هرکجای عکس به جز نقاط ذکر شده قبلی

                pts1 = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
                pts2 = np.float32([[0,0],[320,0],[0,192],[320,192]])

                M = cv2.getPerspectiveTransform(pts1,pts2)

                dst = cv2.warpPerspective(image,M,(320,192))
            cv2.imshow('frame',dst)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                zoomout()
                break
        if percent==100:
            x1 = x - 300
            if x1 < 0:
                x1 = 0
        
            x2 = x + 300
            #if x2 > image.shape[1]:
            #   x2 = 320
            
            y1 = y - 180
        
            if y1 < 0:
                y1 = 0
        
            y2 = y + 180
            
            #if y2 > image.shape[0]:
            #   y2 = 192
            if y2>image.shape[0] and x2>image.shape[1]:
                x2=image.shape[1]
                y2=image.shape[0]
                pts1 = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
                pts2 = np.float32([[0,0],[320,0],[0,192],[320,192]])
                M = cv2.getPerspectiveTransform(pts1,pts2)
                dst = cv2.warpPerspective(image,M,(320,192))
            if y2 > image.shape[0]:
                y2=image.shape[0]
                
                pts1 = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
                pts2 = np.float32([[0,0],[320,0],[0,192],[320,192]])
                M = cv2.getPerspectiveTransform(pts1,pts2)

                dst = cv2.warpPerspective(image,M,(320,192))
            elif  x2 > image.shape[1]:
                x2=image.shape[1]
                pts1 = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
                pts2 = np.float32([[0,0],[320,0],[0,192],[320,192]])
                M = cv2.getPerspectiveTransform(pts1,pts2)
                dst = cv2.warpPerspective(image,M,(320,192))

            
            else:

                pts1 = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
                pts2 = np.float32([[0,0],[320,0],[0,192],[320,192]])

                M = cv2.getPerspectiveTransform(pts1,pts2)

                dst = cv2.warpPerspective(image,M,(320,192))
                

            cv2.imshow('frame',dst)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                zoomout()
                break
        if percent==50:
            x1 = x - 450
            if x1 < 0:
                x1 = 0
        
            x2 = x + 450
            #if x2 > image.shape[1]:
            #   x2 = 320
            
            y1 = y - 290
        
            if y1 < 0:
                y1 = 0
        
            y2 = y + 290
            
            #if y2 > image.shape[0]:
            #   y2 = 192
            if y2>image.shape[0] and x2>image.shape[1]:
                x2=image.shape[1]
                y2=image.shape[0]
                pts1 = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
                pts2 = np.float32([[0,0],[320,0],[0,192],[320,192]])
                M = cv2.getPerspectiveTransform(pts1,pts2)
                dst = cv2.warpPerspective(image,M,(320,192))
            if y2 > image.shape[0]:
                y2=image.shape[0]
                
                pts1 = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
                pts2 = np.float32([[0,0],[320,0],[0,192],[320,192]])
                M = cv2.getPerspectiveTransform(pts1,pts2)

                dst = cv2.warpPerspective(image,M,(320,192))
            elif  x2 > image.shape[1]:
                x2=image.shape[1]
                pts1 = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
                pts2 = np.float32([[0,0],[320,0],[0,192],[320,192]])
                M = cv2.getPerspectiveTransform(pts1,pts2)
                dst = cv2.warpPerspective(image,M,(320,192))

            
            else:

                pts1 = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
                pts2 = np.float32([[0,0],[320,0],[0,192],[320,192]])

                M = cv2.getPerspectiveTransform(pts1,pts2)

                dst = cv2.warpPerspective(image,M,(320,192))
                

            cv2.imshow('frame',dst)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                zoomout()
                break
def zoomout():
    image = cv2.imread(args["image"])
    cv2.imshow('frame',image)
def fblur(x,y):
    cv2.setMouseCallback('frame', mousenone)
    while(True):
        image = cv2.imread(args["image"])
        x1 = x - 100
        x2 = x + 100
        y1 = y - 96
        y2 = y + 96
        blurred=cv2.blur(image[y1:y2,x1:x2],(55,55),0)          # مات کردن محدوده ی دلخواه
        image[y1:y2,x1:x2]=blurred
        cv2.imshow('frame',image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            image = cv2.imread(args["image"])

            cv2.imshow("frame",image)
            break
### PROGRAM STARTS HERE ##################
#cap = cv2.VideoCapture(0)
percent=int(input("which percent to zoom in? 50 , 100, 200?"'\n'))      # گرفتن درصد زوم کردن از کاربر
image = cv2.imread(args["image"])
frame=cv2.resize(image,(640,480))
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame',640,480)
while(True):
    
    cv2.imshow('frame',image)     
    cv2.setMouseCallback('frame', mousecall)
    if cv2.waitKey(1) & 0xFF == ord('q'):               
        break
### PROGRAM ENDS HERE ####################
#cv2.waitKey(0)
cv2.destroyAllWindows()