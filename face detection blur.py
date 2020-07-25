from pyimagesearch.facedetector import FaceDetector
from pyimagesearch.eyetracker import EyeTracker
from pyimagesearch.mouthtracker import mouthtracker
from pyimagesearch import imutils
import argparse
import cv2
import numpy as np
#defining function 
def AddImage(img1,img2,pos):      

            x_offset=pos[0]
            y_offset=pos[1]

            y1, y2 = y_offset, y_offset + img2.shape[0]
            x1, x2 = x_offset, x_offset + img2.shape[1]

            alpha_s = img2[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                img1[y1:y2, x1:x2, c] = (alpha_s * img2[:, :, c] + alpha_l * img1[y1:y2, x1:x2, c])
            return img1
#end defining
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True,help = "path to where the face cascade resides")
ap.add_argument("-e", "--eye", required = True,help = "path to where the eye cascade resides")
ap.add_argument("-v", "--video",help = "path to the (optional) video file")
ap.add_argument("-m","--mouth",required=True,help="path to where the mouth cascade resides")
args = vars(ap.parse_args())
et = EyeTracker(args["face"], args["eye"])
fd = FaceDetector(args["face"])
mt= mouthtracker(args["mouth"])
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
else:
	camera = cv2.VideoCapture(args["video"])
while True:
	(grabbed, frame) = camera.read()
	if args.get("video") and not grabbed:
		break
	frame = imutils.resize(frame, width = 600)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5,minSize = (30, 30))
	frameClone = frame.copy()
	#print(rects)
	for (fX, fY, fW, fH) in faceRects:
		cv2.circle(frameClone, (fX+fW//2, fY+fH//2), fW//2, (120, 0, 200), 3)
		img3 = cv2.imread("img5.png",-1)
		img4=cv2.imread("img6.png",-1)
		#x1=img3.shape[0]
		#y1=img3.shape[1]
		#w=fW//2-(x1//2)
		#h=fH//2-(y1//2)
		rects = et.track1(frameClone)
		eyes_position = []
		for (i,rect) in enumerate(rects[0:2]):          #eye detection
			#rects.remove(rects[0])
			cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0),0)
			eye=frameClone[rect[1]:rect[3],rect[0]:rect[2]]         #limited area of eyes
			img3=cv2.resize(img3,(rect[2]-rect[0],rect[3]-rect[1]))
			img4=cv2.resize(img4,(rect[2]-rect[0],rect[3]-rect[1]))
			mask=np.zeros(frameClone.shape[:2],dtype='uint8')
			cv2.rectangle(mask,(rect[0], rect[1]), (rect[2], rect[3]),255,-1)
			eye=cv2.bitwise_and(frameClone,frameClone,mask=mask)    # provid frame to show eyes
			text="eye"+str(i)               #name of frame
			cv2.imshow(text, eye)
			eyes_position.append([rect[0],rect[1]])         # add eye position to list to use in AddIMage function
		smiles  = mt.track1(frameClone)

		for (sx, sy, sw, sh) in smiles:
			cv2.rectangle(frame, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
			mouth=frameClone[sx:sw,sy:sh]
			mask1=np.zeros(frameClone.shape[:2],dtype='uint8')
			cv2.rectangle(mask1,(sx, sy), ((sx+sw), (sy+sh)),255,-1)
			mouth=cv2.bitwise_and(frameClone,frameClone,mask=mask1)
			cv2.imshow('mouth',mouth)



		blur=cv2.GaussianBlur(frameClone,(33,33),0)					#مات کردن کل عکس به جز چهره
		mask=np.zeros((frameClone.shape[0],frameClone.shape[1],3),dtype="uint8")
		mask=cv2.circle(mask,(fX+fW//2,fY+fH//2),fW//2,(255,255,255),-1)
		out=np.where(mask==np.array([255,255,255]),frameClone,blur)
		try:
                    if eyes_position [0][0] < eyes_position [1][0]:
                            frameClone = AddImage(out,img3,(eyes_position[0][0],eyes_position[0][1]))  	# قرار دادن چشم شخص دیگر روی چشم روی چشم تشخیص داده شده
                            frameClone = AddImage(out,img4,(eyes_position[1][0],eyes_position[1][1]))
                    else:
                        frameClone = AddImage(out,img3,(eyes_position[1][0],eyes_position[1][1]))
                        frameClone = AddImage(out,img4,(eyes_position[0][0],eyes_position[0][1]))
		except:
			        pass
		
	cv2.imshow("Face", frameClone)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()