import sys
from cvzone.ColorModule import ColorFinder
import cvzone
import cv2
import numpy as np
import json
import time

def setTrackbarValues(hsvVals):
    """
    To set Trackbars values.
    """
    cv2.setTrackbarPos("Hue Min", "TrackBars", hsvVals["hmin"])
    cv2.setTrackbarPos("Hue Max", "TrackBars", hsvVals["hmax"])
    cv2.setTrackbarPos("Sat Min", "TrackBars", hsvVals["smin"])
    cv2.setTrackbarPos("Sat Max", "TrackBars", hsvVals["smax"])
    cv2.setTrackbarPos("Val Min", "TrackBars", hsvVals["vmin"])
    cv2.setTrackbarPos("Val Max", "TrackBars", hsvVals["vmax"])
    print("setTrackbarValues: ", hsvVals)
    
def mouseRGB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        img = cv2.resize(param["image"], (480, 270))
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        colorsH = imgHSV[y,x,0]
        colorsS = imgHSV[y,x,1]
        colorsV = imgHSV[y,x,2]
        colors = img[y,x]
        print("HSV Format: ",colors)
        print("Coordinates of pixel: X: ",x,"Y: ",y)
        param["is_clicked"] = True
        param["hsv_val"] = {'hmin': colorsH-10, 'smin': colorsS-10, 'vmin': colorsV-10, 'hmax': colorsH+10, 'smax': colorsS+10, 'vmax': colorsV+10}

#define kernel size  
kernel = np.ones((7,7),np.uint8)

namaVideo = sys.argv[1]
namaWarna = sys.argv[2]
namaHasil = sys.argv[3]

warnaBolaHsv = {}

with open("deliverables/warnaBola/"+namaWarna, 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        warnaBolaHsv[line.strip()] = {'hmin': 0, 'smin': 0, 'vmin': 0, 'hmax': 179, 'smax': 255, 'vmax': 255}

pause = False
param = {"is_clicked" : False, "hsv_val":{'hmin': 0, 'smin': 0, 'vmin': 0, 'hmax': 179, 'smax': 255, 'vmax': 255}}
for warnaBola in warnaBolaHsv:
    hsvVals = {'hmin': 0, 'smin': 193, 'vmin': 99, 'hmax': 17, 'smax': 255, 'vmax': 217}
    myColorFinder = ColorFinder(1)
    setTrackbarValues(hsvVals)
    while True:
        isclosed=0
        cap = cv2.VideoCapture(namaVideo)
        cap.set(3, 1280)
        cap.set(4, 720)

        while True:
            if not(pause):
                success, img = cap.read()
                param["image"] = img
            if np.shape(img) == ():
                 break
            img = cv2.resize(img, (2560, 1440))
            
            if(param["is_clicked"]):
                 setTrackbarValues(param["hsv_val"])
                 param["is_clicked"] = False
            imageColor, mask = myColorFinder.update(img, hsvVals)

            # Remove unnecessary noise from mask
            mask_denoise = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask_denoise = cv2.morphologyEx(mask_denoise, cv2.MORPH_OPEN, kernel)

            imageStack = cvzone.stackImages([img, imageColor, mask, mask_denoise], 2, 0.5)
            imageStack = cv2.resize(imageStack, (960, 540))
            # img = cv2.resize(img, (960, 540))
            # cv2.imshow("pick_color", img)
            cv2.imshow(warnaBola, imageStack)
            cv2.setMouseCallback(warnaBola,mouseRGB, param)

            if cv2.waitKey(2) & 0xFF == ord('p'):
                    pause = not(pause)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                    # cap.release()
                    # cv2.destroyAllWindows()
                    isclosed=1
                    break

            cv2.waitKey(1)
        
        # To break the loop when closed manually
        if isclosed:
            warnaBolaHsv[warnaBola] = myColorFinder.getTrackbarValues()
            cap.release()
            cv2.destroyAllWindows()
            break
        
    time.sleep(1)


with open("deliverables/warnaBola/"+namaHasil+'.json', "w") as outfile:
    json.dump(warnaBolaHsv, outfile)
print(warnaBolaHsv)
# fileResultPickle1 = open("deliverables/warnaBola/"+namaHasil, "wb")
# pickle.dump(warnaBolaHsv, fileResultPickle1)
# fileResultPickle1.close()
    