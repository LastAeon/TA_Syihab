from cvzone.ColorModule import ColorFinder
import cvzone
import cv2
import numpy as np

hsvVals = {'hmin': 0, 'smin': 193, 'vmin': 99, 'hmax': 17, 'smax': 255, 'vmax': 217}

myColorFinder = ColorFinder(1, hsvVals)

#define kernel size  
kernel = np.ones((7,7),np.uint8)

pause = False
minSizeDetectedPixel = 999999

while True:
    #This is to check whether to break the first loop
    isclosed=0
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        if not(pause):
            success, img = cap.read()
        # h, w, _ = img.shape
        if np.shape(img) == ():
                break
        # img = cv2.resize(img, (5120, 2880))
        img = cv2.resize(img, (2560, 1440))
        imageColor, mask = myColorFinder.update(img, hsvVals)

        # Remove unnecessary noise from mask
        mask_denoise = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_denoise = cv2.morphologyEx(mask_denoise, cv2.MORPH_OPEN, kernel)

        imageStack = cvzone.stackImages([img, imageColor, mask, mask_denoise], 2, 0.5)
        contours, _ = cv2.findContours(mask_denoise, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if(len(contours) > 0):
              biggestContour = max(contours, key = cv2.contourArea)
              sizeDetectedPixel = int(cv2.contourArea(biggestContour))
              if(minSizeDetectedPixel > sizeDetectedPixel):
                    minSizeDetectedPixel = sizeDetectedPixel

        imageStack = cv2.resize(imageStack, (960, 540))
        cv2.imshow("imageStack", imageStack)

        if cv2.waitKey(2) & 0xFF == ord('p'):
                pause = not(pause)

        if cv2.waitKey(2) & 0xFF == ord('q'):
                isclosed=1
                break

        cv2.waitKey(1)
    # To break the loop if it is closed manually
    if isclosed:
        break

cap.release()
cv2.destroyAllWindows()

print("minSizeDetectedPixel ", minSizeDetectedPixel)