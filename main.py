from cvzone.ColorModule import ColorFinder
import cvzone
import cv2
import numpy as np

from KalmanFilter import KalmanFilter
from KalmanFilter1D import KalmanFilter1D

 
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
 
myColorFinder = ColorFinder(0)
# hsvVals = {'hmin': 33, 'smin': 72, 'vmin': 126, 'hmax': 58, 'smax': 255, 'vmax': 255} # hijau
hsvVals = {'hmin': 6, 'smin': 42, 'vmin': 53, 'hmax': 32, 'smax': 206, 'vmax': 156}

#define kernel size  
kernel = np.ones((7,7),np.uint8)

#Create KalmanFilter object KF
#KalmanFilter(dt, u_x, u_y, u_z, std_acc, x_std_meas, y_std_meas, z_std_meas)

KF = KalmanFilter(0.1, 1, 1, 1, 1, 0.1, 0.1, 0.1)
KF_X = KalmanFilter1D(0.1, 1, 1, 0.1)
KF_Y = KalmanFilter1D(0.1, 1, 1, 0.1)
KF_Z = KalmanFilter1D(0.1, 1, 1, 0.1)

while True:
    success, img = cap.read()
    # h, w, _ = img.shape
    imageColor, mask = myColorFinder.update(img, hsvVals)
    # print("mask shape:", np.shape(mask), "value:", np.unique(mask))

    # Remove unnecessary noise from mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours) > 0):
        biggestContour = max(contours, key = cv2.contourArea)
        cv2.drawContours(img, biggestContour, -1, (0,255,0), 3)

        # bisa dipake buat approximate lokasi di video depth
        # https://stackoverflow.com/questions/69637673/finding-points-within-a-contour-using-opencv
        # https://stackoverflow.com/questions/70438811/reading-frames-from-two-video-sources-is-not-in-sync-opencv
        x,y,w,h = cv2.boundingRect(biggestContour) 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        # Predict
        # (x, y, z) = KF.predict()
        x, y, z = KF_X.predict(), KF_Y.predict(), KF_Z.predict()
        # print(x,y,z)
        # Draw a rectangle as the predicted object position
        cv2.rectangle(img, (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), (255, 0, 0), 2)

        # Update
        M = cv2.moments(biggestContour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        # (x1, y1, z1) = KF.update([cx, cy, 0])
        x1, y1, z1 = KF_X.update(cx), KF_Y.update(cy), KF_Z.update(0)
    
    imageStack = cvzone.stackImages([img, mask], 2, 0.5)
    cv2.imshow("imageStack", imageStack)


    if cv2.waitKey(2) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cv2.waitKey(1)
 
