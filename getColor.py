from cvzone.ColorModule import ColorFinder
import cvzone
import cv2
import numpy as np


hsvVals = {'hmin': 0, 'smin': 86, 'vmin': 108, 'hmax': 4, 'smax': 231, 'vmax': 255}

myColorFinder = ColorFinder(1, hsvVals)

#define kernel size  
kernel = np.ones((7,7),np.uint8)
pause = False
if(1):
    while True:
        #This is to check whether to break the first loop
        isclosed=0
        cap = cv2.VideoCapture('video/take3/kanan/VID_move.mp4')
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
            # img = cv2.resize(img, (1280, 720))
            # img = cv2.resize(img, (640, 360))
            imageColor, mask = myColorFinder.update(img, hsvVals)

            # Remove unnecessary noise from mask
            mask_denoise = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask_denoise = cv2.morphologyEx(mask_denoise, cv2.MORPH_OPEN, kernel)

            imageStack = cvzone.stackImages([img, imageColor, mask, mask_denoise], 2, 0.5)
            imageStack = cv2.resize(imageStack, (960, 540))
            cv2.imshow("imageStack", imageStack)

            if cv2.waitKey(2) & 0xFF == ord('p'):
                    pause = not(pause)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                    # cap.release()
                    # cv2.destroyAllWindows()
                    isclosed=1
                    break

            cv2.waitKey(1)
        # To break the loop if it is closed manually
        if isclosed:
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        success, img = cap.read()
        # h, w, _ = img.shape
        imageColor, mask = myColorFinder.update(img, hsvVals)

        # Remove unnecessary noise from mask
        mask_denoise = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_denoise = cv2.morphologyEx(mask_denoise, cv2.MORPH_OPEN, kernel)

        imageStack = cvzone.stackImages([img, imageColor, mask, mask_denoise], 2, 0.5)
        cv2.imshow("imageStack", imageStack)


        if cv2.waitKey(2) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

        cv2.waitKey(1)
    