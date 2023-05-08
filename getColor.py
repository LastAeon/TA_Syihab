from cvzone.ColorModule import ColorFinder
import cvzone
import cv2
import numpy as np


myColorFinder = ColorFinder(1)
hsvVals = {'hmin': 0, 'smin': 96, 'vmin': 160, 'hmax': 31, 'smax': 255, 'vmax': 255}
# {'hmin': 33, 'smin': 72, 'vmin': 126, 'hmax': 58, 'smax': 255, 'vmax': 255} # hijau
# hsvVals = {'hmin': 6, 'smin': 42, 'vmin': 53, 'hmax': 32, 'smax': 206, 'vmax': 156} # kulit

#define kernel size  
kernel = np.ones((7,7),np.uint8)
if(1):
    while True:
        #This is to check whether to break the first loop
        isclosed=0
        cap = cv2.VideoCapture('video/tes_31_03_2023_2.mp4')
        cap.set(3, 1280)
        cap.set(4, 720)

        while True:
            success, img = cap.read()
            # h, w, _ = img.shape
            if np.shape(img) == ():
                 break
            imageColor, mask = myColorFinder.update(img, hsvVals)

            # Remove unnecessary noise from mask
            mask_denoise = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask_denoise = cv2.morphologyEx(mask_denoise, cv2.MORPH_OPEN, kernel)

            imageStack = cvzone.stackImages([img, imageColor, mask, mask_denoise], 2, 0.5)
            cv2.imshow("imageStack", imageStack)


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
    