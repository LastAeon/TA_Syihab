from cvzone.ColorModule import ColorFinder
import cvzone
import cv2
import numpy as np

warnaBolaHsv = {
        "kuning_kotak"      : {'hmin': 15, 'smin': 153, 'vmin': 139, 'hmax': 50, 'smax': 255, 'vmax': 255},
        "hijau_tua"         : {'hmin': 64, 'smin': 110, 'vmin': 75, 'hmax': 87, 'smax': 199, 'vmax': 255},    # bola hijau tua
        "biru_muda"         : {'hmin': 98, 'smin': 67, 'vmin': 124, 'hmax': 107, 'smax': 146, 'vmax': 255},   # bola biru muda
        "biru_telur_unta"   : {'hmin': 80, 'smin': 80, 'vmin': 57, 'hmax': 95, 'smax': 227, 'vmax': 255},    # bola biru telur unta
        "oranye_kotak"      : {'hmin': 5, 'smin': 126, 'vmin': 182, 'hmax': 9, 'smax': 241, 'vmax': 255},     # bola oranye
        "pink_kotak"        : {'hmin': 160, 'smin': 41, 'vmin': 107, 'hmax': 170, 'smax': 201, 'vmax': 255},   # bola pink kotak
        "ungu"              : {'hmin': 140, 'smin': 56, 'vmin': 70, 'hmax': 155, 'smax': 137, 'vmax': 255},   # ungu
        "biru_kotak"        : {'hmin': 109, 'smin': 84, 'vmin': 145, 'hmax': 130, 'smax': 255, 'vmax': 255}, # terlalu mirip oren
    }

hsvVals = warnaBolaHsv["kuning_kotak"]

myColorFinder = ColorFinder(1, hsvVals)

#define kernel size  
kernel = np.ones((7,7),np.uint8)
pause = False
if(1):
    while True:
        #This is to check whether to break the first loop
        isclosed=0
        cap = cv2.VideoCapture('video/take4/kanan/VID_DefaultPose.mp4')
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
    