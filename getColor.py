from cvzone.ColorModule import ColorFinder
import cvzone
import cv2
import numpy as np

warnaBolaHsv = {
        "kuning_kotak"      : {'hmin': 18, 'smin': 153, 'vmin': 139, 'hmax': 50, 'smax': 255, 'vmax': 255},
        "hijau_tua"         : {'hmin': 37, 'smin': 36, 'vmin': 85, 'hmax': 77, 'smax': 204, 'vmax': 255},    # bola hijau tua
        "biru_muda"         : {'hmin': 90, 'smin': 20, 'vmin': 104, 'hmax': 104, 'smax': 61, 'vmax': 255},   # bola biru muda
        "biru_telur_unta"   : {'hmin': 83, 'smin': 70, 'vmin': 53, 'hmax': 92, 'smax': 161, 'vmax': 255},    # bola biru telur unta
        "oranye_kotak"      : {'hmin': 0, 'smin': 95, 'vmin': 216, 'hmax': 21, 'smax': 241, 'vmax': 255},     # bola oranye
        "pink_kotak"        : {'hmin': 156, 'smin': 41, 'vmin': 107, 'hmax': 170, 'smax': 148, 'vmax': 255},   # bola pink kotak
        "ungu"              : {'hmin': 147, 'smin': 36, 'vmin': 90, 'hmax': 164, 'smax': 128, 'vmax': 200},   # ungu
        "biru_kotak"        : {'hmin': 89, 'smin': 84, 'vmin': 145, 'hmax': 130, 'smax': 255, 'vmax': 255}, # terlalu mirip oren
    }

hsvVals = warnaBolaHsv["pink_kotak"]

myColorFinder = ColorFinder(1, hsvVals)

#define kernel size  
kernel = np.ones((7,7),np.uint8)
# backSub = cv2.createBackgroundSubtractorMOG2()
backSub = cv2.createBackgroundSubtractorKNN()

namaVideo = ["VID_DefaultPose", "VID_DefaultPose_1", "VID_DefaultPose_2", "VID_DefaultPose_3", 
             "VID_DefPose_Take2", "VID_DefPose_Take2_1",
             "VID_DefPose_Take3", "VID_DefPose_Take3_1", "VID_DefPose_Take3_2", "VID_DefPose_Take3_3"]
pause = False
if(1):
    while True:
        #This is to check whether to break the first loop
        isclosed=0
        cap = cv2.VideoCapture('video/take4/kiri/VID_DefPose_Take2.mp4')
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
            # fgMask = backSub.apply(img)
            # img = cv2.bitwise_and(img, fgMask)
            imageColor, mask = myColorFinder.update(img, hsvVals)

            # Remove unnecessary noise from mask
            mask_denoise = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask_denoise = cv2.morphologyEx(mask_denoise, cv2.MORPH_OPEN, kernel)

            imageStack = cvzone.stackImages([img, imageColor, mask, mask_denoise], 2, 0.5)

            # img_blur = cv2.GaussianBlur(img, (7, 7), 0)
            # imageColor_blur, mask_blur = myColorFinder.update(img, hsvVals)
            # mask_blur_denoise = cv2.morphologyEx(mask_blur, cv2.MORPH_CLOSE, kernel)
            # mask_blur_denoise = cv2.morphologyEx(mask_blur_denoise, cv2.MORPH_OPEN, kernel)
            # imageStack = cvzone.stackImages([img, imageColor, mask, mask_denoise, mask_blur, mask_blur_denoise], 2, 0.5)

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
    