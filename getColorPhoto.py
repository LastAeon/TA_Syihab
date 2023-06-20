from cvzone.ColorModule import ColorFinder
import cvzone
import cv2
import numpy as np

warnaBolaHsv = {
        "kuning_kotak"      : {'hmin': 15, 'smin': 153, 'vmin': 139, 'hmax': 50, 'smax': 255, 'vmax': 255},
        "hijau_tua"         : {'hmin': 64, 'smin': 110, 'vmin': 75, 'hmax': 87, 'smax': 199, 'vmax': 255},    # bola hijau tua
        "biru_muda"         : {'hmin': 90, 'smin': 67, 'vmin': 124, 'hmax': 107, 'smax': 99, 'vmax': 255},   # bola biru muda
        "biru_telur_unta"   : {'hmin': 80, 'smin': 80, 'vmin': 57, 'hmax': 95, 'smax': 227, 'vmax': 255},    # bola biru telur unta
        "oranye_kotak"      : {'hmin': 5, 'smin': 126, 'vmin': 182, 'hmax': 15, 'smax': 241, 'vmax': 255},     # bola oranye
        "pink_kotak"        : {'hmin': 160, 'smin': 41, 'vmin': 107, 'hmax': 170, 'smax': 201, 'vmax': 255},   # bola pink kotak
        "ungu"              : {'hmin': 140, 'smin': 73, 'vmin': 70, 'hmax': 155, 'smax': 137, 'vmax': 255},   # ungu
        "biru_kotak"        : {'hmin': 109, 'smin': 84, 'vmin': 145, 'hmax': 130, 'smax': 255, 'vmax': 255}, # terlalu mirip oren
    }
#define kernel size  
kernel = np.ones((7,7),np.uint8)
pause = False

img = cv2.imread("video/take4/kanan/IMG_DefaultPose.jpg")
# img = cv2.resize(img, (5120, 2880))
img = cv2.resize(img, (2560, 1440))
imgCopy = img.copy()
# img = cv2.resize(img, (1280, 720))
# img = cv2.resize(img, (640, 360))
# myColorFinder = ColorFinder(1, {'hmin': 23, 'smin': 153, 'vmin': 139, 'hmax': 50, 'smax': 255, 'vmax': 255})

for warnaBola in warnaBolaHsv:
    hsvVals = warnaBolaHsv[warnaBola]

    myColorFinder = ColorFinder(1, hsvVals)


    while True:
        imageColor, mask = myColorFinder.update(img, hsvVals)

        # Remove unnecessary noise from mask
        mask_denoise = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_denoise = cv2.morphologyEx(mask_denoise, cv2.MORPH_OPEN, kernel)

        imageStack = cvzone.stackImages([imgCopy, imageColor, mask, mask_denoise], 2, 0.5)
        imageStack = cv2.resize(imageStack, (960, 540))
        cv2.imshow("imageStack", imageStack)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        