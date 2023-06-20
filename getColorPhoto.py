from cvzone.ColorModule import ColorFinder
import cvzone
import cv2
import numpy as np

warnaBolaHsv = {
        "kuning_kotak"      : {'hmin': 23, 'smin': 153, 'vmin': 139, 'hmax': 50, 'smax': 255, 'vmax': 255},
        # "hijau_tua"         : {'hmin': 64, 'smin': 26, 'vmin': 75, 'hmax': 79, 'smax': 199, 'vmax': 255},    # bola hijau tua
        # "biru_muda"         : {'hmin': 89, 'smin': 30, 'vmin': 124, 'hmax': 140, 'smax': 146, 'vmax': 255},   # bola biru muda
        # "biru_telur_unta"   : {'hmin': 80, 'smin': 80, 'vmin': 57, 'hmax': 95, 'smax': 227, 'vmax': 255},    # bola biru telur unta
        # "oranye_kotak"      : {'hmin': 4, 'smin': 120, 'vmin': 184, 'hmax': 24, 'smax': 241, 'vmax': 255},     # bola oranye
        # "pink_kotak"        : {'hmin': 139, 'smin': 35, 'vmin': 155, 'hmax': 170, 'smax': 160, 'vmax': 255},   # bola pink kotak
        # "ungu"              : {'hmin': 140, 'smin': 56, 'vmin': 107, 'hmax': 162, 'smax': 120, 'vmax': 255},   # ungu
        # "merah_kotak"       : {'hmin': 0, 'smin': 86, 'vmin': 108, 'hmax': 2, 'smax': 231, 'vmax': 255}, # terlalu mirip oren
    }
#define kernel size  
kernel = np.ones((7,7),np.uint8)
pause = False

img = cv2.imread("video/take3/kiri/IMG_20230616_150223.jpg")
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
        