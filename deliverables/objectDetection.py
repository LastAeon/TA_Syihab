from cvzone.ColorModule import ColorFinder
import cvzone
import cv2
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter as KalmanFilterLib
from filterpy.kalman import ExtendedKalmanFilter as ExtendedKalmanFilterLib
from skimage import exposure
from skimage.exposure import match_histograms
import json
import sys


class KalmanFilter1D(object):
    def __init__(self, dt, u, std_acc, std_meas):
        self.dt = dt
        self.u = u
        self.std_acc = std_acc

        self.F = np.matrix([[1, self.dt],
                            [0, 1]])
        self.G = np.matrix([[(self.dt**2)/2], 
                            [self.dt]])

        self.H = np.matrix([[1, 0]])

        self.Q = np.matrix([[(self.dt**4)/4, (self.dt**3)/2],
                            [(self.dt**3)/2, self.dt**2]]) * self.std_acc**2

        self.R = std_meas**2

        self.P = np.eye(self.F.shape[1])
        
        self.x = np.matrix([[0], 
                            [0]])

        # print(self.Q)


    def predict(self):
        # Ref :Eq.(9) and Eq.(10)

        # Update time state
        self.x = np.dot(self.F, self.x) + np.dot(self.G, self.u)

        # Calculate error covariance
        # P= A*P*A' + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[0]

    def update(self, z):
        # Ref :Eq.(11) , Eq.(11) and Eq.(13)
        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Eq.(11)

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))  # Eq.(12)

        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P  # Eq.(13)
        # print("self.x : {0}".format(self.x))
        return self.x[0]

def objectDetectionVideo(videoPath, warnaBolaHsv, offset, isLive, isShow, maxFrame = 99999, tremor_threshold=0):
    if(not(isLive)):
        # Create opencv video capture objecttremor_threshold
        # cap = cv2.VideoCapture('video/tes_jarak_bola.mp4')
        cap = cv2.VideoCapture(videoPath)
    else:
        cap = cv2.VideoCapture(0)

    # cap.set(3, 1280)
    # cap.set(4, 720)
    
    myColorFinder = ColorFinder(0)
    
    kalmanFilterHasil = {}

    #define kernel size  
    kernel = np.ones((7,7),np.uint8)

    for i in range(offset):
        success, img = cap.read()

    # init object
    x, y, z = 0, 0, 0
    frameNum = 0

    pastX = -100
    pastY = -100

    try:
        while(frameNum < maxFrame):
            success, img = cap.read()
            img = cv2.resize(img, (2560, 1440))
            imgOutput = img.copy()
            allMask = []
            frameNum += 1
            kalmanFilterHasil[frameNum] = {}
            kalmanFilterHasilLocal = kalmanFilterHasil[frameNum]
            # h, w, _ = img.shape
            for warnaBola in warnaBolaHsv:
                hsvVals = warnaBolaHsv[warnaBola]
                # kalmanFilterHasilLokal = kalmanFilterHasil[warnaBola]
                imageColor, mask = myColorFinder.update(img, hsvVals)
                # print("mask shape:", np.shape(mask), "value:", np.unique(mask))

                # Remove unnecessary noise from mask
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if(len(contours) > 0):
                    biggestContour = max(contours, key = cv2.contourArea)
                    cv2.drawContours(imgOutput, biggestContour, -1, (0,255,0), 3)

                    # bisa dipake buat approximate lokasi di video depth
                    # https://stackoverflow.com/questions/69637673/finding-points-within-a-contour-using-opencv
                    # https://stackoverflow.com/questions/70438811/reading-frames-from-two-video-sources-is-not-in-sync-opencv
                    x,y,w,h = cv2.boundingRect(biggestContour) 
                    cv2.rectangle(imgOutput,(x,y),(x+w,y+h),(255,0,0),2)

                    # Update
                    M = cv2.moments(biggestContour)
                    if(M['m00'] != 0):
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                    else:
                        cx = x
                        cy = y
                    # minimize tremmor
                    if(abs(cx - pastX) < tremor_threshold):
                        cx = pastX
                    if(abs(cy - pastY) < tremor_threshold):
                        cy = pastY
                    pastX = cx
                    pastY = cy
                    cv2.rectangle(imgOutput, (cx-15, cy-15), (cx+15, cy+15), (0, 255, 255), 2)
                    cv2.putText(imgOutput, warnaBola, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(imgOutput, str(cv2.contourArea(biggestContour)), (int(cx), int(cy)-25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
                    kalmanFilterHasilLocal[warnaBola] = [cx, cy]
                else:
                    kalmanFilterHasilLocal[warnaBola] = [-1, -1]
                    pastX = -100
                    pastY = -100
                allMask.append(mask)
            
            if(isShow):
                imgOutput = cv2.resize(imgOutput, (1280, 720))
                cv2.imshow("no_tracking_"+videoPath, imgOutput)

                # imageStack = cvzone.stackImages(allMask, 2, 0.5)
                # imageStack = cv2.resize(imageStack, (1280, 720))
                # cv2.imshow("imageStack", imageStack)


            if cv2.waitKey(2) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break

            cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("error: ", e)
        cap.release()
        cv2.destroyAllWindows()
    return kalmanFilterHasil

def objectTrackingKalmanFilter(videoPath, warnaBolaHsv, offset, isLive, isShow):
    if(not(isLive)):
        # Create opencv video capture object
        # cap = cv2.VideoCapture('video/tes_jarak_bola.mp4')
        cap = cv2.VideoCapture(videoPath)
    else:
        cap = cv2.VideoCapture(0)

    # cap.set(3, 1280)
    # cap.set(4, 720)
    # kalman filter intial attributes
    dt = 0.1
    u = 1
    std_acc = 1
    std_meas = 0.1
    
    myColorFinder = ColorFinder(0)
    
    kalmanFilterBola = {}
    kalmanFilterSupportingAttributes = {}
    kalmanFilterHasil = {}
    for warnaBola in warnaBolaHsv:
        kalmanFilterBola[warnaBola] = {"KF_X" : KalmanFilter1D(dt, u, std_acc, std_meas), "KF_Y" : KalmanFilter1D(dt, u, std_acc, std_meas), "KF_Z" : KalmanFilter1D(dt, u, std_acc, std_meas)}
        kalmanFilterSupportingAttributes[warnaBola] = {"is_active" : False, "is_new" : True, "frame_since_last_detected" : 0}
        # kalmanFilterHasil[warnaBola] = []
    MAXFRAMESINCEDETECTED = 10

    #define kernel size  
    kernel = np.ones((7,7),np.uint8)

    # init object
    x, y, z = 0, 0, 0
    frameNum = 0

    # fixing offset
    for i in range(offset):
        success, img = cap.read()

    try:
        while(True):
            success, img = cap.read()
            img = cv2.resize(img, (2560, 1440))
            imgOutput = img.copy()
            allMask = []
            frameNum += 1
            kalmanFilterHasil[frameNum] = {}
            kalmanFilterHasilLocal = kalmanFilterHasil[frameNum]
            # h, w, _ = img.shape
            for warnaBola in warnaBolaHsv:
                hsvVals = warnaBolaHsv[warnaBola]
                KF_1D = kalmanFilterBola[warnaBola]
                kalmanFilterSupportingAttributesLocal = kalmanFilterSupportingAttributes[warnaBola]
                # kalmanFilterHasilLokal = kalmanFilterHasil[warnaBola]
                imageColor, mask = myColorFinder.update(img, hsvVals)
                # print("mask shape:", np.shape(mask), "value:", np.unique(mask))

                # Remove unnecessary noise from mask
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if(len(contours) > 0):
                    if(not(kalmanFilterSupportingAttributesLocal["is_active"])):
                        kalmanFilterBola[warnaBola] = {"KF_X" : KalmanFilter1D(dt, u, std_acc, std_meas), "KF_Y" : KalmanFilter1D(dt, u, std_acc, std_meas), "KF_Z" : KalmanFilter1D(dt, u, std_acc, std_meas)}
                        KF_1D = kalmanFilterBola[warnaBola]
                        kalmanFilterSupportingAttributesLocal["is_active"] = True

                    kalmanFilterSupportingAttributesLocal["frame_since_last_detected"] = 0

                    biggestContour = max(contours, key = cv2.contourArea)
                    cv2.drawContours(imgOutput, biggestContour, -1, (0,255,0), 3)

                    # bisa dipake buat approximate lokasi di video depth
                    # https://stackoverflow.com/questions/69637673/finding-points-within-a-contour-using-opencv
                    # https://stackoverflow.com/questions/70438811/reading-frames-from-two-video-sources-is-not-in-sync-opencv
                    x,y,w,h = cv2.boundingRect(biggestContour) 
                    cv2.rectangle(imgOutput,(x,y),(x+w,y+h),(255,0,0),2)

                    # Predict
                    # (x, y, z) = KF.predict()
                    x, y, z = KF_1D["KF_X"].predict(), KF_1D["KF_Y"].predict(), KF_1D["KF_Z"].predict()

                    # Update
                    M = cv2.moments(biggestContour)
                    if(M['m00'] != 0):
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                    else:
                        cx = x
                        cy = y
                    # (x1, y1, z1) = KF.update([cx, cy, 0])
                    x1, y1, z1 = KF_1D["KF_X"].update(cx), KF_1D["KF_Y"].update(cy), KF_1D["KF_Z"].update(0)

                    # Draw a rectangle as the predicted object position
                    # cv2.rectangle(imgOutput, (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), (0, 0, 255), 2)
                    cv2.rectangle(imgOutput, (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), (0, 255, 255), 2)
                else:
                    if(kalmanFilterSupportingAttributesLocal["frame_since_last_detected"] > MAXFRAMESINCEDETECTED):
                        kalmanFilterSupportingAttributesLocal["is_active"] = False
                    else:
                        kalmanFilterSupportingAttributesLocal["frame_since_last_detected"] += 1
                        # Predict
                        # (x, y, z) = KF.predict()
                        x, y, z = KF_1D["KF_X"].predict(), KF_1D["KF_Y"].predict(), KF_1D["KF_Z"].predict()

                        cx = x
                        cy = y
                        x1, y1, z1 = KF_1D["KF_X"].update(cx), KF_1D["KF_Y"].update(cy), KF_1D["KF_Z"].update(0)

                        # Draw a rectangle as the predicted object position
                        # cv2.rectangle(imgOutput, (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), (0, 0, 255), 2)
                        cv2.rectangle(imgOutput, (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), (0, 255, 255), 2)
                if(kalmanFilterSupportingAttributesLocal["is_active"]):
                    cv2.putText(imgOutput, warnaBola, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(imgOutput, str(cv2.contourArea(biggestContour)), (int(x), int(y)-25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
                    kalmanFilterHasilLocal[warnaBola] = [x[0,0], y[0,0]]
                    # fileResult1.write("{:.3f}, {:.3f}, ".format(x[0,0], y[0,0]))
                    # print("nilai: {:.3f}, {:.3f}, ".format(x[0,0], y[0,0]))
                    # print("shape: {}, {}, ".format(np.shape(x[0,0]), np.shape(y[0,0])))
                else:
                    # fileResult1.write("-, -, ")
                    kalmanFilterHasilLocal[warnaBola] = [-1, -1]
                allMask.append(mask)
            # fileResult1.write("\n")
            
            imgOutput = cv2.resize(imgOutput, (1280, 720))
            if(isShow):
                cv2.imshow("kalman_filter_"+videoPath, imgOutput)

                # imageStack = cvzone.stackImages(allMask, 2, 0.5)
                # imageStack = cv2.resize(imageStack, (1280, 720))
                # cv2.imshow("imageStack", imageStack)


            if cv2.waitKey(2) & 0xFF == ord('q'):
                    break

            cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("error: ", e)
        cap.release()
        cv2.destroyAllWindows()
    return kalmanFilterHasil
    # fileResult1.close()

def objectTrackingOpticalFlow(videoPath, warnaBolaHsv, offset, isLive, isShow):
    if(not(isLive)):
        # Create opencv video capture object
        # cap = cv2.VideoCapture('video/tes_jarak_bola.mp4')
        cap = cv2.VideoCapture(videoPath)
    else:
        cap = cv2.VideoCapture(0)

    # cap.set(3, 1280)
    # cap.set(4, 720)
    
    myColorFinder = ColorFinder(0)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                  maxLevel = 1,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # kalmanFilterBola = {}
    OpticalFlowSupportingAttributes = {}
    kalmanFilterHasil = {}
    for warnaBola in warnaBolaHsv:
        OpticalFlowSupportingAttributes[warnaBola] = {"is_tracked" : False, "koordinat" : [0, 0]}
    MAXFRAMESINCEDETECTED = 10

    #define kernel size  
    kernel = np.ones((7,7),np.uint8)

    # init object
    x, y, z = 0, 0, 0
    frameNum = 0

    # fixing offset
    for i in range(offset):
        success, img = cap.read()

    try:
        ret, oldFrame = cap.read()
        oldFrame = cv2.resize(oldFrame, (2560, 1440))
        imgOutput = oldFrame.copy()
        oldFrameGray = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
        for warnaBola in warnaBolaHsv:
            hsvVals = warnaBolaHsv[warnaBola]
            OpticalFlowSupportingAttributesLocal = OpticalFlowSupportingAttributes[warnaBola]
            imageColor, mask = myColorFinder.update(oldFrame, hsvVals)
            # print("mask shape:", np.shape(mask), "value:", np.unique(mask))

            # Remove unnecessary noise from mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if(len(contours) > 0):
                OpticalFlowSupportingAttributesLocal["is_tracked"] = True

                biggestContour = max(contours, key = cv2.contourArea)
                cv2.drawContours(imgOutput, biggestContour, -1, (0,255,0), 3)

                # bisa dipake buat approximate lokasi di video depth
                # https://stackoverflow.com/questions/69637673/finding-points-within-a-contour-using-opencv
                # https://stackoverflow.com/questions/70438811/reading-frames-from-two-video-sources-is-not-in-sync-opencv
                x,y,w,h = cv2.boundingRect(biggestContour) 
                cv2.rectangle(imgOutput,(x,y),(x+w,y+h),(255,0,0),2)

                # Update
                M = cv2.moments(biggestContour)
                if(M['m00'] != 0):
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                else:
                    cx = x
                    cy = y
                OpticalFlowSupportingAttributesLocal["koordinat"] = [cx, cy]

                # Draw a rectangle as the predicted object position
                # cv2.rectangle(imgOutput, (int(cx - 15), int(cy - 15)), (int(cx + 15), int(cy + 15)), (0, 255, 255), 2)
        while(True):
            success, img = cap.read()
            img = cv2.resize(img, (2560, 1440))
            imgOutput = img.copy()
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            allMask = []
            frameNum += 1
            kalmanFilterHasil[frameNum] = {}
            kalmanFilterHasilLocal = kalmanFilterHasil[frameNum]
            p0new = []
            # h, w, _ = img.shape
            for warnaBola in warnaBolaHsv:
                hsvVals = warnaBolaHsv[warnaBola]
                OpticalFlowSupportingAttributesLocal = OpticalFlowSupportingAttributes[warnaBola]

                if not(OpticalFlowSupportingAttributesLocal["is_tracked"]) or frameNum%5 == 0:
                    # intinya is p0, kalo gak tracked object detection dulu
                    imageColor, mask = myColorFinder.update(img, hsvVals)
                    # print("mask shape:", np.shape(mask), "value:", np.unique(mask))

                    # Remove unnecessary noise from mask
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if(len(contours) > 0):
                        OpticalFlowSupportingAttributesLocal["is_tracked"] = True

                        biggestContour = max(contours, key = cv2.contourArea)
                        cv2.drawContours(imgOutput, biggestContour, -1, (0,255,0), 3)

                        # bisa dipake buat approximate lokasi di video depth
                        # https://stackoverflow.com/questions/69637673/finding-points-within-a-contour-using-opencv
                        # https://stackoverflow.com/questions/70438811/reading-frames-from-two-video-sources-is-not-in-sync-opencv
                        x,y,w,h = cv2.boundingRect(biggestContour) 
                        cv2.rectangle(imgOutput,(x,y),(x+w,y+h),(255,0,0),2)

                        # Update
                        M = cv2.moments(biggestContour)
                        if(M['m00'] != 0):
                            cx = int(M['m10']/M['m00'])
                            cy = int(M['m01']/M['m00'])
                        else:
                            cx = x
                            cy = y
                        OpticalFlowSupportingAttributesLocal["koordinat"] = [cx, cy]

                        # Draw a rectangle as the predicted object position
                        # cv2.rectangle(imgOutput, (int(cx - 15), int(cy - 15)), (int(cx + 15), int(cy + 15)), (0, 255, 255), 2)
                if OpticalFlowSupportingAttributesLocal["is_tracked"]:
                    p0new.append([OpticalFlowSupportingAttributesLocal["koordinat"]])
                else:
                    p0new.append([[0, 0]])

            # Predict
            # calculate optical flow
            p0 = np.float32(p0new)
            p1, st, err = cv2.calcOpticalFlowPyrLK(oldFrameGray, imgGray, p0, None, **lk_params)

            # save detected points good points
            for idx, warnaBola in enumerate(warnaBolaHsv):
                x = p1[idx, 0, 0]
                y = p1[idx, 0, 1]
                if st[idx, 0] == 1:
                    OpticalFlowSupportingAttributes[warnaBola]["koordinat"] = [x, y]
                else:
                    OpticalFlowSupportingAttributes[warnaBola]["is_tracked"] = False

                # Draw a rectangle as the predicted object position
                cv2.rectangle(imgOutput, (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), (0, 255, 255), 2)
                if(OpticalFlowSupportingAttributes[warnaBola]["is_tracked"]):
                    cv2.putText(imgOutput, warnaBola, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(imgOutput, str(cv2.contourArea(biggestContour)), (int(x), int(y)-25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
                    kalmanFilterHasilLocal[warnaBola] = [x, y]
                    # fileResult1.write("{:.3f}, {:.3f}, ".format(x[0,0], y[0,0]))
                    # print("nilai: {:.3f}, {:.3f}, ".format(x[0,0], y[0,0]))
                    # print("shape: {}, {}, ".format(np.shape(x[0,0]), np.shape(y[0,0])))
                else:
                    # fileResult1.write("-, -, ")
                    kalmanFilterHasilLocal[warnaBola] = [-1, -1]
                allMask.append(mask)
                # fileResult1.write("\n")
            
            oldFrameGray = imgGray.copy()
            imgOutput = cv2.resize(imgOutput, (1280, 720))
            if(isShow):
                cv2.imshow("optical_flow_"+videoPath, imgOutput)

                # imageStack = cvzone.stackImages(allMask, 2, 0.5)
                # imageStack = cv2.resize(imageStack, (1280, 720))
                # cv2.imshow("imageStack", imageStack)


            if cv2.waitKey(2) & 0xFF == ord('q'):
                    break

            cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("error: ", e)
        cap.release()
        cv2.destroyAllWindows()
        # e.print_exc()
    return kalmanFilterHasil
    # fileResult1.close()


OFFSET = {}
warnaBolaTengah = {}
warnaBolaKanan = {}
warnaBolaKiri = {}

if(len(sys.argv)<7):
    print("argumen tidak lengkap")
    exit()

tipe = sys.argv[1]
namaOffset = sys.argv[2]
namaWarna1 = sys.argv[3]
namaWarna2 = sys.argv[4]
namaWarna3 = sys.argv[5]
namaVideo = sys.argv[6]
namaHasil = sys.argv[7]

with open("deliverables/warnaBola/"+namaOffset, "r") as outfile:
    OFFSET = json.load(outfile)
    # print(OFFSET)

with open("deliverables/warnaBola/"+namaWarna1, "r") as outfile:
    warnaBolaTengah = json.load(outfile)
    # print(warnaBolaTengah)

with open("deliverables/warnaBola/"+namaWarna2, "r") as outfile:
    warnaBolaKanan = json.load(outfile)
    # print(warnaBolaKanan)

with open("deliverables/warnaBola/"+namaWarna3, "r") as outfile:
    warnaBolaKiri = json.load(outfile)
    # print(warnaBolaKanan)

namaVideoArray = [namaVideo]
offsetVideoArray = OFFSET["take4"]
for idx, namaVideo in enumerate(namaVideoArray):
    # tracking
    hasilObjectTracking = {}
    if(tipe.lower() == "no_tracking"):
        hasilObjectTracking[0] = objectDetectionVideo('video/take4/tengah/'+namaVideo+'.mp4', warnaBolaTengah, OFFSET["take4"][namaVideo][0], False, True)
        hasilObjectTracking[1] = objectDetectionVideo('video/take4/kanan/'+namaVideo+'.mp4', warnaBolaKanan, OFFSET["take4"][namaVideo][1], False, True)
        hasilObjectTracking[2] = objectDetectionVideo('video/take4/kiri/'+namaVideo+'.mp4', warnaBolaKiri, OFFSET["take4"][namaVideo][2], False, True)
    elif(tipe.lower() == "kalman_filter"):
        hasilObjectTracking[0] = objectTrackingKalmanFilter('video/take4/tengah/'+namaVideo+'.mp4', warnaBolaTengah, OFFSET["take4"][namaVideo][0], False, True)
        hasilObjectTracking[1] = objectTrackingKalmanFilter('video/take4/kanan/'+namaVideo+'.mp4', warnaBolaKanan, OFFSET["take4"][namaVideo][1], False, True)
        hasilObjectTracking[2] = objectTrackingKalmanFilter('video/take4/kiri/'+namaVideo+'.mp4', warnaBolaKiri, OFFSET["take4"][namaVideo][2], False, True)
    elif(tipe.lower() == "optical_flow"):
        hasilObjectTracking[0] = objectTrackingOpticalFlow('video/take4/tengah/'+namaVideo+'.mp4', warnaBolaTengah, OFFSET["take4"][namaVideo][0], False, True)
        hasilObjectTracking[1] = objectTrackingOpticalFlow('video/take4/kanan/'+namaVideo+'.mp4', warnaBolaKanan, OFFSET["take4"][namaVideo][1], False, True)
        hasilObjectTracking[2] = objectTrackingOpticalFlow('video/take4/kiri/'+namaVideo+'.mp4', warnaBolaKiri, OFFSET["take4"][namaVideo][2], False, True)
    else:
        print("metode tidak ada!")
        exit()

    # reformat
    hasilObjectTrackingFormatted = {}
    maxFrame = min(len(hasilObjectTracking[0]), len(hasilObjectTracking[1]), len(hasilObjectTracking[2]))
    markerIndex = hasilObjectTracking[0][1].keys()
    # print(maxFrame, markerIndex)
    initial_size = [2560, 1440]
    actual_size = [1920, 1080]

    for i in range(maxFrame):
        frameContent = {}
        for marker in markerIndex:
            frameContent[marker] = {"c":hasilObjectTracking[0][i+1][marker], "l":hasilObjectTracking[1][i+1][marker], "r":hasilObjectTracking[2][i+1][marker]}
            for camera in frameContent[marker]:
                frameContent[marker][camera] = [frameContent[marker][camera][0]*actual_size[0]/initial_size[0], frameContent[marker][camera][1]*actual_size[1]/initial_size[1]]
        hasilObjectTrackingFormatted[i+1] = frameContent
    # print(hasilObjectTrackingFormatted[1])
    # print(hasilObjectTrackingFormatted)

    # save
    with open("deliverables/warnaBola/"+namaHasil+'.json', "w") as outfile:
        json.dump(hasilObjectTrackingFormatted, outfile)
    # fileResultPickle1 = open("output2d/outputPickle_"+namaVideo+"_with_offset_no_tracking", "wb")
    # pickle.dump(hasilObjectTrackingFormatted, fileResultPickle1)
    # fileResultPickle1.close()
