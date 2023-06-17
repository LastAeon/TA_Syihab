import numpy as np
import cv2 as cv
import argparse
import random

cap = cv.VideoCapture("video/slow_traffic_small.mp4")
cap.set(3, 1280)
cap.set(4, 720)
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 5,
                       blockSize = 10 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
# old_frame = cv.resize(old_frame, (2560, 1440))
# print("old frame", type(old_frame), old_frame.shape, type(old_frame[0,0,0]))

# fromImage = cv.imread("video/foto_1.png")
# fromImage = cv.resize(fromImage, (2560, 1440))
# print("image frame", type(fromImage), fromImage.shape, type(fromImage[0,0,0]))

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
print(p0, type(p0), p0.shape, type(p0[0,0,0]))
# p0 = np.array([[[1, 1, 1], [1, 1, 1]]])
# print(p0, type(p0), p0.shape, type(p0[0,0,0]))
p0 = np.float32([[[374.0, 196.0]], [[315, 198]]])
print(p0, type(p0), p0.shape, type(p0[0,0,0]))

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
errorOnFirstFrame = True
while(1):
    # print("looping!")
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    print(p1, type(p1), p1.shape, type(p1[0,0,0]), st, st.shape, type(st[0,0]))

    # Select good points
    rand = random.randint(1, 5)
    if(rand == 2):
        print("random!")
        st = np.int8([[1],[0]])
    # elif(rand==3):
    #     st = [[1],[0]]
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)

    cv.imshow('frame', img)
    cv.imshow('mask', mask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()