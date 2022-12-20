import cv2
import time
import numpy as np
import time
import imutils

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def prep(img):
    ## ToDo 4.1.2
    #  1. resize using  imutils.resize()
    img = imutils.resize(img, width=600)
    #  2. flip image using cv2.flip()
    img = cv2.flip(img, 1)
    #  3. convert to gray color using cv2.cvtColor()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray, img


def get_trackable_points(gray, img):
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) != 0:

        ## ToDO 4.1.3
        for (x, y, w, h) in faces:
            # draw rectangle
            # (x, y) starting point
            # (x + w, y + h) ending point
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #   slice ROI
            roi_gray = gray[y:y + h, x:x + w]

        global p0
        p0 = cv2.goodFeaturesToTrack(roi_gray, maxCorners=70,
                                     qualityLevel=0.001, minDistance=5)
        # ToDO 4.1.3
        #  covert format of p0 to [[x1,y1],[x2,y2],....]

        # number of row of p0
        row = p0.shape[0]
        # leave this dimension for function to determine
        p0 = p0.reshape(-1)

        p0 = p0.reshape((row, 2))

        #  convert points from ROI to image coordinates
        for a in range(len(p0)):
            p0[a][0] = p0[a][0] + x
            p0[a][1] = p0[a][1] + y
            print("x-axis is %d, y-axis is %d" % (p0[a][0], p0[a][1]))

    return p0, faces, img


def do_track_face(gray_prev, gray, p0):
    p1, isFound, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray, p0,
                                                None,
                                                winSize=(31, 31),
                                                maxLevel=10,
                                                criteria=(
                                                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                    30, 0.03),
                                                flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                                                minEigThreshold=0.00025)
    ## ToDo 4.1.3 - Select valid points from p1
    good_new = []
    if p1 is not None:
        for i in range(len(isFound)):
            if isFound[i] == 1:
                good_new.append(p1[i])
    # return a numpy array of selected points from p0
    p1 = np.array(good_new)

    return p1


frame_rate = 30
prev = 0
gray_prev = None
p0 = []
# cam = cv2.VideoCapture("Face.mp4")
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise Exception("Could not open camera/file")

while True:
    time_elapsed = time.time() - prev

    if time_elapsed > 1. / frame_rate:

        ret_val, img = cam.read()

        if not ret_val:
            cam.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart video
            gray_prev = None  # previous frame
            p0 = []  # previous points
            continue
        prev = time.time()

        gray, img = prep(img)

        if len(p0) <= 10:
            p0, faces, img = get_trackable_points(gray, img)
            gray_prev = gray.copy()

        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            p1 = do_track_face(gray_prev, gray, p0)
            for i in p1:
                cv2.drawMarker(img, (int(i[0]), int(i[1])), [255, 0, 0], 0)
            p0 = p1

        cv2.imshow('Video feed', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cv2.destroyAllWindows()