from imutils.video import VideoStream
import imutils
import cv2
import time
import numpy as np
import colorsys

vs = cv2.VideoCapture("test_video.avi")
time.sleep(2.0)

frame_rate = 30
prev = 0


def prep(input_frame):
    # Blue shapes
    Lower = (90, 50, 70)
    Upper = (128, 255, 255)


    output_frame = imutils.resize(input_frame, width=800)
    # output_frame = input_frame.copy()
    ## Todo 3.1.3 gaussian blur
    blurred = cv2.blur(output_frame, (11, 11))
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    output_mask = cv2.inRange(hsv, Lower, Upper)

    return output_mask, output_frame


def find_cnts(input_mask):
    # input: src_img, contour_mode, approx_method
    # output: contours and hierarchy
    # SIMPLE: two endpoints of the line
    # NONE: all boundary points
    threshold = 100
    canny_output = cv2.Canny(input_mask, threshold, threshold * 2)
    cnts = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts


def contour(input_frame, input_mask):
    cnts = find_cnts(input_mask)

    # c = cnts[5]
    c = max(cnts, key=cv2.contourArea)
    frame1 = input_frame.copy()
    # for c in cnts:
    ((x, y), radius) = cv2.minEnclosingCircle(c)  ## A different contour?
    # print(cnts)

    # Find center of contour using moments in opencvq
    # M = cv2.moments(c)
    # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    # circle
    cv2.circle(frame1, (int(x), int(y)), int(radius), (0, 0, 255), 2)
    cv2.circle(frame1, (int(x), int(y)), 5, (0, 0, 255), -1)

    # draw rectangle with green line
    # contours_poly = cv2.approxPolyDP(c, 3, True)
    # boundRect = cv2.boundingRect(contours_poly)
    # cv2.rectangle(frame1, (int(boundRect[0]), int(boundRect[1])),
    #               (int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])), (0, 255, 0), 1)

    # draw rotate rectangle with blue line
    # rect = cv2.minAreaRect(c)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(frame1, [box], 0, (255, 0, 0), 1)

    # draw Hull with pink line
    # hull = cv2.convexHull(c)
    # cv2.drawContours(frame1, [hull], 0, (255, 0, 255), 1)

    cv2.imshow("Task 02 Circle", frame1)


while True:
    time_elapsed = time.time() - prev

    if time_elapsed > 1. / frame_rate:
        frame = vs.read()
        frame = frame[1]
        if frame is None:
            break
        mask, frame = prep(frame)
        contour(frame, mask)
        cv2.imshow("Frame", frame)
        cv2.imshow("Masked Image", mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        prev = time.time()

cv2.destroyAllWindows()
