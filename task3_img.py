import imutils
import cv2
import colorsys
import numpy as np
from random import randint

dilatation_size = 0
erosion_size = 0
max_elem = 2
max_kernel_size = 21

title_trackbar_element_shape = 'Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n +1'
title_erosion_window = 'Erosion Demo'
title_dilation_window = 'Dilation Demo'

# Task 01
src = cv2.imread('image.png')
# cv2.imshow("Tracking",frame)
# cv2.waitKey(0) & 0xFF
# cv2.destroyAllWindows()


def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE


def erosion(val):
    erosion_size = cv2.getTrackbarPos(title_trackbar_kernel_size, title_erosion_window)
    erosion_shape = cv2.getTrackbarPos(title_trackbar_kernel_size, title_erosion_window)

    element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))

    erosion_dst = cv2.erode(src, element)
    cv2.imshow(title_erosion_window, erosion_dst)


def dilatation(val):
    dilatation_size = cv2.getTrackbarPos(title_trackbar_kernel_size, title_dilation_window)
    dilation_shape = cv2.getTrackbarPos(title_trackbar_kernel_size, title_dilation_window)
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilatation_dst = cv2.dilate(src, element)
    cv2.imshow(title_dilation_window, dilatation_dst)


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


def prep(input_frame):
    # All shapes
    Lower = (0, 86, 6)
    Upper = (150, 255, 255)

    # Original shape
    # Lower = (29, 86, 6)
    # Upper = (64, 255, 255)

    (r, g, b) = (27, 69, 103)
    # 0 255 102
    # (r, g, b) = (255, 0, 0)
    # 0 255 255

    # normalize
    (r, g, b) = (r / 255, g / 255, b / 255)
    # convert to hsv
    (h, s, v) = colorsys.rgb_to_hsv(r, g, b)
    # expand HSV range
    (h, s, v) = (int(h * 179), int(s * 255), int(v * 255))
    print('HSV : ', h, s, v)

    output_frame = imutils.resize(input_frame, width=800)
    # output_frame = input_frame.copy()
    ## Todo 3.1.3 gaussian blur
    blurred = cv2.blur(output_frame, (11, 11))
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    output_mask = cv2.inRange(hsv, Lower, Upper)

    return output_mask, output_frame


def ero_dil():
    cv2.namedWindow(title_erosion_window)
    cv2.createTrackbar(title_trackbar_element_shape, title_erosion_window, 0, max_elem, erosion)
    cv2.createTrackbar(title_trackbar_kernel_size, title_erosion_window, 0, max_kernel_size, erosion)
    cv2.namedWindow(title_dilation_window)
    cv2.createTrackbar(title_trackbar_element_shape, title_dilation_window, 0, max_elem, dilatation)
    cv2.createTrackbar(title_trackbar_kernel_size, title_dilation_window, 0, max_kernel_size, dilatation)
    erosion(0)
    dilatation(0)


def contour(input_frame, input_mask):
    cnts = find_cnts(input_mask)

    # c = cnts[5]
    # c = max(cnts, key=cv2.contourArea)
    frame1 = input_frame.copy()
    frame2 = input_frame.copy()
    for c in cnts:
        ((x, y), radius) = cv2.minEnclosingCircle(c)  ## A different contour?
        # print(cnts)

        # Find center of contour using moments in opencvq
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # circle
        cv2.circle(frame1, (int(x), int(y)), int(radius), (0, 0, 255), 2)
        cv2.circle(frame1, center, 5, (0, 0, 255), -1)

        # draw rectangle with green line
        contours_poly = cv2.approxPolyDP(c, 3, True)
        boundRect = cv2.boundingRect(contours_poly)
        cv2.rectangle(frame1, (int(boundRect[0]), int(boundRect[1])),
                      (int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])), (0, 255, 0), 1)

        # draw rotate rectangle with blue line
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame1, [box], 0, (255, 0, 0), 1)

        # draw Hull with pink line
        hull = cv2.convexHull(c)
        cv2.drawContours(frame1, [hull], 0, (255, 0, 255), 1)

    cv2.imshow("Task 04 Circle & Rectangle", frame1)


# Task 02
frame = cv2.imread('image.png')
mask, frame = prep(frame)
# cv2.imshow("Image", frame)
cv2.imshow("Masked Image", mask)

# Task 03
# ero_dil()

# Task 04
contour(frame, mask)


cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
