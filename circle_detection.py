import cv2 as cv
import numpy as np


def detect_circle(img, masked=None):
    # cv.imshow("before", img)
    # if masked is not None:
    #     cv.imshow("masked", masked)
    # else:
    #     masked = img
    if masked is None:
        masked = img
    # masked = cv.medianBlur(masked, 5)
    masked = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)
    rows = masked.shape[0]

    circles = cv.HoughCircles(masked, cv.HOUGH_GRADIENT, 1, rows / 8,
                             param1=100, param2=5,
                             minRadius=1, maxRadius=4)
    if circles is None:
        return [(0, 0)]

    result = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # print(f"circle[{i}] ({i[0]}, {i[1]}), r = {i[2]}")
            center = (i[0], i[1])
            result.append(center)

    # return [(0,0)]
    # img = cv.circle(img, result[0], 1, (255, 0, 0), 3)
    # cv.imshow(f"after{len(img[0])}", img)
    return result
    #         cv.circle(img, center, 1, (255, 0, 0), 3)

    # cv.waitKey(0)
