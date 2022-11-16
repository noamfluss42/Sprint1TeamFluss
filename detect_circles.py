import sys
import cv2 as cv
import numpy as np


def detect_circles(img, masked=None):
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

    return result
    #         cv.circle(img, center, 1, (255, 0, 0), 3)
    # cv.imshow("after", img)
    # cv.waitKey(0)


def img_difference(img1, img2):
    diff = cv.subtract(img1, img2)
    #cv.imshow("diff", diff)
    cv.waitKey(0)


def get_images(i, m):
    img = cv.imread(f'images and filters/{m} meters new/opencv_frame_{i+1}.png')
    cropped = cv.imread(f'images and filters/{m} meters new/cropped_opencv_frame_{i+1}.png')
    filtered = cv.imread(f'images and filters/{m} meters new/masked_opencv_frame_{i+1}.png')
    return [img, cropped, filtered]


if __name__ == '__main__':
    for i in range(20):
        images = get_images(i, 5)
        print(detect_circles(images[1]))




