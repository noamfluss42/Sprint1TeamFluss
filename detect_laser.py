import cv2
import numpy as np
import math

import image_prep
import image_producer
import distance_calculator

cam = cv2.VideoCapture(1)

cv2.namedWindow("test")

img_counter = 0

# define range of red color in HSV
LOWER_RED = np.array([0, 0, 50])
UPPER_RED = np.array([30, 5, 255])


def mask_color(img, lower_bound, upper_bound):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masked_image = cv2.inRange(hsv, lower_bound, upper_bound)
    return masked_image


while True:
    frame = image_producer.img_from_cam(cam)
    if frame is None:
        break

    cv2.imshow("test", frame)
    cropped_frame = image_prep.crop_img(frame, (50, 150), (100, 330))
    cv2.imshow("cropped", cropped_frame)

    # Threshold the HSV image to get only correct colors
    mask = mask_color(cropped_frame, LOWER_RED, UPPER_RED)
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)

    cv2.imshow("mask", mask)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    # print(cnts)
    center = None
    isBall = False

    # only proceed if at least one contour was found
    # if len(cnts) > 0:
    #     # find the largest contour in the mask, then use
    #     # it to compute the minimum enclosing circle and
    #     # centroid
    #     radius = 0
    #     cnts.sort()#key=cv2.contourArea)
    #     i = len(cnts) - 1
    #     ball_masked = np.zeros([10, 10])
    #     while (not isBall) and i >= 0:
    #         ((x, y), radius) = cv2.minEnclosingCircle(cnts[i])
    #         M = cv2.moments(cnts[i])
    #         center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    #
    #         ball_mask = np.zeros(mask.shape[:2], dtype="uint8")
    #         cv2.circle(ball_mask, (int(x), int(y)), int(radius), 255, -1)
    #         ball_masked = cv2.bitwise_and(mask, ball_mask)
    #
    #         isBall = np.sum(ball_masked == 255) > int(radius) ** 2 * math.pi * 0.8
    #
    #         i -= 1
    #
    #     cv2.imshow("laser", ball_masked)
    # only proceed if the radius meets a minimum size
    # if radius > 10 and isBall:
    #     # draw the circle and centroid on the frame,
    #     # then update the list of tracked points
    #     cv2.putText(frame, str(int(radius)), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
    #     cv2.circle(frame, (int(x), int(y)), int(radius),
    #                (0, 255, 255), 2)
    #     cv2.circle(frame, center, 5, (0, 0, 255), -1)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        cv2.imwrite(f"cropped_{img_name}", cropped_frame)
        cv2.imwrite(f"masked_{img_name}", mask)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
