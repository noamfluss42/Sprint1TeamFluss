import cv2
import numpy as np
import math

import image_prep
import image_producer
import distance_calculator
import circle_detection







# define range of red color in HSV
LOWER_RED = np.array([0, 0, 50])
UPPER_RED = np.array([30, 5, 255])


def mask_color(img, lower_bound, upper_bound):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masked_image = cv2.inRange(hsv, lower_bound, upper_bound)
    return masked_image


def detect_single_laser(img, coord0, coord1):
    cropped_frame = image_prep.crop_img(img, coord0, coord1)
    # mask = mask_color(cropped_frame, LOWER_RED, UPPER_RED)
    laser_cords = circle_detection.detect_circle(cropped_frame)[0]
    # print(laser_cords)
    # print(coord0)
    # return coord0[0], coord0[1]
    return laser_cords[0]+coord0[1], laser_cords[1]+coord0[0]


def main():
    cam = cv2.VideoCapture(0)
    img_counter = 0
    while True:
        frame = image_producer.img_from_cam(cam)
        if frame is None:
            break

        cv2.imshow("test", frame)
        laser0 = detect_single_laser(frame, (50, 150), (100, 330))
        laser1 = detect_single_laser(frame, (100, 200), (200, 430))

        frame = cv2.circle(frame, laser0, 1, (255, 0, 0), 3)
        frame = cv2.circle(frame, laser1, 1, (255, 0, 0), 3)
        cv2.imshow("detected", frame)

        # print(distance_calculator.distance_in_meters(laser0, laser1))

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            # cv2.imwrite(f"cropped_{img_name}", cropped_frame)
            # cv2.imwrite(f"masked_{img_name}", mask)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
