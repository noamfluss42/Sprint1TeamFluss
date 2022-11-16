import cv2
import numpy as np
import math

import image_prep
import image_producer
import distance_calculator
import circle_detection







# define range of red color in HSV
LOWER_RED = np.array([0, 0, 253])
UPPER_RED = np.array([1, 1, 255])

CROP1_P1 = (250, 75)
CROP1_P2 = (370, 140)

CROP2_P1 = (400, 90)
CROP2_P2 = (500, 150)

def mask_color(img, lower_bound, upper_bound):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masked_image = cv2.inRange(hsv, lower_bound, upper_bound)
    return masked_image


def detect_single_laser(img, coord0, coord1): # (x1, y1), (x2, y2)
    # print("coord0, coord1",coord0, coord1)
    cropped_frame = image_prep.crop_img(img, coord0, coord1)
    # cv2.imshow(f"cropped{coord0}", cropped_frame)
    mask = mask_color(cropped_frame, LOWER_RED, UPPER_RED)
    cv2.imshow(f"masked{coord0}", mask)
    laser_cords = circle_detection.detect_circle(cropped_frame, mask)[0]
    # print(laser_cords)
    # print(coord0)

    # return coord1[0], coord1[1]
    return laser_cords[0]+coord0[0], laser_cords[1]+coord0[1]


def main():
    cam = cv2.VideoCapture(1)
    img_counter = 0
    dist = 450
    out = []
    while True:
        frame = image_producer.img_from_cam(cam)
        if frame is None:
            break

        cv2.imshow("test", frame)
        # cv2.imshow("test", frame)
        # laser0 = detect_single_laser(frame, CROP1_P1, CROP1_P2)
        laser1 = detect_single_laser(frame, CROP2_P1, CROP2_P2)

        frame = cv2.rectangle(frame, CROP1_P1, CROP1_P2, (255, 0, 0), 1)
        frame = cv2.rectangle(frame, CROP2_P1, CROP2_P2, (0, 255, 0), 1)
        # frame = cv2.circle(frame, laser0, 1, (0, 0, 255), 3)
        frame = cv2.circle(frame, laser1, 1, (0, 255, 255), 3)
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
            out.append([dist, laser1])
            dist += 50
            print(out)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
