import cv2


def img_from_cam(cam):
    ret, cam_frame = cam.read()
    if not ret:
        print("failed to grab frame")
        return None
    return cam_frame


def img_from_file():
    return cv2.imread("5 meters/opencv_frame_4.png")
