import cv2


def img_from_cam(cam):
    ret, cam_frame = cam.read()
    if not ret:
        print("failed to grab frame")
        return None
    return cam_frame


def img_from_file():
    return cv2.imread("images and filters/5 meters new/opencv_frame_5.png")
