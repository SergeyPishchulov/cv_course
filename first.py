from cv2 import cv2
import numpy as np


def rescale(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def get_mask_with_blue(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([60, 0, 0])
    upper_blue = np.array([180, 255, 255])
    return cv2.inRange(hsv, lower_blue, upper_blue)


def draw_contours(thresh_img, orig_frame):
    thresh_img = thresh_img.copy()
    orig_frame = orig_frame.copy()
    cnts, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 20 and h > 20:
            cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(binary, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return orig_frame, binary


cv = cv2
cap = cv2.VideoCapture('m4.mp4')

if not cap.isOpened():
    print("Error opening video stream or file")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = rescale(frame, scale_percent=45)
    orig_frame = frame

    height, width, _ = frame.shape
    mask_with_blue = get_mask_with_blue(frame)
    frame = cv2.bitwise_and(frame, frame, mask=mask_with_blue)

    thresh = 50
    ret, thresh_img = cv2.threshold(frame[:, :, 0], thresh, 255, cv2.THRESH_BINARY)
    binary = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2RGB)

    orig_frame_w_cntrs, binary_w_cntrs = draw_contours(thresh_img, orig_frame)

    result = cv2.hconcat([orig_frame_w_cntrs, binary_w_cntrs])
    if ret:
        cv2.imshow('Frame', result)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
