from cv2 import cv2
import numpy as np


def rescale(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


cv = cv2
cap = cv2.VideoCapture('m3.mp4')

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Read until video is completed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = rescale(frame, scale_percent=45)
    orig_frame = frame

    height, width, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([60, 0, 0])
    upper_blue = np.array([180, 255, 255])
    mask_with_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    frame = cv2.bitwise_and(frame, frame, mask=mask_with_blue)

    thresh = 50
    ret, thresh_img = cv2.threshold(frame[:, :, 0], thresh, 255, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    binary = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2RGB)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(binary, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # if contours:
    #     x, y, w, h = cv2.boundingRect(contours[0])
    #     # if w > 10 or h > 10:
    #     cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 200, 0), 3)
    #     cv2.rectangle(binary, (x, y), (x + w, y + h), (0, 200, 0), 3)
    result = cv2.hconcat([orig_frame, binary])
    if ret:
        cv2.imshow('Frame', result)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
