from cv2 import cv2
import numpy as np

cv = cv2
cap = cv2.VideoCapture('m2.mp4')

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Read until video is completed
while cap.isOpened():
    ret, frame = cap.read()
    height, width, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([60, 0, 0])
    upper_blue = np.array([180, 255, 255])
    mask_with_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    frame = cv2.bitwise_and(frame, frame, mask=mask_with_blue)

    thresh = 20
    # get threshold image
    ret, thresh_img = cv2.threshold(frame[:, :, 0], thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros(frame.shape)
    # draw the contours on the empty image
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), thickness=3)
    frame = img_contours

    # mask = np.zeros((height + 2, width + 2), np.uint8)
    # # cv2.rectangle(frame, (x, y), (x + w, y + h), color=(5, 65, 65), thickness=10)
    # imgray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # ret, thresh = cv.threshold(imgray, 127, 255, 0)
    # contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # cv.drawContours(frame, contours, -1, (0, 255, 0), 5)
    # print(len(contours))
    if ret:

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
