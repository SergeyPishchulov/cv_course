import cv2
import numpy as np

cap = cv2.VideoCapture('m2.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while (cap.isOpened()):
    ret, frame = cap.read()
    height, width, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([60, 0, 0])
    upper_blue = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(frame, mask, (0, 0), 1)
    # canvas = frame[1:height + 1, 1:width + 1].astype(np.bool)
    # frame = ~frame
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
