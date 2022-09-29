import cv2
import numpy as np
cap = cv2.VideoCapture('m2.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  # Threshold of blue in HSV space
  lower_blue = np.array([60, 0, 0])
  upper_blue = np.array([180, 255, 255])

  # preparing the mask to overlay
  mask = cv2.inRange(hsv, lower_blue, upper_blue)

  # The black region in the mask has the value of 0,
  # so when multiplied with original image removes all non-blue regions
  result = cv2.bitwise_and(frame, frame, mask=mask)
  # blue_chanel = frame[:, :, 0]
  # blue_img = np.zeros(frame.shape)
  #
  # # assign the red channel of src to empty image
  # blue_img[:, :, 0] = blue_chanel
  # frame=blue_chanel
  if ret == True:

    # Display the resulting frame
    cv2.imshow('Frame',result)

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