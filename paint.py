import math

from cv2 import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def get_distance(t1, t2):
    return abs(t1[0] - t2[0]) + abs(t1[1] - t2[1])


def fingers_are_connected(hand_landmarks, image) -> bool:
    lm = hand_landmarks.landmark
    thumb = lm[4]
    index = lm[8]
    thumb_x = int(thumb.x * image.shape[1])
    thumb_y = int(thumb.y * image.shape[0])
    index_x = int(index.x * image.shape[1])
    index_y = int(index.y * image.shape[0])
    distance = get_distance((thumb_x, thumb_y), (index_x, index_y))
    return distance < 10

    # cv2.circle(image, (thumb_x, thumb_y), 5, (0, 255, 0), -1)


def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    """
    @brief      Overlays a transparant PNG onto another image using CV2

    @param      background_img    The background image
    @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
    @param      x                 x location to place the top-left corner of our overlay
    @param      y                 y location to place the top-left corner of our overlay
    @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

    @return     Background image with overlay on top
    """

    bg_img = background_img.copy()

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    # Extract the alpha mask of the RGBA image, convert to RGB
    b, g, r, a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))

    # Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a, 5)

    h, w, _ = overlay_color.shape
    roi = bg_img[y:y + h, x:x + w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))

    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Update the original image with our new ROI
    bg_img[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)

    return bg_img


def add_point(image, hand_landmarks):
    lm = hand_landmarks.landmark
    thumb = lm[4]
    index = lm[8]
    x = int(index.x * image.shape[1])
    y = int(index.y * image.shape[0])
    # Annotate landmarks or do whatever you want.
    drawn_circles.add((x, y))


palete_logo = cv2.imread('palete.png', -1)
palete_logo = cv2.resize(palete_logo, (64, 64), interpolation=cv2.INTER_AREA)
palete_logo_h, palete_logo_w, _ = palete_logo.shape
drawn_circles = set()
# For webcam input:
fingers_connected = False
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1,
        static_image_mode=False) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:  # hand_landmarks - метки одной руки
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    # mp_hands.HAND_CONNECTIONS,
                    list({(3, 4)}),
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            fingers_connected = fingers_are_connected(results.multi_hand_world_landmarks[0], image)
            if fingers_connected:
                add_point(image, results.multi_hand_landmarks[0])
        for x, y in drawn_circles:
            cv2.circle(image, (x, y), 5, (0, 0, 0), -1)

            # print(mp_hands.HAND_CONNECTIONS)
        # Flip the image horizontally for a selfie-view display.
        image = overlay_transparent(image, palete_logo, 0, 0, (palete_logo_w, palete_logo_h))
        # image[0:palete_logo_h, 0:palete_logo_w] = palete_logo
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
