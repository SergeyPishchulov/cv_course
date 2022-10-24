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


def add_point(image, hand_landmarks):
    lm = hand_landmarks.landmark
    thumb = lm[4]
    index = lm[8]
    x = int(index.x * image.shape[1])
    y = int(index.y * image.shape[0])
    # Annotate landmarks or do whatever you want.
    drawn_circles.add((x, y))


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
                # print(hand_landmarks)
            fingers_connected = fingers_are_connected(results.multi_hand_world_landmarks[0], image)
            if fingers_connected:
                add_point(image, results.multi_hand_landmarks[0])
        for x, y in drawn_circles:
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            # print(mp_hands.HAND_CONNECTIONS)
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
