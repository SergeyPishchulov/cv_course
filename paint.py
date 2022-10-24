import math

from cv2 import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def get_distance(t1, t2):
    return math.hypot(t1[0] - t2[0], t1[1] - t2[1])
    # return abs(t1[0] - t2[0]) + abs()


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


def get_index_loc(image, hand_landmarks):
    lm = hand_landmarks.landmark
    thumb = lm[4]
    index = lm[8]
    x = int(index.x * image.shape[1])
    y = int(index.y * image.shape[0])
    return x, y


def is_in_rect(rw, rh, x, y):
    return 0 <= x <= rw and 0 <= y <= rh


def show_palette(image):
    return overlay_transparent(image, palette, int((W - H) / 2), 0, (H, H))


def get_color_bullet_points(image):
    r = 160
    colors_cnt = 10
    angles = [2 * math.pi * angle_num / colors_cnt for angle_num in range(colors_cnt)]
    return [(int(W / 2) + int(r * math.cos(a)), int(H / 2) + int(r * math.sin(a))) for a in angles]


def pick_color(image, ilx, ily):
    closest = None
    picked_color = None
    min_d = 10_000
    bullet_points = get_color_bullet_points(image)
    for (x, y) in bullet_points:
        cv2.circle(image, (x, y), 20, (223, 190, 24), thickness=4)
        if ilx and ily:  # в кадре есть указательный палец
            d = get_distance((x, y), (ilx, ily))
            if d < min_d and get_distance((ilx, ily), (W / 2, H / 2)) < 250:
                min_d = d
                closest = (x, y)
                picked_color = tuple([int (a) for a in image[y, x]])
                # raise Exception(picked_color)

    if closest:
        cv2.circle(image, closest, 20, (223, 190, 24), thickness=-1)
    return image, picked_color


# Annotate landmarks or do whatever you want.

W = 640
H = 480
CURRENT_COLOR = (0, 0, 0)
palette_logo = cv2.imread('palete.png', -1)
palette = cv2.imread('full_palette.png', -1)
palette_logo = cv2.resize(palette_logo, (64, 64), interpolation=cv2.INTER_AREA)
palette = cv2.resize(palette, (H, H), interpolation=cv2.INTER_AREA)
palette_logo_h, palette_logo_w, _ = palette_logo.shape
drawn_circles = set()
# For webcam input:
fingers_connected = False
SHOW_PALETTE = True  # TODO
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
        image = overlay_transparent(image, palette_logo, 0, 0, (palette_logo_w, palette_logo_h))
        ilx, ily = None, None
        fingers_connected = False
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:  # hand_landmarks - метки одной руки
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    # mp_hands.HAND_CONNECTIONS,
                    list({(3, 4), (7, 8)}),
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            ilx, ily = get_index_loc(image, hand_landmarks=results.multi_hand_landmarks[0])
            fingers_connected = fingers_are_connected(results.multi_hand_world_landmarks[0], image)

        if not SHOW_PALETTE and fingers_connected:
            if is_in_rect(palette_logo_w, palette_logo_h, ilx, ily):
                SHOW_PALETTE = True
            else:
                drawn_circles.add((ilx, ily, CURRENT_COLOR))
        if SHOW_PALETTE:
            image = show_palette(image)
            image, picked_color = pick_color(image, ilx, ily)
            if picked_color is not None and fingers_connected:
                SHOW_PALETTE = False
                CURRENT_COLOR = picked_color

        if not SHOW_PALETTE:
            for x, y, color in drawn_circles:
                cv2.circle(image, (x, y), 5, color, -1)

        # image[0:palete_logo_h, 0:palete_logo_w] = palete_logo
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
