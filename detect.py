import cv2
import mediapipe as mp
import numpy as np
import math

KNOWN_DISTANCE = 30.0  
KNOWN_WIDTH = 6.0  

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

desired_width = 1280
desired_height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

def estimate_focal_length(known_distance, known_width, pixel_width):
    return (pixel_width * known_distance) / known_width

def pixel_to_cm(pixel_distance, focal_length):
    return (KNOWN_WIDTH * focal_length) / pixel_distance

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

    focal_length_estimated = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        hand_landmarks = results.multi_hand_landmarks
        handedness = results.multi_handedness

        if hand_landmarks and handedness:
            hands_data = []

            for idx, hand_landmark in enumerate(hand_landmarks):
                mp_drawing.draw_landmarks(
                    image, hand_landmark, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

                hand_label = handedness[idx].classification[0].label
                hand_index_finger_tip = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                hand_data = {
                    'label': hand_label,
                    'landmark': hand_landmark,
                    'index_tip': (hand_index_finger_tip.x, hand_index_finger_tip.y)
                }
                hands_data.append(hand_data)

                h, w, _ = image.shape
                cx, cy = int(hand_index_finger_tip.x * w), int(hand_index_finger_tip.y * h)
                cv2.putText(image, hand_label, (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if len(hands_data) == 2:
                x1, y1 = hands_data[0]['index_tip']
                x2, y2 = hands_data[1]['index_tip']

                x1, y1 = int(x1 * w), int(y1 * h)
                x2, y2 = int(x2 * w), int(y2 * h)

                pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if not focal_length_estimated:
                    focal_length_estimated = estimate_focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, pixel_distance)

                distance_cm = pixel_to_cm(pixel_distance, focal_length_estimated)

                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
