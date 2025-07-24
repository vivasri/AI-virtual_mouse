import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
import time

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Screen resolution
screen_w, screen_h = pyautogui.size()

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Smooth cursor movement
coords_buffer = deque(maxlen=5)

# Gesture state flags
left_click_active = False
right_click_active = False


# Function to count fingers
def count_fingers(hand_landmarks):
    fingers = []
    tip_ids = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    for tip in tip_ids:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return sum(fingers)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Finger coordinates
            x_index = int(hand_landmarks.landmark[8].x * w)
            y_index = int(hand_landmarks.landmark[8].y * h)
            x_thumb = int(hand_landmarks.landmark[4].x * w)
            y_thumb = int(hand_landmarks.landmark[4].y * h)
            x_middle = int(hand_landmarks.landmark[12].x * w)
            y_middle = int(hand_landmarks.landmark[12].y * h)

            # Convert to screen coordinates
            screen_x = np.interp(x_index, [0, w], [0, screen_w])
            screen_y = np.interp(y_index, [0, h], [0, screen_h])
            coords_buffer.append((screen_x, screen_y))

            # Smooth movement
            avg_x = sum(c[0] for c in coords_buffer) / len(coords_buffer)
            avg_y = sum(c[1] for c in coords_buffer) / len(coords_buffer)
            pyautogui.moveTo(avg_x, avg_y)

            # Gesture distances
            distance_pinch = np.hypot(x_index - x_thumb, y_index - y_thumb)
            distance_right_click = np.hypot(x_index - x_middle, y_index - y_middle)

            # LEFT CLICK or DRAG
            if distance_pinch < 30:
                if not left_click_active:
                    pyautogui.mouseDown()
                    left_click_active = True
            else:
                if left_click_active:
                    pyautogui.mouseUp()
                    left_click_active = False

            # RIGHT CLICK
            if distance_right_click < 30:
                if not right_click_active:
                    pyautogui.click(button='right')
                    right_click_active = True
            else:
                right_click_active = False

           
    # Show webcam feed
    cv2.imshow("Virtual Mouse", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

