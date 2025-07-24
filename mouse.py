import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Setup
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip horizontally
    h, w, _ = frame.shape

    # RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Index finger tip
            x_index = int(hand_landmarks.landmark[8].x * w)
            y_index = int(hand_landmarks.landmark[8].y * h)

            # Convert to screen coordinates
            screen_x = np.interp(x_index, [0, w], [0, screen_w])
            screen_y = np.interp(y_index, [0, h], [0, screen_h])

            # Move mouse
            pyautogui.moveTo(screen_x, screen_y)

            # Detect clicking (e.g., pinch)
            x_thumb = int(hand_landmarks.landmark[4].x * w)
            y_thumb = int(hand_landmarks.landmark[4].y * h)

            distance = np.hypot(x_index - x_thumb, y_index - y_thumb)

            if distance < 30:
                pyautogui.click()
                pyautogui.sleep(1)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
