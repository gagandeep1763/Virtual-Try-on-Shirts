import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
shirtFolderPath = r"C:\Users\DELL\PycharmProjects\PythonProject5\Virtual try on shirt\resources\shirts"
listShirts = os.listdir(shirtFolderPath)
fixedRatio = 1.6
imageNumber = 0
imgButton = cv2.imread(r"C:\Users\DELL\PycharmProjects\PythonProject5\Virtual try on shirt\button.png", cv2.IMREAD_UNCHANGED)
buttonPressed = False
buttonCooldown = False

def overlay_image_alpha(bg, overlay, x, y):
    h, w = overlay.shape[:2]
    x, y = max(0, x), max(0, y)
    h, w = min(h, bg.shape[0] - y), min(w, bg.shape[1] - x)
    if h <= 0 or w <= 0:
        return bg
    overlay_image = overlay[:h, :w, :3]
    mask = overlay[:h, :w, 3:] / 255.0
    bg[y:y + h, x:x + w] = (1.0 - mask) * bg[y:y + h, x:x + w] + mask * overlay_image
    return bg

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image_rgb)
    hands_results = hands.process(image_rgb)

    if pose_results.pose_landmarks:
        lm11 = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        lm12 = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lm23 = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

        ih, iw, _ = image.shape
        lm11_px, lm12_px, lm23_px = (int(lm11.x * iw), int(lm11.y * ih)), (int(lm12.x * iw), int(lm12.y * ih)), (int(lm23.x * iw), int(lm23.y * ih))
        shirt_width = int(abs(lm11_px[0] - lm12_px[0]) * fixedRatio)
        shirt_height = int(abs(lm11_px[1] - lm23_px[1]) * 1.3)
        shirt_top_left = (lm12_px[0] - int(shirt_width * 0.2), lm11_px[1] - int(shirt_height * 0.15))
        imgShirtPath = os.path.join(shirtFolderPath, listShirts[imageNumber])
        imgShirt = cv2.imread(imgShirtPath, cv2.IMREAD_UNCHANGED)
        if imgShirt is not None:
            imgShirt = cv2.resize(imgShirt, (shirt_width, shirt_height))
            image = overlay_image_alpha(image, imgShirt, shirt_top_left[0], shirt_top_left[1])
        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            x, y = int(hand_landmarks.landmark[8].x * image.shape[1]), int(hand_landmarks.landmark[8].y * image.shape[0])
            button_x1 = image.shape[1] - imgButton.shape[1] - 20
            button_y1 = image.shape[0] - imgButton.shape[0] - 20
            button_x2 = image.shape[1] - 20
            button_y2 = image.shape[0] - 20

            if button_x1 < x < button_x2 and button_y1 < y < button_y2:
                if not buttonCooldown:
                    imageNumber = (imageNumber + 1) % len(listShirts)
                    buttonCooldown = True
            else:
                buttonCooldown = False

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    image = overlay_image_alpha(image, imgButton, image.shape[1] - imgButton.shape[1] - 20, image.shape[0] - imgButton.shape[0] - 20)
    cv2.imshow('Virtual Try-On', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
