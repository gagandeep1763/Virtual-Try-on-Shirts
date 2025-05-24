import cv2
import os
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

draw_spec = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4)

cap = cv2.VideoCapture(0)
pTime = 0

def rescaleFrame(frame, scale=0.25):
    width = max(1, int(frame.shape[1] * scale))
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

while cap.isOpened():
    success, img = cap.read()
    if not success or img is None:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS, draw_spec, draw_spec)

        for id,ln in enumerate(results.pose_landmarks.landmark):
            h,w,c=img.shape
            print(id,ln)
            cx,cy=int(ln.x*w),int(ln.y*h)
            cv2.circle(img,(cx,cy),20,(255,0,0),cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    frame_resized = rescaleFrame(img)

    cv2.imshow('Pose Estimation', frame_resized)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
