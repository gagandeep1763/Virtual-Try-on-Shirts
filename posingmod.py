import cv2
import time
import posemodule as pm

def rescaleFrame(frame, scale=0.25):
    width = max(1, int(frame.shape[1] * scale))
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

cap = cv2.VideoCapture(r'C:\Users\DELL\PycharmProjects\PythonProject5\Pose estimation\videos\video 5.mp4')
pTime = 0
detector = pm.PoseDetector()

while True:
    success, img = cap.read()
    if img is None:
        print("Frame not read correctly, exiting...")
        break

    img = detector.findPose(img, draw=True)
    lmlist = detector.findPosition(img)

    if lmlist and (len(lmlist) !=0):
        print(lmlist[14])
        cv2.circle(img, (lmlist[14][1], lmlist[14][2]), 25, (0, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)

    frame_resized = rescaleFrame(img)

    cv2.imshow('Pose Estimation', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
