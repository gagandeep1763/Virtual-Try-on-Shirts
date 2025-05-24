import cv2
import mediapipe as mp
import time

class PoseDetector():
    def __init__(self, mode=False, model_complexity=1, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        # Increased thickness and circle radius
        self.draw_spec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=6)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS, self.draw_spec, self.draw_spec)
        return img

    def findPosition(self, img, draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            for id, ln in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(ln.x * w), int(ln.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 0), cv2.FILLED)  # Larger circles
        return lmlist


def rescaleFrame(frame, scale=0.25):  # Increased default scale
    width = max(1, int(frame.shape[1] * scale))
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        if not success or img is None:  # Escape if no frame is received
            print("No frame received, exiting...")
            break

        img = detector.findPose(img, draw=True)
        lmlist = detector.findPosition(img)
        print(lmlist[14])
        cv2.circle(img, (lmlist[14][1],lmlist[14][2]),25, (0, 0, 255), cv2.FILLED)

        if lmlist and len(lmlist) > 0:
            print(lmlist[0])  # Print only first landmark

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


if __name__ == '__main__':
    main()
