import cv2
import mediapipe as mp
import pickle
import numpy as np
from win10toast import ToastNotifier

model_file = 'model.sav'

clf = pickle.load(open(model_file, 'rb'))

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
target_lm_idx = [0, 2, 5, 9, 10, 11, 12]

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
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
    results = pose.process(image)
    
    if results.pose_landmarks is not None:
        # Flip the image horizontally for a selfie-view display.
        lm_result = []
        for lm_idx in target_lm_idx:
            lm_result.append(results.pose_landmarks.landmark[lm_idx].x)
            lm_result.append(results.pose_landmarks.landmark[lm_idx].y)
            lm_result.append(results.pose_landmarks.landmark[lm_idx].z)
            lm_result.append(results.pose_landmarks.landmark[lm_idx].visibility)
        lm_result = np.array(lm_result)[None, :]
        pred = clf.predict(lm_result)[0]

        if pred == 2 or pred == 3:
            # detecting bad or terrible posture
            toast = ToastNotifier()
            toast.show_toast(
                "Posture Remainder",
                "Bad Posture Detected",
                duration=20,
                threaded=False,
            )

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()