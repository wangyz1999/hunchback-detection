import numpy as np
import cv2
import mediapipe as mp
from sklearn.linear_model import LogisticRegressionCV

pos_data = np.load('posture_data.npy')
X = pos_data[:, :-1]
y = pos_data[:, -1]
clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)


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


    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    lm_result = []
    for lm_idx in target_lm_idx:
        lm_result.append(results.pose_landmarks.landmark[lm_idx].x)
        lm_result.append(results.pose_landmarks.landmark[lm_idx].y)
        lm_result.append(results.pose_landmarks.landmark[lm_idx].z)
        lm_result.append(results.pose_landmarks.landmark[lm_idx].visibility)
    lm_result = np.array(lm_result)[None, :]
    pred = clf.predict(lm_result)[0]
    image = cv2.flip(image, 1)
    if pred == 0:
        image = cv2.putText(image, 'Excellent Posture', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 255, 153), 2, cv2.LINE_AA)
    if pred == 1:
        image = cv2.putText(image, 'Okay Posture', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 255, 255), 2, cv2.LINE_AA)
    if pred == 2:
        image = cv2.putText(image, 'Bad Posture', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 153, 255), 2, cv2.LINE_AA)
    if pred == 3:
        image = cv2.putText(image, 'Terrible Posture', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 51, 255), 2, cv2.LINE_AA)

    cv2.imshow('Sitting Posture', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()