import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

target_lm_idx = [0, 2, 5, 9, 10, 11, 12]
lm_results = []
data_count = {
    'excellent': 0,
    'okay': 0,
    'bad': 0,
    'terrible': 0,
}

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
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(33) == ord('a'):
        # Excellent Posture
        lm_result.append(0)
        lm_results.append(np.array(lm_result))
        data_count['excellent'] += 1
        print("Excellent posture collected, total =", data_count['excellent'])
    if cv2.waitKey(33) == ord('b'):
        # Okay Posture
        lm_result.append(1)
        lm_results.append(np.array(lm_result))
        data_count['okay'] += 1
        print("Okay posture collected, total =", data_count['okay'])
    if cv2.waitKey(33) == ord('c'):
        # Bad Posture
        lm_result.append(2)
        lm_results.append(np.array(lm_result))
        data_count['bad'] += 1
        print("Bad posture collected, total =", data_count['bad'])
    if cv2.waitKey(33) == ord('d'):
        # Terrible Posture
        lm_result.append(3)
        lm_results.append(np.array(lm_result))
        data_count['terrible'] += 1
        print("Terrible posture collected, total =", data_count['terrible'])
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

lm_data = np.vstack(lm_results)
np.save('lm_data', lm_data)