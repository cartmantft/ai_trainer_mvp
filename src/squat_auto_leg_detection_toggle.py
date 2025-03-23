import cv2
import mediapipe as mp
import numpy as np

# 0 = 웹캠, 1 = 비디오 파일
VIDEO_MODE = 1
VIDEO_PATH = "sample_squat.mp4"  # VIDEO_MODE = 1일 때만 사용됨

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

DEBUG = True

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def get_leg_angle(landmarks, side='left'):
    if side == 'left':
        hip = [landmarks[23].x, landmarks[23].y]
        knee = [landmarks[25].x, landmarks[25].y]
        ankle = [landmarks[27].x, landmarks[27].y]
    else:
        hip = [landmarks[24].x, landmarks[24].y]
        knee = [landmarks[26].x, landmarks[26].y]
        ankle = [landmarks[28].x, landmarks[28].y]
    return calculate_angle(hip, knee, ankle)

def decide_visible_leg(landmarks):
    left_knee_x = landmarks[25].x
    right_knee_x = landmarks[26].x
    return 'left' if left_knee_x < right_knee_x else 'right'

def log_squat_state(frame_idx, leg_side, angle, direction, count):
    print(
        f"[Frame {frame_idx:04d}] "
        f"Leg: {leg_side:<5} | "
        f"Angle: {angle:6.2f}° | "
        f"Dir: {direction:<4} | "
        f"Total: {count:>2}"
    )

cap = cv2.VideoCapture(VIDEO_PATH if VIDEO_MODE == 1 else 0)

squat_count = 0
direction = "up"
frame_idx = 0
previous_angle = None

DOWN_THRESHOLD = 130
UP_THRESHOLD = 150

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            leg_side = decide_visible_leg(landmarks)
            angle = get_leg_angle(landmarks, side=leg_side)

            if previous_angle is not None:
                if angle < DOWN_THRESHOLD and angle < previous_angle and direction == "up":
                    direction = "down"
                elif angle > UP_THRESHOLD and angle > previous_angle and direction == "down":
                    direction = "up"
                    squat_count += 1

            if DEBUG and frame_idx % 5 == 0:
                log_squat_state(frame_idx, leg_side, angle, direction, squat_count)

            previous_angle = angle

            cv2.putText(image, f"{leg_side.title()} Leg Angle: {int(angle)}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f"Squats: {squat_count}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Squat Detection (Auto Side)', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()