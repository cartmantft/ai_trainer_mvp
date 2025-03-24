import cv2
import mediapipe as mp
import numpy as np

# 설정
VIDEO_MODE = 1
VIDEO_PATH = "./squat_videos/10_Overhead Squat Side View.mp4"
DEBUG = True

DOWN_THRESHOLD = 130
UP_THRESHOLD = 150
BUFFER_LIMIT = 3

# 초기화
cap = cv2.VideoCapture(VIDEO_PATH if VIDEO_MODE == 1 else 0)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def get_leg_metrics(landmarks, side='left'):
    if side == 'left':
        hip = landmarks[23]
        knee = landmarks[25]
        ankle = landmarks[27]
    else:
        hip = landmarks[24]
        knee = landmarks[26]
        ankle = landmarks[28]
    angle = calculate_angle([hip.x, hip.y], [knee.x, knee.y], [ankle.x, ankle.y])
    y_diff = knee.y - hip.y
    return angle, y_diff

def decide_visible_leg(landmarks):
    left_knee_x = landmarks[25].x
    right_knee_x = landmarks[26].x
    return 'left' if left_knee_x < right_knee_x else 'right'

def log_squat_state(frame_idx, leg_side, angle, y_diff, direction, count):
    print(
        f"[Frame {frame_idx:04d}] "
        f"Leg: {leg_side:<5} | "
        f"Angle: {angle:6.2f}° | "
        f"Ydiff: {y_diff:.3f} | "
        f"Dir: {direction:<4} | "
        f"Total: {count:>2}"
    )

# 조건 함수 (angle 기준만 사용)
def is_down_condition(angle):
    return angle < DOWN_THRESHOLD

def is_up_condition(angle):
    return angle > UP_THRESHOLD

# 상태 초기화
squat_count = 0
direction = None
frame_idx = 0
down_buffer = 0
up_buffer = 0

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
            angle, y_diff = get_leg_metrics(landmarks, leg_side)

            if direction is None:
                direction = "down" if is_down_condition(angle) else "up"
                if DEBUG:
                    print(f"🧭 초기 방향 자동 설정: {direction.upper()}")

            if is_down_condition(angle):
                down_buffer += 1
                up_buffer = 0
            elif is_up_condition(angle):
                up_buffer += 1
                down_buffer = 0
            else:
                down_buffer = 0
                up_buffer = 0

            if down_buffer >= BUFFER_LIMIT and direction == "up":
                direction = "down"
                down_buffer = 0
            elif up_buffer >= BUFFER_LIMIT and direction == "down":
                direction = "up"
                squat_count += 1
                up_buffer = 0

            if DEBUG and frame_idx % 5 == 0:
                log_squat_state(frame_idx, leg_side, angle, y_diff, direction, squat_count)

            cv2.putText(image, f"{leg_side.title()} Leg Angle: {int(angle)}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f"Squats: {squat_count}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Squat Detection (Angle Only)', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()