import cv2
import mediapipe as mp
import numpy as np

# --- 설정값 ---
VIDEO_MODE = 1
VIDEO_PATH = "./squat_videos/mistake sqaut.mp4"
DEBUG = True

DOWN_THRESHOLD = 130
UP_THRESHOLD = 150
BUFFER_LIMIT = 3

# --- MediaPipe 초기화 ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- 기본 함수 ---
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
    return angle, hip, knee

def decide_visible_leg(landmarks):
    return 'left' if landmarks[25].x < landmarks[26].x else 'right'

# --- 피드백 조건 정의 ---
feedback_rules = [
    {
        "name": "깊이 부족",
        "priority": 1,
        "condition": lambda angle, direction, **kwargs: angle > 130 and direction == "down",
        "message": "조금 더 앉아보세요!"
    },
    {
        "name": "완전히 올라오지 않음",
        "priority": 1,
        "condition": lambda angle, direction, angle_buffer, **kwargs:
            direction == "up"
            and angle < 150
            and len(angle_buffer) >= 3
            and max(angle_buffer) - min(angle_buffer) < 2,
        "message": "완전히 올라오세요!"
    },
]

def evaluate_feedback(rules, context):
    active_rules = sorted(rules, key=lambda r: r["priority"])
    for rule in active_rules:
        if rule["condition"](**context):
            return rule["message"]
    return ""

# --- 실행 로직 ---
cap = cv2.VideoCapture(VIDEO_PATH if VIDEO_MODE == 1 else 0)
squat_count = 0
direction = None
frame_idx = 0
down_buffer = 0
up_buffer = 0
angle_buffer = []
feedback_message = ""

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
            angle, hip, knee = get_leg_metrics(landmarks, leg_side)

            # 🧭 방향 초기 설정
            if direction is None:
                direction = "down" if angle < DOWN_THRESHOLD else "up"
                if DEBUG:
                    print(f"🧭 초기 방향 자동 설정: {direction.upper()}")

            # 🔁 angle 추세 기록
            angle_buffer.append(angle)
            if len(angle_buffer) > 5:
                angle_buffer.pop(0)

            # 🧩 상태 변화 감지
            if angle < DOWN_THRESHOLD:
                down_buffer += 1
                up_buffer = 0
            elif angle > UP_THRESHOLD:
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

            # 💬 피드백 생성
            context = {
                "angle": angle,
                "direction": direction,
                "angle_buffer": angle_buffer
            }
            feedback_message = evaluate_feedback(feedback_rules, context)

            if DEBUG and frame_idx % 5 == 0:
                print(f"[Frame {frame_idx:04d}] Leg: {leg_side:<5} | Angle: {angle:6.2f}° | Dir: {direction:<4} | Total: {squat_count} | Msg: {feedback_message}")

            # 📺 화면 표시
            cv2.putText(image, f"Squats: {squat_count}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            if feedback_message:
                cv2.putText(image, feedback_message, (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Squat Detection + Feedback', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()