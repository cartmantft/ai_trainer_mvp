import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
from feedback_rule_base import FeedbackInput
from feedback_rules import *

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def decide_visible_leg(landmarks):
    return 'left' if landmarks[25].visibility >= landmarks[26].visibility else 'right'

def determine_direction(angle, direction, up_buffer, down_buffer):
    if direction == "down":
        if angle > 160:
            up_buffer += 1
            if up_buffer > 3:
                return "up", 0, down_buffer
        else:
            up_buffer = 0
    elif direction == "up":
        if angle < 130:
            down_buffer += 1
            if down_buffer > 3:
                return "down", up_buffer, 0
        else:
            down_buffer = 0
    return direction, up_buffer, down_buffer


def main():
    # cap = cv2.VideoCapture(0)  # 캠
    cap = cv2.VideoCapture("./squat_videos/4.엉덩이 뒤로 내밀기.mp4")  # 영상

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    f = open(log_path, "w", encoding="utf-8")

    direction = "up"
    up_buffer = 0
    down_buffer = 0
    squat_count = 0
    angle_buffer = []
    was_down_before = False

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        print("🧭 초기 방향 자동 설정: UP")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if not results.pose_landmarks:
                continue
            landmarks = results.pose_landmarks.landmark
            leg = decide_visible_leg(landmarks)

            hip = [landmarks[23].x, landmarks[23].y] if leg == 'left' else [landmarks[24].x, landmarks[24].y]
            knee = [landmarks[25].x, landmarks[25].y] if leg == 'left' else [landmarks[26].x, landmarks[26].y]
            ankle = [landmarks[27].x, landmarks[27].y] if leg == 'left' else [landmarks[28].x, landmarks[28].y]
            shoulder = [landmarks[11].x, landmarks[11].y] if leg == 'left' else [landmarks[12].x, landmarks[12].y]

            angle = calculate_angle(hip, knee, ankle)
            angle_buffer.append(angle)
            if len(angle_buffer) > 5:
                angle_buffer.pop(0)

            # 방향 전환 판단
            # if direction == "down" and angle > 160:
            #     direction = "up"
            #     squat_count += 1
            # elif direction == "up" and angle < 130:
            #     direction = "down"


            new_direction, up_buffer, down_buffer=determine_direction(angle, direction, up_buffer, down_buffer)
            
            # 방향이 실제로 전환되었을 때 처리
            if new_direction != direction:
                direction = new_direction
                if direction == "up":
                    squat_count += 1
                elif direction == "down":
                    was_down_before = True
            elif direction == "up" and angle > 170:
                was_down_before = False  # 충분히 올라오면 리셋

            input_data = FeedbackInput(
                angle=angle,
                direction=direction,
                angle_buffer=angle_buffer,
                knee_y=knee[1],
                hip_y=hip[1],
                knee_x=knee[0],
                ankle_x=ankle[0],
                shoulder_y=shoulder[1],
                shoulder_x=shoulder[0],
                hip_x=hip[0],
                l_ankle_x=landmarks[27].x,
                r_ankle_x=landmarks[28].x,
                was_down_before= was_down_before
            )

            feedback_rules = [
                KneeTooForward(),
                TorsoBentForward(),
                IncompleteUp(),
                NotDeepEnough(),
                FeetNotAligned(),
                HipTooFarBack()
            ]

            feedback_msg = ""
            for rule in sorted(feedback_rules, key=lambda r: r.priority):
                if rule.check(input_data):
                    feedback_msg = rule.message
                    break

            # log = f"[{datetime.now().strftime('%H:%M:%S')}] [Frame {frame_count:04}] Leg: {leg:<5} | Angle: {angle:.2f}° | Dir: {direction:<4} | Total: {squat_count} | Msg: {feedback_msg} | KneeX: {knee[0]:.2f} | AnkleX: {ankle[0]:.2f}"
            log = (
                f"[{datetime.now().strftime('%H:%M:%S')}] [Frame {frame_count:04}] "
                f"Leg: {leg:<5} | Angle: {angle:.2f}° | Dir: {direction:<4} | Total: {squat_count} | Msg: {feedback_msg} | "
                f"KneeX: {knee[0]:.2f} | AnkleX: {ankle[0]:.2f} | KneeY: {knee[1]:.2f} | HipY: {hip[1]:.2f} | "
                f"ShoulderY: {shoulder[1]:.2f} | ShoulderX: {shoulder[0]:.2f} | HipX: {hip[0]:.2f} | "
                f"L_AnkleX: {landmarks[27].x:.2f} | R_AnkleX: {landmarks[28].x:.2f} | "
                f"ΔAngle: {(max(angle_buffer)-min(angle_buffer)):.2f}"
            )
            
            
            print(log)
            f.write(log + "\n")

            cv2.imshow('Squat Tracker', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    f.close()

if __name__ == "__main__":
    main()
