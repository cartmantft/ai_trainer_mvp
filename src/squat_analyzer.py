import mediapipe as mp
import numpy as np
import src.messages as msg

mp_pose = mp.solutions.pose

class SquatAnalyzer:
    def __init__(self, thresholds, logger):
        self.thresholds = thresholds
        self.logger = logger
        self.stage = None
        self.counter = 0
        self.active_side = None
        self.initial_heel_y = None
        self.logger.debug("SquatAnalyzer 초기화됨")
        self.logger.info(f"사용 임계값: {thresholds}")

    def detect_side(self, landmarks):
        """어느 쪽 측면인지 판단합니다 (LEFT 또는 RIGHT)."""
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_x = l_shoulder.x
        r_x = r_shoulder.x
        x_diff = abs(l_x - r_x)
        threshold = self.thresholds['side_x_diff']
        l_vis = l_shoulder.visibility
        r_vis = r_shoulder.visibility

        self.logger.debug(f"Side Detection: L_X={l_x:.3f}, R_X={r_x:.3f}, XDiff={x_diff:.3f}, Threshold={threshold:.2f}")

        if x_diff < threshold:
            self.logger.debug("Side view detected (XDiff < Threshold).")
            if l_vis >= r_vis:
                self.logger.debug("Active side: LEFT (L_Vis >= R_Vis)")
                return "LEFT"
            else:
                self.logger.debug("Active side: RIGHT (R_Vis > L_Vis)")
                return "RIGHT"
        else:
            self.logger.debug("Front view detected (XDiff >= Threshold).")
            return None

    def calculate_angle(self, a, b, c):
        """3점 a, b, c 사이의 각도를 계산합니다."""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def analyze(self, landmarks, active_side, initial_heel_y):
        """스쿼트 자세를 분석하고, stage, counter, feedback_list를 업데이트합니다."""
        feedback_list = []

        if active_side == "LEFT":
            knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
        elif active_side == "RIGHT":
            knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]
        else:
            self.logger.debug("No active side detected.")
            return self.stage, self.counter, feedback_list

        # 각도 계산
        knee_angle = self.calculate_angle(hip, knee, ankle)
        torso_vertical_angle = self.calculate_angle(hip, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])

        self.logger.debug(f"Angles: Knee={knee_angle:.2f}, TorsoVertical={torso_vertical_angle:.2f}")

        # 상태 업데이트 확인
        if self.stage == "None" or self.stage == None:
          if knee_angle > self.thresholds['up_pose_check_angle']:
            self.stage = "UP"
            initial_heel_y = heel.y # 시작자세 y 초기화
            self.logger.info(f"Stage -> UP. Initial heel_y ({active_side}) set/reset: {initial_heel_y:.3f}")

        elif self.stage == 'UP' and knee_angle < self.thresholds['knee_bent']:  # 임계값 100
            self.stage = "DOWN"
            self.counter += 1
            self.logger.info(f"Stage -> DOWN (Knee {knee_angle:.2f} < 100). Counter increment.")
            self.logger.info(f"REPS: {self.counter}")

        elif self.stage == 'DOWN' and knee_angle > self.thresholds['up_pose_check_angle']:  # 임계값 170
            self.stage = "UP"
            initial_heel_y = heel.y # 시작자세 y 초기화
            self.logger.info(f"Stage -> UP. Initial heel_y ({active_side}) set/reset: {initial_heel_y:.3f}")

        # 발꿈치 들림 확인
        # 발꿈치 들림 확인
        if initial_heel_y is not None:
            heel_lift_diff = abs(heel.y - initial_heel_y)
            if heel_lift_diff > self.thresholds['heel_lift_diff']: #임계값 0.03
              feedback_list.append(msg.HEEL_LIFT)
              self.logger.debug(f"Feedback added: {msg.HEEL_LIFT}")

        # 스쿼트 깊이 확인 (힙의 y 좌표가 무릎의 y 좌표보다 낮아야 함)
        depth_check_extra = self.thresholds['depth_check_extra']
        if self.stage == "DOWN":
            if hip.y > (knee.y - depth_check_extra * 0.01):
              feedback_list.append(msg.SQUAT_DEEPER)
              self.logger.debug(f"Feedback added: {msg.SQUAT_DEEPER}")

        self.logger.debug(f"Final Feedback List for Frame: {feedback_list}")
        return self.stage, self.counter, feedback_list
