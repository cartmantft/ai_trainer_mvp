# squat_analyzer.py
import mediapipe as mp
import logging
from utils import get_landmark_coords, calculate_angle, calculate_vertical_angle

mp_pose = mp.solutions.pose # MediaPipe Pose 솔루션 사용

class SquatAnalyzer:
    def __init__(self, thresholds, logger):
        """
        스쿼트 분석기 초기화
        :param thresholds: 각 오류 감지를 위한 임계값 딕셔너리
        :param logger: 로깅 객체
        """
        self.thresholds = thresholds
        self.logger = logger
        
        # 상태 변수
        self.counter = 0
        self.stage = None
        self.active_side = None
        self.initial_heel_y = None
        
        self.logger.debug("SquatAnalyzer 초기화됨")
        self.logger.info(f"사용 임계값: {self.thresholds}")

    def _detect_side(self, landmarks):
        """어깨 X좌표 기준으로 측면 감지"""
        vis_threshold = self.thresholds['visibility'] * 0.8 # 측면 감지용 가시성 임계값
        x_diff_threshold = self.thresholds['side_x_diff']
        
        left_shoulder_coord = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, vis_threshold)
        right_shoulder_coord = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, vis_threshold)
        
        is_side_view = False
        current_active_side = None
        
        if left_shoulder_coord and right_shoulder_coord:
            shoulder_x_diff = abs(left_shoulder_coord[0] - right_shoulder_coord[0])
            self.logger.debug(f"Side Detection: L_X={left_shoulder_coord[0]:.3f}, R_X={right_shoulder_coord[0]:.3f}, XDiff={shoulder_x_diff:.3f}, Threshold={x_diff_threshold}")
            
            if shoulder_x_diff < x_diff_threshold:
                is_side_view = True
                self.logger.debug(f"Side view detected (XDiff < Threshold).")
                left_vis = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
                right_vis = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
                if left_vis >= right_vis: 
                    current_active_side = 'LEFT'; self.logger.debug("Active side: LEFT (L_Vis >= R_Vis)")
                else: 
                    current_active_side = 'RIGHT'; self.logger.debug("Active side: RIGHT (R_Vis > L_Vis)")
            else: self.logger.debug(f"Not a side view (XDiff >= Threshold).")
        else: self.logger.warning("Shoulder landmarks for side detection not visible.")
            
        return is_side_view, current_active_side

    def _update_stage_and_counter(self, knee_angle, heel_y):
        """무릎 각도를 기준으로 스쿼트 상태(UP/DOWN) 및 카운터 업데이트"""
        knee_straight_angle = self.thresholds['knee_straight']
        knee_bent_angle = self.thresholds['knee_bent']
        
        self.logger.debug(f"State Update Check: Current Stage='{self.stage}', KneeAngle={knee_angle:.2f}")
        
        if knee_angle > knee_straight_angle:
            if self.stage == 'DOWN' or self.initial_heel_y is None: 
                if heel_y is not None: 
                    self.initial_heel_y = heel_y
                    self.logger.info(f"Stage -> UP. Initial heel_y ({self.active_side}) set/reset: {self.initial_heel_y:.3f}")
                else: self.logger.warning("Cannot set initial_heel_y in UP state, heel not visible.")
            self.stage = "UP"
        elif knee_angle < knee_bent_angle and self.stage == 'UP':
            self.logger.info(f"Stage -> DOWN (Knee {knee_angle:.2f} < {knee_bent_angle}). Counter increment.")
            self.stage = "DOWN"
            self.counter += 1
            self.logger.info(f"REPS: {self.counter}")
        # else: 중간 상태, stage 변경 없음

    def _check_posture(self, landmarks, knee_angle, torso_vertical_angle):
        """현재 상태에 따라 자세 오류 검사 및 피드백 생성"""
        feedback = []
        lean_error_detected = False

        # 1. 발뒤꿈치 들림 체크
        heel_coord = get_landmark_coords(landmarks, 
                                       mp_pose.PoseLandmark.LEFT_HEEL if self.active_side == 'LEFT' else mp_pose.PoseLandmark.RIGHT_HEEL, 
                                       self.thresholds['visibility'] * 0.8)
        if self.initial_heel_y is not None and heel_coord is not None:
            heel_lift_diff = self.initial_heel_y - heel_coord[1] # 현재 Y가 작으면 들린 것
            lift_threshold = self.thresholds['heel_lift_diff']
            self.logger.debug(f"Heel Lift Check: current_y={heel_coord[1]:.3f}, initial_y={self.initial_heel_y:.3f}, diff={heel_lift_diff:.3f}, threshold={lift_threshold}")
            if heel_lift_diff > lift_threshold: # 초기 위치보다 많이 올라갔으면
                feedback.append("발뒤꿈치를 바닥에 붙이세요.")
                self.logger.debug("Feedback added: 발뒤꿈치를 바닥에 붙이세요.")
        else: self.logger.debug("Heel Lift Check: Skipped (initial_heel_y or current heel is None)")

        # 2. 상체 숙임 체크 (완전히 서있지 않을 때)
        if knee_angle < self.thresholds['up_pose_check_angle']: 
            lean_threshold = self.thresholds['lean_angle']
            self.logger.debug(f"Lean Check: torso_angle={torso_vertical_angle:.2f}, threshold={lean_threshold}")
            if torso_vertical_angle > lean_threshold: 
                feedback.append("상체를 덜 숙이세요.") 
                self.logger.debug("Feedback added: 상체를 덜 숙이세요.")
                lean_error_detected = True 
        else: self.logger.debug(f"Lean Check skipped (Up pose, KneeAngle={knee_angle:.2f})")
             
        # 3. 깊이 체크 (DOWN 상태, 최저점 부근, 상체 숙임 오류 없을 때)
        if not lean_error_detected:
            if self.stage == 'DOWN' and knee_angle < (self.thresholds['knee_bent'] + self.thresholds['depth_check_extra']): 
                hip_coord = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP if self.active_side == 'LEFT' else mp_pose.PoseLandmark.RIGHT_HIP, self.thresholds['visibility'])
                knee_coord = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE if self.active_side == 'LEFT' else mp_pose.PoseLandmark.RIGHT_KNEE, self.thresholds['visibility'])
                
                if hip_coord and knee_coord: # 좌표 둘 다 있어야 비교 가능
                    depth_offset = self.thresholds['depth_offset_y']
                    threshold_y = knee_coord[1] - depth_offset
                    self.logger.debug(f"Depth Check: hip_y={hip_coord[1]:.3f}, knee_y={knee_coord[1]:.3f}, threshold_y={threshold_y:.3f}")
                    if hip_coord[1] < threshold_y: # 엉덩이가 무릎보다 높으면
                        feedback.append("더 깊이 앉으세요.")
                        self.logger.debug("Feedback added: 더 깊이 앉으세요.")
            elif self.stage == 'DOWN':
                 self.logger.debug(f"Depth Check skipped (DOWN but ascending, KneeAngle={knee_angle:.2f})")
        else:
             self.logger.debug(f"Depth Check skipped due to lean error detected.")

        if not feedback: 
             self.logger.debug("Posture check: No issues detected.")
             
        return feedback


    def analyze(self, landmarks):
        """
        주어진 랜드마크로 스쿼트 자세 분석 수행
        :param landmarks: MediaPipe PoseLandmark 리스트
        :return: stage, counter, active_side, feedback_list 튜플
        """
        self.logger.debug("Analyzing landmarks...")
        feedback = []
        
        is_side_view, current_active_side = self._detect_side(landmarks)

        if not is_side_view:
            feedback.append("측면 자세를 취해주세요.")
            if self.active_side is not None: 
                self.logger.info("Side view lost, resetting state.")
                self.initial_heel_y = None
            self.active_side = None
            self.stage = None # 측면 아니면 상태도 리셋
            # self.counter 는 유지
        else:
            # 측면 정보 업데이트
            if self.active_side != current_active_side:
                self.logger.info(f"Active side detected: {current_active_side}. Resetting initial_heel_y.")
                self.initial_heel_y = None 
            self.active_side = current_active_side

            # 사용할 랜드마크 Enum 결정
            shoulder_lm, hip_lm, knee_lm, ankle_lm, heel_lm = (
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL) 
                if self.active_side == 'LEFT' else 
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL)
            )

            # 주요 랜드마크 좌표 추출
            vis_threshold = self.thresholds['visibility']
            shoulder = get_landmark_coords(landmarks, shoulder_lm, vis_threshold)
            hip = get_landmark_coords(landmarks, hip_lm, vis_threshold)
            knee = get_landmark_coords(landmarks, knee_lm, vis_threshold)
            ankle = get_landmark_coords(landmarks, ankle_lm, vis_threshold)
            
            required_coords = {'shoulder': shoulder, 'hip': hip, 'knee': knee, 'ankle': ankle}
            if all(coord is not None for coord in required_coords.values()):
                # 각도 계산
                knee_angle = calculate_angle(hip, knee, ankle)
                torso_vertical_angle = calculate_vertical_angle(shoulder, hip)

                if knee_angle is not None and torso_vertical_angle is not None:
                    self.logger.debug(f"Angles: Knee={knee_angle:.2f}, TorsoVertical={torso_vertical_angle:.2f}")
                    
                    # 현재 발뒤꿈치 Y 좌표 (상태 업데이트 및 자세 검사용)
                    heel_y = get_landmark_coords(landmarks, heel_lm, vis_threshold * 0.8)[1] if get_landmark_coords(landmarks, heel_lm, vis_threshold * 0.8) else None

                    # 상태 업데이트 및 카운터
                    self._update_stage_and_counter(knee_angle, heel_y)
                    
                    # 자세 검사 및 피드백 생성
                    feedback = self._check_posture(landmarks, knee_angle, torso_vertical_angle)
                else:
                    self.logger.warning("Angle calculation failed.")
                    feedback.append("각도 계산 오류.")
            else:
                 self.logger.warning("Main landmarks not visible.")
                 feedback.append("몸 전체가 잘 보이도록 거리를 조절하세요.")
                 if self.active_side is not None: 
                     self.logger.info("Main landmarks missing, resetting initial_heel_y.")
                     self.initial_heel_y = None
                 self.stage = None # 주요 랜드마크 안 보이면 상태 리셋
                 # self.counter 는 유지

        return self.stage, self.counter, self.active_side, feedback