# ai_trainer_mvp/main.py (Refactored - Feedback Priority Applied)

import cv2
import mediapipe as mp
import numpy as np
import math
import logging 
import datetime 
import sys 

# --- 로깅 설정 ---
# (이전과 동일)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_filename = f"ai_trainer_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = logging.getLogger()
logger.setLevel(logging.DEBUG) 
try:
    file_handler = logging.FileHandler(log_filename, encoding='utf-8') 
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Error creating file handler: {e}") 
try:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG) 
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
except Exception as e:
    print(f"Error creating console handler: {e}") 
logger.info("===== AI Trainer MVP Logging Start (UTF-8 Enabled) =====")


# --- MediaPipe Pose 모델 초기화 ---
# (이전과 동일)
mp_pose = mp.solutions.pose
try:
    pose = mp_pose.Pose(static_image_mode=False, 
                        model_complexity=1, 
                        smooth_landmarks=True,
                        enable_segmentation=False, 
                        smooth_segmentation=True, 
                        min_detection_confidence=0.5, 
                        min_tracking_confidence=0.5)
    logger.info("MediaPipe Pose 모델 초기화 성공")
except Exception as e:
    logger.error(f"MediaPipe Pose 모델 초기화 실패: {e}", exc_info=True)
    exit()
mp_drawing = mp.solutions.drawing_utils


# --- 웹캠 초기화 ---
# (이전과 동일)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("웹캠을 열 수 없습니다.")
    exit()
else:
    logger.info("웹캠 초기화 성공")


# --- 스쿼트 카운터 및 상태 변수 ---
# (이전과 동일)
counter = 0
stage = None 
feedback_list = [] 
active_side = None 
initial_heel_y = None 
logger.debug("카운터 및 상태 변수 초기화")


# --- 각도 및 좌표 계산 함수 ---
# (이전과 동일)
def calculate_angle(a, b, c, landmark_names=None):
    if a is None or b is None or c is None: return None
    try:
        a_np, b_np, c_np = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c_np[1] - b_np[1], c_np[0] - b_np[0]) - np.arctan2(a_np[1] - b_np[1], a_np[0] - b_np[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0: angle = 360 - angle
        return angle
    except Exception as e: return None

def calculate_vertical_angle(a, b, landmark_names=None):
    if a is None or b is None: return None
    try:
        a_np, b_np = np.array(a), np.array(b)
        vertical_vector = np.array([0, 1]) 
        line_vector = b_np - a_np
        norm = np.linalg.norm(line_vector)
        if norm == 0: return 90.0 
        line_vector_norm = line_vector / norm
        dot_product = np.clip(np.dot(line_vector_norm, vertical_vector), -1.0, 1.0) 
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        return angle_deg 
    except Exception as e: return None

# --- 랜드마크 좌표 추출 함수 ---
# (이전과 동일)
def get_landmark_coords(landmarks, landmark_enum, visibility_threshold):
    try:
        lm = landmarks[landmark_enum.value]
        lm_name = landmark_enum.name 
        if lm.visibility > visibility_threshold:
            coords = [lm.x, lm.y]
            # logger.debug(f"Landmark {lm_name}: Coords=[{coords[0]:.3f}, {coords[1]:.3f}], Visibility={lm.visibility:.3f} > {visibility_threshold}")
            return coords
        else:
            # logger.debug(f"Landmark {lm_name}: Visibility {lm.visibility:.3f} <= {visibility_threshold}. Returning None.")
            return None
    except IndexError:
        logger.warning(f"Landmark index {landmark_enum.value} ({landmark_enum.name}) out of range.")
        return None
    except Exception as e:
        logger.error(f"Error getting landmark {landmark_enum.name}: {e}", exc_info=False)
        return None

# --- 오류 감지 임계값 설정 ---
DEPTH_THRESHOLD_Y_OFFSET = 0.03 
LEAN_ANGLE_THRESHOLD = 50 
HEEL_LIFT_THRESHOLD_Y_DIFF = 0.03
VISIBILITY_THRESHOLD = 0.6 
SIDE_VIEW_X_DIFF_THRESHOLD = 0.15 
KNEE_STRAIGHT_ANGLE = 160 
KNEE_BENT_ANGLE = 100   
UP_POSE_CHECK_ANGLE_THRESHOLD = 170 
DEPTH_CHECK_EXTRA_ANGLE = 15 

logger.info(f"Thresholds set: DepthOffsetY={DEPTH_THRESHOLD_Y_OFFSET}, LeanAngle={LEAN_ANGLE_THRESHOLD}, HeelLiftDiff={HEEL_LIFT_THRESHOLD_Y_DIFF}, Visibility={VISIBILITY_THRESHOLD}, SideXDiff={SIDE_VIEW_X_DIFF_THRESHOLD}, KneeStraight={KNEE_STRAIGHT_ANGLE}, KneeBent={KNEE_BENT_ANGLE}, UpPoseCheckAngle={UP_POSE_CHECK_ANGLE_THRESHOLD}, DepthCheckExtra={DEPTH_CHECK_EXTRA_ANGLE}")


# --- 메인 루프 ---
frame_count = 0
while cap.isOpened():
    frame_count += 1
    logger.debug(f"--- Frame {frame_count} Start ---")
    ret, frame = cap.read()
    if not ret:
        logger.error("프레임 읽기 실패. 루프 종료.")
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    try:
        results = pose.process(image)
        logger.debug("Pose processing executed.")
    except Exception as e:
        logger.error(f"Pose processing failed: {e}", exc_info=True)
        continue 

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    feedback_list = []
    current_active_side = None 

    try:
        if results.pose_landmarks:
            logger.debug("Pose landmarks detected.")
            landmarks = results.pose_landmarks.landmark

            # --- 측면 감지 ---
            # (이전과 동일)
            left_shoulder_coord = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, VISIBILITY_THRESHOLD * 0.8)
            right_shoulder_coord = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, VISIBILITY_THRESHOLD * 0.8)
            is_side_view = False
            if left_shoulder_coord and right_shoulder_coord:
                shoulder_x_diff = abs(left_shoulder_coord[0] - right_shoulder_coord[0])
                logger.debug(f"Side Detection: L_Shoulder_X={left_shoulder_coord[0]:.3f}, R_Shoulder_X={right_shoulder_coord[0]:.3f}, ActualXDiff={shoulder_x_diff:.3f}, XDiffThreshold={SIDE_VIEW_X_DIFF_THRESHOLD}")
                if shoulder_x_diff < SIDE_VIEW_X_DIFF_THRESHOLD:
                    is_side_view = True
                    logger.debug(f"Side view detected (XDiff < Threshold).")
                    left_shoulder_vis = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
                    right_shoulder_vis = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
                    if left_shoulder_vis >= right_shoulder_vis: current_active_side = 'LEFT'; logger.debug("Determined active side: LEFT (L_Vis >= R_Vis)")
                    else: current_active_side = 'RIGHT'; logger.debug("Determined active side: RIGHT (R_Vis > L_Vis)")
                else: logger.debug(f"Not a side view (XDiff >= Threshold).")
            else: logger.warning("Shoulder landmarks for side detection not visible.")

            if not is_side_view:
                feedback_list.append("측면 자세를 취해주세요.")
                if active_side is not None: logger.info("Side view lost or undetermined, resetting initial_heel_y."); initial_heel_y = None
                active_side = None

            # --- 활성 측면이 감지된 경우 ---
            if current_active_side:
                if active_side != current_active_side:
                    logger.info(f"Active side is {current_active_side}. Resetting initial_heel_y if needed.")
                    initial_heel_y = None 
                active_side = current_active_side

                if active_side == 'LEFT':
                    shoulder_lm, hip_lm, knee_lm, ankle_lm, heel_lm = (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL)
                else:
                    shoulder_lm, hip_lm, knee_lm, ankle_lm, heel_lm = (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL)

                logger.debug(f"Getting main landmarks for {active_side} side using VISIBILITY_THRESHOLD={VISIBILITY_THRESHOLD}...")
                shoulder = get_landmark_coords(landmarks, shoulder_lm, VISIBILITY_THRESHOLD)
                hip = get_landmark_coords(landmarks, hip_lm, VISIBILITY_THRESHOLD)
                knee = get_landmark_coords(landmarks, knee_lm, VISIBILITY_THRESHOLD)
                ankle = get_landmark_coords(landmarks, ankle_lm, VISIBILITY_THRESHOLD)
                heel = get_landmark_coords(landmarks, heel_lm, VISIBILITY_THRESHOLD * 0.8) 

                required_landmarks = {'shoulder': shoulder, 'hip': hip, 'knee': knee, 'ankle': ankle}
                required_landmarks_visible = all(coord is not None for coord in required_landmarks.values())
                logger.debug(f"Main required landmarks visible: {required_landmarks_visible}")

                if required_landmarks_visible:
                    logger.debug("Calculating angles...")
                    knee_angle = calculate_angle(hip, knee, ankle) 
                    torso_vertical_angle = calculate_vertical_angle(shoulder, hip)
                    
                    if knee_angle is not None and torso_vertical_angle is not None:
                        logger.debug(f"Calculated Angles: Knee={knee_angle:.2f} deg, TorsoVertical={torso_vertical_angle:.2f} deg")
                        
                        # --- 스쿼트 상태 판정 및 카운터 ---
                        logger.debug(f"State Check: Current Stage='{stage}', KneeAngle={knee_angle:.2f}")
                        if knee_angle > KNEE_STRAIGHT_ANGLE:
                            if stage == 'DOWN' or initial_heel_y is None: 
                                if heel is not None: initial_heel_y = heel[1]; logger.info(f"Stage changed to UP. Initial heel_y ({active_side}) set/reset: {initial_heel_y:.3f}")
                                else: logger.warning("Cannot set initial_heel_y in UP state because heel landmark is not visible.")
                            stage = "UP"
                        elif knee_angle < KNEE_BENT_ANGLE and stage == 'UP':
                            logger.info(f"Stage changed to DOWN (Knee Angle {knee_angle:.2f} < {KNEE_BENT_ANGLE}). Counter incremented.")
                            stage = "DOWN"
                            counter += 1
                            logger.info(f"REPS: {counter}")
                        
                        # --- 잘못된 자세 감지 및 피드백 (단계별 적용 + 우선순위) ---
                        logger.debug("Checking for incorrect posture based on stage...")
                        
                        # 플래그: 상체 숙임 오류가 감지되었는지 여부
                        lean_error_detected = False 
                        
                        # 발뒤꿈치 들림 체크 (항상)
                        if initial_heel_y is not None and heel is not None:
                            heel_lift_condition = heel[1] < initial_heel_y - HEEL_LIFT_THRESHOLD_Y_DIFF
                            logger.debug(f"Heel Lift Check: current_heel_y={heel[1]:.3f}, initial_heel_y={initial_heel_y:.3f}, threshold_diff={HEEL_LIFT_THRESHOLD_Y_DIFF}. Condition Met: {heel_lift_condition}")
                            if heel_lift_condition:
                                feedback_list.append("발뒤꿈치를 바닥에 붙이세요.")
                                logger.debug("Feedback added: 발뒤꿈치를 바닥에 붙이세요.")
                        else: logger.debug("Heel Lift Check: Skipped (initial_heel_y or current heel is None)")

                        # 상체 숙임 체크 (완전히 서있지 않을 때 우선적으로)
                        if knee_angle < UP_POSE_CHECK_ANGLE_THRESHOLD: 
                            lean_condition = torso_vertical_angle > LEAN_ANGLE_THRESHOLD 
                            logger.debug(f"Lean Check (during movement): torso_vertical_angle={torso_vertical_angle:.2f}, threshold={LEAN_ANGLE_THRESHOLD}. Condition Met: {lean_condition}")
                            if lean_condition:
                                feedback_list.append("상체를 덜 숙이세요.") 
                                logger.debug("Feedback added: 상체를 덜 숙이세요.")
                                lean_error_detected = True # 상체 숙임 오류 감지됨
                        else: logger.debug(f"Lean Check skipped (Up pose, KneeAngle={knee_angle:.2f} >= {UP_POSE_CHECK_ANGLE_THRESHOLD})")
                             
                        # 깊이 체크 (DOWN 상태이고, 최저점 부근이며, *상체 숙임 오류가 없을 때만*)
                        # ==================== 수정된 부분 ====================
                        if not lean_error_detected and stage == 'DOWN' and knee_angle < (KNEE_BENT_ANGLE + DEPTH_CHECK_EXTRA_ANGLE): 
                            depth_condition = hip[1] < knee[1] - DEPTH_THRESHOLD_Y_OFFSET 
                            logger.debug(f"Depth Check (Lean OK, DOWN stage near bottom): hip_y={hip[1]:.3f}, knee_y={knee[1]:.3f}, threshold_y={knee[1] - DEPTH_THRESHOLD_Y_OFFSET:.3f}. Condition Met: {depth_condition}")
                            if depth_condition:
                                feedback_list.append("더 깊이 앉으세요.")
                                logger.debug("Feedback added: 더 깊이 앉으세요.")
                        elif lean_error_detected:
                             logger.debug(f"Depth Check skipped due to lean error detected.")
                        elif stage == 'DOWN': # DOWN이지만 최저점 부근 아님
                             logger.debug(f"Depth Check skipped (in DOWN stage but ascending, KneeAngle={knee_angle:.2f} >= {KNEE_BENT_ANGLE + DEPTH_CHECK_EXTRA_ANGLE})")
                        # ===================================================

                        if not feedback_list: 
                             logger.debug("Posture check: No issues detected in this frame.")

                    else: 
                        logger.warning("Angle calculation failed, skipping posture check.")
                        feedback_list.append("각도 계산 오류.")
                        
                else: 
                     logger.warning("Main required landmarks not visible, skipping calculations.")
                     feedback_list.append("몸 전체가 잘 보이도록 거리를 조절하세요.")
                     if active_side is not None: 
                         logger.info("Main landmarks missing, resetting initial_heel_y.")
                         initial_heel_y = None
                     
        else: 
            logger.debug("Pose landmarks not detected in this frame.")
            feedback_list.append("자세를 감지할 수 없습니다.")
            if active_side is not None: logger.info("No landmarks detected, resetting initial_heel_y."); initial_heel_y = None
            active_side = None
            if stage is not None: logger.info("No landmarks detected, resetting stage."); stage = None 

    except Exception as e:
        logger.error(f"메인 루프 처리 중 오류 발생: {e}", exc_info=True) 
        feedback_list.append("처리 중 오류 발생")
        if active_side is not None: initial_heel_y = None
        active_side = None
        stage = None

    # --- 화면 표시 ---
    # (이전과 동일)
    box_width = 280
    cv2.rectangle(image, (0, 0), (box_width, 75), (245, 117, 16), -1)
    cv2.putText(image, 'REPS', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, str(counter), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'STAGE', (105, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, stage if stage else "READY", (105, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'SIDE', (205, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, active_side if active_side else "N/A", (205, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    feedback_y_start = 95 
    feedback_box_height = 30 + len(feedback_list) * 25 
    if feedback_box_height > 30 : 
        overlay = image.copy()
        cv2.rectangle(overlay, (0, feedback_y_start - 25), (box_width, feedback_y_start + feedback_box_height - 25), (50, 50, 50), -1) 
        alpha = 0.6 
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
    cv2.putText(image, 'FEEDBACK', (15, feedback_y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if not feedback_list and active_side: 
         cv2.putText(image, "Good form!", (15, feedback_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    else: 
        for i, msg in enumerate(feedback_list):
            is_warning = "측면" in msg or "감지" in msg or "오류" in msg or "거리" in msg or "계산" in msg
            color = (0, 180, 255) if is_warning else (50, 50, 255) 
            cv2.putText(image, msg, (15, feedback_y_start + i*25 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA) 
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                   mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                   )               
    cv2.imshow('AI Trainer MVP Feed', image)

    logger.debug(f"Final Feedback List for Frame {frame_count}: {feedback_list}")
    logger.debug(f"--- Frame {frame_count} End ---")

    if cv2.waitKey(10) & 0xFF == ord('q'):
        logger.info("종료 키('q') 입력됨. 루프 종료.")
        break

# --- 종료 처리 ---
# (이전과 동일)
logger.info("===== AI Trainer MVP Logging End =====")
cap.release()
logger.info("웹캠 리소스 해제됨.")
cv2.destroyAllWindows()
logger.info("OpenCV 창 닫힘.")
if 'pose' in locals() and pose is not None: 
    pose.close()
    logger.info("MediaPipe Pose 리소스 해제됨.")
for handler in logger.handlers[:]:
    handler.close()
    logger.removeHandler(handler)