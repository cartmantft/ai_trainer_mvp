# main.py
import cv2
import mediapipe as mp
import logging
from utils import setup_logging 
from squat_analyzer import SquatAnalyzer 
from visualization import draw_landmarks_and_feedback 

# --- 로깅 설정 ---
logger = setup_logging()

# --- MediaPipe Pose 모델 초기화 ---
mp_pose = mp.solutions.pose
pose = None 
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

# --- 웹캠/비디오 파일 초기화 ---
VIDEO_SOURCE = 0 
# VIDEO_SOURCE = "path/to/your/video.mp4" 
cap = cv2.VideoCapture(VIDEO_SOURCE) 
if not cap.isOpened():
    logger.error(f"비디오 소스 {VIDEO_SOURCE} 를 열 수 없습니다.")
    exit()
else:
    # 웹캠/비디오의 원본 해상도 얻기 (리사이즈 비율 계산용)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"비디오 소스 {VIDEO_SOURCE} 초기화 성공 (원본 해상도: {original_width}x{original_height})")

# --- 임계값 설정 ---
thresholds = {
    'depth_offset_y': 0.03, 
    'lean_angle': 50, 
    'heel_lift_diff': 0.03,
    'visibility': 0.6, 
    'side_x_diff': 0.15, 
    'knee_straight': 160, 
    'knee_bent': 100,   
    'up_pose_check_angle': 170, 
    'depth_check_extra': 15, 
    'feedback_font_size': 18, 
    'info_box_width': 300 
}

# --- 분석기 인스턴스 생성 ---
analyzer = SquatAnalyzer(thresholds, logger)

# --- 메인 루프 ---
frame_count = 0
window_name = 'AI Trainer MVP Feed' 

# ==================== 추가: 원하는 출력 너비 설정 ====================
DESIRED_WIDTH = 960 
# =================================================================

while cap.isOpened():
    frame_count += 1
    logger.debug(f"--- Frame {frame_count} Start ---")
    
    ret, frame = cap.read()
    if not ret:
        logger.error(f"프레임 {frame_count} 읽기 실패. 루프 종료.")
        break

    # 이미지 처리
    frame = cv2.flip(frame, 1)
    
    # ==================== 추가: 원본 비율 유지하며 리사이즈 ====================
    if original_width > 0 and original_height > 0: # 원본 해상도 확인 후
        aspect_ratio = original_height / original_width
        desired_height = int(DESIRED_WIDTH * aspect_ratio)
        # 리사이즈 (분석 전에 리사이즈하면 처리 속도 약간 향상 가능)
        frame = cv2.resize(frame, (DESIRED_WIDTH, desired_height), interpolation=cv2.INTER_AREA) 
        logger.debug(f"Frame resized to: {DESIRED_WIDTH}x{desired_height}")
    # =======================================================================

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False 
    
    # MediaPipe 처리
    try:
        results = pose.process(image_rgb)
        logger.debug("Pose processing executed.")
    except Exception as e:
        logger.error(f"Pose processing failed: {e}", exc_info=True)
        continue 

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) 
    image_bgr.flags.writeable = True

    # 스쿼트 분석
    feedback_list = [] 
    stage, counter, active_side = analyzer.stage, analyzer.counter, analyzer.active_side 
    
    if results.pose_landmarks:
        try:
            stage, counter, active_side, feedback_list = analyzer.analyze(results.pose_landmarks.landmark)
        except Exception as e:
            logger.error(f"분석 중 오류 발생: {e}", exc_info=True)
            feedback_list.append("분석 오류 발생")
    else:
        logger.debug("Pose landmarks not detected.")
        feedback_list.append("자세를 감지할 수 없습니다.")
        analyzer.active_side = None
        analyzer.initial_heel_y = None
        analyzer.stage = None

    # 결과 시각화
    try:
        output_image = draw_landmarks_and_feedback(image_bgr, results, stage, counter, active_side, feedback_list)
    except Exception as e:
         logger.error(f"시각화 중 오류 발생: {e}", exc_info=True)
         output_image = image_bgr 

    # 화면 표시
    cv2.imshow(window_name, output_image)
    
    # --- 삭제: cv2.resizeWindow 호출 ---
    # try:
    #     desired_width = 960  
    #     desired_height = 720 
    #     cv2.resizeWindow(window_name, desired_width, desired_height)
    # except Exception as e:
    #     logger.debug(f"cv2.resizeWindow 오류 (무시 가능): {e}") 
    #     pass
    # -----------------------------------

    logger.debug(f"Final Feedback List for Frame {frame_count}: {feedback_list}")
    logger.debug(f"--- Frame {frame_count} End ---")

    # 종료 조건
    if cv2.waitKey(10) & 0xFF == ord('q'):
        logger.info("종료 키('q') 입력됨. 루프 종료.")
        break

# --- 종료 처리 ---
# ... (이전과 동일) ...
logger.info("===== AI Trainer MVP Logging End =====")
cap.release()
logger.info("비디오 캡처 리소스 해제됨.")
cv2.destroyAllWindows()
logger.info("OpenCV 창 닫힘.")
if pose is not None: 
    pose.close()
    logger.info("MediaPipe Pose 리소스 해제됨.")
for handler in logging.getLogger().handlers[:]:
    handler.close()
    logging.getLogger().removeHandler(handler)