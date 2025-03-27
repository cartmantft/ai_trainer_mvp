# main.py
import cv2
import mediapipe as mp
import logging
from utils import setup_logging # 로깅 설정 함수 임포트
from squat_analyzer import SquatAnalyzer # 분석 클래스 임포트
from visualization import draw_landmarks_and_feedback # 시각화 함수 임포트

# --- 로깅 설정 ---
logger = setup_logging()

# --- MediaPipe Pose 모델 초기화 ---
mp_pose = mp.solutions.pose
pose = None # try 블록 밖에서 초기화
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
# VIDEO_SOURCE = 0 # 웹캠 사용 시
VIDEO_SOURCE = "./squat_videos/과도한 상체 숙임.mp4" # 영상 파일 사용 시 경로 지정

cap = cv2.VideoCapture(VIDEO_SOURCE) 
if not cap.isOpened():
    logger.error(f"비디오 소스 {VIDEO_SOURCE} 를 열 수 없습니다.")
    exit()
else:
    logger.info(f"비디오 소스 {VIDEO_SOURCE} 초기화 성공")

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
    'depth_check_extra': 15 
}

# --- 분석기 인스턴스 생성 ---
analyzer = SquatAnalyzer(thresholds, logger)

# --- 메인 루프 ---
frame_count = 0
while cap.isOpened():
    frame_count += 1
    logger.debug(f"--- Frame {frame_count} Start ---")
    
    ret, frame = cap.read()
    if not ret:
        logger.error(f"프레임 {frame_count} 읽기 실패. 루프 종료.")
        break

    # 이미지 처리
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False # 성능 향상
    
    # MediaPipe 처리
    try:
        results = pose.process(image_rgb)
        logger.debug("Pose processing executed.")
    except Exception as e:
        logger.error(f"Pose processing failed: {e}", exc_info=True)
        continue 

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # 화면 표시용 BGR 이미지
    image_bgr.flags.writeable = True

    # 스쿼트 분석
    feedback_list = [] # 매 프레임 초기화
    stage, counter, active_side = None, analyzer.counter, None # 기본값 설정
    
    if results.pose_landmarks:
        try:
            stage, counter, active_side, feedback_list = analyzer.analyze(results.pose_landmarks.landmark)
        except Exception as e:
            logger.error(f"분석 중 오류 발생: {e}", exc_info=True)
            feedback_list.append("분석 오류 발생")
            # 필요시 analyzer 상태 리셋 고려
    else:
        logger.debug("Pose landmarks not detected.")
        feedback_list.append("자세를 감지할 수 없습니다.")
        # analyzer 상태 리셋 (active_side 등)
        analyzer.active_side = None
        analyzer.initial_heel_y = None
        analyzer.stage = None

    # 결과 시각화
    try:
        output_image = draw_landmarks_and_feedback(image_bgr, results, stage, counter, active_side, feedback_list)
    except Exception as e:
         logger.error(f"시각화 중 오류 발생: {e}", exc_info=True)
         output_image = image_bgr # 오류 시 원본 이미지 표시

    # 화면 표시
    cv2.imshow('AI Trainer MVP Feed', output_image)

    logger.debug(f"Final Feedback List for Frame {frame_count}: {feedback_list}")
    logger.debug(f"--- Frame {frame_count} End ---")

    # 종료 조건
    if cv2.waitKey(10) & 0xFF == ord('q'):
        logger.info("종료 키('q') 입력됨. 루프 종료.")
        break

# --- 종료 처리 ---
logger.info("===== AI Trainer MVP Logging End =====")
cap.release()
logger.info("비디오 캡처 리소스 해제됨.")
cv2.destroyAllWindows()
logger.info("OpenCV 창 닫힘.")
if pose is not None: 
    pose.close()
    logger.info("MediaPipe Pose 리소스 해제됨.")

# 로깅 핸들러 닫기
for handler in logging.getLogger().handlers[:]:
    handler.close()
    logging.getLogger().removeHandler(handler)