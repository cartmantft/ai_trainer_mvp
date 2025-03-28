import cv2
import mediapipe as mp
import logging
import datetime
import src.messages as msg
from utils import setup_logging, smooth_value  # smooth_value 함수 import
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
                        min_detection_confidence=0.7, # 높임
                        min_tracking_confidence=0.7) # 높임
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
    'heel_lift_diff': 0.05, # 높임
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

# ==================== 추가: VideoRecorder 클래스 ====================
class VideoRecorder:
    def __init__(self, output_path="output.mp4", fps=20.0, resolution=(640, 480)):
        """
        웹캠 영상을 녹화하는 클래스입니다.

        Args:
            output_path (str, optional): 저장될 영상 파일 경로. Defaults to "output.mp4".
            fps (float, optional): 프레임 속도. Defaults to 20.0.
            resolution (tuple, optional): 해상도 (width, height). Defaults to (640, 480).
        """
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.is_recording = False
        self.video_writer = None

    def start_recording(self):
        """녹화를 시작합니다."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 (MP4)
        self.video_writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, self.resolution)
        self.is_recording = True
        print(msg.RECORDING_START_PROMPT)

    def stop_recording(self):
        """녹화를 중지합니다."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
        print(msg.RECORDING_STOP_PROMPT)

    def write_frame(self, frame):
        """프레임을 영상에 기록합니다."""
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)
# =======================================================================

# ==================== 추가: 녹화 관련 변수 초기화 ====================
recording = False  # 녹화 상태 변수
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"output_{timestamp}.mp4"
video_recorder = None
# =======================================================================

# ==================== 추가: 창 크기 조절 가능하도록 설정 ====================
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# =======================================================================

# active_side 변수를 SquatAnalyzer 외부에서 관리하도록 변경
active_side = None
initial_heel_y = None

# 이전 프레임 랜드마크 변수 초기화
previous_landmarks = None

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
    if original_width > 0 and original_height > 0:  # 원본 해상도 확인 후
        aspect_ratio = original_height / original_width
        desired_height = int(DESIRED_WIDTH * aspect_ratio)
        # 리사이즈 (분석 전에 리사이즈하면 처리 속도 약간 향상 가능)
        frame = cv2.resize(frame, (DESIRED_WIDTH, desired_height), interpolation=cv2.INTER_AREA)
        logger.debug(f"Frame resized to: {DESIRED_WIDTH}x{desired_height}")
    # =======================================================================

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    # Pose processing
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
    stage, counter = analyzer.stage, analyzer.counter

    # active_side가 None일 때만 Side Detection 실행
    if results.pose_landmarks and active_side is None:
        try:
            active_side = analyzer.detect_side(results.pose_landmarks.landmark)
            logger.info(f"Active side detected: {active_side}. Resetting initial_heel_y.")
            initial_heel_y = None  # active_side가 변경될 때마다 초기 heel_y 값을 재설정
        except Exception as e:
            logger.error(f"{msg.ERROR_SIDE_DETECTION}: {e}", exc_info=True)
            feedback_list.append(msg.ERROR_SIDE_DETECTION)
            active_side = None  # 오류 발생 시 active_side를 None으로 설정

    if results.pose_landmarks:
        try:
            landmark_list = results.pose_landmarks.landmark

            # 랜드마크 좌표를 부드럽게 처리
            if previous_landmarks:
                for i in range(len(landmark_list)):
                    landmark_list[i].x = smooth_value(landmark_list[i].x, previous_landmarks[i].x)
                    landmark_list[i].y = smooth_value(landmark_list[i].y, previous_landmarks[i].y)
                    landmark_list[i].z = smooth_value(landmark_list[i].z, previous_landmarks[i].z)
            previous_landmarks = landmark_list # 현재 랜드마크를 이전 랜드마크로 저장

            stage, counter, feedback_list = analyzer.analyze(landmark_list, active_side, initial_heel_y)

        except Exception as e:
            logger.error(f"{msg.ERROR_ANALYSIS}: {e}", exc_info=True)
            feedback_list.append(msg.ERROR_ANALYSIS)
    else:
        logger.debug("Pose landmarks not detected.")
        feedback_list.append(msg.ERROR_POSE_DETECTION)
        analyzer.active_side = None
        initial_heel_y = None
        analyzer.stage = None
        previous_landmarks = None # 랜드마크가 없을 경우 이전 랜드마크 정보 초기화


    # 결과 시각화
    try:
        output_image = draw_landmarks_and_feedback(image_bgr, results, stage, counter, active_side, feedback_list)
    except Exception as e:
        logger.error(f"시각화 중 오류 발생: {e}", exc_info=True)
        output_image = image_bgr

    # ==================== 추가: 창 크기에 맞춰 리사이즈 ====================
    window_width = cv2.getWindowImageRect(window_name)[2]
    window_height = cv2.getWindowImageRect(window_name)[3]
    resized_frame = cv2.resize(output_image, (window_width, window_height))
    # =======================================================================

    # 화면 표시
    cv2.imshow(window_name, resized_frame)

    # ==================== 추가: 녹화 시작/중지 기능 ====================
    key = cv2.waitKey(10) & 0xFF
    if  key == ord('s'):  # 's' 키를 누르면 녹화 시작/중지
        if not recording:
            # 해상도, FPS 설정
            fps = 20.0
            resolution = (window_width, window_height) # 현재 창 크기
            video_recorder = VideoRecorder(output_path=output_path, fps=fps, resolution=resolution)

            video_recorder.start_recording()
            recording = True
            logger.info(msg.LOG_RECORDING_START)
        else:
            video_recorder.stop_recording()
            recording = False
            logger.info(msg.LOG_RECORDING_STOP)
    elif key == ord('q'):
        logger.info("종료 키('q') 입력됨. 루프 종료.")
        break
    # =======================================================================

    # ==================== 추가: 녹화 중 프레임 기록 ====================
    if recording:
        video_recorder.write_frame(output_image) # 리사이즈 안된 원본 이미지
    # =======================================================================

    logger.debug(f"Final Feedback List for Frame {frame_count}: {feedback_list}")
    logger.debug(f"--- Frame {frame_count} End ---")


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

# ==================== 추가: 종료 시 녹화 중이면 중지 ====================
if recording and video_recorder:
    video_recorder.stop_recording()
    logger.info(msg.LOG_RECORDING_STOP_EXIT)
# =======================================================================

for handler in logging.getLogger().handlers[:]:
    handler.close()
    logging.getLogger().removeHandler(handler)
