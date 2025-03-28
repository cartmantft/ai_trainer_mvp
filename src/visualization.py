# visualization.py
import cv2
import mediapipe as mp
import numpy as np
# ==================== 추가된 부분 ====================
from PIL import Image, ImageDraw, ImageFont 
import logging # 로깅 추가 (폰트 로드 확인용)
import os      # 폰트 경로 확인용
import sys     # OS 확인용
# ===================================================

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 

# ==================== 추가된 부분: 폰트 로드 ====================
logger = logging.getLogger(__name__) # visualization 모듈용 로거

def load_korean_font(font_size=20):
    """한글 지원 폰트를 로드합니다. 실패 시 None 반환"""
    # Windows 우선 탐색
    font_path_win = "C:/Windows/Fonts/malgun.ttf" # 맑은 고딕
    # macOS 우선 탐색 (Apple SD Gothic Neo)
    font_path_mac = "/System/Library/Fonts/Supplemental/AppleGothic.ttf" 
    # Linux (Nanum 폰트 설치 필요)
    font_path_linux = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    
    font_path = None
    if sys.platform == "win32":
        if os.path.exists(font_path_win):
            font_path = font_path_win
    elif sys.platform == "darwin": # macOS
        if os.path.exists(font_path_mac):
            font_path = font_path_mac
    elif sys.platform.startswith("linux"):
        if os.path.exists(font_path_linux):
            font_path = font_path_linux
            
    # 대체 폰트 경로 (예: 프로젝트 내부에 fonts 폴더 만들고 NanumGothic.ttf 넣기)
    font_path_local = "fonts/NanumGothic.ttf" 
    if font_path is None and os.path.exists(font_path_local):
         font_path = font_path_local

    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
            logger.info(f"성공적으로 폰트 로드: {font_path}")
            return font
        except IOError:
            logger.error(f"폰트 파일을 찾았으나 로드할 수 없습니다: {font_path}")
            return None
        except Exception as e:
            logger.error(f"폰트 로드 중 알 수 없는 오류: {e}")
            return None
    else:
        logger.warning("한글 지원 폰트(맑은 고딕, Apple SD Gothic Neo, 나눔고딕)를 찾을 수 없습니다. 피드백에 한글이 깨질 수 있습니다.")
        # 기본 Pillow 폰트 로드 시도 (영문만 가능)
        try:
            font = ImageFont.load_default()
            return font
        except IOError:
             logger.error("기본 폰트 로드 실패.")
             return None

# 폰트 로드 (스크립트 시작 시 한 번만 로드)
KOREAN_FONT_SIZE = 18 # 폰트 크기 조절 가능
korean_font = load_korean_font(KOREAN_FONT_SIZE) 
# ==========================================================

def draw_landmarks_and_feedback(image, results, stage, counter, active_side, feedback_list):
    """랜드마크, 정보 박스, 피드백을 이미지에 그립니다."""
    
    # 랜드마크 그리기 (OpenCV 함수 유지)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                   mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                   )               

    # 정보 박스 그리기 (OpenCV 함수 유지)
    box_width = 300 # 약간 늘림
    cv2.rectangle(image, (0, 0), (box_width, 75), (245, 117, 16), -1)
    cv2.putText(image, 'REPS', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, str(counter), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'STAGE', (115, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA) # 위치 조정
    cv2.putText(image, stage if stage else "READY", (115, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA) # 위치 조정
    cv2.putText(image, 'SIDE', (215, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA) # 위치 조정
    cv2.putText(image, active_side if active_side else "N/A", (215, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA) # 위치 조정

    # 피드백 표시
    feedback_y_start = 95 
    line_height = KOREAN_FONT_SIZE + 10 # 폰트 크기에 맞춰 줄 간격 조정

    # 피드백 영역 배경 (반투명) - 약간 넓게
    feedback_area_height = 30 + len(feedback_list) * line_height 
    if feedback_area_height > 30 : 
        overlay = image.copy()
        cv2.rectangle(overlay, (0, feedback_y_start - 25), (box_width + 50, feedback_y_start + feedback_area_height - 20), (50, 50, 50), -1) 
        alpha = 0.6 
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
    cv2.putText(image, 'FEEDBACK', (15, feedback_y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # ==================== 수정된 부분: Pillow로 한글 피드백 그리기 ====================
    if korean_font: # 폰트 로드 성공 시
        # OpenCV 이미지를 Pillow 이미지로 변환 (한 번만)
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        if not feedback_list and active_side: 
             # 'Good form!' 메시지 그리기
             msg = "Good form!"
             text_color_rgb = (0, 255, 0) # Green
             draw.text((15, feedback_y_start + 10), msg, font=korean_font, fill=text_color_rgb)
        else: 
            for i, msg in enumerate(feedback_list):
                is_warning = "측면" in msg or "감지" in msg or "오류" in msg or "거리" in msg or "계산" in msg or "몸 전체" in msg
                # BGR -> RGB 순서로 변경
                color_rgb = (255, 180, 0) if is_warning else (255, 50, 50) # 주황색 경고, 빨간색 피드백
                
                y_pos = feedback_y_start + i * line_height + 10
                draw.text((15, y_pos), msg, font=korean_font, fill=color_rgb)
                
        # Pillow 이미지를 다시 OpenCV 이미지로 변환
        image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
    else: # 폰트 로드 실패 시, 기존 OpenCV 방식으로 영문만 시도 (한글 깨짐)
         if not feedback_list and active_side: 
             cv2.putText(image, "Good form!", (15, feedback_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
         else:
             for i, msg in enumerate(feedback_list):
                is_warning = "측면" in msg or "감지" in msg or "오류" in msg or "거리" in msg or "계산" in msg
                color = (0, 180, 255) if is_warning else (50, 50, 255) 
                # OpenCV는 한글 깨짐
                cv2.putText(image, msg, (15, feedback_y_start + i*25 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA) 
    # ===============================================================================

    return image