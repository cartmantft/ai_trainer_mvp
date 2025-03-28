# utils.py
import numpy as np
import math
import logging
import datetime
import os


def smooth_value(current_value, previous_value, weight=0.3):
    """
    이동 평균 필터를 적용하여 값을 부드럽게 만듭니다.

    Args:
        current_value (float): 현재 값.
        previous_value (float): 이전 값.
        weight (float, optional): 가중치 (0 ~ 1 사이 값). 값이 클수록 현재 값에 더 많은 비중을 둡니다. Defaults to 0.3.

    Returns:
        float: 부드러워진 값.
    """
    if previous_value is None:
        return current_value
    return (1 - weight) * previous_value + weight * current_value

# --- 각도 및 좌표 계산 함수 ---
def calculate_angle(a, b, c):
    """세 점 사이의 각도를 계산하는 함수 (a, b, c는 [x, y] 좌표)"""
    if a is None or b is None or c is None: return None
    try:
        a_np, b_np, c_np = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c_np[1] - b_np[1], c_np[0] - b_np[0]) - \
                  np.arctan2(a_np[1] - b_np[1], a_np[0] - b_np[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0: angle = 360 - angle
        return angle
    except Exception as e: 
        logging.error(f"Error calculating angle: {e}", exc_info=False)
        return None

def calculate_vertical_angle(a, b):
    """두 점을 잇는 선과 수직선(아래방향) 사이 각도 계산"""
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
    except Exception as e: 
        logging.error(f"Error calculating vertical angle: {e}", exc_info=False)
        return None

# --- 랜드마크 좌표 추출 함수 ---
def get_landmark_coords(landmarks, landmark_enum, visibility_threshold):
    """랜드마크 좌표 추출 및 로깅 (선택적)"""
    logger = logging.getLogger(__name__) # 함수 내에서 로거 가져오기
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

# --- 로깅 설정 함수 ---
def setup_logging():
    """로깅 설정을 초기화하고 로거 객체를 반환"""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_filename = f"ai_trainer_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger() # 루트 로거 가져오기
    logger.setLevel(logging.DEBUG) 
    
    # 기존 핸들러 제거 (중복 로깅 방지)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 파일 핸들러
    try:
        logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(logs_dir, log_filename), encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error creating file handler: {e}") 

    # 콘솔 핸들러
    try:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG) 
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)
    except Exception as e:
        print(f"Error creating console handler: {e}") 
        
    logger.info("===== AI Trainer MVP Logging Start (UTF-8 Enabled) =====")
    return logger