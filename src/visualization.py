# visualization.py
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose # landmark 그리기 위해 필요

def draw_landmarks_and_feedback(image, results, stage, counter, active_side, feedback_list):
    """랜드마크, 정보 박스, 피드백을 이미지에 그립니다."""
    
    # 랜드마크 그리기
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                   mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                   )               

    # 정보 박스 그리기
    box_width = 280
    cv2.rectangle(image, (0, 0), (box_width, 75), (245, 117, 16), -1)
    cv2.putText(image, 'REPS', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, str(counter), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'STAGE', (105, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, stage if stage else "READY", (105, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'SIDE', (205, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, active_side if active_side else "N/A", (205, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # 피드백 표시
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
            # 화면에 한글 깨짐 이슈는 별도 처리 필요 (PIL 사용 등)
            # 여기서는 일단 영문으로 변경하거나 ?로 표시될 수 있음
            is_warning = "측면" in msg or "감지" in msg or "오류" in msg or "거리" in msg or "계산" in msg
            color = (0, 180, 255) if is_warning else (50, 50, 255) 
            try:
                # 간단히 ascii 로 변환 시도, 실패 시 ? 처리 (임시 방편)
                msg_display = msg.encode('ascii', 'replace').decode('ascii') 
            except:
                msg_display = msg # 실패 시 원본 (깨질 수 있음)
                
            cv2.putText(image, msg_display, (15, feedback_y_start + i*25 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA) 

    return image