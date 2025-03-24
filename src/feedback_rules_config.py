# 스쿼트 실시간 피드백 조건 → 메시지 매핑 구조 예시 (개선 버전)

feedback_rules = [
    {
        "name": "깊이 부족",
        "priority": 1,
        "condition": lambda angle, direction, **kwargs: angle > 130 and direction == "down",
        "message": "조금 더 앉아보세요!"
    },
    {
        "name": "완전히 올라오지 않음",
        "priority": 1,
        "condition": lambda angle, direction, angle_buffer, **kwargs:
            direction == "up"
            and angle < 150
            and len(angle_buffer) >= 3
            and max(angle_buffer) - min(angle_buffer) < 2,
        "message": "완전히 올라오세요!"
    },
    # ... (기타 피드백은 그대로 유지)
]