from typing import Optional

def evaluate_feedback(feedback_rules, context: dict) -> Optional[str]:
    """
    실시간 프레임 데이터(context)를 받아서
    피드백 조건이 충족되면 해당 메시지를 반환.
    우선순위(priority) 순으로 정렬 후 가장 먼저 조건 만족한 항목 출력.

    Parameters:
        feedback_rules (list): 조건/메시지 매핑 리스트
        context (dict): 실시간 데이터 (angle, direction, visibility 등)

    Returns:
        str | None: 조건 만족 시 메시지, 없으면 None
    """
    sorted_rules = sorted(feedback_rules, key=lambda r: r.get("priority", 99))

    for rule in sorted_rules:
        try:
            if rule["condition"](**context):
                return rule["message"]
        except Exception:
            continue

    return None
