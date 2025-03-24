항상 세부 정보 표시

복사
# Save markdown file summarizing feedback items and detection conditions

feedback_markdown = """
# 🧠 스쿼트 피드백 항목 정의 및 감지 조건 정리

StanLab | AI 기반 자세 분석 피드백 설계  
(2025-03 기준)

---

## ✅ 피드백 항목 분류

스쿼트 자세 분석에서 사용자에게 제공할 수 있는 피드백을  
중요도에 따라 분류하고, 각 항목에 대한 감지 조건을 정의함.

---

### 🔥 1순위: 핵심 동작 피드백

| 항목명         | 설명                          | 감지 조건 예시 |
|----------------|-------------------------------|----------------|
| **깊이 부족**   | 충분히 앉지 않음               | `angle > 130` 상태가 N프레임 이상 지속 |
| **올라오지 않음** | 위로 복귀 미흡                | `angle < 150` 상태가 유지되며 up 전환 없음 |
| **흐름 이상**    | 내려간 뒤 또 내려가는 패턴      | `direction == "down"` 상태에서 down 조건 재진입 |

---

### ⚠️ 2순위: 자세(Form) 피드백

| 항목명         | 설명                          | 감지 조건 예시 |
|----------------|-------------------------------|----------------|
| **허리 숙임 과다** | 상체가 너무 숙여짐             | `shoulder.y > hip.y` 또는 `shoulder-hip` 각도 < 30° |
| **무릎 정렬 문제** | 무릎이 안쪽/바깥쪽으로 치우침  | `abs(knee.x - hip.x) > threshold` |
| **다리 균형 문제** | 좌우 움직임 불균형             | `abs(left_leg_angle - right_leg_angle) > threshold` |

---

### 🧭 3순위: 흐름/보조 피드백

| 항목명         | 설명                          | 감지 조건 예시 |
|----------------|-------------------------------|----------------|
| **속도 너무 빠름** | 프레임 간 angle 변화 급격함     | `abs(angle - prev_angle) > X` |
| **멈춤/정지**    | 일정 시간 동안 움직임 없음       | `angle 변화량 < 2°` over `n` frames |
| **시선 정면 문제** | 측면 촬영 아닌 정면 시도         | `nose.x` 중앙, `ear` 거리 작음 |
| **landmark 인식 실패** | 포즈 인식 실패 또는 landmark 누락 | `landmark.visibility < 0.5` 프레임 반복 |

---

## ✅ 예시: Python 감지 조건 매핑 구조

```python
feedback_rules = [
  {
    "name": "깊이 부족",
    "condition": lambda angle, *_: angle > 130,
    "message": "조금 더 앉아보세요!"
  },
  {
    "name": "올라오지 않음",
    "condition": lambda angle, direction: angle < 150 and direction == "down",
    "message": "끝까지 올라와주세요!"
  },
  {
    "name": "허리 숙임 과다",
    "condition": lambda hip_y, shoulder_y: shoulder_y > hip_y,
    "message": "허리를 더 세워주세요!"
  },
]