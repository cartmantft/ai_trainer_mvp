from feedback_rule_base import FeedbackRule, FeedbackInput

class IncompleteUp(FeedbackRule):
    def __init__(self):
        super().__init__("완전히 올라오지 않음", 2, "완전히 올라오세요!")

    def check(self, input: FeedbackInput) -> bool:
        return (
            input.direction == "up"
            and input.was_down_before  # ✅ 핵심 조건
            and input.angle < 150
            and len(input.angle_buffer) >= 5
            and max(input.angle_buffer) - min(input.angle_buffer) < 2
            and max(input.angle_buffer) < 145
        )

class NotDeepEnough(FeedbackRule):
    def __init__(self):
        super().__init__("깊이 부족", 2, "조금 더 앉아보세요!")

    def check(self, input: FeedbackInput) -> bool:
        return input.direction == "down" and input.angle > 110 and input.knee_y < input.hip_y

class KneeTooForward(FeedbackRule):
    def __init__(self):
        super().__init__("무릎 앞으로 나감", 1, "무릎이 너무 앞으로 나갔어요!")

    def check(self, input: FeedbackInput) -> bool:
        return abs(input.knee_x - input.ankle_x) > 0.06

class TorsoBentForward(FeedbackRule):
    def __init__(self):
        super().__init__("허리 숙임 과도", 1, "허리를 세워주세요!")

    def check(self, input: FeedbackInput) -> bool:
        return input.direction == "down" and (input.shoulder_y - input.hip_y) > -0.05

class FeetNotAligned(FeedbackRule):
    def __init__(self):
        super().__init__("발끝 정렬 안 맞음", 2, "발끝 정렬이 맞지 않아요!")

    def check(self, input: FeedbackInput) -> bool:
        return input.l_ankle_x is not None and input.r_ankle_x is not None and \
               abs(input.l_ankle_x - input.r_ankle_x) > 0.05

class HipTooFarBack(FeedbackRule):
    def __init__(self):
        super().__init__("엉덩이 뒤로 내밀기", 2, "엉덩이가 너무 뒤로 빠졌어요!")

    def check(self, input: FeedbackInput) -> bool:
        return input.direction == "down" and (input.hip_x - input.ankle_x) > 0.1