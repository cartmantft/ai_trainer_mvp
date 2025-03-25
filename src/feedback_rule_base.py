from abc import ABC, abstractmethod

class FeedbackInput:
    def __init__(self, angle, direction, angle_buffer, knee_y, hip_y,
                 knee_x, ankle_x, shoulder_y, shoulder_x, hip_x,
                 l_ankle_x=None, r_ankle_x=None, was_down_before=False):
        self.angle = angle
        self.direction = direction
        self.angle_buffer = angle_buffer
        self.knee_y = knee_y
        self.hip_y = hip_y
        self.knee_x = knee_x
        self.ankle_x = ankle_x
        self.shoulder_y = shoulder_y
        self.shoulder_x = shoulder_x
        self.hip_x = hip_x
        self.l_ankle_x = l_ankle_x
        self.r_ankle_x = r_ankle_x
        self.was_down_before = was_down_before 

class FeedbackRule(ABC):
    def __init__(self, name, priority, message):
        self.name = name
        self.priority = priority
        self.message = message

    @abstractmethod
    def check(self, input: FeedbackInput) -> bool:
        pass