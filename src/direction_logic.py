def determine_direction(angle, prev_angle, hip_y, knee_y, current_direction):
    DOWN_THRESHOLD = 130
    UP_THRESHOLD = 150
    MIN_HIP_Y = 0.55
    MIN_KNEE_Y = 0.60

    new_direction = current_direction

    if angle < DOWN_THRESHOLD and hip_y > MIN_HIP_Y and knee_y > MIN_KNEE_Y and current_direction == "up":
        new_direction = "down"
    elif angle > UP_THRESHOLD and current_direction == "down":
        new_direction = "up"

    return new_direction