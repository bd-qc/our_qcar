import os
import numpy as np
import cv2
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.environment_outdoors import QLabsEnvironmentOutdoors

def filter_yellow_lane(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lower_yellow = np.array([25, 150, 100])
    upper_yellow = np.array([35, 240, 255])
    mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    cv2.imshow("Camera", mask)
    return mask

def filter_sidewalk(image, road_color_bgr=[217, 212, 217]):
    image_float = image.astype(np.float32)
    color_distance = np.linalg.norm(image_float - np.array(road_color_bgr, dtype=np.float32), axis=2)
    sidewalk_mask = (color_distance > 60).astype(np.uint8) * 255

    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    yellow_mask = cv2.inRange(hls, np.array([10, 50, 50]), np.array([40, 255, 255]))
    sidewalk_mask = cv2.bitwise_and(sidewalk_mask, cv2.bitwise_not(yellow_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sidewalk_mask = cv2.morphologyEx(sidewalk_mask, cv2.MORPH_CLOSE, kernel)
    sidewalk_mask = cv2.morphologyEx(sidewalk_mask, cv2.MORPH_OPEN, kernel)
    return sidewalk_mask

def estimate_lane_center(yellow_mask, sidewalk_mask, height):
    row_fractions = [0.65, 0.75, 0.85]
    centers = []

    for frac in row_fractions:
        row = int(height * frac)
        yellow_row = yellow_mask[row, :]
        sidewalk_row = sidewalk_mask[row, :]
        yellow_idx = np.where(yellow_row > 200)[0]
        sidewalk_idx = np.where(sidewalk_row > 200)[0]

        if len(yellow_idx) > 0 and len(sidewalk_idx) > 0:
            left = int(np.mean(yellow_idx))
            right = int(np.mean(sidewalk_idx))
            center = (left + right) // 2
        elif len(yellow_idx) > 0:
            left = int(np.mean(yellow_idx))
            center = left + 60  # assume lane width
        elif len(sidewalk_idx) > 0:
            right = int(np.mean(sidewalk_idx))
            center = right - 60
        else:
            continue  # no features, skip
        centers.append((center, row))

    if len(centers) >= 2:
        pts = np.array(centers)
        x, y = pts[:, 0], pts[:, 1]
        fit     = np.polyfit(y, x, deg=2)
        target_row = int(height * 0.8)
        predicted_center = int(np.polyval(fit, target_row))
        return predicted_center, fit
    return None, None

def move(car, currentSpeed, goalSpeed, target_center_x, frame):
    frame_center = frame.shape[1] // 2
    deviation = target_center_x - frame_center
    steering_gain = 0.1
    steering_angle = np.clip(deviation * steering_gain, -60, 60)

    if not hasattr(move, "prev_angle"):
        move.prev_angle = 0
    alpha = 0.3
    steering_angle = alpha * steering_angle + (1 - alpha) * move.prev_angle
    move.prev_angle = steering_angle

    speed_gain = 0.1
    currentSpeed += speed_gain * (goalSpeed - currentSpeed)
    currentSpeed = max(currentSpeed, 0)

    car.set_velocity_and_request_state_degrees(currentSpeed, steering_angle, True, False, False, currentSpeed < goalSpeed, False)
    return currentSpeed, steering_angle

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    qlabs = QuanserInteractiveLabs(); cv2.startWindowThread()
    if not qlabs.open("localhost"):
        print("Unable to connect to QLabs")
        return

    qlabs.destroy_all_spawned_actors()
    currentSpeed = 0
    goalSpeed = 5

    env = QLabsEnvironmentOutdoors(qlabs)
    car = QLabsQCar2(qlabs)
    car.spawn([22.349, 15.945, 0.324], [0, 0, -30], [1, 1, 1], waitForConfirmation=True)
    car.set_led_strip_uniform([199, 21, 133], True)
    car.possess()

    while True:
        success, image = car.get_image(4)
        if not success:
            continue

        height, width = image.shape[:2]
        yellow_mask = filter_yellow_lane(image)
        sidewalk_mask = filter_sidewalk(image)

        lane_center, curve_fit = estimate_lane_center(yellow_mask, sidewalk_mask, height)

        if lane_center is None:
            # Fallback: follow center of the black road region
            road_mask = cv2.bitwise_not(sidewalk_mask)
            row = int(height * 0.8)
            black_indices = np.where(road_mask[row, :] > 200)[0]
            if len(black_indices) > 0:
                lane_center = int(np.mean(black_indices))
            else:
                lane_center = width // 2

        # Draw lookahead points from curve
        if curve_fit is not None:
            for frac in [0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
                y = int(height * frac)
                x = int(np.polyval(curve_fit, y))
                cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
                
        currentSpeed, steering_angle = move(car, currentSpeed, goalSpeed, lane_center, image)

        cv2.imshow("Camera", image)
        cv2.imshow("Camera 2", yellow_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main()
