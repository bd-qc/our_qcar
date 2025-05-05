import os
import torch
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

def filter_sidewalk(image, road_bgr_lower=[215, 210, 160], road_bgr_upper=[265, 265, 265]):
    # Create a road mask by filtering within a BGR color range
    road_mask = cv2.inRange(image, np.array(road_bgr_lower), np.array(road_bgr_upper))

    # Invert to get sidewalk candidate mask (anything not road)
    sidewalk_mask = cv2.bitwise_not(road_mask)

    # Remove yellow lane lines to avoid mistaking them as sidewalk
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    yellow_mask = cv2.inRange(hls, np.array([10, 50, 50]), np.array([40, 255, 255]))
    sidewalk_mask = cv2.bitwise_and(sidewalk_mask, cv2.bitwise_not(yellow_mask))

    # Morphological operations to remove noise
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
        
        #-------------------------REMOVE OUTLIERS THAT PICK UP ON THE ROAD MASK------------------------
        sidewalk_idx = np.where(sidewalk_row > 200)[0]
        if len(sidewalk_idx) > 0:
            # Group into connected components and keep the largest one
            components = np.split(sidewalk_idx, np.where(np.diff(sidewalk_idx) > 10)[0]+1)
            largest_component = max(components, key=len)
            sidewalk_idx = largest_component
        #-------------------------REMOVE OUTLIERS THAT PICK UP ON THE ROAD MASK------------------------

        if len(yellow_idx) > 0 and len(sidewalk_idx) > 0:
            left = int(np.mean(yellow_idx))
            right = int(np.mean(sidewalk_idx))
            center = ((left + 150) + (right+ 40)) // 2
            print("both")
        elif len(yellow_idx) > 0:
            left = int(np.mean(yellow_idx))
            center = left + 180  # assume lane width
            print("yellow")
        elif len(sidewalk_idx) > 0:
            right = int(np.mean(sidewalk_idx))
            center = right + 20
            print("sidewalk")
        else:
            print("none")
            continue  # no features, skip
        centers.append((center, row))

    if len(centers) >= 2:
        pts = np.array(centers)
        x, y = pts[:, 0], pts[:, 1]
        fit = np.polyfit(y, x, deg=2)
        target_row = int(height * 0.869)
        predicted_center = int(np.polyval(fit, target_row))
        return predicted_center, fit
    return None, None

def move(car, currentSpeed, goalSpeed, target_center_x, frame):
    frame_center = frame.shape[1] // 2
    deviation = target_center_x - frame_center

    # Reference speed where current settings work best
    reference_speed = 5.0

    # --- Dynamic Steering Gain ---
    base_gain = 0.123  # good at 5 m/s
    min_gain = 0.07    # softer at low speeds
    max_gain = 0.13    # slightly softer at very high speeds

    speed_ratio = np.clip(currentSpeed / reference_speed, 0, 2)

    if speed_ratio <= 1:
        # Slower than 5 m/s → reduce gain linearly
        steering_gain = min_gain + (base_gain - min_gain) * speed_ratio
    else:
        # Faster than 5 m/s → decay gain slightly to avoid oversteering
        steering_gain = base_gain - (base_gain - max_gain) * (speed_ratio - 1)

    steering_angle = np.clip(deviation * steering_gain, -60, 60)

    # --- Dynamic Smoothing ---
    base_alpha = 0.3  # good at 5 m/s
    min_alpha = 0.15  # more reactive at low speeds
    max_alpha = 0.5   # more stable at high speeds

    if speed_ratio <= 1:
        alpha = min_alpha + (base_alpha - min_alpha) * speed_ratio
    else:
        alpha = base_alpha + (max_alpha - base_alpha) * min(speed_ratio - 1, 1)

    # Smooth steering
    if not hasattr(move, "prev_angle"):
        move.prev_angle = 0
    steering_angle = alpha * steering_angle + (1 - alpha) * move.prev_angle
    move.prev_angle = steering_angle

    # --- Speed control ---
    speed_gain = 0.1
    currentSpeed += speed_gain * (goalSpeed - currentSpeed)
    currentSpeed = max(currentSpeed, 0)

    car.set_velocity_and_request_state_degrees(
        currentSpeed, steering_angle,
        False, False, False,
        currentSpeed < goalSpeed, False
    )

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
    car.spawn([2.675, 15.458, 0.464], [0, 0, 80], [1, 1, 1], waitForConfirmation=True)
    car.set_led_strip_uniform([199, 21, 133], True)
    car.possess()

    while True:
        success, image = car.get_image(4)
        success, depth_image = car.get_image(car.CAMERA_DEPTH)

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
        cv2.imshow("Sidewalk Mask", sidewalk_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main()
