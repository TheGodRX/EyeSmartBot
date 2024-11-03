import cv2
import pygame
import numpy as np
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Smart Bot Eye")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Eye parameters
EYE_CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
EYE_RADIUS = 100
PUPIL_BASE_RADIUS = 30
BLINK_INTERVAL_MIN = 2000  # Min time between blinks in milliseconds
BLINK_INTERVAL_MAX = 5000  # Max time between blinks in milliseconds
BLINK_DURATION_MIN = 100  # Min blink duration in milliseconds
BLINK_DURATION_MAX = 300  # Max blink duration in milliseconds
DYNAMIC_PUPIL_MOVEMENT_STEP = 0.1  # Step for pupil movement smoothness

# Eye state variables
blink_timer = 0
blink_interval = random.randint(BLINK_INTERVAL_MIN, BLINK_INTERVAL_MAX)
pupil_radius = PUPIL_BASE_RADIUS
blink_in_progress = False
current_eye_radius = EYE_RADIUS
current_pupil_offset = (0, 0)
target_pupil_offset = (0, 0)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Background subtractor for detecting movement
backSub = cv2.createBackgroundSubtractorMOG2()

# Function to draw the eye
def draw_eye(screen, eye_center, eye_radius, pupil_offset, pupil_radius):
    # Draw the white part of the eye
    pygame.draw.circle(screen, WHITE, eye_center, eye_radius)
    # Calculate pupil position
    pupil_center = (eye_center[0] + pupil_offset[0], eye_center[1] + pupil_offset[1])
    # Draw the black pupil
    pygame.draw.circle(screen, BLACK, pupil_center, pupil_radius)

# Function to handle smooth eye movement using linear interpolation
def lerp(a, b, t):
    return a + (b - a) * t

def smooth_eye_movement(current_offset, target_offset, step_size=DYNAMIC_PUPIL_MOVEMENT_STEP):
    new_x = lerp(current_offset[0], target_offset[0], step_size)
    new_y = lerp(current_offset[1], target_offset[1], step_size)
    return new_x, new_y

# Main loop
running = True
clock = pygame.time.Clock()

while running:
    delta_time = clock.get_time()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update blink timer
    blink_timer += delta_time
    if blink_timer >= blink_interval and not blink_in_progress:
        blink_duration = random.randint(BLINK_DURATION_MIN, BLINK_DURATION_MAX)
        blink_timer = 0
        blink_interval = random.randint(BLINK_INTERVAL_MIN, BLINK_INTERVAL_MAX)
        blink_in_progress = True
        blink_start_time = pygame.time.get_ticks()

    # Handle blinking
    if blink_in_progress:
        elapsed_blink_time = pygame.time.get_ticks() - blink_start_time
        if elapsed_blink_time < blink_duration:
            # Simulate eye closing by reducing the eye radius
            current_eye_radius = EYE_RADIUS * (1 - elapsed_blink_time / blink_duration)
            pupil_radius = max(0, PUPIL_BASE_RADIUS * (1 - elapsed_blink_time / blink_duration))
        else:
            current_eye_radius = EYE_RADIUS  # Restore eye radius
            pupil_radius = PUPIL_BASE_RADIUS  # Restore pupil size
            blink_in_progress = False

    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera.")
        continue

    # Apply background subtraction to detect movement
    fg_mask = backSub.apply(frame)
    fg_mask = cv2.medianBlur(fg_mask, 5)
    _, thresh = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

    # Find contours of the detected moving objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize target point for pupil movement
    target_pupil_offset = (0, 0)
    if contours:
        # Get the largest contour to follow
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        target_center = (x + w // 2, y + h // 2)

        # Reverse the x-axis to make pupil look in the opposite direction
        target_pupil_offset = (
            -(target_center[0] - EYE_CENTER[0]) / 5,  # Inverted for opposite direction
            (target_center[1] - EYE_CENTER[1]) / 5
        )

    # Update current pupil offset using smooth movement
    current_pupil_offset = smooth_eye_movement(current_pupil_offset, target_pupil_offset)

    # Clear screen
    screen.fill(BLACK)

    # Draw the eye
    if blink_in_progress:  # If blinking, draw closed eye
        pygame.draw.rect(screen, BLACK, (EYE_CENTER[0] - current_eye_radius, EYE_CENTER[1] - current_eye_radius // 2,
                                           current_eye_radius * 2, current_eye_radius))
    else:
        draw_eye(screen, EYE_CENTER, current_eye_radius, current_pupil_offset, pupil_radius)

    # Update display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

# Release the webcam and close OpenCV
cap.release()
pygame.quit()
