import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance as dist
import pygame  # Import pygame for alarm sound
import threading
from collections import deque  # For blink rate calculation

# Initialize pygame mixer with larger buffer
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
alarm_sound = pygame.mixer.Sound("alarm.wav")  # Preload sound

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Eye and Mouth landmarks for EAR and MAR calculation
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 17, 18, 81, 311]  # Define mouth landmark indices

# Function to calculate EAR
def calculate_ear(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])  # Vertical distance 1
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])  # Vertical distance 2
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])  # Horizontal distance
    EAR = (A + B) / (2.0 * C)
    return EAR

# Function to calculate MAR (updated formula)
def calculate_mar(mouth_landmarks):
    # Vertical distances
    CD = dist.euclidean(mouth_landmarks[1], mouth_landmarks[5])  # Distance between points 14 and 18
    EF = dist.euclidean(mouth_landmarks[2], mouth_landmarks[4])  # Distance between points 17 and 81
    GH = dist.euclidean(mouth_landmarks[0], mouth_landmarks[3])  # Distance between points 13 and 311

    # Horizontal distance
    AB = dist.euclidean(mouth_landmarks[0], mouth_landmarks[3])  # Distance between points 13 and 14

    # MAR calculation
    MAR = (CD + EF + GH) / (3.0 * AB)
    return MAR

# Function to play alarm
def play_alarm():
    if not pygame.mixer.get_busy():  # Ensure only one instance plays
        threading.Thread(target=alarm_sound.play, daemon=True).start()

# Function to calculate blink rate
def calculate_blink_rate(ear_history, blink_thresh):
    blinks = 0
    for i in range(1, len(ear_history)):
        if ear_history[i] < blink_thresh and ear_history[i - 1] >= blink_thresh:
            blinks += 1
    blink_rate = blinks * 2  # Assuming 30 FPS, 30 frames = 1 second
    return blink_rate

# Thresholds and constants
EAR_THRESH = 0.2  # Threshold for EAR to detect closed eyes
MAR_THRESH = 0.43  # Updated MAR threshold for yawning detection (as per the file)
BLINK_RATE_THRESH = 6  # Blinks per minute threshold for drowsiness
WAIT_TIME = 2  # Time in seconds for drowsiness
D_TIME = 0  # Initialize cumulative drowsiness time
ALARM_PLAYING = False  # To prevent overlapping alarm sounds

# For blink rate calculation
EAR_HISTORY = deque(maxlen=30)  # Store last 30 frames (1 second at 30 FPS)

# For dynamic MAR threshold
BASELINE_MAR = None
MAR_HISTORY = deque(maxlen=30)  # Store last 30 frames for MAR baseline
YAWN_DURATION_THRESH = 1.5  # Yawning duration threshold (in seconds)

cap = cv2.VideoCapture(2)
t1 = 0
yawn_start_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Unable to read frame. Exiting...")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = [(int(face_landmarks.landmark[pt].x * w), int(face_landmarks.landmark[pt].y * h)) for pt in LEFT_EYE]
            right_eye = [(int(face_landmarks.landmark[pt].x * w), int(face_landmarks.landmark[pt].y * h)) for pt in RIGHT_EYE]
            mouth = [(int(face_landmarks.landmark[pt].x * w), int(face_landmarks.landmark[pt].y * h)) for pt in MOUTH]

            # Live feed frontend eyes and mouth landmarks
            # for point in left_eye + right_eye + mouth:
            #     cv2.circle(frame, point, 2, (255, 0, 0), -1)

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(mouth)

            # Add EAR to history for blink rate calculation
            EAR_HISTORY.append(avg_ear)

            # Calculate blink rate
            blink_rate = calculate_blink_rate(EAR_HISTORY, EAR_THRESH)
            if blink_rate > BLINK_RATE_THRESH:
                cv2.putText(frame, "HIGH BLINK RATE DETECTED!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                play_alarm()

            # Dynamic MAR threshold based on baseline
            if BASELINE_MAR is None:
                MAR_HISTORY.append(mar)
                if len(MAR_HISTORY) == 30:  # After 1 second, set baseline
                    BASELINE_MAR = np.mean(MAR_HISTORY)
                    dynamic_mar_thresh = BASELINE_MAR + 0.05 # Set threshold slightly above normal MAR
            else:
                dynamic_mar_thresh = BASELINE_MAR + 0.05  # Adjust threshold dynamically

                # Yawning detection with duration
                if mar > dynamic_mar_thresh:
                    if yawn_start_time == 0:
                        yawn_start_time = time.time()  # Start timing/counting
                    else:
                        yawn_duration = time.time() - yawn_start_time
                        if yawn_duration >= YAWN_DURATION_THRESH:
                            cv2.putText(frame, "PROLONGED YAWNING DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            play_alarm()
                else:
                     yawn_start_time = 0


            if avg_ear < EAR_THRESH:
                if t1 == 0:
                    t1 = time.time()
                else:
                    t2 = time.time()
                    T = t2 - t1
                    D_TIME += T
                    t1 = t2

                if D_TIME >= WAIT_TIME:
                    cv2.putText(frame, "DROWSINESS DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if not ALARM_PLAYING:
                        ALARM_PLAYING = True
                        play_alarm()
            else:
                D_TIME = 0
                t1 = 0
                ALARM_PLAYING = False

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(frame, f"Blink Rate: {blink_rate:.2f}", (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()