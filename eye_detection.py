
import cv2
import mediapipe as mp
import time
import math

MODEL_PATH = "face_landmarker.task"

# Six landmarks around each eye:
# [outer, upper-outer, upper-inner, inner, lower-inner, lower-outer]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.20
CLOSED_SECONDS = 1.5

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def eye_aspect_ratio(landmarks, indices, width, height):
    points = []
    for index in indices:
        lm = landmarks[index]
        points.append((lm.x * width, lm.y * height))

    vertical_1 = distance(points[1], points[5])
    vertical_2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit(1)

closed_start = None
start_time = time.monotonic()
previous_timestamp = -1

with FaceLandmarker.create_from_options(options) as landmarker:
    print("CONTEXTRA: Eye-closure estimation started!")
    print("Press Q to quit.")

    while True:
        success, frame = camera.read()
        if not success:
            print("Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int((time.monotonic() - start_time) * 1000)
        timestamp = max(timestamp, previous_timestamp + 1)
        previous_timestamp = timestamp

        result = landmarker.detect_for_video(mp_image, timestamp)

        status = "NO FACE - CHECK CAMERA"
        color = (0, 0, 255)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_ear = eye_aspect_ratio(
                landmarks, LEFT_EYE, width, height
            )
            right_ear = eye_aspect_ratio(
                landmarks, RIGHT_EYE, width, height
            )
            ear = (left_ear + right_ear) / 2.0

            # Mark the eye landmarks.
            for index in LEFT_EYE + RIGHT_EYE:
                lm = landmarks[index]
                x = int(lm.x * width)
                y = int(lm.y * height)
                cv2.circle(frame, (x, y), 3, (255, 200, 0), -1)

            now = time.monotonic()

            if ear < EAR_THRESHOLD:
                if closed_start is None:
                    closed_start = now

                duration = now - closed_start

                if duration >= CLOSED_SECONDS:
                    status = "WARNING: PROLONGED EYE CLOSURE"
                    color = (0, 0, 255)
                else:
                    status = "EYES MAY BE CLOSED"
                    color = (0, 165, 255)
            else:
                closed_start = None
                status = "EYES OPEN"
                color = (0, 255, 0)

            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        cv2.putText(
            frame,
            status,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        cv2.imshow("CONTEXTRA | Eye Closure Prototype", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()