
import cv2
import mediapipe as mp
import time

MODEL_PATH = "face_landmarker.task"

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

with FaceLandmarker.create_from_options(options) as landmarker:
    print("CONTEXTRA: Live face detection started!")
    print("Press Q to quit.")

    start_time = time.monotonic()
    previous_timestamp = -1

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

        if result.face_landmarks:
            for landmark in result.face_landmarks[0]:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            status = "FACE DETECTED"
            color = (0, 255, 0)
        else:
            status = "NO FACE DETECTED"
            color = (0, 0, 255)

        cv2.putText(
            frame,
            "CONTEXTRA | " + status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.imshow("CONTEXTRA - Live Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()