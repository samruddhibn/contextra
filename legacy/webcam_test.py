
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Webcam could not be opened.")
    print("Check camera permissions or try another camera index.")
    raise SystemExit(1)

print("CONTEXTRA: Webcam connected!")
print("Press Q to close the webcam window.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read a webcam frame.")
        break

    cv2.putText(
        frame,
        "CONTEXTRA - LIVE CAMERA",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("CONTEXTRA | Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()