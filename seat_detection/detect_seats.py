from ultralytics import YOLO
import cv2

def detect_seats():
    model = YOLO('yolov8s.pt')  # general model

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera error")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()

        cv2.imshow("YOLO Seat Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_seats()
