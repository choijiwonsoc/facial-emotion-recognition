import cv2
from ultralytics import YOLO
cap = cv2.VideoCapture(0)

model = YOLO("/Users/choijiwon/Downloads/best-faces.pt")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence_threshold = 0.5
            if box.conf > confidence_threshold:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Face", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # Display the frame
    cv2.imshow("YOLOv8 Emotion Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

