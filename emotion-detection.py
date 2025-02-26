from tensorflow.keras.models import load_model
import cv2
import numpy as np
from ultralytics import YOLO

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']

face_model = YOLO("/Users/choijiwon/Downloads/best-faces.pt")
emotion_model = load_model("/Users/choijiwon/expression/model_78.h5")

cap = cv2.VideoCapture(0)
def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face/255.0
    face = np.expand_dims(face, axis = -1)
    face = np.expand_dims(face, axis = 0)
    return face

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = face_model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]
            face_processed = preprocess_face(face)

            emotion_prediction = emotion_model.predict(face_processed)
            emotion_label = emotion_labels[np.argmax(emotion_prediction)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Real time emotion detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()