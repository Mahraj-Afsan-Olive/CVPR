import cv2 as cv
print(cv.__version__)
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

model = load_model('C:/Users/HP/OneDrive/Documents/Desktop/cvprmid/face_recognition_model.keras')

# Load class labels from the pickle file
class_names =['AZIZ', 'MAHEDI', 'OLIVE', 'ZAFRUL']

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define padding percentage
padding_ratio = 0.2

# Initialize webcam
webcam = cv.VideoCapture(0)

# Queue for smoothing predictions
prediction_queue = deque(maxlen=5)

while True:
    # Capture frame-by-frame
    ret, frame = webcam.read()

    # Convert the frame to grayscale for face detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)  # Normalize lighting

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(50, 50))

    # Loop over detected faces
    for (x, y, w, h) in faces:
        # Calculate padding
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)

        # Expand the bounding box
        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w + pad_x, frame.shape[1])
        y2 = min(y + h + pad_y, frame.shape[0])

        # Extract the face
        face_img = frame[y1:y2, x1:x2]

        # Preprocess the face image
        face_img_resized = cv.resize(face_img, (256, 256))
        face_img_array = np.expand_dims(face_img_resized / 255.0, axis=0)

        # Get predictions
        predictions = model.predict(face_img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        # Add to the queue for smoothing
        prediction_queue.append(predicted_class)
        most_common_prediction = max(set(prediction_queue), key=prediction_queue.count)

        if confidence > 0.8:
            class_name = class_names[most_common_prediction]
            color = (0, 255, 0)  # Green
        else:
            class_name = 'No Prediction'
            color = (0, 0, 255)  # Red

        # Draw the bounding box and label
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(frame, f'{class_name} ({confidence:.2f})', (x1, y1 - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the video frame
    cv.imshow('Webcam Face Recognition', frame)

    # Break if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv.destroyAllWindows()