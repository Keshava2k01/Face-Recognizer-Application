import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model('vggface_model.h5')

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the LabelEncoder object
le = LabelEncoder()
le.classes_ = np.load('label_encoder_classes.npy')

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = frame[y:y+h, x:x+w]

        # Resize the face ROI to match the input size of the model
        face_roi_resized = cv2.resize(face_roi, (224, 224))

        # Preprocess the face ROI
        img = image.img_to_array(face_roi_resized)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Predict the label of the face
        prediction = model.predict(img)
        person = le.inverse_transform([np.argmax(prediction)])[0]

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, person, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
