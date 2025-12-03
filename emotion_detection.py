import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Define emotion labels (adjust these based on your model's training)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the pre-trained emotion detection model
emotion_model = load_model('face_model.h5')

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit the application")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract the face ROI (Region of Interest)
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize face to 48x48 (standard size for emotion models)
        face_roi = cv2.resize(face_roi, (48, 48))
        
        # Normalize pixel values
        face_roi = face_roi.astype('float32') / 255.0
        
        # Convert to array and expand dimensions
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)
        
        # Make prediction
        emotion_prediction = emotion_model.predict(face_roi, verbose=0)
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]
        
        # Display the emotion label on the frame
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)
        
        # Optional: Display confidence score
        confidence = np.max(emotion_prediction) * 100
        cv2.putText(frame, f"{confidence:.1f}%", (x, y+h+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
