# %%
import cv2
import numpy as np
from keras.models import load_model

# %%
# Load the pre-trained model
model = load_model('model_file.h5')

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dictionary to map labels to emotions
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# %%
video = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)

    for x, y, w, h in faces:
        # Extract the region of interest (face)
        sub_face_img = gray[y:y + h, x:x + w]

        # Resize and normalize the face image
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))

        # Predict emotion from the face image
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        print(label)

        # Draw rectangles around the face and label the emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release the video capture and destroy all windows
video.release()
cv2.destroyAllWindows()



