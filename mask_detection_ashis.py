import cv2
import numpy as np
import cvlib as cv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = load_model('mask_detector.model')

cap = cv2.VideoCapture(0)

classes = ['Mask', 'No Mask']

while cap.isOpened():
    success, frame = cap.read()
    face, confidence = cv.detect_face(frame)

    for idx, f in enumerate(face):

        (h, w) = frame.shape[:2]
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w-1, endX), min(h-1, endY))

        face_crop = frame[startY:endY, startX:endX]

        if face_crop.shape[0] < 10 and face_crop.shape[1] < 10:
            continue

        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_crop = cv2.resize(face_crop, (224, 224))
        face_crop = img_to_array(face_crop)
        face_crop = preprocess_input(face_crop)

        face_crop = np.array([face_crop], dtype="float32")

        conf = model.predict(face_crop)[0]

        idx = np.argmax(conf)
        label = classes[idx]

        if label == 'Mask':
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        label = "{}: {:.2f}%".format(label, conf[idx]*100)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, label, (startX, startY-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Gender Detection', frame)

    if cv2.waitkey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    
