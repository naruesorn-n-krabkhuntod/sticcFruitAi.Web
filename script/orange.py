from keras.models import load_model
import cv2
import numpy as np
import time
np.set_printoptions(suppress=True)
model = load_model("model/orange.h5", compile=False)
class_names = ["good", "bad", "none"]
camera = cv2.VideoCapture(0)
assert camera.isOpened(), "Error reading video file"

while True:
    ret, image = camera.read()
    show = image.copy()
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]


    if index == 0:
        cv2.rectangle(show, (0,0), (100,50), (0,255,0), -1)
        cv2.imshow("Webcam Image", show)
        time.sleep(0.1)
    elif index == 1:
        cv2.rectangle(show, (0,0), (100,50), (0,0,255), -1)
        cv2.imshow("Webcam Image", show)
        time.sleep(0.1)
        
    cv2.imshow("Webcam Image", show)
    print("Class:", class_name)
    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()