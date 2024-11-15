from flask import Flask, jsonify, render_template, request
import os
from keras.models import load_model
import cv2
import numpy as np
import time
np.set_printoptions(suppress=True)
orange_model = load_model("model/orange.h5", compile=False)
class_names = ["good", "bad", "none"]
camera = cv2.VideoCapture(0)
assert camera.isOpened(), "Error reading video file"


app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/orange', methods=['GET'])
def get_tasks():
    # os.system("C:/ProgramData/anaconda3/envs/conda-env/python.exe script/orange.py")
    startDetector(orange_model)
    return jsonify({'tasks': "success"})



def startDetector(model):
    camera = cv2.VideoCapture(0)
    while True:
        cv2.namedWindow("sticc camera", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("sticc camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty("sticc camera", cv2.WND_PROP_TOPMOST, 1)
        ret, image = camera.read()
        show = image.copy()
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        if index == 0:
            cv2.rectangle(show, (0,0), (100,50), (0,255,0), -1)
            cv2.imshow("sticc camera", show)
            time.sleep(0.1)
        elif index == 1:
            cv2.rectangle(show, (0,0), (100,50), (0,0,255), -1)
            cv2.imshow("sticc camera", show)
            time.sleep(0.1)
        cv2.imshow("sticc camera", show)
        print("Class:", class_name)
        if cv2.waitKey(1) == 27:
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(debug=True)