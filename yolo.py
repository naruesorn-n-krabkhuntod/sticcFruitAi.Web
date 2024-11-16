# import module
import cv2, time
import numpy as np
from ultralytics import YOLO


# setup
print("setup project")
class_names = ["helmet", "no helmet"]
np.set_printoptions(suppress=True)



# load model
print("loading... model")
model = YOLO("yolov8n.pt")


# load source
print("import source")
cap =cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"

# main loop
print("start detection loop")
while cap.isOpened():
    success, im1 = cap.read()
    if not success or cv2.waitKey(1) == 27 : break


    image = cv2.resize(im1, (300, 300), interpolation=cv2.INTER_AREA)
    tracks = model.track(image, persist=False, show=True, verbose=False)

    
    # if tracks[0].boxes.xywh.cpu() is not None:
    #     for box, clss, id, conf in zip(tracks[0].boxes.xywh.cpu(), tracks[0].boxes.cls.cpu().tolist(), tracks[0].boxes.id.int().cpu().tolist(), tracks[0].boxes.conf.tolist()):
    #         x1, y1, x2, y2 = int(box[0])-int(box[2] //2), int(box[1])-int(box[3]//2), int(box[0])+int(box[2] //2), int(box[1])+int(box[3]//2)
    #         center_x = np.clip(((x1 + x2) // 2), 1, 1279)
    #         center_y = np.clip((y2), 1, 719)
    #         center_color = im_detect[center_y, center_x]
cap.release()
cv2.destroyAllWindows()