import torch
import cv2

model = torch.hub.load("yolov5", "yolov5n", source="local")
cap = cv2.VideoCapture('video.mp4')


while True:
    img = cap.read()[1]
    if img is None:
        break
    result = model(img)
    data_frame = result.pandas().xyxy[0]
    indexes = data_frame.index
    for index in indexes:
        x1 = int(data_frame['xmin'][index])
        y1 = int(data_frame['ymin'][index])
        x2 = int(data_frame['xmax'][index])
        y2 = int(data_frame['ymax'][index])
        label = data_frame['name'][index]
        conf = data_frame['confidence'][index]
        text = label + ' ' + str(conf.round(decimals=2))

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

    cv2.imshow('Video', img)
    if cv2.waitKey(30) & 0xFF == ord('q'): break