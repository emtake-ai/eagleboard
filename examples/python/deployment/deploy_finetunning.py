import os
import cv2
import time
import numpy as np 
import threading
from lne_tflite import interpreter as lt

class AiMF_NPU():
    def __init__(self):
        self.cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)
        self.weight = "./LNE/Detection/kevin_body.lne"
        self.interpreter = lt.Interpreter(self.weight)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.output_details.sort(key=lambda _: np.prod(_["shape"][-3:]))
        self.output_tensors = [_["index"] for _ in self.output_details]
        self.input_shape = self.input_details[0]["shape"]
        self.height = self.input_shape[1]
        self.width = self.input_shape[2]

        # ✅ Single class only : 여기에 class 이름을 넣으세요 ["kevin", "dog", "cat"...]
        self.class_names = ["kevin", "dog", "cat"]
        self.colors = np.array([[0, 255, 0]], dtype=np.uint8)   # green box

    # -------------------------------
    # NMS functions
    # -------------------------------
    def _postprocess_iou(self, scores, classes, boxes):
        boxes_valid   = [True]*len(boxes)
        boxes_area    = [_[4]*_[5] for _ in boxes]
        score_orders  = np.argsort(scores)[::-1]

        for i, id_i in enumerate(score_orders):
            if not boxes_valid[id_i]:
                continue

            for id_j in score_orders[i+1:]:
                if (not boxes_valid[id_j]) or (classes[id_i] != classes[id_j]):
                    continue

                x0  = max(boxes[id_i][0], boxes[id_j][0])
                y0  = max(boxes[id_i][1], boxes[id_j][1])
                x1  = min(boxes[id_i][2], boxes[id_j][2])
                y1  = min(boxes[id_i][3], boxes[id_j][3])
                if (x1 <= x0 or y1 <= y0):
                    continue

                area_ixj  = (x1-x0)*(y1-y0)
                area_iuj  = boxes_area[id_i] + boxes_area[id_j] - area_ixj
                if (area_ixj >= 0.5*area_iuj):   # tightened IoU threshold
                    boxes_valid[id_j] = False

        detects = []
        for i, id_i in enumerate(score_orders):
            if i == 100:
                break
            if boxes_valid[id_i]:
                detects.append(id_i)

        return [
            np.asarray(scores )[detects],
            np.asarray(classes)[detects],
            np.asarray(boxes  )[detects]
        ]

    def postprocess_yolov7(self, predicts):
        YOLO_ANCHORS  = [
            [[116, 90], [156,198], [373,326]],    # large
            [[ 30, 61], [ 62, 45], [ 59,119]],    # medium
            [[ 10, 13], [ 16, 30], [ 33, 23]]     # small
        ]
        scores, classes, boxes  = [],[],[]

        for p, anchors in enumerate(YOLO_ANCHORS):
            if p == 0:
                sample  = 32
                GH      = self.input_shape[-3]//32
                GW      = self.input_shape[-2]//32
                nboxes  = GH*GW*3
            else:
                sample /= 2
                GH     *= 2
                GW     *= 2
                nboxes *= 4

            for boxid in range(nboxes):
                P       = predicts[p][boxid]
                score   = P[4] * P[5]   # single class only

                if score >= 0.3:
                    a     =  boxid % 3
                    xg    = (boxid//3) % GW
                    yg    = (boxid//3) // GW
                    xc    = (xg + P[0]*2 - 0.5) * sample       
                    yc    = (yg + P[1]*2 - 0.5) * sample       
                    w     = (P[2]*2)**2 * anchors[a][0]
                    h     = (P[3]*2)**2 * anchors[a][1]
                    boxes.append([
                        max(xc-w/2, 0),
                        max(yc-h/2, 0),
                        min(xc+w/2, 416),
                        min(yc+h/2, 416),
                        w,
                        h
                    ])
                    classes.append(0)   # always class 0 ("kevin")
                    scores.append(score)

        return self._postprocess_iou(scores, classes, boxes)

    # -------------------------------
    # Drawing
    # -------------------------------
    def post_draw(self, img, classes, scores, boxes, model_input_height, model_input_width):
        img = cv2.resize(img, (500, 375))
        h, w, _ = img.shape
        h_ratio = h / model_input_height
        w_ratio = w / model_input_width

        font_face = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale = 1
        font_thick = 1

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = np.ascontiguousarray(img)

        for i in range(len(classes)):
            label = f'{self.class_names[classes[i]]} {scores[i]:.2f}'
            left = max(0, int(np.round(boxes[i][0] * w_ratio)))
            top = max(0, int(np.round(boxes[i][1] * h_ratio)))
            right = min(w, int(np.round(boxes[i][2] * w_ratio)))
            bottom = min(h, int(np.round(boxes[i][3] * h_ratio)))

            color = (0, 255, 0)   # green box

            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            cv2.putText(img, label, (left, top-2), font_face, font_scale, (0,0,0),
                        font_thick, cv2.LINE_AA)

        return img

    # -------------------------------
    # Main worker
    # -------------------------------
    def work(self):
        while True:
            ret, frame = self.cap.read()
            frame_rgb = cv2.resize(frame, (416, 416))
            frame_rgb = cv2.cvtColor(frame_rgb,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0   # shift color space from BGR to RGB    
            frame_rgb = np.expand_dims(frame_rgb, axis=0).astype(np.float32)

            self.interpreter.set_tensor(self.input_details[0]['index'], frame_rgb)
            self.interpreter.invoke()
            outputs = [self.interpreter.get_tensor(_) for _ in self.output_tensors]
            predicts = [np.reshape(_, (-1, len(self.class_names)+5)) for _ in outputs]

            scores, classes, boxes = self.postprocess_yolov7(predicts)
            detected_image = self.post_draw(frame, classes, scores, boxes, 416, 416)
            cv2.imshow('d', detected_image)
            cv2.waitKey(1)

# -------------------------------
# Flask + threading
# -------------------------------
npu = AiMF_NPU()
t2 = threading.Thread(target=npu.work, daemon=True).start()

while True:
    time.sleep(1)
