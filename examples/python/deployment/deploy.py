import numpy as np
import cv2
from lne_tflite import interpreter as lt
import threading
import time

# ==========================================
# NPU PIPELINE CLASS
# ==========================================
class AiMF_Pipeline:
    def __init__(self):
        # ---------------- Capture ----------------
        self.cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)

        # ---------------- NPU ----------------
        self.weight = "./LNE/Detection/yolov7_tiny_4x8_0_1.lne"
        
        # ---------------- Original loading weight  ----------------
        #interpreter = tf.lite.Interpreter(model_path="model.tflite")
        #interpreter.allocate_tensors()  
        
        # ---------------- actually LG npu loading weight  ----------------
        self.interpreter = lt.Interpreter(self.weight)
        self.interpreter.allocate_tensors()

        # ---------------- model information  ----------------
        
        # getting input shape $& outupt shape
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.output_details.sort(key=lambda x: np.prod(x["shape"][-3:]))
        self.output_tensors = [x["index"] for x in self.output_details]
        self.input_shape = self.input_details[0]["shape"]

        # ---------------- Labels ----------------
        with open("./ObjectDetection/labels/coco.names") as f:
            self.class_names = [c.strip() for c in f.readlines()]

        self.colors = np.random.randint(0,255,(len(self.class_names),3),dtype=np.uint8)

    # ==========================================
    # YOLO POST (그대로 유지) : Yolov7-tiny만 유효 
    # ==========================================
    def postprocess_yolov7(self, predicts):
        YOLO_ANCHORS = [
            [[116,90],[156,198],[373,326]],
            [[30,61],[62,45],[59,119]],
            [[10,13],[16,30],[33,23]]
        ]

        scores, classes = [], []
        boxes_xywh, boxes_xyxy = [], []

        num_cls = len(self.class_names)
        sample = 32
        GH = self.input_shape[-3] // 32
        GW = self.input_shape[-2] // 32
        nboxes = GH * GW * 3

        for p in range(len(YOLO_ANCHORS)):
            anchors = YOLO_ANCHORS[p]
            pred = predicts[p]

            for boxid in range(nboxes):
                P = pred[boxid]
                conf = P[4]
                if conf < 0.35: continue

                cls_scores = P[5:5+num_cls]
                classid = np.argmax(cls_scores)
                score = conf * cls_scores[classid]
                if score < 0.40: continue

                a  = boxid % 3
                xg = (boxid // 3) % GW
                yg = (boxid // 3) // GW

                xc = (xg + P[0] * 2.0 - 0.5) * sample
                yc = (yg + P[1] * 2.0 - 0.5) * sample
                w  = (P[2] * 2.0) ** 2 * anchors[a][0]
                h  = (P[3] * 2.0) ** 2 * anchors[a][1]

                x1,y1 = xc-w/2, yc-h/2
                x2,y2 = xc+w/2, yc+h/2

                boxes_xywh.append([x1,y1,w,h])
                boxes_xyxy.append([x1,y1,x2,y2])
                scores.append(float(score))
                classes.append(int(classid))

            sample //= 2
            GH *= 2; GW *= 2; nboxes *= 4

        if len(boxes_xywh) == 0:
            return [],[],[]

        idx = cv2.dnn.NMSBoxes(boxes_xywh, scores, 0.30, 0.45)
        if len(idx) == 0:
            return [],[],[]

        final_scores, final_classes, final_boxes = [],[],[]
        for i in idx.flatten():
            final_scores.append(scores[i])
            final_classes.append(classes[i])
            final_boxes.append(boxes_xyxy[i])

        return np.array(final_scores), np.array(final_classes), np.array(final_boxes)

    def draw(self, img, classes, scores, boxes):
        for i,c in enumerate(classes):
            x0,y0,x1,y1 = map(int, boxes[i])
            color = tuple(int(v) for v in self.colors[c])
            cv2.rectangle(img,(x0,y0),(x1,y1),color,1)
            cv2.putText(img,f"{self.class_names[c]} {scores[i]:.2f}",
                        (x0,y0-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)
        return img

    # ==========================================
    # Thread 1: Capture
    # ==========================================
    
    # ==========================================
    # Thread 2: NPU
    # ==========================================
    def npu_loop(self):
        while True:
            ret, frame = self.cap.read() # getting one frame from camera
            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0   # shift color space from BGR to RGB
            rgb = np.expand_dims(rgb,0) # set the one frame for input shape

            self.interpreter.set_tensor(self.input_details[0]["index"], rgb) # allocate the one frame to the NPU Memory
            self.interpreter.invoke() # run the dpu for inference

            outputs = [self.interpreter.get_tensor(i) for i in self.output_tensors] # get the output from model
            predicts = [np.reshape(o,(-1,len(self.class_names)+5)) for o in outputs] # reshape the output and get the result.
            scores, classes, boxes = self.postprocess_yolov7(predicts)
            frame = self.draw(frame, classes, scores, boxes)
            _, jpeg = cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),50])

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   jpeg.tobytes() + b"\r\n")
           
pipeline = AiMF_Pipeline()
threading.Thread(target=pipeline.npu_loop, daemon=True).start()

while True:
    time.sleep(1)
