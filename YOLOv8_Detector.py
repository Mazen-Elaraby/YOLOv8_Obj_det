import numpy as np
import os
from time import time
import cv2
import torch
import ultralytics
from ultralytics import YOLO
import supervision as sv


class Detector:
    def __init__(self, capture_index):

        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        
        self.CLASS_NAMES_DICT = self.model.model.names
    
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=2, text_thickness=1, text_scale=0.5)
        
    def load_model(self):

        model = YOLO("yolov8m.pt")  
        model.fuse()

        return model
    
    def predict(self, frame):
    
        results = self.model(frame)
        
        return results
    
    def plot_bboxes(self, results, frame):
    
        # Extract & Setup detections for visualization
        detections = sv.Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )
        
        # Format labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _ in detections]
        
        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        
        return frame
    
    def __call__(self):

        if self.capture_index:
            video_path = os.path.join('.', 'data', 'people.mp4')
            video_out_path = os.path.join('.', 'out.mp4')
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
                          (width, height))
        else:
            #Real-time Webcam stream
            cap = cv2.VideoCapture(self.capture_index)
            assert cap.isOpened()
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            
            start_time = time()
            
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
                
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv8 Detection', frame)

            if self.capture_index:
                cap_out.write(frame) #write to file

            if cv2.waitKey(5) & 0xFF == 27: #27 -> escape key
                
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        
    

ultralytics.checks()
detector = Detector(capture_index=1)
detector()
