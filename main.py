import cv2
import numpy as np

class RealTimeProcessor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        # Initialize modules
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load YOLOv4-tiny for Object Detection
        self.net = cv2.dnn.readNet("models/yolov4-tiny.weights", "models/yolov4-tiny.cfg")
        
        # Determine the output layer names
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        with open("models/coco.names", "r") as f:
            self.CLASSES = [line.strip() for line in f.readlines()]
            
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        
        # State
        self.mode = 'face' # Modes: normal, face, object

    def run(self):
        print("Starting Real-time Image Processor.")
        print("Press 'f' for Face Detection mode.")
        print("Press 'o' for Object Detection mode.")
        print("Press 'n' for Normal mode.")
        print("Press 'q' to Quit.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
                
            frame = cv2.flip(frame, 1) # Mirror for natural feel
            output = frame.copy()

            # Process based on mode
            if self.mode == 'face':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                for (x, y, w, h) in faces:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    
            elif self.mode == 'object':
                (H, W) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                self.net.setInput(blob)
                layerOutputs = self.net.forward(self.ln)

                boxes = []
                confidences = []
                classIDs = []

                for output_layer in layerOutputs:
                    for detection in output_layer:
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        if confidence > 0.5:
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")

                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)

                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

                if len(idxs) > 0:
                    for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        color = [int(c) for c in self.COLORS[classIDs[i]]]
                        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
                        text = f"{self.CLASSES[classIDs[i]]}: {confidences[i]:.4f}"
                        cv2.putText(output, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display mode on screen
            cv2.putText(output, f"Mode: {self.mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('Real-time Processor', output)

            # Handle Key Presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                self.mode = 'normal'
            elif key == ord('f'):
                self.mode = 'face'
            elif key == ord('o'):
                self.mode = 'object'

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = RealTimeProcessor()
    app.run()