import cv2
import numpy as np

class RealTimeProcessor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        # Initialize modules
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load MobileNet SSD for Object Detection
        self.net = cv2.dnn.readNetFromCaffe(
            "models/MobileNetSSD_deploy.prototxt", 
            "models/MobileNetSSD_deploy.caffemodel"
        )
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
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
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                self.net.setInput(blob)
                detections = self.net.forward()

                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        label = f"{self.CLASSES[idx]}: {confidence * 100:.2f}%"
                        cv2.rectangle(output, (startX, startY), (endX, endY), self.COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(output, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

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