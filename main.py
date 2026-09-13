import cv2

class RealTimeProcessor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        # Initialize modules
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # State
        self.mode = 'face' # Modes: normal, face

    def run(self):
        print("Starting Real-time Image Processor.")
        print("Press 'f' for Face Detection mode.")
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

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = RealTimeProcessor()
    app.run()