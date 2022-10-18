
import cv2
from deepface import DeepFace
from rescale_frame import rescale_frame


class FaceDetection:

    def __init__(self):
        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
       # self.path = path
        #assertIsInstance(self.faceCascade)


    def run_camera(self):

        cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("cannot open webcam")

        while True:

            ret, frame = cap.read()

            result = DeepFace.analyze(frame, enforce_detection=False, actions=["emotion"])

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.faceCascade.detectMultiScale(gray, 1.1,4)

            #draw rectangles around face
            for(x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(frame, 
                        result['dominant_emotion'], 
                        (50,50),
                        font, 3,
                        (0,0,255),
                        2,
                        cv2.LINE_4)
            cv2.imshow('original video', frame)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()




def main():
    test = FaceDetection()
    test.run_camera()


if __name__ == "__main__":
    main()