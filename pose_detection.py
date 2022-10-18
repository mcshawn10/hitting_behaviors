from hashlib import new
import cv2
import mediapipe as mp
import numpy as np



class PoseDetection:
    def __init__(self, path):
        self.path = path
    
    def rescale_frame(self, frame, percent):
      width = int(frame.shape[1] * percent/100)
      height = int(frame.shape[0] * percent/100)
      dim = (width, height)
      return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def run_camera(self):
      mp_drawing = mp.solutions.drawing_utils
      mp_drawing_styles = mp.solutions.drawing_styles
      mp_pose = mp.solutions.pose
#mp_faceMesh = mp.solutions.face_mesh

      cap = cv2.VideoCapture(self.path)
      # width  = int(cap.get(3))  # float `width`
      # height = int(cap.get(4))  # float `height`
      
      # new_width = width//2
      # new_height = height//2

      
      
      
      
      assert cap.isOpened()
      with mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
          success, image = cap.read()
          if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
          image = self.rescale_frame(image, 50)
          # To improve performance, optionally mark the image as not writeable to
          # pass by reference.
          image.flags.writeable = False
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          results = pose.process(image)

          # Draw the pose annotation on the image.
          image.flags.writeable = True
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          mp_drawing.draw_landmarks(
              image,
              results.pose_landmarks,
              mp_pose.POSE_CONNECTIONS,
              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
          # Flip the image horizontally for a selfie-view display.
          cv2.imshow('MediaPipe Pose', image) #cv2.flip(image, 1)
          if cv2.waitKey(5) & 0xFF == 27:
            break
        cap.release()
        cv2.destroyAllWindows()


def main():
  test = PoseDetection(r"video_data/back_hit.mp4")
  test.run_camera()      
if __name__ == "__main__":
  main()