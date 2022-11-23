import numpy as np
import pandas as pd
#import tensorflow as tf 
import cv2 
import os
import time


print(cv2.__version__)

def monitor(sample):
    assert os.path.exists(sample)
    #EffNet = tf.keras.model.load_model("./EffNet_1")
    font = cv2.FONT_HERSHEY_SIMPLEX

    cap = cv2.VideoCapture(sample)
    
    if not cap.isOpened():
        raise IOError("cannot open webcam")
    
    while cap.isOpened():

        success, frame = cap.read()
        
        #assert success

        # fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
        # print(fps)
        # assert fps >0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # duration = frame_count/fps #in seconds
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #interval = int(fps*5)
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

        cv2.imshow('original video', frame)

        '''if current frame == interval, make a prediction on the video'''
        

        if current_frame % 20 == 0:

            cv2.putText(frame, 
                        "Prediction", 
                        (25,50),
                        font, 2,
                        (0,0,255),
                        1,
                        cv2.LINE_4)

            cv2.imshow('original video', frame)
        

        

                #preds = EffNet.predict(np.expand_dims(frame, axis=0))[0]


        if cv2.waitKey(5) & 0xFF == 27:
            print("broked")
            break
        
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def main():
    pass


if __name__ == "__main__":

    monitor("video_data/back_hit.mp4")