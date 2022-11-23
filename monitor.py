import numpy as np
import pandas as pd
import tensorflow as tf 
import cv2 
import os
import time


def display_prediction(frame, categories, i, font):
    RED, GREEN = (0,0,255),(0,255,0)
    if i == 1:
        cv2.putText(frame, 
                    categories[i], 
                    (5,25),
                    font, 1,
                    GREEN,
                    1,
                    cv2.LINE_4)
    else:
        cv2.putText(frame, 
                    categories[i], 
                    (5,25),
                    font, 1,
                    RED,
                    1,
                    cv2.LINE_4)

def monitor(sample):
    assert os.path.exists(sample)
    EffNet = tf.keras.models.load_model("./EffNet_1")
    font = cv2.FONT_HERSHEY_SIMPLEX
    categories = ["HITTING", "none"]
    cap = cv2.VideoCapture(sample)
    
    if not cap.isOpened():
        raise IOError("cannot open webcam")

    hitting_count = 0
    
    while cap.isOpened():

        success, frame = cap.read()
        
        #assert success

        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

        input_arr = frame.astype("float32") / 255

        prediction = EffNet.predict(np.expand_dims(input_arr, axis=0))
        #cv2.imshow('original video', frame)
        #print(prediction)

        i = 0

        if prediction[0][0] <= 0.5:
            i = 0
        else: i = 1

        #if current_frame % 20 == 0:

        # cv2.putText(frame, 
        #             categories[i], 
        #             (5,25),
        #             font, 1,
        #             (0,0,255),
        #             1,
        #             cv2.LINE_4)
        display_prediction(frame, categories, i,font)
        cv2.imshow('original video', frame)
    

        

                #preds = EffNet.predict(np.expand_dims(frame, axis=0))[0]


        if cv2.waitKey(5) & 0xFF == 27:
            
            break
        
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def main():
    pass


if __name__ == "__main__":

    monitor("video_data/back_hit.mp4")