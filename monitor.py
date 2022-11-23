import numpy as np
import pandas as pd
import tensorflow as tf 
import cv2 
import os



def monitor(sample):

    EffNet = tf.keras.model.load_model("./EffNet_1")

    cap = cv2.VideoCapture(sample)

    while cap.isOpened():

        success, frame = cap.imread()

        fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps #in seconds



        '''monitor for every 5 seconds, then make prediction'''

        frame = frame = cv2.resize(frame, (224, 224)).astype("float32")

        preds = EffNet.predict(np.expand_dims(frame, axis=0))[0]


def main():
    pass


if __name__ == "__main__":

    main()