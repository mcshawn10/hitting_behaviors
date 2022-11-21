import cv2
import mediapipe as mp
import numpy as np


def rescale_frame(frame, percent):
    #width = int(frame.shape[1] * percent/100)
    #height = int(frame.shape[0] * percent/100)
    dim = (224, 224)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)