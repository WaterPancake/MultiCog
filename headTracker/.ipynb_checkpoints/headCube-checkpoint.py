import cv2
import time
import mediapipe as mp
from mediapipe import solutions
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import math
from PIL import Image


face_mesh = mp.solutions.face_mesh


cap = cv2.VideoCapture(0)


FACE_OUTLINE_INDICES = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]
