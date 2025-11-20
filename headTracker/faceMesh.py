from sys import base_exec_prefix
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import BaseOptions, vision
import numpy as np


WHITE = (255,255,255)
"""
For extracting only the facial landmarks of person to be saved for training purposes. 
"""


model_path = "face_landmarker.task"
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_mesh = FaceLandmarker.create_from_options(options)


def landmark_to_np(landmarks, w, h,):
    return np.array([[l.x * w * 1.75, l.y * h * 0.55, l.z] for l in landmarks],dtype=int)



def view_ahead():
    return 
def view_above():
    return 
def view_aside():
    return 


"""
MAIN LOOP 
"""

cap = cv2.VideoCapture(0)
# w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

_, frame = cap.read()
w, h = frame.shape[:2]
print(w, h)

blank_img = np.zeros((int(w),int(h),3), dtype=int) # matches the resolution of screen, change later to a cropped version

# ignore linter error if present
# mp_face_mesh = mp.solutions.face_mesh

# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True, f
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )
#
#
# 

cv2.namedWindow("Face Mesh", cv2.WINDOW_FULLSCREEN)
cv2.resizeWindow("Face Mesh", w, h)
FACE_BOUNDING_IDX = [10, 152, 127, 356] # top, bottom, face right, face left

# FACIAL_OUTLINE = 
# 478 landmarks
# To Do: graph connecting them?

frame_count = 0
while cap.isOpened():

    _, frame = cap.read()

    if not _:
        continue

    frame_count += 1
    
    frame = cv2.flip(frame, 1)
    # results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    results = face_mesh.detect_for_video(mp_frame, frame_count)
    # if results.multi_face_landmarks:
    if results.face_landmarks:
        # results = results.multi_face_landmarks[0].landmark
        results = results.face_landmarks[0]


    # displaying 720 x 720
        np_results = landmark_to_np(results, w, h)

        # blank_img = np.zeros((w,h,3), dtype=np.int8) # matches the resolution of screen, change later to a cropped version

        for landmark in np_results:
            cv2.circle(frame, tuple(landmark[:2]), radius=3, thickness=2, color=WHITE)        
        cv2.imshow("Face Mesh", frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows
