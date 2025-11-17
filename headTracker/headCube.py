import cv2
import time
import mediapipe as mp
from mediapipe import solutions
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from PIL import Image
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import math

"""
COLOR CONSTANT represend as (B,G,R)
"""
RED = (0,0,255)
BLUE = (255,0,0)
GREEN = (0,255,0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)

def landmark_to_np(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h], dtype=int)

def landmarks_to_np(landmarks, idx, w, h):
    points = []
    
    for idx in idx:
        landmark = landmarks[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))
    return points

cap = cv2.VideoCapture(0)  # 0 for default webcam
w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

"""
Landmark Constnat
"""

FACE_OUTLINE_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

LEFT_EYE_ENDS = [263, 362] # outer, inner
RIGHT_EYE_ENDS = [33, 133] # inner, outer


LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [474, 475, 476, 477]

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_IRIS = [469, 470, 471, 472]

FACE_LANDMARKS = [356, 127, 10, 152] # 

"""
HELPER FUNCTIONS:
"""


# def face_distance():


# def extrapolate_eyes(landmarks, annotated_img):
#     h, w, _ = annotated_img.shape
#     # left eye
#     a = (landmarks[33].x * w, landmarks[33].y * h)
#     b = (landmarks[133].x * w, landmarks[133].y * h)

#     R_radius = math.sqrt(((a[0] - b[0])**2) + ((a[1] - b[1])**2)) / 2

#     R_center = np.array([
#         (a[0] + b[0]) / 2,
#         (a[1] + b[1]) / 2 ], dtype=int)

#     # right eye    
#     a = (landmarks[362].x * w, landmarks[362].y * h)
#     b = (landmarks[263].x * w, landmarks[263].y * h)

#     L_radius = math.sqrt(((a[0] - b[0])**2) + ((a[1] - b[1])**2)) / 2

#     L_center = np.array([
#         (a[0] + b[0]) / 2,
#         (a[1] + b[1]) / 2 ], dtype=int)
    
#     return int(L_radius), L_center, int(R_radius), R_center

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

def gaze_direction(eye_points, iris_center):
    
    left_point = min(eye_points, key=lambda p: p[0])
    right_point = max(eye_points, key=lambda p: p[0])

    eye_width = math.sqrt(((left_point[0] - right_point[0])**2) + ((left_point[1] - right_point[1])**2)) / 2


    iris_relative = (iris_center - left_point[0]) / eye_width # not sure what the math is doing here

    return iris_relative


def get_gaze_vector(eye_points, iris_center):

    eye_center = np.mean(eye_points, axis=0).astype(int) # taking average of all eye 

    gaze_vector = (
        iris_center[0] - eye_center[0],
        iris_center[1] - eye_center[1]
    )
    

    return eye_center, gaze_vector

def draw_gaze_projection(frame, eye_center, gaze_vector, magnitude = 10):

    # projection_point = np.array([
    #     eye_center[0] + gaze_vector[0] * magnitude,
    #     eye_center[1] + gaze_vector[1] * magnitude        
    #                             ], dtype=int)

    projection_point = (
       int(eye_center[0] + gaze_vector[0] * magnitude),
       int(eye_center[1] + gaze_vector[1] * magnitude)        
    )
    # cv2.line(frame, tuple(eye_center), tuple(projection_point), thickness=3, color=YELLOW)
    


    cv2.arrowedLine(frame, tuple(eye_center), tuple(projection_point), thickness=3, color=YELLOW)




# def eyes(annotated_img, landmarks):
#     A = 75
#     h, w, _ = annotated_img.shape

#     # RIGHT
#     a = (landmarks[33].x * w, landmarks[33].y * h)
#     b = (landmarks[133].x * w, landmarks[133].y * h)


#     rr = int(math.sqrt(((a[0] - b[0])**2) + ((a[1] - b[1])**2)) / 2)

#     rc = np.array([
#         (a[0] + b[0]) / 2,
#         (a[1] + b[1]) / 2 ], dtype=int)


#     # iris 
#     ri = landmark_to_np(results[468], w, h)

#     cv2.circle(annotated_img, center=ri, color=RED, radius=2, thickness=2)
#     cv2.circle(annotated_img, center=rc, color=BLUE, radius=3, thickness=3)
#     cv2.circle(annotated_img, center=rc, color=BLUE, radius=rr, thickness=2)

#     # projecting
#     rm = (rc[1] - ri[1]) / (rc[0] - ri[0])

#     if rm > 0:
#         rm = np.clip(rm, a_min = 0.01, a_max=100)
#     else:
#         rm = np.clip(rm, a_min=-100, a_max=-0.01)

#     print(rm)

#     if rm < 0:
#         ra = np.array([
#             rc[0] + A,
#             rc[1] + (A * rm)
#                       ], dtype = int)
#     else:
#         ra = np.array([
#             rc[0] - A,
#             rc[1] - (A * rm)
#                       ], dtype = int)

#     cv2.line(annotated_img, pt1=rc, pt2=ra, color=CYAN, thickness=2)

#     # LEFT
#     a = (landmarks[362].x * w, landmarks[362].y * h)
#     b = (landmarks[263].x * w, landmarks[263].y * h)


#     rr = int(math.sqrt(((a[0] - b[0])**2) + ((a[1] - b[1])**2)) / 2)

#     rc = np.array([
#         (a[0] + b[0]) / 2,
#         (a[1] + b[1]) / 2 ], dtype=int)


#     # iris 
#     ri = landmark_to_np(results[473], w, h)

#     cv2.circle(annotated_img, center=ri, color=RED, radius=2, thickness=2)
#     cv2.circle(annotated_img, center=rc, color=BLUE, radius=3, thickness=3)
#     cv2.circle(annotated_img, center=rc, color=BLUE, radius=rr, thickness=2)

#     # projecting
#     rm = (rc[1] - ri[1]) / (rc[0] - ri[0])

#     if rm > 0:
#         rm = np.clip(rm, a_min = 0.01, a_max=100)
#     else:
#         rm = np.clip(rm, a_min=-100, a_max=-0.01)

#     print(rm)

#     if rm < 0:
#         ra = np.array([
#             rc[0] + A,
#             rc[1] + (A * rm)
#                       ], dtype = int)
#     else:
#         ra = np.array([
#             rc[0] - A,
#             rc[1] - (A * rm)
#                       ], dtype = int)

#     cv2.line(annotated_img, pt1=rc, pt2=ra, color=CYAN, thickness=2)


#     return annotated_img

"""
Lc = Left (eye) center
Lr = Left (eye) iris (center)
"""
    

# landmakrs should be the NormalizedLandmarkList
# def landmarks_to_np(landmarks, w, h):
#     return np.array([[l.x * w, l.y * h, l.z] for l in landmarks.landmark])
# p

def square_face(landmarks, annotated_img):
    h, w, _ = annotated_img.shape
    # variables represen
    # face left
    # face right
    # top 
    # bottom
    # FACE_IDX = [356, 127, 10, 152]

    top_left = np.array([landmarks[127].x * w, landmarks[10].y * h], dtype=int)
    bottom_right = np.array([landmarks[356].x * w, landmarks[152].y * h], dtype=int)

    return top_left, bottom_right


# now it actually make it a cube
def plot_face(annotated_img, results):
    GREEN = (127,225, 0)
    PURPLE = (0,255,127)
    
    top_left, bottom_right = square_face(results, annotated_img)        
    cv2.rectangle(annotated_img, pt1=top_left, pt2=bottom_right, thickness=4, color=(127, 255, 0))

    # center of face rectangle
    rectangle_center = np.array([(top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1])/ 2], dtype = int)
    cv2.circle(annotated_img, center=rectangle_center, radius=4, thickness=4, color = (0, 255, 127)) #COLORED 

    # nose
    np_landmark = landmark_to_np(results[4], w, h)
    nose_center  = (int(np_landmark[0]), int(np_landmark[1]))
    cv2.circle(annotated_img, center=nose_center, radius=4, thickness=4, color=(255, 0, 127))
    cv2.line(annotated_img, pt1=rectangle_center, pt2=nose_center, thickness=3, color=(127, 0, 127)) # COLORED purple

    # drawing the square silloute but using the nose as the center index
    diff = nose_center - rectangle_center
    

    nose_top_left = top_left + diff
    nose_bottom_right = bottom_right + diff
    cv2.rectangle(annotated_img, pt1=nose_top_left, pt2=nose_bottom_right, thickness= 4, color=GREEN)

    # head_w = bottom_right[0] - top_left[0]
    # head_h = top_left[1] - bottom_right[1]


    top_right = np.array([top_left[0],bottom_right[1]],dtype=int)
    bottom_left = np.array([bottom_right[0],top_left[1]],dtype=int)

    nose_top_right = top_right + diff
    nose_bottom_left = bottom_left + diff

    cv2.line(annotated_img, top_left, nose_top_left, thickness=4, color=GREEN)
    cv2.line(annotated_img, bottom_right, nose_bottom_right, thickness=4, color=GREEN)

    cv2.line(annotated_img, top_right, nose_top_right, thickness=4, color=GREEN)
    cv2.line(annotated_img, bottom_left, nose_bottom_left, thickness=4, color=GREEN)

    



    return annotated_img



def draw_eyes(frame, eye_points, eye_center, iris_points, iris_center, color):

    for _, point in enumerate(eye_points):
        p =  (point[0], point[1])
        cv2.circle(frame, p, radius=1, color=RED, thickness=2)

    for _, point in enumerate(iris_points):
        p =  (point[0], point[1])
        cv2.circle(frame, p, radius=1, color=BLUE, thickness=2)

        

"""
MAIN LOOP
"""
while cap.isOpened():
    _, img = cap.read()

    if not _:
        print("error reading img")
        continue

    img = cv2.flip(img, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    annotated_img = img

    h, w, _ = annotated_img.shape

    img_center = np.array([
        w/2, h/2
    ], dtype=int)
    
    cv2.circle(annotated_img, center=img_center, radius=3, thickness=3, color=(0,0,0))
    
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if results.multi_face_landmarks:
        results = results.multi_face_landmarks[0].landmark

        """
        new
        """

        left_eye_points = landmarks_to_np(results, LEFT_EYE, w, h)
        left_iris_points = landmarks_to_np(results, LEFT_IRIS, w, h)
            

        left_iris_center = np.mean(left_iris_points, axis=0).astype(int)
        left_eye_center, left_gaze_vector = get_gaze_vector(left_eye_points, left_iris_center)

        draw_eyes(annotated_img, left_eye_points, left_eye_center, left_iris_points, left_iris_center, color=RED)
        
        
        draw_gaze_projection(annotated_img, left_eye_center, left_gaze_vector, magnitude=5)

        
        right_eye_points = landmarks_to_np(results, RIGHT_EYE, w, h)
        right_iris_points = landmarks_to_np(results, RIGHT_IRIS, w, h)



        

    
        # annotated_img = plot(annotated_img, results)

        
        
        
        
        """
        EYES 
        """

        # annotated_img = eyes(annotated_img, results)

        
        # annotated_img = plot_gaze(annotated_img, results)


        """
        FACE
        """

        annotated_img = plot_face(annotated_img, results)


        
        # for idx, landmark in enumerate(FACE_LANDMARKS):
        #     np_landmark = landmark_to_np(results[landmark], w, h)
        #     x, y = int(np_landmark[0]), int(np_landmark[1])
        #     cv2.circle(annotated_img, (x, y), radius=3, thickness=4, color=(0, 255, 255))

        # for idx, landmark in enumerate(LANDMARKS):
        #     np_landmark = landmark_to_np(results[landmark], w, h)
        #     x, y = int(np_landmark[0]), int(np_landmark[1])
        #     cv2.circle(
        #         annotated_img, (x, y), radius=2, thickness=1, color=(127, 255, 0)
        #     )
        # for idx, landmark in enumerate(EYE_LANDMARKS):
        #     np_landmark = landmark_to_np(results[landmark], w, h)
        #     x, y = int(np_landmark[0]), int(np_landmark[1])
        #     cv2.circle(
        #         annotated_img, (x, y), radius=2, thickness=2, color=(0, 0, 255)
        #     )

        

        cv2.imshow("Face Outline", annotated_img)
    else:
        # cv2.imshow("Face Outline(fail)", annotated_img)
        continue


cap.release()
cv2.destroyAllWindows()
