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

from sys import base_exec_prefix
from mediapipe.tasks import python
from mediapipe.tasks.python import BaseOptions, vision

"""
TO DO:

"""

"""
COLOR CONSTANT represend as (B,G,R)
"""
RED = (0, 0, 255)
BLUE = (255, 0, 0)
PURPLE = (255, 0, 255)
GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)


# def adjusted_landmark_to_np(
#     landmarks,
#     w,
#     h,
# ):
#     return np.array([[l.x * w * 1.775, l.y * h * 0.57] for l in landmarks], dtype=int)


def adjusted_landmark_to_np(
    landmarks,
    idx,
    w,
    h,
):
    points = []
    for idx in idx:
        landmark = landmarks[idx]
        x = int(landmark.x * w * 1.775)
        y = int(landmark.y * h * 0.555)
        # x = int(landmark.x * w)
        # y = int(landmark.y * h)
        points.append((x, y))
    return points


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

# ignore linter error if present
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

"""
Landmark Constnat
"""

FACE_OUTLINE_LANDMARKS = [
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

LEFT_EYE = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,  # old
    359,
    467,
    260,
    259,
    257,
    258,
    286,
    414,
    463,
    341,
    256,
    252,
    255,
    254,
    339,
    255,
]
LEFT_IRIS = [474, 475, 476, 477, 473]


RIGHT_EYE = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,  ## end
    130,
    247,
    30,
    29,
    27,
    28,
    56,
    190,
    243,
    112,
    26,
    22,
    23,
    24,
    110,
    25,
]
RIGHT_IRIS = [469, 470, 471, 472, 468]

FACE_LANDMARKS = [356, 127, 10, 152]  #

"""
HELPER FUNCTIONS:
"""


# still not used
def gaze_direction(eye_points, iris_center):
    left_point = min(eye_points, key=lambda p: p[0])
    right_point = max(eye_points, key=lambda p: p[0])

    eye_width = (
        math.sqrt(
            ((left_point[0] - right_point[0]) ** 2)
            + ((left_point[1] - right_point[1]) ** 2)
        )
        / 2
    )

    iris_relative = (
        iris_center - left_point[0]
    ) / eye_width  # not sure what the math is doing here

    return iris_relative


def get_gaze_vector(eye_points, iris_center):
    eye_center = np.mean(eye_points, axis=0).astype(
        int
    )  # taking average of all eye points

    gaze_vector = np.array(
        [iris_center[0] - eye_center[0], iris_center[1] - eye_center[1]], dtype=float
    )

    return eye_center, gaze_vector


def draw_gaze_projection(frame, eye_center, gaze_vector, magnitude=10):
    projection_point = np.array(eye_center + gaze_vector * magnitude, dtype=int)

    cv2.arrowedLine(
        frame, tuple(eye_center), tuple(projection_point), color=YELLOW, thickness=3
    )


def square_face(landmarks, annotated_img):
    h, w, _ = annotated_img.shape
    top_left = np.array([landmarks[127].x * w, landmarks[10].y * h], dtype=int)
    bottom_right = np.array([landmarks[356].x * w, landmarks[152].y * h], dtype=int)

    return top_left, bottom_right


# now it actually make it a cube
def plot_face(annotated_img, results):
    GREEN = (127, 225, 0)

    top_left, bottom_right = square_face(results, annotated_img)
    cv2.rectangle(
        annotated_img,
        tuple(top_left),
        tuple(bottom_right),
        thickness=4,
        color=(127, 255, 0),
    )

    # center of face rectangle
    rectangle_center = np.array(
        [(top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1]) / 2],
        dtype=int,
    )
    cv2.circle(
        annotated_img,
        tuple(rectangle_center),
        radius=4,
        thickness=4,
        color=(0, 255, 127),
    )  # COLORED

    # nose
    np_landmark = landmark_to_np(results[4], w, h)
    nose_center = (int(np_landmark[0]), int(np_landmark[1]))
    cv2.circle(
        annotated_img, center=nose_center, radius=4, thickness=4, color=(255, 0, 127)
    )
    cv2.line(
        annotated_img,
        tuple(rectangle_center),
        tuple(nose_center),
        thickness=3,
        color=(127, 0, 127),
    )  # COLORED purple

    # drawing the square silloute but using the nose as the center index
    diff = nose_center - rectangle_center

    nose_top_left = top_left + diff
    nose_bottom_right = bottom_right + diff
    cv2.rectangle(
        annotated_img,
        tuple(nose_top_left),
        tuple(nose_bottom_right),
        thickness=4,
        color=GREEN,
    )

    # head_w = bottom_right[0] - top_left[0]
    # head_h = top_left[1] - bottom_right[1]

    top_right = np.array([top_left[0], bottom_right[1]], dtype=int)
    bottom_left = np.array([bottom_right[0], top_left[1]], dtype=int)

    nose_top_right = top_right + diff
    nose_bottom_left = bottom_left + diff

    cv2.line(
        annotated_img, tuple(top_left), tuple(nose_top_left), thickness=4, color=GREEN
    )
    cv2.line(
        annotated_img,
        tuple(bottom_right),
        tuple(nose_bottom_right),
        thickness=4,
        color=GREEN,
    )

    cv2.line(
        annotated_img, tuple(top_right), tuple(nose_top_right), thickness=4, color=GREEN
    )
    cv2.line(
        annotated_img,
        tuple(bottom_left),
        tuple(nose_bottom_left),
        thickness=4,
        color=GREEN,
    )

    return annotated_img


def draw_eyes(frame, eye_points, eye_center, iris_points, iris_center, color):
    for _, point in enumerate(eye_points):
        p = (point[0], point[1])
        cv2.circle(frame, p, radius=1, color=RED, thickness=2)

    for _, point in enumerate(iris_points):
        p = (point[0], point[1])
        cv2.circle(frame, p, radius=1, color=CYAN, thickness=2)


"""
using math from: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
Pass center points
"""


def gaze_focus(left_iris, left_eye, right_iris, right_eye):
    x_1, y_1 = left_iris
    x_2, y_2 = left_eye
    x_3, y_3 = right_iris
    x_4, y_4 = right_eye

    A = (x_3 - x_1) * (y_3 - y_4) - (y_3 - y_1) * (x_3 - x_4)
    B = (x_1 - x_2) * (y_3 - y_4) - (y_1 - y_2) * (x_3 - x_4)

    if math.isnan(B):
        B = 1 / 1e9

    t = A / B

    if math.isinf(t):
        t = 1e9

    F_x = x_1 + t * (x_2 - x_1)
    F_y = y_1 + t * (y_2 - y_1)

    return np.array([F_x, F_y], dtype=int)


def run():
    model_path = "face_landmarker.task"
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.7,
    )

    face_mesh = FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    w, h = frame.shape[:2]

    frame_count = 0
    while cap.isOpened():
        _, frame = cap.read()

        if not _:
            continue

        frame_count += 1
        # frame = cv2.flip(frame, 1)

        # this version requires that the detector be feed in a specific format
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = face_mesh.detect_for_video(mp_frame, frame_count)
        frame = mp_frame.numpy_view().copy()  # retore to np array
        frame.flags.writeable = True

        if results.face_landmarks:
            results = results.face_landmarks[0]

            left_eye_points = adjusted_landmark_to_np(results, LEFT_EYE, w, h)
            left_iris_points = adjusted_landmark_to_np(results, LEFT_IRIS, w, h)

            left_iris_center = np.mean(left_iris_points, axis=0).astype(int)
            left_eye_center, left_gaze_vector = get_gaze_vector(
                left_eye_points, left_iris_center
            )

            draw_gaze_projection(frame, left_eye_center, left_gaze_vector, magnitude=20)

            draw_eyes(
                frame,
                left_eye_points,
                left_eye_center,
                left_iris_points,
                left_iris_center,
                color=RED,
            )

            """"
            RIGHT EYE 
            """

            right_eye_points = adjusted_landmark_to_np(results, RIGHT_EYE, w, h)
            right_iris_points = adjusted_landmark_to_np(results, RIGHT_IRIS, w, h)

            right_iris_center = np.mean(right_iris_points, axis=0).astype(int)
            right_eye_center, right_gaze_vector = get_gaze_vector(
                right_eye_points, right_iris_center
            )

            draw_gaze_projection(
                frame, right_eye_center, right_gaze_vector, magnitude=20
            )

            draw_eyes(
                frame,
                right_eye_points,
                right_eye_center,
                right_iris_points,
                right_iris_center,
                color=RED,
            )

            cv2.circle(
                frame,
                tuple(right_eye_center),
                radius=2,
                thickness=2,
                color=PURPLE,
            )

            cv2.circle(
                frame,
                tuple(right_iris_center),
                radius=2,
                thickness=2,
                color=BLUE,
            )
            """
            IN DEVELOPMENT
            """
            # gaze_focal_point = gaze_focus(
            #     left_iris_center, left_eye_center, right_iris_center, right_eye_center
            # )
            # cv2.circle(
            #     frame, tuple(gaze_focal_point), radius=4, thickness=4, color=GREEN
            # )

            cv2.imshow("Face Outline", frame)
        else:
            continue
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# def old_ver():
#     while cap.isOpened():
#         _, img = cap.read()

#         if not _:
#             print("error reading img")
#             continue

#         img = cv2.flip(img, 1)
#         # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         annotated_img = img

#         h, w, _ = annotated_img.shape

#         img_center = np.array([w / 2, h / 2], dtype=int)

#         cv2.circle(
#             annotated_img, tuple(img_center), radius=3, thickness=3, color=(0, 0, 0)
#         )

#         results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#         if results.multi_face_landmarks:
#             results = results.multi_face_landmarks[0].landmark

#             """
#             new
#             """

#             left_eye_points = landmarks_to_np(LEFT_EYE, LEFT_EYE, w, h)
#             left_iris_points = landmarks_to_np(results, LEFT_IRIS, w, h)

#             left_iris_center = np.mean(left_iris_points, axis=0).astype(int)
#             left_eye_center, left_gaze_vector = get_gaze_vector(
#                 left_eye_points, left_iris_center
#             )

#             # draw_eyes(
#             #     annotated_img,
#             #     left_eye_points,
#             #     left_eye_center,
#             #     left_iris_points,
#             #     left_iris_center,
#             #     color=RED,
#             # )

#             draw_gaze_projection(
#                 annotated_img, left_eye_center, left_gaze_vector, magnitude=20
#             )

#             """"
#             RIGHT EYE
#             """

#             right_eye_points = landmarks_to_np(results, RIGHT_EYE, w, h)
#             right_iris_points = landmarks_to_np(results, RIGHT_IRIS, w, h)

#             right_iris_center = np.mean(right_iris_points, axis=0).astype(int)
#             right_eye_center, right_gaze_vector = get_gaze_vector(
#                 right_eye_points, right_iris_center
#             )

#             # draw_eyes(annotated_img, right_eye_points, right_eye_center, right_iris_points, right_iris_center, color=RED)

#             draw_gaze_projection(
#                 annotated_img, right_eye_center, right_gaze_vector, magnitude=20
#             )

#             draw_eyes(
#                 annotated_img,
#                 right_eye_points,
#                 right_eye_center,
#                 right_iris_points,
#                 right_iris_center,
#                 color=RED,
#             )

#             cv2.circle(
#                 annotated_img,
#                 tuple(right_iris_center),
#                 radius=2,
#                 thickness=2,
#                 color=CYAN,
#             )
#             cv2.circle(
#                 annotated_img,
#                 tuple(right_eye_center),
#                 radius=2,
#                 thickness=2,
#                 color=RED,
#             )

#             gaze_interception(
#                 left_iris_points, left_eye_points, right_iris_points, right_eye_points
#             )

#             # putting a cube on the face
#             annotated_img = plot_face(annotated_img, results)

#             """
#             pt_1 and pt_2 are shape [,2]
#             """

#             def dist(pt_1, pt_2):
#                 x_sd = pow(pt_1[0], pt_2[0], 2)
#                 y_sd = pow(pt_1[1], pt_2[1], 2)

#                 return int(math.sqrt(x_sd + y_sd))

#             cv2.imshow("Face Outline", annotated_img)
#         else:
#             continue
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


run()
