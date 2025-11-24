from sys import base_exec_prefix
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import BaseOptions, vision
import numpy as np


# Color constant
WHITE = (255, 255, 255)

"""
For extracting only the facial landmarks of person to be saved for training purposes. 
"""


def adjusted_landmark_to_np(
    landmarks,
    w,
    h,
):
    return np.array([[l.x * w * 1.775, l.y * h * 0.57] for l in landmarks], dtype=int)


def landmark_to_np(
    landmarks,
    w,
    h,
):
    return np.array([[l.x * w, l.y * h] for l in landmarks], dtype=int)


"""
NAME
"""


def faceMesh_V0():
    # initlaizing model
    model_path = "face_landmarker.task"
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
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
        frame = cv2.flip(frame, 1)

        # this version requires that the detector be feed in a specific format
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = face_mesh.detect_for_video(mp_frame, frame_count)
        frame = mp_frame.numpy_view().copy()  # retore to np array
        frame.flags.writeable = True

        if results.face_landmarks:
            results = results.face_landmarks[0]
            np_results = adjusted_landmark_to_np(results, w, h)

            for landmark in np_results:
                cv2.circle(frame, tuple(landmark), radius=3, thickness=2, color=WHITE)
            cv2.imshow("Face Mesh", frame)

        # for termination
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def faceMesh_V1():
    cv2.namedWindow("Face Mesh", cv2.WINDOW_NORMAL)
    # initalizing model
    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    w, h = frame.shape[:2]

    while cap.isOpened():
        _, frame = cap.read()

        if not _:
            continue

        frame = cv2.flip(frame, 1)
        results = face_mesh.process(frame)

        if results.multi_face_landmarks:
            results = results.multi_face_landmarks[0].landmark  # use with version two
            np_results = adjusted_landmark_to_np(results, w, h)

            frame = np.zeros((w, h, 4), dtype=np.int8)
            for landmark in np_results:
                cv2.circle(frame, tuple(landmark), radius=3, thickness=2, color=WHITE)
            cv2.imshow("Face Mesh", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# faceMesh_V1()
faceMesh_V0()
