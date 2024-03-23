import json
import cv2
import mediapipe as mp
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
mp_face_mesh = mp.solutions.face_mesh


def generate_face_landmarks_dataset(filepath, outputpath):
    # create a VideoCapture object
    cap = cv2.VideoCapture(filepath)

    # check if the VideoCapture object was successfully opened
    if cap.isOpened() == False:
        print("Error opening video stream or file")
        raise TypeError

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    landmarks_per_frame = {}
    frame_idx = 0

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            # read the next frame from the video file
            success, image = cap.read()

            if not success:
                print("Video file finished. Total Frames: %d" % (cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                break

            image.flags.writeable = False
            results = face_mesh.process(image)

            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                    landmarks_list = []
                    for landmark in face.landmark:
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)
                        z = landmark.z
                        landmarks_list.append((x, y, z))

                landmarks_per_frame[frame_idx] = landmarks_list
                frame_idx += 1

            key = cv2.waitKey(25)
            if key == ord('q'):
                break

        cap.release()

    print(outputpath)
    with open(outputpath, "w") as outfile:
        json.dump(landmarks_per_frame, outfile)


if __name__ == "__main__":
    filepath = 'GuessWhatTest.mp4'
    outputpath = 'GuessWhatTest.json'
    generate_face_landmarks_dataset(filepath, outputpath)
