import os
import dlib
import cv2
import numpy as np
import pickle
from colorama import Fore
import glob

pickle_dir = "pickles"

detector_path = "models/mmod_human_face_detector.dat"
predictor_path = "models/shape_predictor_5_face_landmarks.dat"
face_recognizer_path = "models/dlib_face_recognition_resnet_model_v1.dat"
face_labels_path = "labels.pickle"

font = cv2.FONT_HERSHEY_SIMPLEX

def load_face_labels():
    face_labels_file = open(face_labels_path)
    face_labels = pickle.load(face_labels)
    return face_labels

def load_face_id_vectors():
    number_faces = len(glob.glob(os.path.join(pickle_dir, "*.pickle")))

    face_id_vectors = np.zeros(shape=(number_faces, 128))
    face_ids = ["" for i in range(number_faces)]

    count = 0
    for face_id_vector_pickle in glob.glob(os.path.join(pickle_dir, "*.pickle")):
        face_id_vector_file = open(face_id_vector_pickle, 'rb')     
        face_id_vectors[count] = pickle.load(face_id_vector_file)
        face_ids[count] = face_id_vector_pickle.split("/")[-1].split(".")[0]
        count += 1

    return (face_id_vectors, face_ids)


def get_closest_face(face, faces):
    deltas = faces - face
    dist_squared = np.einsum('ij, ij -> i', deltas, deltas)
    return (np.argmin(dist_squared), (dist_squared[np.argmin(dist_squared)] ** 0.5))


def get_camera():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def load_models():
    detector = dlib.cnn_face_detection_model_v1(detector_path)
    shape_predictor = dlib.shape_predictor(predictor_path)
    face_recognizer = dlib.face_recognition_model_v1(face_recognizer_path)
    return (detector, shape_predictor, face_recognizer)


def scan_faces(face_ids, face_id_vectors):
    camera = get_camera()

    detector, shape_predictor, face_recognizer = load_models()

    scanned_faces = []
    scanned_face_id_vectors = []
    scanned_face_ids = []

    while True:
        status, image = camera.read()
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = detector(image, 0)

        faces = dlib.rectangles()
        faces.extend([detection.rect for detection in detections])

        for i, face in enumerate(faces):
            cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

            shape = shape_predictor(image, face)            
            face_chip = dlib.get_face_chip(image, shape)

            scanned_face_id_vector = face_recognizer.compute_face_descriptor(face_chip)
            closest_index, closest_distance = get_closest_face(scanned_face_id_vector, face_id_vectors)
            if closest_distance < 0.6:
                scanned_face_id = face_ids[closest_index]
            else:
                scanned_face_id = "UNKNOWN"

            cv2.putText(image, scanned_face_id, (face.left(), face.top() - 5), font, 0.6, (255, 255, 255), 1)

            if scanned_face_id != "UNKNOWN" and scanned_face_id not in scanned_face_ids:
                scanned_faces.append(face)
                scanned_face_id_vectors.append(scanned_face_id_vector)
                scanned_face_ids.append(scanned_face_id)

        cv2.imshow("Scanning Faces", image)

        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break

    camera.release()

    return (scanned_faces, scanned_face_id_vectors)


def identify_faces(scanned_face_id_vectors, face_ids, face_id_vectors):
    scanned_face_ids = []
    for scanned_face_id_vector in scanned_face_id_vectors:
        closest_index, closest_distance = get_closest_face(scanned_face_id_vector, face_id_vectors)
        if closest_distance < 0.6:
            scanned_face_ids.append(face_ids[closest_index])
        else:
            scanned_face_ids.append("UNKNOWN")
    return scanned_face_ids


def main():
    face_id_vectors, face_ids = load_face_id_vectors()

    scanned_faces, scanned_face_id_vectors = scan_faces(face_ids, face_id_vectors)
    scanned_face_ids = identify_faces(scanned_face_id_vectors, face_ids, face_id_vectors)

    print(scanned_face_ids)

    
main()