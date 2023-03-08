import os
import sys
import dlib
import cv2
import glob
import numpy as np
import pickle

model = "models/mmod_human_face_detector.dat"
dataset_dir = "dataset"
predictor_path = "models/shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "models/dlib_face_recognition_resnet_model_v1.dat"

cnn_face_detector = dlib.cnn_face_detection_model_v1(model)
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

window = dlib.image_window()

camera = cv2.VideoCapture(0)
# camera.set(3, 100)
# camera.set(4, 100)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)

save_dir = "pickles"

face_vectors = []
ids = []

def save_object(object, name):
    try:
        with open(os.path.join(save_dir, str(name) + ".pickle"), "wb") as f:
            pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


for image_file in glob.glob(os.path.join(dataset_dir, "*.jpg")):
    image = dlib.load_rgb_image(image_file)
    
    window.clear_overlay()
    window.set_image(image)

    detections = cnn_face_detector(image, 0)

    faces = dlib.rectangles()
    faces.extend([detection.rect for detection in detections])

    if len(faces) == 0:
        print(image_file)

    for k, d in enumerate(faces):
        shape = sp(image, d)
        window.clear_overlay()
        window.add_overlay(d)
        window.add_overlay(shape)

        face_chip = dlib.get_face_chip(image, shape)        

        face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)                
        face_vectors.append(face_descriptor_from_prealigned_image)

        ids.append(image_file.split(".")[1])  

    # dlib.hit_enter_to_continue()

face_ids = {}


for i in range(len(ids)):
    id = ids[i]
    face_vector = np.append(np.asarray(face_vectors[i]), 1)
    if id not in face_ids.keys():
        face_ids[id] = face_vector
    else:
        face_ids[id] += face_vector
    # print(type(np.asarray(face_vector)))

for id in face_ids.keys():
    face_ids[id] = np.delete(face_vector / face_vector[-1], -1, 0)
    save_object(face_ids[id], id)