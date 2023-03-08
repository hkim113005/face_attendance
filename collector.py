import os
import dlib
import cv2
import numpy as np
import pickle
from colorama import Fore

save_dir = "pickles"

detector_path = "models/mmod_human_face_detector.dat"
predictor_path = "models/shape_predictor_5_face_landmarks.dat"
face_recognizer_path = "models/dlib_face_recognition_resnet_model_v1.dat"


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f' \r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def save_object(object, name):
    try:
        with open(os.path.join(save_dir, str(name) + ".pickle"), "wb") as f:
            pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print(Fore.RED + "Error during pickling object (Possibly unsupported):", ex + Fore.RESET)


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


def scan_face(samples):
    camera = get_camera()

    window = dlib.image_window()

    detector, shape_predictor, face_recognizer = load_models()

    face_id_vectors = []

    count = 0
    while True:
        status, image = camera.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        window.clear_overlay()
        window.set_image(image)

        detections = detector(image, 0)

        faces = dlib.rectangles()
        faces.extend([detection.rect for detection in detections])

        if len(faces) == 0:
            printProgressBar(count, samples, prefix = Fore.RED + " Progress:", suffix = "No faces detected..." + Fore.RESET, length = 40)
            continue
        elif len(faces) != 1:
            printProgressBar(count, samples, prefix = Fore.RED + " Progress:", suffix = "Too many faces detected..." + Fore.RESET, length = 40)

        for i, face in enumerate(faces):
            shape = shape_predictor(image, face)

            window.clear_overlay()
            window.add_overlay(face)
            window.add_overlay(shape)

            face_chip = dlib.get_face_chip(image, shape)
            face_descriptor = face_recognizer.compute_face_descriptor(face_chip)

            face_id_vectors.append(face_descriptor)

        count += 1

        printProgressBar(count, samples, prefix = " Progress:" + Fore.GREEN, suffix = Fore.RESET + "Complete                    ", length = 40)

        if count >= samples:
            break

    
    camera.release()

    return face_id_vectors


def save_face(face_id_vectors, id):
    face_id_vector = np.sum(np.asarray([np.asarray(face_vector) for face_vector in face_id_vectors]), axis = 0) / (np.asarray(face_id_vectors).shape[0])
    save_object(face_id_vector, id)


def main():
    print("\n\n==========================================================================================\n")
    id = input("\n Enter User ID and press <RETURN>: ")
    
    print("\n Initiating capture sequence. Please look at the camera and wait.\n")
    save_face(scan_face(5), id)
    print("\n Facial ID Vector saved.")

    key = input("\n Enter <y> and press <RETURN> to register another User: ")
    if key == "y" or key == "Y":
        main()
    else:
        print("\n\n==========================================================================================\n\n")


main()