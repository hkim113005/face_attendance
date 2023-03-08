import os
import sys
import dlib

model = "mmod_human_face_detector.dat"
dataset = "dataset"

cnn_face_detector = dlib.cnn_face_detection_model_v1(model)
window = dlib.image_window()

for image_file in os.listdir(dataset):
    print("Processing file: {}".format(image_file))
    image = dlib.load_rgb_image(os.path.join(dataset, image_file))

    detections = cnn_face_detector(image, 1)

    print("Number of faces detected: {}".format(len(detections)))
    for i, detection in enumerate(detections):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
            i, detection.rect.left(), detection.rect.top(), detection.rect.right(), detection.rect.bottom(), detection.confidence))

    rects = dlib.rectangles()
    rects.extend([detection.rect for detection in detections])

    window.clear_overlay()
    window.set_image(image)
    window.add_overlay(rects)
    dlib.hit_enter_to_continue()