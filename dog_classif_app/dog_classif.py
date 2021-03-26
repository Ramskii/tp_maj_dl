import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import tensorflow as tf
import cv2
from collections import OrderedDict
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required = True)
parser.add_argument("-v", "--view", action="store_true")
args = parser.parse_args()

with open("./model/class_indices.json", "r") as f:
    class_indices = json.load(f)

model = tf.keras.models.load_model('./model/vgg_v2.h5')

img = cv2.imread(args.image)
img_resized = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)

pred = model.predict(np.array([img_resized]))

dogs = OrderedDict()
for dog, indice in class_indices.items():
    dogs[dog] = round(pred[0][indice], 4)
dogs = sorted(dogs.items(), key=lambda x:x[1], reverse=True)

for dog, proba in dogs[:5]:
    print(dog, proba)

if args.view:
	cv2.imshow("img", img)
	cv2.waitKey(0)