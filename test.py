
import tensorflow as tf
from PIL import Image
import numpy as np
import imutils
import cv2
import os

model = tf.keras.models.load_model('./model/model.h5') #Load the trained model

list = os.listdir('./test/') #Get a list of file names in the test folder
list.sort()

for file in list:
    if file.endswith('jpg') or file.endswith('png'):
        image_path = './test/' + file
        image = Image.open(image_path)
        image = image.resize((64, 64)) #Change the size of the picture to meet the requirements of feeding the network
        image = np.array(image).reshape(-1, 64, 64, 1)  #Converted to numpy type data, normalized processing

        prediction = model.predict(image)
        final_prediction = [result.argmax() for result in prediction][0]
        a = np.max(prediction)
        print(a)
        print(['circle', 'no'][final_prediction])
        

        image = cv2.imread(image_path)
        image = imutils.resize(image, width=450)
        cv2.imshow('', image)
        cv2.waitKey(0)
         