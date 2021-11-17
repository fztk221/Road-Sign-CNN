import numpy as np
import pickle
import cv2
import pandas as pd
from preprocessing import preprocessing

# Parameters
model_file = "my_model_trained.p"
# some camera parameters
frameWidth = 640
frameHeight = 480
brightness = 140
threshold = 0.85
font = cv2.FONT_HERSHEY_TRIPLEX  # good looking font

# setting up camera
cap = cv2.VideoCapture(0)  # returns video from the first webcam
# setting camera parameters
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
fps = cap.get(cv2.CAP_PROP_FPS)  # getting frame rate

# loading in camera
pickle_in = open(model_file, "rb")  # reading in pickle file "rb"=read byte
model = pickle.load(pickle_in)

data = pd.read_csv('labels.csv')
class_names = data['Name']

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    # Displaying text on image
    cv2.putText(imgOriginal, "Class: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, "Probability: ", (20, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, "FPS: " + str(fps), (20, 470), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue = np.amax(predictions)
    # Displaying prediction values when probability exceeds threshold
    if probabilityValue > threshold:
        cv2.putText(imgOriginal, str(classIndex) + " " + str(class_names[int(classIndex)]), (120, 35), font, 0.5,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (255, 0, 0), 2,
                    cv2.LINE_AA)

    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
