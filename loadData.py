import numpy as np
import cv2
import os
import pickle_data as p

"""Script for loading in and Pickling Data"""

# Parameters
path = "myData"  # path to data folder
dataset_path = 'datasets/'  # path to location to save dataset

# import images
count = 0
images = []
classNo = []
myList = os.listdir(path)  # lists of directories within mydata directory
print("Total Classes Detected:", len(myList))  # should detect 43 classes (0-42)
num_of_classes = len(myList)
print("importing Classes.....")
for x in range(0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count))  # creates a list of images from a directory
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)  # reads in an image
        images.append(curImg)  # adds the image to the image list
        classNo.append(count)  # adds the class value to the corresponding spot in the classNo list
    print(count, end=" ")
    count += 1  # increments the count to move onto the next class
print(" ")
images = np.array(images)
classNo = np.array(classNo)

# Storing the sets for later use
file_names = ["images.p", "classNo.p"]
datasets = [images, classNo]
p.save_pickles(file_names, datasets, dataset_path)
