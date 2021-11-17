from matplotlib import pyplot as plt
import pickle_data as p
"""Script to view data distribution by class"""

# Parameters
file_names = ['images.p', 'classNo.p']
path = 'datasets/'  # path to data files
save_path= 'plots'


# loading in data and determining number of classes
datasets = p.load_in_pickles(file_names, path)
images = datasets[0]
classNo = datasets[1]
num_of_classes= len(set(classNo))

totals = []
class_labels_int = range(0, num_of_classes)
for i in class_labels_int:
    x_curr_class = images[classNo == i]
    totals.append(len(x_curr_class))
plt.bar(class_labels_int, totals)
plt.xlabel('Class Number')
plt.ylabel('Number of Images')
plt.title('Data Distribution by Class')
plt.show()


