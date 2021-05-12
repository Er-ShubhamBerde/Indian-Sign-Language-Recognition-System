from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

from skimage.io import imread
from skimage.transform import resize
import skimage
from sklearn import metrics
import pickle
import time
import sys

DATASET_DIR = "./ISL DATASET/"
dimension=(64, 64)

def load_image_files(container_path):
    """
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset
    
    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to
        
    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

choice = input("Do you want to train SVM (Y/N)? (Save file will be overwritten if present) ")
if(choice != "Y"):
    sys.exit()

print("images loading, please wait...")

image_dataset = load_image_files(DATASET_DIR)
print("images loaded")

X_train, X_test, y_train, y_test = train_test_split(image_dataset.data, image_dataset.target, test_size=0.15,random_state=42)
print("training and testing set created")

model = svm.SVC(C=10, kernel="rbf", gamma=0.001, probability=True)
model.fit(X_train, y_train)
print("SVM model trained")

y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Classification report for - \n{}:\n{}\n".format(model, metrics.classification_report(y_test, y_pred)))

with open('ISL_SVM_SAVEFILE.pkl', 'wb') as f:
    pickle.dump(model, f)  
