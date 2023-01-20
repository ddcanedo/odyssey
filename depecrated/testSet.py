import os
import shutil
import numpy as np
from tkinter import filedialog


folder = filedialog.askdirectory(title = "Select the dataset")

data = []

for folders in os.listdir(folder):
    data.append(folder + '/' + folders)
    if "test" not in os.listdir(folder + '/' + folders + '/images'):
        os.makedirs(folder + '/' + folders + "/images/test") 
    if "test" not in os.listdir(folder + '/' + folders + '/labels'):
        os.makedirs(folder + '/' + folders + "/labels/test") 


for img in os.listdir(data[0]+"/images/train"):
    if np.random.rand(1) < 0.2:
        for f in data:
            shutil.move(f+"/images/train/"+img, f+"/images/test/")
            shutil.move(f+"/labels/train/"+img.split('.')[0]+'.txt', f+"/labels/test/")