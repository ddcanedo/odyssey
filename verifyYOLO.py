# Author: Daniel Canedo
#
# Summary: Verifies the dataset YOLO annotations
# 
# Detailed Description:
# 1. Opens the dataset folder
# 2. Searches for the labels and the associated images
# 3. Opens the labels (text files) and converts the annotations from YOLO format to image pixels
# 4. Displays the bounding boxes

import cv2
import os
import math
from tkinter import filedialog
import matplotlib.pyplot as plt

def main():
	# Asks the user to select the dataset folder
	# For example:
	#
	# dataset/					<- This folder must be selected!
	# |-- DTMs/
	# 	  |-- images
	#     	  |-- train
	#         |-- val
	#	  |-- labels
	#     	  |-- train
	#         |-- val
	#
	# |-- LRMs/              
	# 	  |-- images
	#     	  |-- train
	#         |-- val
	#	  |-- labels
	#     	  |-- train
	#         |-- val
	#
	# |-- ...
	datasetPath = filedialog.askdirectory(title = "Path to the dataset") + '/'

	trainval = ''
	while trainval != "train" and trainval != "val":
		trainval = input("train/val: ")

	# Saves all the folders in the same directory
	folders = []
	for f in os.listdir(datasetPath):
		folders.append(f)


	labelsPath = datasetPath + folders[0] + '/labels/' + trainval + '/'

	labels = []

	# Appends the labels' path to a list
	for file in (os.listdir(labelsPath)):
		if file.split('.')[-1] == 'txt':
			labels.append(labelsPath + file)

	objects = 0

	# Iterates over the labels
	for label in labels:
		# Gets the path to the image associated to the labels
		aux = label.split('/')[-1]
		images = []

		for f in folders:
			if(os.path.exists(datasetPath + f + '/images/' + trainval + '/' + aux.split('.')[0] + '.png')):
				images.append(datasetPath + f + '/images/' + trainval + '/' + aux.split('.')[0] + '.png')

		print(aux.split('.')[0] + '.png')

		# Opens both the labels and images
		f = open(label, "r")
		loaded_imgs = []
		for img in images:
			loaded_imgs.append(cv2.imread(img))
		
		# Extracts the bounding boxes from the labels, which are in YOLO format
		if loaded_imgs:
			for line in f:
				class_label = line.split(' ')[0]
				color = (255,0,0)
		
				objects+=1

				x = float(line.split(' ')[1])
				y = float(line.split(' ')[2])
				w = float(line.split(' ')[3])
				h = float(line.split(' ')[4])

				# Converts from YOLO to image pixels
				tl = (round((x-w/2)*loaded_imgs[0].shape[1]), round((y-h/2)*loaded_imgs[0].shape[0]))
				br = (round((x+w/2)*loaded_imgs[0].shape[1]), round((y+h/2)*loaded_imgs[0].shape[0]))

				for img in loaded_imgs:
					cv2.rectangle(img, tl, br, color, 1)
				

			# Displays the annotations
			fig = plt.figure(figsize=(14, 14))
			for i in range(1, len(loaded_imgs)+1):
				fig.add_subplot(1, len(loaded_imgs), i)
				plt.imshow(loaded_imgs[i-1])
			plt.show()

		
		f.close()

	print('objects: ', objects)

if __name__ == "__main__":
	main()