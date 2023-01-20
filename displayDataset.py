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




	labelsPath = datasetPath + '/labels/' + trainval + '/'

	labels = []

	# Appends the labels' path to a list
	for file in (os.listdir(labelsPath)):
		if file.split('.')[-1] == 'txt':
			labels.append(labelsPath + file)

	objects = 0
	classes = {}
	# Iterates over the labels
	for label in labels:
		# Gets the path to the image associated to the labels
		aux = label.split('/')[-1]
		image = datasetPath + '/images/' + trainval + '/' + aux.split('.')[0] + '.png'


		# Opens both the labels and images
		f = open(label, "r")

		print(image.split("/")[-1])

		img = cv2.imread(image)
		
		# Extracts the bounding boxes from the labels, which are in YOLO format
		
		color = (0,0,255)
		for line in f:

			class_label = line.split(' ')[0]

			if class_label not in classes.keys():
				classes[class_label] = 0

			classes[class_label] += 1

			x = float(line.split(' ')[1])
			y = float(line.split(' ')[2])
			w = float(line.split(' ')[3])
			h = float(line.split(' ')[4])

			# Converts from YOLO to image pixels
			tl = (round((x-w/2)*img.shape[1]), round((y-h/2)*img.shape[0]))
			br = (round((x+w/2)*img.shape[1]), round((y+h/2)*img.shape[0]))


			cv2.rectangle(img, tl, br, color, 1)
				
		f.close()

		cv2.imshow('img', img)
		cv2.waitKey()

	
	print(classes)


if __name__ == "__main__":
	main()