# Author: Daniel Canedo
#
# Summary: Creates a usable dataset for YOLOv5 from annotated TIF images
# 
# Detailed Description:
# 1. Loads TIF images obtained through different sensors from a folder and their corresponding object annotation files (CSV)
# 2. Breaks the TIF images into smaller RGB images with uniform resolutions
# 3. Converts the object annotations (polygons) into bounding boxes
# 4. Converts the object annotations (GIS reference) to YOLO annotations and associates them with the respective RBG images
# 5. Saves the RBG images and respective YOLO annotations such that YOLOv5 can use them directly

import os
import sys
import math
import csv
import numpy as np 
from tkinter import filedialog
import rasterio
from PIL import Image
import random
import collections
import laspy
Image.MAX_IMAGE_PIXELS = None

# Creates dataset directories in YOLOv5 format if they do not exist yet in the same folder as this script
# TODO: cleaner implementation of this function?
def createDatasetDir(datasetPath, imagesPath, labelsPath, imagesTrainPath, imagesValPath, labelsTrainPath, labelsValPath):
	if not os.path.exists(datasetPath):
		os.makedirs(datasetPath)
		for i in range(len(imagesPath)):
			os.makedirs(imagesPath[i])
			os.makedirs(labelsPath[i])
		for i in range(len(imagesTrainPath)):
			os.makedirs(imagesTrainPath[i])
			os.makedirs(imagesValPath[i])
			os.makedirs(labelsTrainPath[i])
			os.makedirs(labelsValPath[i])
	else:
		for i in range(len(imagesPath)):
			if not os.path.exists(imagesPath[i]):
				os.makedirs(imagesPath[i])
				for x in range(len(imagesTrainPath)):
					os.makedirs(imagesTrainPath[x])
					os.makedirs(imagesValPath[x])
			else:
				for j in range(len(imagesTrainPath)):
					if not os.path.exists(imagesTrainPath[j]):
						os.makedirs(imagesTrainPath[j])
					if not os.path.exists(imagesValPath[j]):
						os.makedirs(imagesValPath[j])
			
			if not os.path.exists(labelsPath[i]):
				os.makedirs(labelsPath[i])
				for x in range(len(labelsTrainPath)):
					os.makedirs(labelsTrainPath[x])
					os.makedirs(labelsValPath[x])
			else:
				for j in range(len(labelsTrainPath)):
					if not os.path.exists(labelsTrainPath[j]):
						os.makedirs(labelsTrainPath[j])
					if not os.path.exists(labelsValPath[j]):
						os.makedirs(labelsValPath[j])


# Converts the polygons in the CSV annotations files to bounding boxes
def poly2bb(row, xMinImg, xMaxImg, yMaxImg, yMinImg):
	raw = row.split(",")
	raw[0] = raw[0].strip(raw[0][:raw[0].find('(')] + '(((')
	raw[len(raw)-1] = raw[len(raw)-1].strip(')))')
	xPoints = []
	yPoints = []
	for i in range(0, len(raw)):
	    xPoints.append(float(raw[i].split(" ")[0]))
	    yPoints.append(float(raw[i].split(" ")[1]))

	xMin = min(xPoints)
	xMax = max(xPoints)
	yMin = min(yPoints)
	yMax = max(yPoints)

	# The bounding box must be within the image limits
	if xMin < xMinImg:
		xMin = xMinImg
	if xMax > xMaxImg:
		xMax = xMaxImg
	if yMin < yMinImg:
		yMin = yMinImg
	if yMax > yMaxImg:
		yMax = yMaxImg

	return (xMin, xMax, yMin, yMax)


# Converts coordinates from GIS reference to image pixels
def map(value, min1, max1, min2, max2):
	return (((value - min1) * (max2 - min2)) / (max1 - min1)) + min2


# Returns a list with 100% visible objects and their respective labels
def checkVisibility(image, crop, processedObjects, bbs):
	visibleObjects = []
	objectLabels = []

	for bb in bbs:
		label = bbs[bb]

		# The object is 100% inside the cropped image
		if bb[0] >= crop[0] and bb[1] <= crop[1] and bb[2] >= crop[2] and bb[3] <= crop[3]:
			# The object was already processed
			if bb in processedObjects:
				return [], []
			else:
				visibleObjects.append(bb)
				objectLabels.append(label)
		# The object is 100% outside the cropped image
		elif bb[1] < crop[0] or bb[0] > crop[1] or bb[3] < crop[2] or bb[2] > crop[3]:
			continue
		# The object is partially visible
		else:
			return [], []

	# Update list of processed objects
	processedObjects.extend(visibleObjects)
	return visibleObjects, objectLabels


# Saves the corresponding point cloud to the cropped region
def pointCloudCrop(dataInfo, pointClouds, labels, imagesTrainPath, imagesValPath, labelsTrainPath, labelsValPath, xMinImg, xMaxImg, yMinImg, yMaxImg, width, height, resolution):
	print("Generating point clouds...")
	# Gets one cloud to later use its header to write an empty .las file
	tmp = ''
	for cloud in os.listdir(pointClouds):
		tmp = pointClouds + '/' + cloud
		break

	# Iterates over data to get paths for the point cloud and its label
	for key in dataInfo:
		labelName = os.path.split(key)[1].split('.')[0] + '.txt'
		imgName = os.path.split(key)[1].split('.')[0] + '.las'
		if 'train' in key:	
			cloudName = [path for path in imagesTrainPath if os.path.split(pointClouds)[1] in path][0] + imgName
			cloudLabel = [path for path in labelsTrainPath if os.path.split(pointClouds)[1] in path][0] + labelName
		else:
			cloudName = [path for path in imagesValPath if os.path.split(pointClouds)[1] in path][0] + imgName
			cloudLabel = [path for path in labelsValPath if os.path.split(pointClouds)[1] in path][0] + labelName

		# Creates empty .las file to later populate it with points
		with laspy.open(tmp) as f:
			w = laspy.open(cloudName, mode='w', header = f.header)
			w.close()

		count = 0
		# Opens the previously created .las file to append points
		with laspy.open(cloudName, mode = 'a') as w:
			# Iterates over the point clouds
			for cloud in os.listdir(pointClouds):
				with laspy.open(pointClouds + '/' + cloud) as f:
					las = f.read()
					# Get the extent of the cropped image 
					# Gets bounding box in 2D reference
					xMin = int(map(dataInfo[key][0][0], 0, width, xMinImg, xMaxImg))
					xMax = int(map(dataInfo[key][0][1], 0, width, xMinImg, xMaxImg))
					yMin = int(map(dataInfo[key][0][3], height, 0, yMinImg, yMaxImg))
					yMax = int(map(dataInfo[key][0][2], height, 0, yMinImg, yMaxImg))

					# Gets bounding box in GIS reference
					#xMin = int(map(dataInfo[key][1][i][0], 0, width, xMinImg, xMaxImg))
					#xMax = int(map(dataInfo[key][1][i][1], 0, width, xMinImg, xMaxImg))
					#yMax = int(map(dataInfo[key][1][i][2], height, 0, yMinImg, yMaxImg))
					#yMin = int(map(dataInfo[key][1][i][3], height, 0, yMinImg, yMaxImg))

					# Checks if there is an overlap with the cropped image and the point cloud
					if xMin <= f.header.x_max and xMax >= f.header.x_min and yMin <= f.header.y_max and yMax >= f.header.y_min:
						# Appends the points of the overlapping region to the previously created .las file
						          
						x, y = las.points[las.classification == 2].x.copy(), las.points[las.classification == 2].y.copy()
						mask = (x >= xMin) & (x <= xMax) & (y >= yMin) & (y <= yMax)
						roi = las.points[las.classification == 2][mask]
						w.append_points(roi)
						# Updates the previously created .las file header
						if count == 0:
							w.header.x_min = np.min(roi.x)
							w.header.y_min = np.min(roi.y)
							w.header.z_min = np.min(roi.z)
							w.header.x_max = np.max(roi.x)
							w.header.y_max = np.max(roi.y)
							w.header.z_max = np.max(roi.z)
						else:
							if w.header.x_min > np.min(roi.x):
								w.header.x_min = np.min(roi.x)
							if w.header.y_min > np.min(roi.y):
								w.header.y_min = np.min(roi.y)
							if w.header.z_min > np.min(roi.z):
								w.header.z_min = np.min(roi.z)
							if w.header.x_max < np.max(roi.x):
								w.header.x_max = np.max(roi.x)
							if w.header.y_max < np.max(roi.y):
								w.header.y_max = np.max(roi.y)
							if w.header.z_max < np.max(roi.z):
								w.header.z_max = np.max(roi.z)

						count += 1			

		# If .las file was not populated with points, deletes it
		if count == 0:
			os.remove(cloudName)				
		else:
			# Save labels kitti format
			with laspy.open(cloudName) as f:
				labelCount = 0
				txtFile = open(cloudLabel, "a+")
				las = f.read()
				for i in range(len(dataInfo[key][1])):
					# Gets bounding box in 2D reference
					xMinLabel = int(map(dataInfo[key][1][i][0], dataInfo[key][0][0], dataInfo[key][0][1], 0, resolution))
					xMaxLabel = int(map(dataInfo[key][1][i][1], dataInfo[key][0][0], dataInfo[key][0][1], 0, resolution))
					yMinLabel = int(map(dataInfo[key][1][i][2], dataInfo[key][0][2], dataInfo[key][0][3], 0, resolution))
					yMaxLabel = int(map(dataInfo[key][1][i][3], dataInfo[key][0][2], dataInfo[key][0][3], 0, resolution))

					# Gets bounding box in GIS reference
					xMin = int(map(dataInfo[key][1][i][0], 0, width, xMinImg, xMaxImg))
					xMax = int(map(dataInfo[key][1][i][1], 0, width, xMinImg, xMaxImg))
					yMax = int(map(dataInfo[key][1][i][2], height, 0, yMinImg, yMaxImg))
					yMin = int(map(dataInfo[key][1][i][3], height, 0, yMinImg, yMaxImg))
         
					
					if xMin <= f.header.x_max and xMax >= f.header.x_min and yMin <= f.header.y_max and yMax >= f.header.y_min:
						x, y = las.points.x.copy(), las.points.y.copy()
						# Gets points within the bounding box
						mask = (x >= xMin) & (x <= xMax) & (y >= yMin) & (y <= yMax)
						roi = las.points[las.classification == 2][mask]

						# 3D dimension: height, width, length 
						w = np.max(roi.x) - np.min(roi.x) 
						l = np.max(roi.y) - np.min(roi.y) 
						h = np.max(roi.z) - np.min(roi.z) 

						#print(str((xMin, xMax, yMax, yMin)) + ': ' + str(h))

						# Object location
						centerX = (np.max(roi.x) + np.min(roi.x))/2
						centerY = (np.max(roi.y) + np.min(roi.y))/2
						centerZ = (np.max(roi.z) + np.min(roi.z))/2

						# Write in kitti format
						txtFile.write(str(labels.index(dataInfo[key][2][i])) + " 0 0 0 " + str(xMinLabel) + " " + str(yMinLabel) + " " + str(xMaxLabel) + " " + str(yMaxLabel) + " " + str(h) + " " + str(w) + " " + str(l) + " " + str(centerX) + " " + str(centerY) + " " + str(centerZ) + " 0" + "\n")
						labelCount += 1

				txtFile.close()

			# If there are no labels, delete both .las and .txt files
			if labelCount == 0:
				os.remove(cloudLabel)
				os.remove(cloudName)
			else:
				print(cloudName.split('/')[-1] + " generated!")



# Saves the corresponding point cloud to the cropped region
def pointCloud(dataInfo, pointClouds, labels, imagesTrainPath, imagesValPath, labelsTrainPath, labelsValPath, xMinImg, xMaxImg, yMinImg, yMaxImg, width, height, resolution):
	print("Generating point clouds...")
	# Gets one cloud to later use its header to write an empty .las file
	tmp = ''
	for cloud in os.listdir(pointClouds):
		tmp = pointClouds + '/' + cloud
		break

	# Iterates over data to get paths for the point cloud and its label
	for key in dataInfo:
		labelName = os.path.split(key)[1].split('.')[0] + '.txt'
		imgName = os.path.split(key)[1].split('.')[0] + '.las'
		if 'train' in key:	
			cloudName = [path for path in imagesTrainPath if os.path.split(pointClouds)[1] in path][0] + imgName
		else:
			cloudName = [path for path in imagesValPath if os.path.split(pointClouds)[1] in path][0] + imgName

		# Creates empty .las file to later populate it with points
		clouds = {}
		with laspy.open(tmp) as f:
			for i in range(len(dataInfo[key][1])):
				c = cloudName.split('.')[0] + str(i) + '.las'
				clouds[c] = 0
				w = laspy.open(c, mode='w', header = f.header)
				w.close()

		# Opens the previously created .las file to append points
		# Iterates over the point clouds
		for cloud in os.listdir(pointClouds):
			with laspy.open(pointClouds + '/' + cloud) as f:
				las = f.read()
				for i in range(len(dataInfo[key][1])):
					# Gets bounding box in 2D reference
					xMinLabel = map(dataInfo[key][1][i][0], dataInfo[key][0][0], dataInfo[key][0][1], 0, resolution)
					xMaxLabel = map(dataInfo[key][1][i][1], dataInfo[key][0][0], dataInfo[key][0][1], 0, resolution)
					yMinLabel = map(dataInfo[key][1][i][2], dataInfo[key][0][2], dataInfo[key][0][3], 0, resolution)
					yMaxLabel = map(dataInfo[key][1][i][3], dataInfo[key][0][2], dataInfo[key][0][3], 0, resolution)

					# Gets bounding box in GIS reference
					xMin = map(dataInfo[key][1][i][0], 0, width, xMinImg, xMaxImg)
					xMax = map(dataInfo[key][1][i][1], 0, width, xMinImg, xMaxImg)
					yMax = map(dataInfo[key][1][i][2], height, 0, yMinImg, yMaxImg)
					yMin = map(dataInfo[key][1][i][3], height, 0, yMinImg, yMaxImg)
					# Checks if there is an overlap with the cropped image and the point cloud
					if xMin <= f.header.x_max and xMax >= f.header.x_min and yMin <= f.header.y_max and yMax >= f.header.y_min:
						# Appends the points of the overlapping region to the previously created .las file
						          
						x, y = las.points[las.classification == 2].x.copy(), las.points[las.classification == 2].y.copy()
						mask = (x >= xMin) & (x <= xMax) & (y >= yMin) & (y <= yMax)
						roi = las.points[las.classification == 2][mask]

						with laspy.open(cloudName.split('.')[0] + str(i) + '.las', mode = 'a') as w:
							w.append_points(roi)
							# Updates the previously created .las file header
							if clouds[cloudName.split('.')[0] + str(i) + '.las'] == 0:
								w.header.x_min = np.min(roi.x)
								w.header.y_min = np.min(roi.y)
								w.header.z_min = np.min(roi.z)
								w.header.x_max = np.max(roi.x)
								w.header.y_max = np.max(roi.y)
								w.header.z_max = np.max(roi.z)
							else:
								if w.header.x_min > np.min(roi.x):
									w.header.x_min = np.min(roi.x)
								if w.header.y_min > np.min(roi.y):
									w.header.y_min = np.min(roi.y)
								if w.header.z_min > np.min(roi.z):
									w.header.z_min = np.min(roi.z)
								if w.header.x_max < np.max(roi.x):
									w.header.x_max = np.max(roi.x)
								if w.header.y_max < np.max(roi.y):
									w.header.y_max = np.max(roi.y)
								if w.header.z_max < np.max(roi.z):
									w.header.z_max = np.max(roi.z)

						clouds[cloudName.split('.')[0] + str(i) + '.las'] += 1

		# If .las file was not populated with points, deletes it
		for c in clouds:
			with laspy.open(c) as f:
				las = f.read()
				if clouds[c] == 0:
					os.remove(c)				
				else:
					print(c + " generated!")



def main():
	random.seed(1)
	# Gets the path to the directory where the image data is located
	#
	# For example:
	#
	# Images/
	# |-- DTMs/
	# 	  |-- Image1.tif
	#	  |-- Image2.tif
	#     |-- Image3.tif
	#     |-- Image4.tif
	#
	# |-- LRMs/              <- This folder must be selected because it has the annotations!
	# 	  |-- Image1.tif
	#	  |-- Image1.csv
	# 	  |-- Image2.tif
	#	  |-- Image2.csv
	#
	# |-- ...
	folder = filedialog.askdirectory(title = "Select the folder with the annotations")

	# Saves all the images path to a list
	annotations = []
	for file in os.listdir(folder):
		if file.endswith(".csv"):
			annotations.append(folder + '/' + file)

	# Checks if the list is empty = no image was found in the selected folder
	if not annotations:
		sys.exit("No annotations found in the selected folder.")


	pointClouds = filedialog.askdirectory(title = "Select the folder with the point clouds")

	# Saves all the folders in the same directory as the previous selected folder, except the selected folder
	folders = []
	folders.append(pointClouds)
	for f in os.listdir(os.path.split(folder)[0]):
		folders.append(os.path.split(folder)[0] + '/' + f)



	# Create dataset folder in YOLOv5 format considering all the different types of images
	datasetPath = filedialog.askdirectory(title = "Path to save the dataset") + '/'
	datasetPath += input('Dataset folder name: ') + '/'
	imagesPath = []
	labelsPath = []
	imagesTrainPath = []
	imagesValPath = []
	labelsTrainPath = []
	labelsValPath = []
	for f in folders:
		imagesPath.append(datasetPath + os.path.split(f)[1] + '/' + 'images/')
		labelsPath.append(datasetPath + os.path.split(f)[1] + '/' + 'labels/')
	for p in imagesPath:
		imagesTrainPath.append(p + 'train/')
		imagesValPath.append(p + 'val/')
	for p in labelsPath:
		labelsTrainPath.append(p + 'train/')
		labelsValPath.append(p + 'val/')

	createDatasetDir(datasetPath, imagesPath, labelsPath, imagesTrainPath, imagesValPath, labelsTrainPath, labelsValPath)

	# Update folders list for later 
	folders = [path for path in folders if os.path.split(folder)[1] not in path]

	resolution = 0
	while resolution <= 0:
		resolution = int(input("Image resolution for the dataset (greater than 0): "))

	validationSize = 0
	while validationSize > 50 or validationSize < 10:
		validationSize = int(input("Validation set size [10, 50]%: "))
	
	labels = []
	print('Creating the dataset, please wait...')
	# Iterates over the images
	for annotation in annotations:

		processedObjects = []
		dataInfo = {}

		# Checks if there is a image associated with the annotation, they must have the same name!
		image = annotation.split('.')[0] + '.tif'
		if os.path.exists(image):
			print('Processing', image)
			# Load image
			img = Image.open(image)
			geoRef = rasterio.open(image)
			width, height = img.size
			print(img.size)

			# Parse corners of the image (GIS reference)
			#Left, Bottom, Right, Top
			xMinImg = geoRef.bounds[0]
			xMaxImg = geoRef.bounds[2]
			yMinImg = geoRef.bounds[1]
			yMaxImg = geoRef.bounds[3]

			# Checks if input resolution is bigger than width or height
			if resolution > min(width, height):
				resolution = min(width, height)

			bbs = {}
			# Open the CSV file containing the object annotations
			with open(annotation) as csvfile:
				reader = csv.DictReader(csvfile)
				# This considers that polygons are under the column name "WKT" and labels are under the column name "Id"
				for row in reader:

					# Converts polygon to bounding box
					bb = poly2bb(row['WKT'], xMinImg, xMaxImg, yMaxImg, yMinImg)
					# Maps coordinates from GIS reference to image pixels
					xMin = int(map(bb[0], xMinImg, xMaxImg, 0, width))
					xMax = int(map(bb[1], xMinImg, xMaxImg, 0, width))
					yMax = int(map(bb[2], yMinImg, yMaxImg, height, 0))
					yMin = int(map(bb[3], yMinImg, yMaxImg, height, 0))

					bbs[(xMin, xMax, yMin, yMax)] = row['Id']


			for bb in bbs:

				if bb not in processedObjects:
					# Randomizes a list of unique points covering a range around the object
					x = list(range(bb[1]-resolution//2, bb[0]+resolution//2))
					y = list(range(bb[3]-resolution//2, bb[2]+resolution//2))
					random.shuffle(x)
					random.shuffle(y)

					visibleObjects = []
					objectLabels = []
					crop = []

					# Iterates over the list of random points
					for i in x:
						if visibleObjects:
							break
						for j in y:
							# Gets a region of interest expanding the random point into a region of interest with a certain resolution
							crop = (i-resolution//2, i+resolution//2, j-resolution//2, j+resolution//2)
							# Checks if that region of interest only covers 100% visible objects
							visibleObjects, objectLabels = checkVisibility(image, crop, processedObjects, bbs)
							if visibleObjects:
								break

					# If we obtain a list of visible objects within a region of interest, we save it
					if visibleObjects:
						# Use the coordinates as the name of the image and text files
						coords = '('+ str(crop[0]) + '_' + str(crop[1]) + '_' + str(crop[2]) + '_' + str(crop[3]) + ')'

						# Training/Validation split
						if np.random.uniform(0,1) > validationSize/100:
							labelPath = labelsTrainPath
							imagePath = imagesTrainPath				
						else:
							labelPath = labelsValPath
							imagePath = imagesValPath


						# Writes the image
						croppedImg = img.crop((crop[0], crop[2], crop[1], crop[3]))
						imgPath = [path for path in imagePath if os.path.split(folder)[1] in path][0]
						imgName = imgPath + image.split('.')[0].split('/')[-1] + coords + '.png'
						croppedImg.save(imgName)
						print(image.split('.')[0].split('/')[-1] + coords + '.png')
						# Stores information to a dictionary for later use
						dataInfo[imgName] = [crop, visibleObjects, objectLabels]

						for label in labelPath:
							if label.split('/')[-4] != pointClouds.split('/')[-1]:
								txtFile = open(label + image.split('.')[0].split('/')[-1] + coords + ".txt", "a+")
								for i in range(len(visibleObjects)):						
									# Maps the object in YOLO format
									centerX = (visibleObjects[i][0] + visibleObjects[i][1])/2.0
									centerX = map(centerX, crop[0], crop[1], 0, 1)
									centerY = (visibleObjects[i][2] + visibleObjects[i][3])/2.0
									centerY = map(centerY, crop[2], crop[3], 0, 1)
									w  = (centerX - map(visibleObjects[i][0], crop[0], crop[1], 0, 1)) * 2.0
									h = (centerY - map(visibleObjects[i][2], crop[2], crop[3], 0, 1)) * 2.0

									# Saves unique labels to a list for later use the index as the class label (YOLO format)
									if objectLabels[i] not in labels:
										labels.append(objectLabels[i])

									# Writes/Appends the annotations to a text file that has the same name of the respective image
									txtFile.write(str(labels.index(objectLabels[i])) + " " + str(centerX) + " " + str(centerY) + " " +str(w)+ " "+ str(h) + "\n")

								txtFile.close()
			img.close()
		
			# Goes through the other image data using the stored cropping information to quickly crop them in the same positions
			for f in folders:
				if f != pointClouds:
					newImage = os.path.split(image)[1]
					if os.path.exists(f + '/' + newImage):
						print('Processing', f + '/' + newImage)
						# Load image
						img = Image.open(f + '/' + newImage)
					
						for key in dataInfo:
							croppedImg = img.crop((dataInfo[key][0][0], dataInfo[key][0][2], dataInfo[key][0][1], dataInfo[key][0][3]))
							imgName = os.path.split(key)[1]
							if 'train' in key:
								croppedImg.save([path for path in imagesTrainPath if os.path.split(f)[1] in path][0] + imgName)
							else:
								croppedImg.save([path for path in imagesValPath if os.path.split(f)[1] in path][0] + imgName)

						img.close()
				else:
					# Writes the point cloud corresponding to the cropped region
					pointCloud(dataInfo, pointClouds, labels, imagesTrainPath, imagesValPath, labelsTrainPath, labelsValPath, xMinImg, xMaxImg, yMinImg, yMaxImg, width, height, resolution)

					

if __name__ == "__main__":
	main()