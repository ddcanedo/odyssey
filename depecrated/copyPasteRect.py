# 1. Load TIF image
# 2. Load image and labels
# 3. Get cropped sites
# 4. Apply random rotation to the cropped sites
# 5. Get 640x640 images from TIF which intersect LBR polygons
# 6. Copy paste cropped sites into those images inside polygons
# 7. Saves new images and labels
# 8. Return to step 2
# 9. After all images are used, return to step 1 if there are more TIF images

# TODO: attempt with polygons instead of bounding boxes

from tkinter import filedialog
import os
import sys
import csv
from shapely.geometry import Polygon
from shapely.wkt import loads
from PIL import Image
import rasterio
import random
import numpy as np

csv.field_size_limit(sys.maxsize)
Image.MAX_IMAGE_PIXELS = None

# Creates a dataset folder in YOLO format
def createDatasetDir(datasetPath, imagesPath, labelsPath, imagesTrainPath, imagesValPath, labelsTrainPath, labelsValPath):
	if not os.path.exists(datasetPath):
		os.makedirs(datasetPath)
		os.makedirs(imagesPath)
		os.makedirs(labelsPath)
		os.makedirs(imagesTrainPath)
		os.makedirs(imagesValPath)
		os.makedirs(labelsTrainPath)
		os.makedirs(labelsValPath)
	else:
		if not os.path.exists(imagesPath):
			os.makedirs(imagesPath)
			os.makedirs(imagesTrainPath)
			os.makedirs(imagesValPath)
		else:
			if not os.path.exists(imagesTrainPath):
				os.makedirs(imagesTrainPath)
			if not os.path.exists(imagesValPath):
				os.makedirs(imagesValPath)
			
		if not os.path.exists(labelsPath):
			os.makedirs(labelsPath)
			os.makedirs(labelsTrainPath)
			os.makedirs(labelsValPath)
		else:
			if not os.path.exists(labelsTrainPath):
				os.makedirs(labelsTrainPath)
			if not os.path.exists(labelsValPath):
				os.makedirs(labelsValPath)

# Converts coordinates from GIS reference to image pixels
def map(value, min1, max1, min2, max2):
	return (((value - min1) * (max2 - min2)) / (max1 - min1)) + min2

def intersection(bb, crop):
    return not (bb[1] < crop[0] or bb[0] > crop[1] or bb[3] < crop[2] or bb[2] > crop[3])

def convert2Pixels(annotations, region, width, height, resolution, xMinImg, yMinImg, xMaxImg, yMaxImg):

	bbs = []
	for coordinates in annotations[region]:

		cropExtent = coordinates.split("_")

		for bb in annotations[region][coordinates]:

			xMin = map(bb[0][0], 0, resolution, int(cropExtent[0]), int(cropExtent[1]))
			xMax = map(bb[0][2], 0, resolution, int(cropExtent[0]), int(cropExtent[1]))
			yMin = map(bb[0][1], 0, resolution, int(cropExtent[2]), int(cropExtent[3]))
			yMax = map(bb[0][3], 0, resolution, int(cropExtent[2]), int(cropExtent[3]))

			bbs.append((xMin, xMax, yMin, yMax, bb[1]))

	return bbs

# Return the LBR polygons that intersect this region
def LBRroi(polygons, bb):
	p = Polygon([(bb[0], bb[2]), (bb[0], bb[3]), (bb[1], bb[3]), (bb[1], bb[2])])
	intersection = []
	for polygon in polygons:
		if polygon.intersects(p):
			intersection.append(polygon.intersection(p))
	return intersection

# Checks if roi is inside polyong
def LBR(roi, bb):
	p = Polygon([(bb[0], bb[2]), (bb[0], bb[3]), (bb[1], bb[3]), (bb[1], bb[2])])
	for polygon in roi:
		if polygon.contains(p):
			return True
	return False

def containsBb(bbs, crop):
	for bb in bbs:
		if intersection(bb, crop):
			return True
	return False


def main():
	random.seed(1)

	datasetPath = filedialog.askdirectory(title = "Select the original dataset")

	if not os.path.isfile(datasetPath + "/paths.txt"):
		sys.exit("Paths to TIF images are missing.")


	print("Path to save the augmented dataset.")
	print("<If you want to save in a existing dataset, select that dataset folder path and skip the next step of naming it.>")
	augmentedDataset = filedialog.askdirectory(title = "Path to save the augmented dataset") + '/'
	print("\n")
	augmentedDataset += input('Augmented dataset folder name: ') + '/'
	print("\n")

	# Creates dataset folder in YOLOv5 format
	imagesPath = augmentedDataset + "images/"
	labelsPath = augmentedDataset + "labels/"
	imagesTrainPath = imagesPath + "train/"
	imagesValPath = imagesPath + "val/"
	labelsTrainPath = labelsPath + "train/"
	labelsValPath = labelsPath + "val/"

	createDatasetDir(augmentedDataset, imagesPath, labelsPath, imagesTrainPath, imagesValPath, labelsTrainPath, labelsValPath)

	# Get the paths to the TIF images
	pathFile = open(datasetPath + "/paths.txt", "r")
	images = pathFile.read().splitlines()
	pathFile.close()

	trainVal = ""
	while trainVal != "train" and trainVal != "val":
		trainVal = input("Augment training set or validation set? (train/val): ")
		imgSavePath = imagesPath+trainVal+"/"
		labelSavePath = labelsPath+trainVal+"/"
		previousSize = len(os.listdir(imgSavePath))
	print("\n")


	originalSize = len(os.listdir(datasetPath+"/LRM/images/"+trainVal))

	confirmation = ""
	print("Final dataset = TIF images ("+str(len(images))+") * dataset original size ("+str(originalSize)+ ") * X")
	while confirmation != "yes":
		imageAugmentations = 0
		while imageAugmentations <= 0:
			imageAugmentations = int(input("X: "))
		finalSize = len(images)*originalSize*imageAugmentations
		confirmation = input("Final dataset size is " + str(finalSize) + " maximum. Proceed? (yes/no): ")
	print("\n")

	objectAugmentations = 0
	while objectAugmentations <= 0:
		objectAugmentations = int(input("How many copy-paste augmentations per image: "))

	polygonsCsv = "Segmentation.csv"
	# Get the polygons from the LBR model
	polygons = []
	with open(polygonsCsv) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			p = loads(row["WKT"])
			polygons.append(p)

	for image in os.listdir(datasetPath + "/LRM/images/train/"): 
		img = Image.open(datasetPath + "/LRM/images/train/" + image)
		resolution = img.size[0]
		img.close()
		break


	# Get the annotations from the dataset and save them into a dictionary
	# which key is the region in which the object belongs
	annotations = {}
	for folder in os.listdir(datasetPath + "/LRM/labels/"): 
		for labels in os.listdir(datasetPath + "/LRM/labels/" + folder):
			label = open(datasetPath + "/LRM/labels/" + folder + "/" + labels, "r")

			region = labels.split("(")[0]

			#coordinates = labels.split("_")
			#coordinates[0] = coordinates[0].strip(coordinates[0][:coordinates[0].find("(")]).strip("(")
			#coordinates[len(coordinates)-1] = coordinates[len(coordinates)-1][:len(coordinates[len(coordinates)-1])-5]
			coordinates = labels.split("(")[1].split(")")[0]

			if region not in annotations:
				annotations[region] = {}

			annotations[region][coordinates] = []
			for line in label:
				classLabel = int(line.split(' ')[0])
				x = float(line.split(' ')[1])
				y = float(line.split(' ')[2])
				w = float(line.split(' ')[3])
				h = float(line.split(' ')[4])
				# Converts from YOLO to image pixels
				bb = (round((x-w/2)*resolution), round((y-h/2)*resolution), round((x+w/2)*resolution), round((y+h/2)*resolution))
				annotations[region][coordinates].append((bb,classLabel))

	f = open('test.csv', 'w')
	writer = csv.writer(f)
	writer.writerow(['WKT', 'Id'])
	geometric = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270, Image.TRANSPOSE, Image.TRANSVERSE]
	# Iterate over TIF images
	for image in images:
		# Load image
		img = Image.open(image)

		geoRef = rasterio.open(image)
		width, height = img.size

		# Parse corners of the image (GIS reference)
		xMinImg = geoRef.bounds[0]
		xMaxImg = geoRef.bounds[2]
		yMinImg = geoRef.bounds[1]
		yMaxImg = geoRef.bounds[3]

		loops = imageAugmentations*originalSize

		region = image.split("/")[-1].split(".")[0]
		bbs = convert2Pixels(annotations, region, width, height, resolution, xMinImg, yMinImg, xMaxImg, yMaxImg)



		# First generate a cropped image that does not contain annotated sites
		x = list(range(0, width-resolution))
		y = list(range(0, height-resolution))

		augmentations = 0
	
		while augmentations < loops and len(x)>0 and len(y)>0:
			print(augmentations)
			i = random.choice(x)
			j = random.choice(y)
			x.remove(i)
			y.remove(j)
			# Gets a region of interest
			crop = (i, i+resolution, j, j+resolution)

			if containsBb(bbs, crop) == False:

				croppedImg = img.crop((crop[0], crop[2], crop[1], crop[3]))

				# Maps the cropped image extent from pixels to real world coordinates
				xMin = map(crop[0], 0, width, xMinImg, xMaxImg)
				xMax = map(crop[1], 0, width, xMinImg, xMaxImg)
				yMax = map(crop[2], height, 0, yMinImg, yMaxImg)
				yMin = map(crop[3], height, 0, yMinImg, yMaxImg)
				coords = "("+ str(int(xMin)) + "_" + str(int(xMax)) + "_" + str(int(yMin)) + "_" + str(int(yMax)) + ")"

				# Return the LBR polygons that intersect this region
				roiPolygons = LBRroi(polygons, (xMin,xMax,yMin,yMax))

				# Generate a random location to copy-paste
				copies = 0
				copyBbs = []
				previousBbs = []
				if len(roiPolygons) > 0:
					xx = list(range(i, i+resolution))
					yy = list(range(j, i+resolution))

					while copies < objectAugmentations and len(xx)>0 and len(yy)>0:
						# Pick a random bb
						bb = random.choice(bbs)
						ii = random.choice(xx)
						jj = random.choice(yy)
						xx.remove(ii)
						yy.remove(jj)

						croppedBb = img.crop((bb[0],bb[2],bb[1],bb[3]))

						# Geometric
						g = random.randint(0, len(geometric)-1)
						croppedBb = croppedBb.transpose(geometric[g])

						# New width/height
						w, h = croppedBb.size

						croppedPosition = img.crop((ii,jj,ii+w,jj+h))
						croppedPosition = croppedPosition.convert("L")
						blackPixels = 0
						pixels = croppedPosition.getdata()
						for pixel in pixels:
							if pixel == 0:
								blackPixels+=1

						# Probably in an out of bounds region, so just ignore
						if blackPixels/len(pixels) >= 0.2:
							continue



						xMinBb = map(ii, 0, width, xMinImg, xMaxImg)
						xMaxBb = map(ii+w, 0, width, xMinImg, xMaxImg)
						yMaxBb = map(jj, height, 0, yMinImg, yMaxImg)
						yMinBb = map(jj+h, height, 0, yMinImg, yMaxImg)

						lbr = LBR(roiPolygons, (xMinBb,xMaxBb,yMinBb,yMaxBb))
						intersecting = containsBb(previousBbs, (ii,ii+w,jj,jj+h))
						

						if lbr == True and intersecting == False and len(xx):
							copyXmin = map(ii, crop[0], crop[1], 0, resolution)
							copyXmax = map(ii+w, crop[0], crop[1], 0, resolution)
							copyYmin = map(jj, crop[2], crop[3], 0, resolution)
							copyYmax = map(jj+h, crop[2], crop[3], 0, resolution)
							


							croppedImg.paste(croppedBb, (int(copyXmin),int(copyYmin)))
							copyBbs.append((copyXmin, copyXmax, copyYmin, copyYmax, bb[4]))
							previousBbs.append((ii,ii+w,jj,jj+h))
							copies += 1


			
							string = "MULTIPOLYGON (((" + str(xMinBb) + " " + str(yMinBb) + "," + str(xMinBb) + " " + str(yMaxBb) + "," + str(xMaxBb) + " " + str(yMaxBb) + "," + str(xMaxBb) + " " + str(yMinBb) + ")))"

					
							writer.writerow([string, bb[4]])


			
				# Save image and labels
				if copies > 0:
					augmentations += 1
					previousSize += 1
					imgName = imgSavePath + region + coords + str(previousSize) + ".png"
					
					croppedImg.save(imgName)

					# Writes the label file
					txtFile = open(labelSavePath + region + coords + str(previousSize) + ".txt", "a+")
					for bb in copyBbs:						
						# Maps the object in YOLO format
						centerX = (bb[0] + bb[1])/2.0
						centerX = map(centerX, 0, resolution, 0, 1)
						centerY = (bb[2] + bb[3])/2.0
						centerY = map(centerY, 0, resolution, 0, 1)
						w = (centerX - map(bb[0], 0, resolution, 0, 1)) * 2.0
						h = (centerY - map(bb[2], 0, resolution, 0, 1)) * 2.0

						# Writes/Appends the annotations to a text file that has the same name of the respective image
						txtFile.write(str(bb[4]) + " " + str(centerX) + " " + str(centerY) + " " +str(w)+ " "+ str(h) + "\n")

					txtFile.close()

		img.close()

	f.close()

if __name__ == "__main__":
	main()