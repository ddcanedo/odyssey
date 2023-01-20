# After creating a dataset with fusionDataset.py
#
# Runtime:
#
# [Select the original dataset]
#
# dataset/ 			<- This folder must be selected
# |-- DTMs/ 
# |-- LRMs/              	    
# |-- LAS/
# |-- paths.txt
#
#
#
# [Path to save the augmented dataset]
# Here you can choose a new path or select the path of the original dataset
# if you want the augmented images to be saved in the same folder
#
#
#
# [Augment training set or validation set? (train/val)]
# Type "train" to augment the training set, type "val" to augment the validation set
#
#
#
# [Final dataset = TIF images (Y) * dataset original size (Z) * X]
# Here the script calculates Y and Z, and expects you to input X (>= 1). This controls
# the final size of the augmented dataset
#
#
#
# [Final dataset size is K maximum. Proceed? (yes/no)]
# Here the script calculates the maximum dataset size based on the previous formula
# and asks the user for confirmation. Type "yes" to proceed.
#
#
#
# [How many copy-paste augmentations per image]
# Here the the script expects the user to input a number of copy-paste augmentations per image.
# For instance, if the user inputs 10, each image will have 10 copy-pasted objects maximum.


from tkinter import filedialog
import os
import cv2
import sys
import csv
from shapely.geometry import Polygon, box
from shapely.wkt import loads
from PIL import Image, ImageDraw, ImageFilter
import rasterio
import random
import numpy as np
import albumentations as A
import torch
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device

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

	polys = {}
	for coordinates in annotations[region]:

		cropExtent = coordinates.split("_")

		for poly in annotations[region][coordinates]:

			if poly[1] not in polys.keys():
				polys[poly[1]] = []

			cPoly = []
			for p in poly[0]:
				x = map(p[0], 0, resolution, int(cropExtent[0]), int(cropExtent[1]))
				y = map(p[1], 0, resolution, int(cropExtent[2]), int(cropExtent[3]))
				cPoly.append((x,y))

			#polys.append((cPoly, poly[1]))
			polys[poly[1]].append(cPoly)


	return polys

# Return the LBR polygons that intersect this region
def LBRroi(polygons, bb):
	p = Polygon([(bb[0], bb[2]), (bb[0], bb[3]), (bb[1], bb[3]), (bb[1], bb[2])])
	intersection = []
	for polygon in polygons:
		if polygon.intersects(p):
			intersection.append(polygon.intersection(p))
	return intersection

# Checks if roi is inside polygon
def LBR(roi, bb):
	p = Polygon([(bb[0], bb[2]), (bb[0], bb[3]), (bb[1], bb[3]), (bb[1], bb[2])])
	for polygon in roi:
		if polygon.contains(p):
			return True
	return False

def intersectsPoly(polys, bb):
	crop = Polygon([(bb[0], bb[2]), (bb[0], bb[3]), (bb[1], bb[3]), (bb[1], bb[2])])
	for poly in polys:
		p = Polygon(poly)
		if p.intersects(crop):
			return True
	return False

# Convert the detection to real world coordinates
def convert2GIS(xyxy, cropExtent, width, height, resolution, xMinImg, yMinImg, xMaxImg, yMaxImg):

	xMin = map(int(xyxy[0]), 0, resolution, cropExtent[0], cropExtent[1])
	xMax = map(int(xyxy[2]), 0, resolution, cropExtent[0], cropExtent[1])
	yMin = map(int(xyxy[3]), 0, resolution, cropExtent[2], cropExtent[3])
	yMax = map(int(xyxy[1]), 0, resolution, cropExtent[2], cropExtent[3])

	xMin = map(xMin, 0, width, xMinImg, xMaxImg)
	xMax = map(xMax, 0, width, xMinImg, xMaxImg)
	yMax = map(yMax, height, 0, yMinImg, yMaxImg)
	yMin = map(yMin, height, 0, yMinImg, yMaxImg)

	return (xMin, xMax, yMin, yMax)

def inference(croppedImg, dt, model, stride, imgsz, device, conf_thres, iou_thres, classes, agnostic_nms, max_det, crop, width, height, resolution, xMinImg, yMinImg, xMaxImg, yMaxImg, roiPolygons):
	# Run inference
	with dt[0]:
		im = croppedImg.convert('RGB')
		im = np.array(im)
		display = im.copy()
		im = letterbox(im, imgsz, stride=stride, auto=True)[0]
		# Convert
		im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW
		im = np.ascontiguousarray(im)
		im = torch.from_numpy(im).to(model.device)
		im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
		im /= 255  # 0 - 255 to 0.0 - 1.0
		if len(im.shape) == 3:
			im = im[None]  # expand for batch dim

	with dt[1]:
		pred = model(im, augment=False, visualize=False)

	# NMS
	with dt[2]:
		pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

	for x, det in enumerate(pred):
		if len(det):
			# Rescale boxes from img_size to im0 size
			det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], display.shape).round()

			for *xyxy, conf, cls in reversed(det):
				#cv2.rectangle(display, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,255), 1)
				#cv2.imshow('image', display)
				#cv2.waitKey()
				# Convert detection to real world coordinates
				GISbb = convert2GIS(xyxy, crop, width, height, resolution, xMinImg, yMinImg, xMaxImg, yMaxImg)
				
				# Chekcs if the detection is intersecting a LBR polygon
				lbr = LBR(roiPolygons, GISbb)
				if lbr:
					return True

	return False


def main():
	#random.seed(1)

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

	imgsz=(resolution, resolution)  # inference size (pixels)
	conf_thres=0.25  # confidence threshold
	iou_thres=0.45  # NMS IOU threshold
	max_det=1000  # maximum detections per image
	agnostic_nms=False
	classes=None
	bs = 1
	device = "0"
	device = select_device(device)

	# YOLO model
	weights = "da.pt"
	# Load YOLO model
	model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
	stride, names, pt = model.stride, model.names, model.pt
	imgsz = check_img_size(imgsz, s=stride)  # check image size
	model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
	seen, windows, dt = 0, [], (Profile(), Profile(), Profile())


	polyLabels = datasetPath + "/LRM/labels/" + trainVal + "Poly/"
	# Get the annotations from the dataset and save them into a dictionary
	# which key is the region in which the object belongs
	annotations = {}
	for labels in os.listdir(polyLabels):
		label = open(polyLabels + labels, "r")

		region = labels.split("(")[0]

		#coordinates = labels.split("_")
		#coordinates[0] = coordinates[0].strip(coordinates[0][:coordinates[0].find("(")]).strip("(")
		#coordinates[len(coordinates)-1] = coordinates[len(coordinates)-1][:len(coordinates[len(coordinates)-1])-5]
		coordinates = labels.split("(")[1].split(")")[0]

		if region not in annotations:
			annotations[region] = {}

		annotations[region][coordinates] = []
		for line in label:
			l = line.split(' ')
			classLabel = int(l[0])

			poly = []
			for i in range(1, len(l), 2):
				poly.append((float(l[i]), float(l[i+1])))

			annotations[region][coordinates].append((poly,classLabel))


	croppedSites = {}
	count = 0
	for image in images:
		region = image.split("/")[-1].split(".")[0]
		if region in annotations.keys():
			# Load image
			img = Image.open(image)

			geoRef = rasterio.open(image)
			width, height = img.size

			# Parse corners of the image (GIS reference)
			xMinImg = geoRef.bounds[0]
			xMaxImg = geoRef.bounds[2]
			yMinImg = geoRef.bounds[1]
			yMaxImg = geoRef.bounds[3]

			polys = convert2Pixels(annotations, region, width, height, resolution, xMinImg, yMinImg, xMaxImg, yMaxImg)

			for k in polys:
				if k not in croppedSites.keys():
					croppedSites[k] = []
				for poly in polys[k]:
					minX, minY, maxX, maxY  = Polygon(poly).bounds
					croppedPoly = img.crop((minX,minY,maxX,maxY))
					# Avoid using really tiny sites for the augmentations, since it's used smoothing later on
					if croppedPoly.width * croppedPoly.height >= 50:
						croppedSites[k].append((croppedPoly, poly))
						count += 1

			img.close()

	print("Sites stored:", count)
	polyClasses = list(croppedSites.keys())


	if trainVal == "train":
		remaining = "val"
	else:
		remaining = "train"

	polyLabels = datasetPath + "/LRM/labels/" + remaining + "Poly/"
	# Get the annotations from the dataset and save them into a dictionary
	# which key is the region in which the object belongs
	for labels in os.listdir(polyLabels):
		label = open(polyLabels + labels, "r")

		region = labels.split("(")[0]

		coordinates = labels.split("(")[1].split(")")[0]

		if region not in annotations:
			annotations[region] = {}

		annotations[region][coordinates] = []
		for line in label:
			l = line.split(' ')
			classLabel = int(l[0])

			poly = []
			for i in range(1, len(l), 2):
				poly.append((float(l[i]), float(l[i+1])))

			annotations[region][coordinates].append((poly,classLabel))



	f = open('test.csv', 'w')
	writer = csv.writer(f)
	writer.writerow(['WKT', 'Id'])
	geometric = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270, Image.TRANSPOSE, Image.TRANSVERSE]
	
	# Declares an augmentation pipeline
	transform = A.Compose([
		A.Flip(p=1),
	], bbox_params=A.BboxParams(format='pascal_voc',label_fields=['classLabels']))

	augLoops = round(imageAugmentations*originalSize*0.9)
	bgLoops = round(imageAugmentations*originalSize*0.1)
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


		region = image.split("/")[-1].split(".")[0]
		polys = {}
		if region in annotations.keys():
			polys = convert2Pixels(annotations, region, width, height, resolution, xMinImg, yMinImg, xMaxImg, yMaxImg)


		# First generate a cropped image that does not contain annotated sites
		x = list(range(0, width-resolution))
		y = list(range(0, height-resolution))

		augmentations = 0
		
		while augmentations < augLoops and len(x)>0 and len(y)>0:
			i = random.choice(x)
			j = random.choice(y)
			x.remove(i)
			y.remove(j)
			# Gets a region of interest
			crop = (i, i+resolution, j, j+resolution)

			flatList = []
			for subList in list(polys.values()):
				flatList.extend(subList)


			if intersectsPoly(flatList, (crop[0], crop[1], crop[2], crop[3])) == False:
			
				croppedImg = img.crop((crop[0], crop[2], crop[1], crop[3]))

				# Maps the cropped image extent from pixels to real world coordinates
				xMin = map(crop[0], 0, width, xMinImg, xMaxImg)
				xMax = map(crop[1], 0, width, xMinImg, xMaxImg)
				yMax = map(crop[2], height, 0, yMinImg, yMaxImg)
				yMin = map(crop[3], height, 0, yMinImg, yMaxImg)
				coords = "("+ str(int(xMin)) + "_" + str(int(xMax)) + "_" + str(int(yMin)) + "_" + str(int(yMax)) + ")"

				# Return the LBR polygons that intersect this region
				roiPolygons = LBRroi(polygons, (xMin,xMax,yMin,yMax))

				if inference(croppedImg, dt, model, stride, imgsz, device, conf_thres, iou_thres, classes, agnostic_nms, max_det, crop, width, height, resolution, xMinImg, yMinImg, xMaxImg, yMaxImg, roiPolygons):
					continue

				# Generate a random location to copy-paste
				copies = 0
				copyBbs = []
				copyLabels = []
				previousPolys = []
				if len(roiPolygons) > 0:
					
					xx = list(range(i, i+resolution))
					yy = list(range(j, i+resolution))

					while copies < objectAugmentations and len(xx)>0 and len(yy)>0:
						# restore list 
						if not polyClasses:
							polyClasses = list(croppedSites.keys())

						# Pick a random bb, class balanced
						polyClass = random.choice(polyClasses)
						croppedPoly, poly = random.choice(croppedSites[polyClass])
						ii = random.choice(xx)
						jj = random.choice(yy)
						xx.remove(ii)
						yy.remove(jj)
						
						minX, minY, maxX, maxY  = Polygon(poly).bounds

						# Geometric
						g = random.randint(0, len(geometric)-1)
						croppedPoly = croppedPoly.transpose(geometric[g])

						w,h = croppedPoly.size

						# Out of bounds
						if ii+w > width or jj+h > height:
							continue

						mask = Image.new("L", croppedPoly.size, 0)
						draw = ImageDraw.Draw(mask)

						cPoly = []
						for p in poly:
							cX = map(p[0], minX, maxX, 0, w)
							cY = map(p[1], minY, maxY, 0, h)
							cPoly.append((cX,cY))

						draw.polygon(cPoly, fill=255)

						mask = mask.filter(ImageFilter.SMOOTH)

						croppedPosition = img.crop((ii,jj,ii+w,jj+h))
						croppedPosition = croppedPosition.convert("L")
						blackPixels = 0
						pixels = croppedPosition.getdata()
						for pixel in pixels:
							if pixel == 0:
								blackPixels+=1

						# Probably in an region without data, so just ignore
						if blackPixels/len(pixels) >= 0.2:
							continue

						xMinBb = map(ii, 0, width, xMinImg, xMaxImg)
						xMaxBb = map(ii+w, 0, width, xMinImg, xMaxImg)
						yMaxBb = map(jj, height, 0, yMinImg, yMaxImg)
						yMinBb = map(jj+h, height, 0, yMinImg, yMaxImg)

						lbr = LBR(roiPolygons, (xMinBb,xMaxBb,yMinBb,yMaxBb))
						intersecting = intersectsPoly(previousPolys, (ii,ii+w, jj,jj+h))

						if lbr == True and intersecting == False and len(xx):

							previousPolys.append(((ii, jj), (ii, jj+h), (ii+w, jj+h), (ii+w, jj)))

							copyXmin = map(ii, crop[0], crop[1], 0, resolution)
							copyXmax = map(ii+w, crop[0], crop[1], 0, resolution)
							copyYmin = map(jj, crop[2], crop[3], 0, resolution)
							copyYmax = map(jj+h, crop[2], crop[3], 0, resolution)
							

							croppedImg.paste(croppedPoly, (int(copyXmin),int(copyYmin)), mask)
							copyBbs.append([copyXmin, copyYmin, copyXmax, copyYmax])
							copyLabels.append(polyClass)
							polyClasses.remove(polyClass)
							
							copies += 1
			
							string = "MULTIPOLYGON (((" + str(xMinBb) + " " + str(yMinBb) + "," + str(xMinBb) + " " + str(yMaxBb) + "," + str(xMaxBb) + " " + str(yMaxBb) + "," + str(xMaxBb) + " " + str(yMinBb) + ")))"

							writer.writerow([string, polyClass])

			
				# Save image and labels
				if copies > 0:
					#print(augmentations)
					augmentations+= 1
					previousSize += 1
					imgName = imgSavePath + region + coords + str(previousSize) + ".png"
					
					transformed = transform(image=np.array(croppedImg), bboxes=copyBbs, classLabels = copyLabels)
					transformedImage = transformed['image']
					transformedBboxes = transformed['bboxes']
					transformedClassLabels = transformed['classLabels']

					Image.fromarray(transformedImage).save(imgName)

					# Writes the label file
					txtFile = open(labelSavePath + region + coords + str(previousSize) + ".txt", "a+")
					for z in range(len(transformedBboxes)):						
						# Maps the object in YOLO format
						centerX = (transformedBboxes[z][0] + transformedBboxes[z][2])/2.0
						centerX = map(centerX, 0, resolution, 0, 1)
						centerY = (transformedBboxes[z][1] + transformedBboxes[z][3])/2.0
						centerY = map(centerY, 0, resolution, 0, 1)
						w = (centerX - map(transformedBboxes[z][0], 0, resolution, 0, 1)) * 2.0
						h = (centerY - map(transformedBboxes[z][1], 0, resolution, 0, 1)) * 2.0

						# Writes/Appends the annotations to a text file that has the same name of the respective image
						txtFile.write(str(transformedClassLabels[z]) + " " + str(centerX) + " " + str(centerY) + " " +str(w)+ " "+ str(h) + "\n")

					txtFile.close()


		# First generate a cropped image that does not contain annotated sites
		x = list(range(0, width-resolution))
		y = list(range(0, height-resolution))
		bgCount = 0 

		while bgCount < bgLoops:
			i = random.choice(x)
			j = random.choice(y)
			x.remove(i)
			y.remove(j)
			# Gets a region of interest
			crop = (i, i+resolution, j, j+resolution)


			flatList = []
			for subList in list(polys.values()):
				flatList.extend(subList)

			if intersectsPoly(flatList, (crop[0], crop[1], crop[2], crop[3])) == False:

				# Maps the cropped image extent from pixels to real world coordinates
				xMin = map(crop[0], 0, width, xMinImg, xMaxImg)
				xMax = map(crop[1], 0, width, xMinImg, xMaxImg)
				yMax = map(crop[2], height, 0, yMinImg, yMaxImg)
				yMin = map(crop[3], height, 0, yMinImg, yMaxImg)

				# Return the LBR polygons that intersect this region
				roiPolygons = LBRroi(polygons, (xMin,xMax,yMin,yMax))

				# 10% background images
				bgImg = img.crop((crop[0], crop[2], crop[1], crop[3]))


				if inference(bgImg, dt, model, stride, imgsz, device, conf_thres, iou_thres, classes, agnostic_nms, max_det, crop, width, height, resolution, xMinImg, yMinImg, xMaxImg, yMaxImg, roiPolygons):
					continue

				grayImg = bgImg.convert("L")
				blackPixels = 0
				pixels = grayImg.getdata()
				for pixel in pixels:
					if pixel == 0:
						blackPixels+=1

				# Probably in an region without data, so just ignore
				if blackPixels/len(pixels) < 0.2:
					bgCount += 1
					previousSize += 1
					# Maps the cropped image extent from pixels to real world coordinates
					xMin = map(crop[0], 0, width, xMinImg, xMaxImg)
					xMax = map(crop[1], 0, width, xMinImg, xMaxImg)
					yMax = map(crop[2], height, 0, yMinImg, yMaxImg)
					yMin = map(crop[3], height, 0, yMinImg, yMaxImg)
					coords = "("+ str(int(xMin)) + "_" + str(int(xMax)) + "_" + str(int(yMin)) + "_" + str(int(yMax)) + ")"
					imgName = imgSavePath + region + coords + str(previousSize) + ".png"
					
					transformed = transform(image=np.array(bgImg), bboxes=[], classLabels = [])
					transformedImage = transformed['image']

					Image.fromarray(transformedImage).save(imgName)
		
					#print(imgName)

		img.close()

	f.close()

if __name__ == "__main__":
	main()